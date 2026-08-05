#include "autoware_shoulder_pullover_manager/trajectory_planner.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace autoware::shoulder_pullover_manager
{

using autoware_perception_msgs::msg::PredictedObject;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_perception_msgs::msg::Shape;
using autoware_planning_msgs::msg::Trajectory;
using autoware_planning_msgs::msg::TrajectoryPoint;

namespace
{
double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

geometry_msgs::msg::Quaternion quaternionFromYaw(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

/// Solves the 3x3 linear system A*x=b via Cramer's rule -- small and fixed
/// in size, so a closed-form solve is simpler and dependency-free compared
/// to pulling in a linear-algebra library for this alone.
bool solve3x3(const double A[3][3], const double b[3], double x[3])
{
  const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                      A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  if (std::abs(det) < 1e-12) {
    return false;
  }

  auto detWithColumn = [&](int col) {
    double M[3][3];
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        M[r][c] = (c == col) ? b[r] : A[r][c];
      }
    }
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  };

  x[0] = detWithColumn(0) / det;
  x[1] = detWithColumn(1) / det;
  x[2] = detWithColumn(2) / det;
  return true;
}

/// Circular over-approximation of an object's radius from its Shape.
double objectRadius(const Shape & shape, double fallback)
{
  switch (shape.type) {
    case Shape::CYLINDER:
      return 0.5 * shape.dimensions.x;
    case Shape::BOUNDING_BOX:
    case Shape::POLYGON:
    default: {
      const double half_x = 0.5 * shape.dimensions.x;
      const double half_y = 0.5 * shape.dimensions.y;
      if (half_x <= 0.0 && half_y <= 0.0) {
        return fallback;
      }
      return std::hypot(half_x, half_y);
    }
  }
}
}  // namespace

double PullOverTrajectoryPlanner::Quintic::position(double t) const
{
  return c0 + c1 * t + c2 * t * t + c3 * t * t * t + c4 * t * t * t * t +
         c5 * t * t * t * t * t;
}

double PullOverTrajectoryPlanner::Quintic::velocity(double t) const
{
  return c1 + 2.0 * c2 * t + 3.0 * c3 * t * t + 4.0 * c4 * t * t * t + 5.0 * c5 * t * t * t * t;
}

double PullOverTrajectoryPlanner::Quintic::acceleration(double t) const
{
  return 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t * t + 20.0 * c5 * t * t * t;
}

PullOverTrajectoryPlanner::Quintic PullOverTrajectoryPlanner::solveQuintic(
  double q0, double v0, double a0, double qT, double vT, double aT, double duration)
{
  Quintic q;
  q.c0 = q0;
  q.c1 = v0;
  q.c2 = a0 / 2.0;

  const double T = duration;
  const double T2 = T * T;
  const double T3 = T2 * T;
  const double T4 = T3 * T;
  const double T5 = T4 * T;

  const double A[3][3] = {
    {T3, T4, T5}, {3.0 * T2, 4.0 * T3, 5.0 * T4}, {6.0 * T, 12.0 * T2, 20.0 * T3}};
  const double b[3] = {
    qT - q.c0 - q.c1 * T - q.c2 * T2, vT - q.c1 - 2.0 * q.c2 * T, aT - 2.0 * q.c2};

  double x[3] = {0.0, 0.0, 0.0};
  if (solve3x3(A, b, x)) {
    q.c3 = x[0];
    q.c4 = x[1];
    q.c5 = x[2];
  }
  // If the (extremely unlikely, T > 0 always) singular case is hit, the
  // trajectory degrades to a 2nd-order (constant-acceleration) motion --
  // still a valid, if suboptimal, candidate that the downstream constraint
  // check will evaluate on its own merits rather than silently accepting.
  return q;
}

PullOverTrajectoryPlanner::PullOverTrajectoryPlanner(const TrajectoryPlannerParams & params)
: params_(params)
{
}

std::optional<Trajectory> PullOverTrajectoryPlanner::buildCandidate(
  const KinematicState & start, const geometry_msgs::msg::Pose & goal_pose, double turn_radius,
  double duration, const rclcpp::Time & stamp) const
{
  // The maneuver's *shape*: a constant-radius Dubins connector, curvature
  // bounded by construction (see class docs for why this replaced two
  // independent Cartesian quintics). Geometry only depends on start/goal
  // pose and turn_radius, not on `duration` -- if it's infeasible here it's
  // infeasible for every duration this candidate loop will try at this
  // radius.
  const DubinsPath path(start.pose, goal_pose, turn_radius);
  if (!path.valid()) {
    return std::nullopt;
  }

  // Deliberately zero start/goal acceleration: injecting a noisy live
  // acceleration estimate as a hard boundary condition risks a
  // poorly-conditioned fit more than it helps -- see class-level docs.
  constexpr double kZeroAccel = 0.0;

  // *Speed* along that fixed shape: a scalar minimum-jerk quintic over arc
  // length s, from the vehicle's current speed to a small nonzero terminal
  // approach speed (keeps the profile well-defined right up to t=T; the
  // actual last sample below is still force-zeroed to satisfy the PID
  // longitudinal controller's need for an explicit stop point -- see the
  // framework report's control/validation contract section).
  const Quintic qs = solveQuintic(
    0.0, start.speed, kZeroAccel, path.length(), params_.terminal_approach_speed, kZeroAccel,
    duration);

  Trajectory trajectory;
  trajectory.header.stamp = stamp;
  trajectory.header.frame_id = "map";

  const int num_samples = std::max(2, static_cast<int>(std::round(duration / params_.dt)) + 1);
  trajectory.points.reserve(static_cast<std::size_t>(num_samples));

  for (int i = 0; i < num_samples; ++i) {
    const bool is_last = (i + 1 == num_samples);
    const double t = std::min(duration, i * params_.dt);
    // Clamped defensively; solveQuintic's boundary conditions already put
    // s(duration) == path.length() exactly, up to floating-point error.
    const double s = std::clamp(qs.position(t), 0.0, path.length());
    const DubinsPath::PathPoint path_point = path.pointAt(s);

    TrajectoryPoint point;
    point.pose.position.x = path_point.x;
    point.pose.position.y = path_point.y;
    point.pose.position.z = goal_pose.position.z;  // Ground height interpolated at the goal.
    // Exact analytic heading from the Dubins geometry -- unlike the old
    // per-axis fit, well-defined at any speed including v=0, no
    // position-delta finite-differencing or near-zero-speed workaround
    // needed here.
    point.pose.orientation = quaternionFromYaw(path_point.yaw);

    const double speed = std::max(0.0, qs.velocity(t));
    point.longitudinal_velocity_mps = static_cast<float>(is_last ? 0.0 : speed);
    point.lateral_velocity_mps = 0.0F;
    // Tangential acceleration is exact and direct now that s(t) *is* the
    // longitudinal coordinate -- no vx/vy projection needed.
    point.acceleration_mps2 = static_cast<float>(is_last ? 0.0 : qs.acceleration(t));

    const auto time_from_start = rclcpp::Duration::from_seconds(t);
    point.time_from_start.sec = static_cast<int32_t>(time_from_start.seconds());
    point.time_from_start.nanosec =
      static_cast<uint32_t>(time_from_start.nanoseconds() % 1000000000LL);

    trajectory.points.push_back(point);
  }

  return trajectory;
}

bool PullOverTrajectoryPlanner::satisfiesKinematicConstraints(
  const Trajectory & trajectory, std::string * failure_reason) const
{
  double previous_lateral_accel = 0.0;
  bool have_previous = false;

  for (std::size_t i = 0; i + 1 < trajectory.points.size(); ++i) {
    const double speed = trajectory.points[i].longitudinal_velocity_mps;

    // Below this speed, path curvature computed from finite differences
    // (dyaw/segment_length) is numerically ill-conditioned -- segment_length
    // itself shrinks toward zero near the end of any stopping maneuver, so
    // even a physically harmless heading adjustment reads as enormous
    // curvature purely from dividing by a near-zero distance. A
    // near-stationary vehicle can point its wheels sharply with zero real
    // dynamic risk (unlike at speed, where the same geometric curvature
    // really would be dangerous), so this is a deliberate physical gate,
    // not a loophole: skip curvature/lateral-accel/lateral-jerk here, but
    // still fall through to the longitudinal-accel check below, which
    // remains meaningful regardless of speed.
    if (speed >= params_.min_speed_for_curvature_check) {
      const auto & p0 = trajectory.points[i].pose.position;
      const auto & p1 = trajectory.points[i + 1].pose.position;
      const double segment_length = std::hypot(p1.x - p0.x, p1.y - p0.y);

      // Curvature via heading change over segment length -- consistent with
      // planning_validator's own finite-difference approach (see report).
      const double yaw0 = yawFromQuaternion(trajectory.points[i].pose.orientation);
      const double yaw1 = yawFromQuaternion(trajectory.points[i + 1].pose.orientation);
      double dyaw = yaw1 - yaw0;
      while (dyaw > M_PI) dyaw -= 2.0 * M_PI;
      while (dyaw < -M_PI) dyaw += 2.0 * M_PI;

      if (segment_length >= 1e-3) {
        const double curvature = dyaw / segment_length;
        if (std::abs(curvature) > params_.max_curvature) {
          if (failure_reason) {
            std::ostringstream oss;
            oss << "curvature " << curvature << " 1/m exceeds max " << params_.max_curvature
                << " 1/m at point " << i << " (speed=" << speed << " m/s)";
            *failure_reason = oss.str();
          }
          return false;
        }

        const double lateral_accel = speed * speed * curvature;
        if (std::abs(lateral_accel) > params_.max_lateral_accel) {
          if (failure_reason) {
            std::ostringstream oss;
            oss << "lateral accel " << lateral_accel << " m/s^2 exceeds max "
                << params_.max_lateral_accel << " m/s^2 at point " << i << " (speed=" << speed
                << " m/s)";
            *failure_reason = oss.str();
          }
          return false;
        }

        if (have_previous) {
          const double lateral_jerk = (lateral_accel - previous_lateral_accel) / params_.dt;
          if (std::abs(lateral_jerk) > params_.max_lateral_jerk) {
            if (failure_reason) {
              std::ostringstream oss;
              oss << "lateral jerk " << lateral_jerk << " m/s^3 exceeds max "
                  << params_.max_lateral_jerk << " m/s^3 at point " << i;
              *failure_reason = oss.str();
            }
            return false;
          }
        }
        previous_lateral_accel = lateral_accel;
        have_previous = true;
      }
    } else {
      // Below the speed gate: don't carry a stale lateral_accel across the
      // gap into whatever high-speed segment might follow (e.g. if the
      // trajectory somehow re-accelerates) -- the jerk check should not
      // compare across a skipped region.
      have_previous = false;
    }

    const double longitudinal_accel = trajectory.points[i].acceleration_mps2;
    if (
      longitudinal_accel > params_.max_longitudinal_accel ||
      longitudinal_accel < params_.min_longitudinal_accel) {
      if (failure_reason) {
        std::ostringstream oss;
        oss << "longitudinal accel " << longitudinal_accel << " m/s^2 outside ["
            << params_.min_longitudinal_accel << ", " << params_.max_longitudinal_accel
            << "] at point " << i;
        *failure_reason = oss.str();
      }
      return false;
    }
  }
  return true;
}

bool PullOverTrajectoryPlanner::isCollisionFree(
  const Trajectory & trajectory, const PredictedObjects & objects,
  const rclcpp::Time & trajectory_start_time, std::string * failure_reason) const
{
  for (const PredictedObject & object : objects.objects) {
    const double radius = objectRadius(object.shape, params_.default_object_radius);
    const double required_clearance = params_.ego_footprint_radius + radius + params_.collision_margin;

    // Prefer the highest-confidence predicted path; fall back to the
    // object's static initial pose if none was provided (e.g. a
    // not-yet-tracked detection).
    const auto & paths = object.kinematics.predicted_paths;
    const auto best_path_it = std::max_element(
      paths.begin(), paths.end(),
      [](const auto & a, const auto & b) { return a.confidence < b.confidence; });

    for (const auto & trajectory_point : trajectory.points) {
      geometry_msgs::msg::Point object_position;
      if (best_path_it != paths.end() && !best_path_it->path.empty()) {
        const double time_step_s = rclcpp::Duration(best_path_it->time_step).seconds();
        const double t = rclcpp::Duration(trajectory_point.time_from_start).seconds();
        const std::size_t index = time_step_s > 1e-3
                                     ? std::min(
                                         best_path_it->path.size() - 1,
                                         static_cast<std::size_t>(std::round(t / time_step_s)))
                                     : 0;
        object_position = best_path_it->path[index].position;
      } else {
        object_position = object.kinematics.initial_pose_with_covariance.pose.position;
      }

      const double distance = std::hypot(
        trajectory_point.pose.position.x - object_position.x,
        trajectory_point.pose.position.y - object_position.y);
      if (distance < required_clearance) {
        if (failure_reason) {
          std::ostringstream oss;
          oss << "would come within " << distance << "m of a tracked object (needs "
              << required_clearance << "m clearance: ego_radius=" << params_.ego_footprint_radius
              << " + object_radius=" << radius << " + margin=" << params_.collision_margin << ")";
          *failure_reason = oss.str();
        }
        return false;
      }
    }
  }
  (void)trajectory_start_time;  // Reserved for future absolute-time alignment refinements.
  return true;
}

std::optional<Trajectory> PullOverTrajectoryPlanner::plan(
  const KinematicState & start, const geometry_msgs::msg::Pose & goal_pose,
  const PredictedObjects & objects, const rclcpp::Time & stamp, std::string * failure_reason) const
{
  const double distance = std::hypot(
    goal_pose.position.x - start.pose.position.x, goal_pose.position.y - start.pose.position.y);
  const double average_speed_estimate = std::max(start.speed, 1.0) / 1.5;
  const double initial_guess =
    std::clamp(distance / average_speed_estimate, params_.min_duration, params_.max_duration);

  // Logged on every failure below -- a large mismatch here, relative to
  // `distance`, is a direct signal that the goal requires a turn tighter
  // than a single quintic segment (or this vehicle) can deliver, as opposed
  // to a threshold-tuning problem. Not used in any feasibility decision.
  double start_yaw = yawFromQuaternion(start.pose.orientation);
  double goal_yaw = yawFromQuaternion(goal_pose.orientation);
  double heading_delta = goal_yaw - start_yaw;
  while (heading_delta > M_PI) heading_delta -= 2.0 * M_PI;
  while (heading_delta < -M_PI) heading_delta += 2.0 * M_PI;

  // Outer loop: try increasingly wide Dubins turn radii, tightest first.
  // Inner loop: for each radius, try increasing durations starting from the
  // distance-based estimate. Both dimensions genuinely matter and neither
  // alone reliably finds a feasible candidate -- see
  // path_turn_radius_step's docs for why radius isn't just fixed at the
  // tightest safe value.
  for (double turn_radius = params_.min_path_turn_radius;
       turn_radius <= params_.max_path_turn_radius;
       turn_radius += params_.path_turn_radius_step) {
    for (double duration = initial_guess; duration <= params_.max_duration;
         duration += params_.duration_step) {
      auto candidate = buildCandidate(start, goal_pose, turn_radius, duration, stamp);
      if (!candidate.has_value()) {
        continue;
      }
      std::string reason;
      if (!satisfiesKinematicConstraints(*candidate, &reason)) {
        if (failure_reason) {
          std::ostringstream oss;
          oss << "radius=" << turn_radius << "m duration=" << duration << "s: " << reason
              << " [start=(" << start.pose.position.x << "," << start.pose.position.y << ","
              << start_yaw << "rad,v=" << start.speed << ") goal=(" << goal_pose.position.x << ","
              << goal_pose.position.y << "," << goal_yaw << "rad) heading_delta=" << heading_delta
              << "rad distance=" << distance << "m]";
          *failure_reason = oss.str();
        }
        continue;
      }
      if (!isCollisionFree(*candidate, objects, stamp, &reason)) {
        if (failure_reason) {
          std::ostringstream oss;
          oss << "radius=" << turn_radius << "m duration=" << duration << "s: " << reason;
          *failure_reason = oss.str();
        }
        continue;
      }
      return candidate;
    }
  }
  return std::nullopt;
}

}  // namespace autoware::shoulder_pullover_manager
