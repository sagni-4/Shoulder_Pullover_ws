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
  const KinematicState & start, const PathShape & shape,
  const geometry_msgs::msg::Pose & goal_pose, double duration, const rclcpp::Time & stamp) const
{
  // Terminal acceleration is still hardcoded zero: each individual candidate
  // should settle smoothly by its own end regardless of how it started, and
  // that boundary is never re-examined mid-flight the way the start one is.
  constexpr double kZeroAccel = 0.0;

  // Start acceleration comes from `start.accel`, NOT a live sensor reading
  // (see KinematicState::accel's docs) -- a raw IMU/odometry-derived
  // acceleration estimate was deliberately rejected as a hard boundary
  // condition (too noisy, risks a poorly-conditioned fit). But hardcoding
  // it to zero on *every* 100ms replanning cycle turned out to have its own,
  // worse failure mode, found live 2026-08-10: a minimum-jerk profile
  // starting from a0=0 necessarily has near-zero acceleration for the first
  // slice of its own duration by construction, and since a fresh multi-second
  // candidate is solved from scratch every single cycle, the vehicle only
  // ever lives through that first, deliberately-gentlest slice before it's
  // discarded and replaced -- the larger acceleration/deceleration any given
  // candidate *would* command several seconds in is never reached in
  // practice. Verified live: ego stuck at exactly 0.0 m/s for 480+
  // consecutive seconds, 4.6m short of goal, while a validly-shaped
  // accelerate-then-decelerate trajectory was successfully replanned and
  // published every cycle the entire time. Fix: the caller (see
  // PullOverManagerNode::seedAccelerationFromLastTrajectory) passes in the
  // *previously-commanded* trajectory's own predicted acceleration at "now"
  // -- a model-based warm start, not a new noise source -- so real
  // acceleration can actually accumulate across cycles instead of
  // re-zeroing every time.
  //
  // *Speed* along the (already-built, already-validated) fixed shape: a
  // scalar minimum-jerk quintic over arc length s, from the vehicle's
  // current speed to a small nonzero terminal approach speed (keeps the
  // profile well-defined right up to t=T; the actual last sample below is
  // still force-zeroed to satisfy the PID longitudinal controller's need
  // for an explicit stop point -- see the framework report's
  // control/validation contract section).
  const Quintic qs = solveQuintic(
    0.0, start.speed, start.accel, shape.length(), params_.terminal_approach_speed, kZeroAccel,
    duration);

  Trajectory trajectory;
  trajectory.header.stamp = stamp;
  trajectory.header.frame_id = "map";

  const int num_samples = std::max(2, static_cast<int>(std::round(duration / params_.dt)) + 1);
  trajectory.points.reserve(static_cast<std::size_t>(num_samples));

  // Live-verified crash (2026-08-10): the [0, shape.length()] clamp below is
  // only meant as floating-point-noise defense (see the comment on it), but
  // with a genuinely nonzero start.accel (see the warm-start docs above) a
  // quintic solved from a small v0 and a meaningfully negative a0 can
  // legitimately dip *below* s=0 for real, for more than one sample near
  // t=0 -- not noise. Clamping those all to exactly 0.0 silently produces
  // several consecutive trajectory points at the identical position, which
  // is fine for this planner's own checks but crashed a downstream Autoware
  // component outright (autoware_interpolation:
  // `std::invalid_argument("Either base_keys or query_keys is not
  // sorted.")`, uncaught, took down the whole /control/control_container --
  // vehicle_cmd_gate and the actual controller included -- mid-maneuver).
  // Physically, arc length along a forward path should never run backward
  // or stall mid-maneuver like this anyway (that's what kDecelerating is
  // for), so reject any candidate whose *raw, unclamped* position ever
  // drops meaningfully below 0 rather than silently repairing it -- the
  // existing duration search above already tries the next candidate.
  constexpr double kMinRawPosition = -1e-3;
  for (int i = 0; i < num_samples; ++i) {
    const double t_check = std::min(duration, i * params_.dt);
    if (qs.position(t_check) < kMinRawPosition) {
      return std::nullopt;
    }
  }

  // Second, independent guard for the SAME downstream crash (live-verified
  // recurring 2026-08-12, after the reactive controller-departure fix in
  // PullOverManagerNode finally let a real v0=0 departure happen for the
  // first time -- this exact non-strictly-increasing-arc-length input had
  // never actually reached a downstream Autoware component before that,
  // since the vehicle never used to depart at all). The negative-dip check
  // above only catches a meaningfully *negative* raw position from a
  // nonzero start acceleration; it says nothing about a genuine v0=0, a0=0
  // start, whose quintic is flat (zero position, slope, and curvature) at
  // t=0 by construction. All the downstream check
  // (autoware_interpolation::validateKeys, via isIncreasing) actually
  // requires is strict `>` between consecutive keys -- no minimum step size.
  // An earlier version of this guard used 1e-4 (0.1mm) as a "trust this as
  // real progress" heuristic, but that's larger than a genuine v0=0,a0=0
  // quintic's first ~100ms sample for any but the shortest candidate
  // durations (its leading term is cubic in t from a standing start), so it
  // was silently discarding *every* long-duration candidate for a true
  // full-stop departure -- live-verified 2026-08-12: the spiral shape solved
  // fine but never produced a single candidate across its whole duration
  // sweep once the vehicle was genuinely at v=0,a=0, which is exactly the
  // state the reactive controller-departure override is meant to unstick.
  // Use an epsilon just above floating-point noise (arc lengths here are
  // O(1-100m), so ~1e-9 is ~1e5x the double-precision noise floor) instead --
  // still rejects genuine clamped-duplicate ties, no longer rejects real
  // sub-millimeter forward creep.
  constexpr double kMinArcLengthStep = 1e-9;
  double previous_s = std::clamp(qs.position(0.0), 0.0, shape.length());
  for (int i = 1; i < num_samples; ++i) {
    const double t_check = std::min(duration, i * params_.dt);
    const double s = std::clamp(qs.position(t_check), 0.0, shape.length());
    if (s - previous_s < kMinArcLengthStep) {
      return std::nullopt;
    }
    previous_s = s;
  }

  for (int i = 0; i < num_samples; ++i) {
    const bool is_last = (i + 1 == num_samples);
    const double t = std::min(duration, i * params_.dt);
    // Clamped defensively for floating-point noise only now that the
    // meaningful-dip case above is rejected outright -- solveQuintic's
    // boundary conditions already put s(duration) == shape.length()
    // exactly, up to floating-point error.
    const double s = std::clamp(qs.position(t), 0.0, shape.length());
    const PathPoint path_point = shape.pointAt(s);

    TrajectoryPoint point;
    point.pose.position.x = path_point.x;
    point.pose.position.y = path_point.y;
    point.pose.position.z = goal_pose.position.z;  // Ground height interpolated at the goal.
    // Exact analytic heading from the shape geometry -- unlike the old
    // per-axis fit, well-defined at any speed including v=0, no
    // position-delta finite-differencing or near-zero-speed workaround
    // needed here.
    point.pose.orientation = quaternionFromYaw(path_point.yaw);

    // Floor only the *published reference*, never the true v0 boundary
    // condition the quintic itself was solved with -- see
    // min_departure_reference_speed's docs for why a genuine v0=0 sample
    // needs this to avoid a mutual deadlock with stock Autoware's own
    // control stack. Gated by remaining arc length (see
    // final_approach_distance's docs): only while genuinely far from the
    // goal, so the last final_approach_distance meters can still decelerate
    // smoothly all the way to a real stop instead of being artificially
    // held above the floor right up to the last sample.
    const double speed = std::max(0.0, qs.velocity(t));
    const double remaining_arc_length = shape.length() - s;
    const double departure_floor = remaining_arc_length > params_.final_approach_distance
                                      ? params_.min_departure_reference_speed
                                      : 0.0;
    point.longitudinal_velocity_mps =
      static_cast<float>(is_last ? 0.0 : std::max(speed, departure_floor));
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

    // Unconditional (not speed-gated like the checks below) -- this is the
    // check that catches the speed itself running away, so it must never be
    // skippable. See max_speed's docs for the live near-miss that motivated
    // this: a receding-horizon quintic solved from a growing v0 can overshoot
    // well above both its own boundary speeds in the middle samples.
    if (speed > params_.max_speed) {
      if (failure_reason) {
        std::ostringstream oss;
        oss << "speed " << speed << " m/s exceeds max " << params_.max_speed << " m/s at point "
            << i;
        *failure_reason = oss.str();
      }
      return false;
    }

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
  const PredictedObjects & objects, const rclcpp::Time & stamp, std::string * failure_reason,
  bool * blocked_by_traffic, const ShapeHint * preferred_hint, ShapeHint * used_hint) const
{
  if (blocked_by_traffic) {
    *blocked_by_traffic = false;
  }
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

  // Shared by both shape attempts below: for a given (already-built,
  // already-validated) shape, try increasing durations starting from the
  // distance-based estimate, returning the first feasible, collision-free
  // candidate. `shape_label` is just for the reason string. Writes into
  // `out_reason` (a *per-attempt* variable the caller controls) rather
  // than the outer failure_reason directly -- both the spiral and the
  // Dubins fallback call this, and if the caller always wrote straight
  // into failure_reason, whichever ran second would silently erase the
  // first one's diagnosis even when the *first* one is the interesting
  // one (e.g. spiral shape valid, only its own duration search failed --
  // live-tested, see project memory 2026-08-05).
  // Single-candidate check, shared by the duration sweep below and the
  // preferred_hint fast path -- factored out so both go through identical
  // logic rather than risking the two drifting apart.
  const auto tryOneCandidate = [&](
                                  const PathShape & shape, double duration,
                                  const std::string & shape_label,
                                  std::string * out_reason) -> std::optional<Trajectory> {
    auto candidate = buildCandidate(start, shape, goal_pose, duration, stamp);
    if (!candidate.has_value()) {
      return std::nullopt;
    }
    std::string reason;
    if (!satisfiesKinematicConstraints(*candidate, &reason)) {
      if (out_reason) {
        std::ostringstream oss;
        oss << shape_label << " duration=" << duration << "s: " << reason << " [start=("
            << start.pose.position.x << "," << start.pose.position.y << "," << start_yaw
            << "rad,v=" << start.speed << ") goal=(" << goal_pose.position.x << ","
            << goal_pose.position.y << "," << goal_yaw << "rad) heading_delta=" << heading_delta
            << "rad distance=" << distance << "m]";
        *out_reason = oss.str();
      }
      return std::nullopt;
    }
    if (!isCollisionFree(*candidate, objects, stamp, &reason)) {
      if (out_reason) {
        std::ostringstream oss;
        oss << shape_label << " duration=" << duration << "s: " << reason;
        *out_reason = oss.str();
      }
      if (blocked_by_traffic) {
        *blocked_by_traffic = true;
      }
      return std::nullopt;
    }
    // A real success -- matches the doc comment's contract of leaving
    // blocked_by_traffic false (not just unset) whenever plan() succeeds,
    // even if an earlier duration/radius attempt had set it true.
    if (blocked_by_traffic) {
      *blocked_by_traffic = false;
    }
    return candidate;
  };

  const auto tryDurations = [&](
                              const PathShape & shape, const std::string & shape_label,
                              std::string * out_reason,
                              double * out_duration) -> std::optional<Trajectory> {
    for (double duration = initial_guess; duration <= params_.max_duration;
         duration += params_.duration_step) {
      if (auto candidate = tryOneCandidate(shape, duration, shape_label, out_reason);
          candidate.has_value()) {
        if (out_duration) {
          *out_duration = duration;
        }
        return candidate;
      }
    }
    return std::nullopt;
  };

  // Attempt 0 (fast path, only if the caller passed one): retry *exactly*
  // last cycle's successful shape+duration before searching for anything
  // new -- see ShapeHint's docs for why this matters (reference-trajectory
  // stability across the receding-horizon replanning cycle, not just goal
  // stability). Deliberately a single tryOneCandidate call, not a sweep --
  // if this exact combination no longer works, fall through to the normal
  // fresh search below rather than searching *around* it.
  if (preferred_hint) {
    const CurvatureSpiralPath preferred_spiral(start.pose, goal_pose, params_.max_curvature);
    const DubinsPath preferred_dubins(start.pose, goal_pose, preferred_hint->turn_radius);
    const bool preferred_shape_valid =
      preferred_hint->use_spiral ? preferred_spiral.valid() : preferred_dubins.valid();
    if (preferred_shape_valid) {
      const PathShape & preferred_shape =
        preferred_hint->use_spiral ? static_cast<const PathShape &>(preferred_spiral)
                                    : static_cast<const PathShape &>(preferred_dubins);
      if (auto candidate = tryOneCandidate(
            preferred_shape, preferred_hint->duration, "preferred (repeated)", nullptr);
          candidate.has_value()) {
        if (used_hint) {
          *used_hint = *preferred_hint;
        }
        return candidate;
      }
    }
  }

  // Attempt 1 (preferred): variable-curvature spiral. A single shape, no
  // radius-style search needed -- see CurvatureSpiralPath's class docs for
  // why the Newton solve has one dominant solution rather than a family to
  // search over.
  std::string spiral_shape_diagnostic;
  const CurvatureSpiralPath spiral(
    start.pose, goal_pose, params_.max_curvature, &spiral_shape_diagnostic);
  std::string spiral_duration_reason;  // Only set if the shape was valid but no duration worked.
  double spiral_duration = 0.0;
  if (spiral.valid()) {
    if (auto candidate = tryDurations(spiral, "spiral", &spiral_duration_reason, &spiral_duration);
        candidate.has_value()) {
      if (used_hint) {
        *used_hint = ShapeHint{true, 0.0, spiral_duration};
      }
      return candidate;
    }
  }

  // Attempt 2 (fallback): constant-radius Dubins, searched tight to wide --
  // see path_turn_radius_step's docs for why radius isn't just fixed at
  // the tightest safe value.
  std::string dubins_reason;
  for (double turn_radius = params_.min_path_turn_radius;
       turn_radius <= params_.max_path_turn_radius;
       turn_radius += params_.path_turn_radius_step) {
    const DubinsPath path(start.pose, goal_pose, turn_radius);
    if (!path.valid()) {
      continue;
    }
    std::ostringstream label;
    label << "dubins radius=" << turn_radius << "m";
    double dubins_duration = 0.0;
    if (auto candidate = tryDurations(path, label.str(), &dubins_reason, &dubins_duration);
        candidate.has_value()) {
      if (used_hint) {
        *used_hint = ShapeHint{false, turn_radius, dubins_duration};
      }
      return candidate;
    }
  }

  // Both shapes exhausted -- report the full chain rather than whichever
  // ran last. This is exactly the kind of information that turned out to
  // matter live: "spiral shape was fine, only its own duration search
  // failed" and "spiral never even converged" look identical if only the
  // Dubins fallback's reason survives -- see project memory for how much
  // offline re-derivation that ambiguity cost before this was added.
  if (failure_reason) {
    std::ostringstream oss;
    oss << spiral_shape_diagnostic;
    if (!spiral_duration_reason.empty()) {
      oss << " | " << spiral_duration_reason;
    }
    oss << " | then " << dubins_reason;
    *failure_reason = oss.str();
  }
  return std::nullopt;
}

}  // namespace autoware::shoulder_pullover_manager
