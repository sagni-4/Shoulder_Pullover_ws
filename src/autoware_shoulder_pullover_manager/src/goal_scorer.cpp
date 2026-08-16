#include "autoware_shoulder_pullover_manager/goal_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autoware::shoulder_pullover_manager
{

namespace
{
/// Planar (x, y) Euclidean distance between two poses. All geometry in this
/// class operates in the map plane -- z is ignored, matching how the rest of
/// the behavior-planning stack treats lanelet-following maneuvers.
double planarDistance(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  return std::hypot(dx, dy);
}

/// Yaw angle from a geometry_msgs quaternion, computed directly rather than
/// pulling in a tf2 dependency for a single scalar.
double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

/// Inverse of yawFromQuaternion -- a planar (roll=pitch=0) orientation.
geometry_msgs::msg::Quaternion quaternionFromYaw(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw / 2.0);
  q.w = std::cos(yaw / 2.0);
  return q;
}
}  // namespace

GoalScorer::GoalScorer(const GoalScorerParams & params) : params_(params) {}

double GoalScorer::estimateCurvature(const nav_msgs::msg::Path & path, std::size_t index)
{
  if (index == 0 || index + 1 >= path.poses.size()) {
    return 0.0;
  }
  const auto & p0 = path.poses[index - 1].pose.position;
  const auto & p1 = path.poses[index].pose.position;
  const auto & p2 = path.poses[index + 1].pose.position;

  const double a = planarDistance(p0, p1);
  const double b = planarDistance(p1, p2);
  const double c = planarDistance(p0, p2);
  if (a < 1e-6 || b < 1e-6 || c < 1e-6) {
    return 0.0;
  }

  // Twice the signed area of the triangle (p0, p1, p2) via the 2D cross
  // product -- the standard Menger-curvature construction.
  const double cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
  const double area2 = std::abs(cross);
  if (area2 < 1e-9) {
    return 0.0;  // Colinear -- zero curvature, not a numerical singularity.
  }
  // Menger curvature: kappa = 4 * Area / (a * b * c).
  return (2.0 * area2) / (a * b * c);
}

double GoalScorer::continuityRunLength(const nav_msgs::msg::Path & path, std::size_t index) const
{
  double run_length = 0.0;
  for (std::size_t i = index; i + 1 < path.poses.size(); ++i) {
    const double gap = planarDistance(path.poses[i].pose.position, path.poses[i + 1].pose.position);
    if (gap > params_.max_continuity_gap) {
      break;
    }
    run_length += gap;
  }
  return run_length;
}

double GoalScorer::requiredForwardProgress(double lateral_offset, double heading_change) const
{
  // Guarded against a non-positive curvature cap (a misconfiguration) by falling back to
  // the flat floor rather than dividing by zero.
  if (params_.max_maneuver_curvature <= 1e-6) {
    return params_.min_maneuver_forward_progress;
  }
  // Two independent demands on the same path length; the binding one wins. See the two
  // coefficients' docs for the derivations (lateral: peak curvature of a quintic lateral
  // shift; heading: integral of a cubic-curvature spiral's own curvature profile).
  const double lateral_minimum = std::sqrt(
    params_.lateral_shift_curvature_coefficient * std::abs(lateral_offset) /
    params_.max_maneuver_curvature);
  const double heading_minimum = params_.heading_change_curvature_coefficient *
                                  std::abs(heading_change) / params_.max_maneuver_curvature;
  // Both are hard kinematic limits; the safety factor turns them into something the vehicle
  // can actually track with margin to spare -- see forward_progress_safety_factor's docs.
  const double kinematic_minimum =
    params_.forward_progress_safety_factor * std::max(lateral_minimum, heading_minimum);
  return std::max(params_.min_maneuver_forward_progress, kinematic_minimum);
}

bool GoalScorer::sweptFootprintFits(
  double lateral_offset, double forward_distance, double half_width) const
{
  if (forward_distance <= 1e-3) {
    return false;
  }
  const double d = std::abs(lateral_offset);
  // Sample the quintic lateral shift and take the worst sideways reach past the shoulder
  // centerline. y_remaining is how much sideways travel is still owed at u (the vehicle is
  // still on the lane side by that much), so it credits back the distance not yet closed.
  // 40 samples is far finer than the metre-scale margins involved here.
  constexpr int kSamples = 40;
  double worst_reach = -std::numeric_limits<double>::max();
  for (int i = 1; i <= kSamples; ++i) {
    const double u = static_cast<double>(i) / kSamples;
    const double u2 = u * u;
    const double u3 = u2 * u;
    const double u4 = u3 * u;
    const double u5 = u4 * u;
    const double progress = 10.0 * u3 - 15.0 * u4 + 6.0 * u5;
    const double y_remaining = d * (1.0 - progress);
    const double theta = std::atan(d / forward_distance * (30.0 * u2 - 60.0 * u3 + 30.0 * u4));
    const double reach = params_.vehicle_half_width * std::cos(theta) +
                          params_.vehicle_front_length * std::sin(theta) - y_remaining;
    worst_reach = std::max(worst_reach, reach);
  }
  return worst_reach <= half_width;
}

double GoalScorer::windowedHalfWidth(
  const nav_msgs::msg::Path & path, const std::vector<float> & half_widths,
  std::size_t index) const
{
  std::vector<double> window{static_cast<double>(half_widths[index])};

  // Backward half of the window -- same walk/gap rules as smoothedHeading().
  double back_span = 0.0;
  for (std::size_t i = index; i > 0; --i) {
    const double seg_len = planarDistance(
      path.poses[i].pose.position, path.poses[i - 1].pose.position);
    if (seg_len > params_.max_continuity_gap ||
        back_span + seg_len > params_.width_window_half_length) {
      break;
    }
    back_span += seg_len;
    window.push_back(static_cast<double>(half_widths[i - 1]));
  }

  // Forward half of the window.
  double forward_span = 0.0;
  for (std::size_t i = index; i + 1 < path.poses.size(); ++i) {
    const double seg_len = planarDistance(
      path.poses[i].pose.position, path.poses[i + 1].pose.position);
    if (seg_len > params_.max_continuity_gap ||
        forward_span + seg_len > params_.width_window_half_length) {
      break;
    }
    forward_span += seg_len;
    window.push_back(static_cast<double>(half_widths[i + 1]));
  }

  // Nearest-rank quantile -- see width_window_quantile's docs for why this is not a strict
  // minimum. Cheap: these windows hold a handful of points.
  std::sort(window.begin(), window.end());
  const std::size_t rank = static_cast<std::size_t>(
    std::clamp(params_.width_window_quantile, 0.0, 1.0) *
    static_cast<double>(window.size() - 1));
  return window[rank];
}

geometry_msgs::msg::Quaternion GoalScorer::smoothedHeading(
  const nav_msgs::msg::Path & path, std::size_t index) const
{
  double sum_x = 0.0;
  double sum_y = 0.0;

  // Backward half of the window.
  double back_span = 0.0;
  for (std::size_t i = index; i > 0; --i) {
    const auto & p1 = path.poses[i].pose.position;
    const auto & p0 = path.poses[i - 1].pose.position;
    const double seg_x = p1.x - p0.x;
    const double seg_y = p1.y - p0.y;
    const double seg_len = std::hypot(seg_x, seg_y);
    if (seg_len > params_.max_continuity_gap || back_span + seg_len > params_.heading_smoothing_distance) {
      break;
    }
    sum_x += seg_x;
    sum_y += seg_y;
    back_span += seg_len;
  }

  // Forward half of the window.
  double forward_span = 0.0;
  for (std::size_t i = index; i + 1 < path.poses.size(); ++i) {
    const auto & p0 = path.poses[i].pose.position;
    const auto & p1 = path.poses[i + 1].pose.position;
    const double seg_x = p1.x - p0.x;
    const double seg_y = p1.y - p0.y;
    const double seg_len = std::hypot(seg_x, seg_y);
    if (seg_len > params_.max_continuity_gap ||
        forward_span + seg_len > params_.heading_smoothing_distance) {
      break;
    }
    sum_x += seg_x;
    sum_y += seg_y;
    forward_span += seg_len;
  }

  if (std::hypot(sum_x, sum_y) < 1e-6) {
    // No usable neighbor segment (isolated point) -- fall back to the raw
    // per-point orientation rather than an undefined atan2(0, 0).
    return path.poses[index].pose.orientation;
  }
  return quaternionFromYaw(std::atan2(sum_y, sum_x));
}

std::optional<ScoredCandidate> GoalScorer::selectBestGoal(
  const nav_msgs::msg::Path & centerline_map, const geometry_msgs::msg::Pose & ego_pose,
  double ego_speed_mps, const std::vector<float> & half_widths) const
{
  if (centerline_map.poses.empty()) {
    return std::nullopt;
  }

  // Width data is only usable if index-aligned with this exact centerline -- a size
  // mismatch means the two topics are momentarily out of step (see the halfwidth topic's
  // docs in shoulder_centerline_node) and per-index lookups would be meaningless.
  const bool have_width_data =
    !half_widths.empty() && half_widths.size() == centerline_map.poses.size();
  if (!have_width_data && params_.require_width_data) {
    return std::nullopt;  // Fail closed -- see require_width_data's docs.
  }
  // Minimum scaled half-width a candidate's window must measure to fit the vehicle.
  const double required_half_width = params_.vehicle_half_width + params_.goal_width_margin;

  const double ego_yaw = yawFromQuaternion(ego_pose.orientation);
  const double ego_forward_x = std::cos(ego_yaw);
  const double ego_forward_y = std::sin(ego_yaw);

  const double min_stopping_distance =
    (ego_speed_mps * ego_speed_mps) / (2.0 * std::max(params_.comfortable_deceleration, 1e-3)) +
    params_.stopping_margin;

  std::optional<ScoredCandidate> best;

  for (std::size_t i = 0; i < centerline_map.poses.size(); ++i) {
    const auto & candidate_position = centerline_map.poses[i].pose.position;

    const double distance = planarDistance(ego_pose.position, candidate_position);
    if (distance > params_.max_lookahead_distance || distance < min_stopping_distance) {
      continue;  // Either unreachably far, or too close to decelerate comfortably.
    }
    if (distance > params_.max_reachable_distance) {
      // Farther than any duration up to trajectory_max_duration can cover while respecting
      // trajectory_max_speed -- see max_reachable_distance's docs. A hard kinematic ceiling,
      // not a search gap, so reject here rather than let PullOverTrajectoryPlanner discover it
      // the slow way (max_consecutive_planning_failures_ cycles later).
      continue;
    }

    // Reject candidates that are not meaningfully ahead of the vehicle --
    // avoids selecting stale waypoints behind ego on the ever-accumulating
    // map-frame trail, *and* (see min_maneuver_forward_progress's docs)
    // near-degenerate "almost pure sidestep" candidates whose required
    // curvature is hypersensitive to exactly how much forward room remains.
    const double to_candidate_x = candidate_position.x - ego_pose.position.x;
    const double to_candidate_y = candidate_position.y - ego_pose.position.y;
    const double forward_alignment =
      to_candidate_x * ego_forward_x + to_candidate_y * ego_forward_y;
    // Cheap floor first (also rejects anything not meaningfully ahead of ego); the full
    // lateral+heading requirement is applied below, once this candidate's smoothed heading
    // -- and therefore the heading change it actually demands -- is known.
    if (forward_alignment < params_.min_maneuver_forward_progress) {
      continue;
    }
    // Perpendicular component: how far sideways this candidate asks the vehicle to move.
    const double lateral_offset =
      std::abs(-ego_forward_y * to_candidate_x + ego_forward_x * to_candidate_y);

    const double continuity_length = continuityRunLength(centerline_map, i);
    if (continuity_length < params_.min_continuity_length) {
      continue;  // Not enough confirmed shoulder ahead of this point yet.
    }

    // Measured-width gate -- the vehicle's actual footprint (plus margin) must fit within
    // the measured shoulder over a window the length of the vehicle, not at one point.
    // Checked before the (much more expensive) spiral solve below. See
    // vehicle_half_width/width_measurement_scale/width_window_half_length's docs for the
    // live-verified wedge this prevents and the scale factor's derivation.
    double scaled_half_width = 0.0;
    if (have_width_data) {
      scaled_half_width = params_.width_measurement_scale *
                          windowedHalfWidth(centerline_map, half_widths, i);
      if (scaled_half_width < required_half_width) {
        continue;
      }
    }

    // Maneuver feasibility from ego's actual current pose -- see class
    // docs for why this is checked here rather than left entirely to
    // PullOverTrajectoryPlanner. Deliberately the same shape generator
    // (CurvatureSpiralPath) and the same cap (max_maneuver_curvature
    // should match trajectory_planner's max_curvature) -- a candidate
    // this rejects would be rejected by the planner's own preferred shape
    // too, just discovered late instead of up front.
    //
    // Must be checked against the *smoothed* heading, not the raw
    // per-point one: the candidate actually locked in below (and later
    // handed to PullOverTrajectoryPlanner) uses smoothedHeading(), which
    // can require meaningfully more curvature than the raw, noisier
    // per-point heading did (live-verified 2026-08-13 -- a candidate
    // passed this gate at curvature 0.26 against the raw heading, but the
    // real solver, targeting the smoothed heading actually committed to,
    // needed 0.42, well past max_maneuver_curvature, and failed 30
    // consecutive replanning cycles before giving up).
    auto goal_pose = centerline_map.poses[i].pose;
    goal_pose.orientation = smoothedHeading(centerline_map, i);

    // Pull the goal inboard, toward the side ego is approaching from -- see
    // goal_inboard_bias's docs for why aiming at the raw centerline lands the vehicle against
    // the shoulder's far line. The direction is derived per candidate from which side ego is
    // actually on, taken perpendicular to the goal's own (shoulder-tangent) heading, so it
    // works for a shoulder on either side without a hand-set left/right flag. Computed at
    // selection time, while ego is still in the lane and the sign is unambiguous; the goal
    // pose is fixed from here on, so it cannot flip later as ego crosses over.
    if (params_.goal_inboard_bias > 0.0) {
      const double goal_yaw = yawFromQuaternion(goal_pose.orientation);
      const double normal_x = -std::sin(goal_yaw);
      const double normal_y = std::cos(goal_yaw);
      const double ego_side =
        (ego_pose.position.x - goal_pose.position.x) * normal_x +
        (ego_pose.position.y - goal_pose.position.y) * normal_y;
      const double sign = (ego_side >= 0.0) ? 1.0 : -1.0;
      goal_pose.position.x += sign * params_.goal_inboard_bias * normal_x;
      goal_pose.position.y += sign * params_.goal_inboard_bias * normal_y;
    }

    // Full reachability requirement, now that the heading this candidate demands is known:
    // the path must be long enough to BOTH translate the vehicle sideways onto the shoulder
    // and rotate it onto the shoulder's tangent (see requiredForwardProgress). Checked
    // before the spiral solve below, which is far more expensive -- and, unlike the solver,
    // this states the reason a too-close goal is hopeless rather than discovering it as a
    // curvature overflow several cycles later.
    double heading_change = yawFromQuaternion(goal_pose.orientation) - ego_yaw;
    while (heading_change > M_PI) heading_change -= 2.0 * M_PI;
    while (heading_change < -M_PI) heading_change += 2.0 * M_PI;
    if (forward_alignment < requiredForwardProgress(lateral_offset, heading_change)) {
      continue;
    }
    // The vehicle must fit the shoulder for the whole approach, not just at the goal -- see
    // sweptFootprintFits(). Only meaningful with real width data; when width gating is off
    // there is nothing to check the sweep against.
    if (have_width_data && !sweptFootprintFits(lateral_offset, forward_alignment, scaled_half_width)) {
      continue;
    }

    const CurvatureSpiralPath maneuver(
      ego_pose, goal_pose, params_.max_maneuver_curvature);
    if (!maneuver.valid()) {
      continue;
    }

    // A *separate* gate from the curvature one above -- see
    // weight_jerk_ease's docs for why peak curvature alone isn't enough.
    // Approximate (speed close to ego_speed_mps still holds at the
    // maneuver's very start, per PullOverTrajectoryPlanner's a0=0
    // boundary condition), not an exact replica of the discretized check.
    const double estimated_initial_jerk =
      ego_speed_mps * ego_speed_mps * ego_speed_mps * maneuver.initialCurvatureRate();
    if (std::abs(estimated_initial_jerk) > params_.max_maneuver_jerk) {
      continue;
    }

    const double curvature = estimateCurvature(centerline_map, i);

    const double continuity_norm =
      std::clamp(continuity_length / params_.reference_run_length, 0.0, 1.0);
    const double curvature_norm =
      std::clamp(curvature / params_.max_reference_curvature, 0.0, 1.0);
    const double maneuver_norm =
      std::clamp(maneuver.peakCurvature() / params_.max_maneuver_curvature, 0.0, 1.0);
    const double jerk_norm =
      std::clamp(std::abs(estimated_initial_jerk) / params_.max_maneuver_jerk, 0.0, 1.0);
    // Prefer wider shoulder beyond the bare gate (0 at exactly-required width, 1 at
    // required + reference_extra_half_width) -- see weight_width's docs. Neutral 0 when
    // width gating is inactive so weights still sum consistently.
    const double width_norm =
      have_width_data
        ? std::clamp(
            (scaled_half_width - required_half_width) / params_.reference_extra_half_width,
            0.0, 1.0)
        : 0.0;

    const double score = params_.weight_continuity * continuity_norm +
                          params_.weight_curvature * (1.0 - curvature_norm) +
                          params_.weight_maneuver_ease * (1.0 - maneuver_norm) +
                          params_.weight_jerk_ease * (1.0 - jerk_norm) +
                          params_.weight_width * width_norm;

    if (!best.has_value() || score > best->score) {
      ScoredCandidate candidate;
      // goal_pose already carries the smoothed heading (see above) --
      // reuse it as-is so the locked-in candidate is exactly what the
      // feasibility gate just validated.
      candidate.pose = goal_pose;
      candidate.score = score;
      candidate.distance_from_ego = distance;
      candidate.curvature = curvature;
      candidate.continuity_length = continuity_length;
      candidate.maneuver_curvature = maneuver.peakCurvature();
      candidate.maneuver_initial_jerk = estimated_initial_jerk;
      candidate.measured_half_width = scaled_half_width;
      best = candidate;
    }
  }

  return best;
}

}  // namespace autoware::shoulder_pullover_manager
