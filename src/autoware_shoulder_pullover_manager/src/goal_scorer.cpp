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
  double ego_speed_mps) const
{
  if (centerline_map.poses.empty()) {
    return std::nullopt;
  }

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

    // Reject candidates that are not meaningfully ahead of the vehicle --
    // avoids selecting stale waypoints behind ego on the ever-accumulating
    // map-frame trail, *and* (see min_maneuver_forward_progress's docs)
    // near-degenerate "almost pure sidestep" candidates whose required
    // curvature is hypersensitive to exactly how much forward room remains.
    const double to_candidate_x = candidate_position.x - ego_pose.position.x;
    const double to_candidate_y = candidate_position.y - ego_pose.position.y;
    const double forward_alignment =
      to_candidate_x * ego_forward_x + to_candidate_y * ego_forward_y;
    if (forward_alignment < params_.min_maneuver_forward_progress) {
      continue;
    }

    const double continuity_length = continuityRunLength(centerline_map, i);
    if (continuity_length < params_.min_continuity_length) {
      continue;  // Not enough confirmed shoulder ahead of this point yet.
    }

    // Maneuver feasibility from ego's actual current pose -- see class
    // docs for why this is checked here rather than left entirely to
    // PullOverTrajectoryPlanner. Deliberately the same shape generator
    // (CurvatureSpiralPath) and the same cap (max_maneuver_curvature
    // should match trajectory_planner's max_curvature) -- a candidate
    // this rejects would be rejected by the planner's own preferred shape
    // too, just discovered late instead of up front.
    const CurvatureSpiralPath maneuver(
      ego_pose, centerline_map.poses[i].pose, params_.max_maneuver_curvature);
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

    const double score = params_.weight_continuity * continuity_norm +
                          params_.weight_curvature * (1.0 - curvature_norm) +
                          params_.weight_maneuver_ease * (1.0 - maneuver_norm) +
                          params_.weight_jerk_ease * (1.0 - jerk_norm);

    if (!best.has_value() || score > best->score) {
      ScoredCandidate candidate;
      candidate.pose = centerline_map.poses[i].pose;
      // Replace the raw per-point heading with a locally-smoothed tangent --
      // see smoothedHeading()'s docs. Position stays exactly the raw
      // waypoint's; only orientation changes.
      candidate.pose.orientation = smoothedHeading(centerline_map, i);
      candidate.score = score;
      candidate.distance_from_ego = distance;
      candidate.curvature = curvature;
      candidate.continuity_length = continuity_length;
      candidate.maneuver_curvature = maneuver.peakCurvature();
      candidate.maneuver_initial_jerk = estimated_initial_jerk;
      best = candidate;
    }
  }

  return best;
}

}  // namespace autoware::shoulder_pullover_manager
