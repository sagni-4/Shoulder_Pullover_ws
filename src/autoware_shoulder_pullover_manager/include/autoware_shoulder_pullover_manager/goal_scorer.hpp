#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/path.hpp>

#include <optional>

namespace autoware::shoulder_pullover_manager
{

/// Tunable parameters for candidate shoulder-goal scoring.
///
/// This is a deliberately *classical* (point-estimate, no probability)
/// implementation of the goal-scoring formula from the project's pull-over
/// MRM framework report. It only uses fields that shoulder_centerline_node
/// actually publishes today (position + heading per waypoint on
/// shoulder_centerline_path_map) -- the report's "maturity" and "lateral
/// clearance" terms are intentionally omitted here rather than faked, since
/// no topic currently exposes per-point sample-confidence or shoulder width.
/// Adding those requires extending shoulder_centerline_node's published
/// interface first; see project memory (pullover_mrm_framework.md).
struct GoalScorerParams
{
  /// Candidates farther than this (straight-line, meters) from the current
  /// ego position are not considered -- keeps the search local to what the
  /// vehicle can actually reach soon, and avoids picking a stale waypoint
  /// from a completely different part of the accumulated map-frame trail.
  double max_lookahead_distance{60.0};

  /// Comfortable deceleration (m/s^2) used to compute the minimum feasible
  /// stopping distance at the vehicle's current speed. Matches the order of
  /// magnitude Autoware's own comfortable_stop MRM behavior targets.
  double comfortable_deceleration{2.0};

  /// Extra fixed margin (meters) added on top of the physical stopping
  /// distance, so the maneuver has room to actually execute the lane
  /// change(s) and shoulder entry, not just brake to a halt exactly at the
  /// goal.
  double stopping_margin{5.0};

  /// A run of consecutive centerline waypoints ahead of a candidate is only
  /// considered "continuous" while consecutive spacing stays below this
  /// (meters). A gap larger than this usually means the accumulated trail's
  /// ragged growing tip, or a genuine break in shoulder detection -- see
  /// project memory on why the tip of the accumulated trail is the noisiest
  /// part.
  double max_continuity_gap{3.0};

  /// Hard minimum continuous run length (meters) ahead of a candidate for it
  /// to be considered at all. Below this, there is not demonstrably enough
  /// confirmed shoulder ahead to actually park in.
  double min_continuity_length{8.0};

  /// Run length (meters) at or above which the continuity score term
  /// saturates to 1.0.
  double reference_run_length{20.0};

  /// Local centerline curvature (1/m) at or above which the curvature
  /// penalty term saturates to 1.0. Chosen well below planning_validator's
  /// hard curvature limit so scoring prefers gentle entry points long before
  /// validator rejection would ever become a concern.
  double max_reference_curvature{0.3};

  /// Score weight for the continuity/run-length term.
  double weight_continuity{0.6};

  /// Score weight for the (inverted) curvature term.
  double weight_curvature{0.4};
};

/// A single scored candidate shoulder-goal, with the raw measurements kept
/// alongside the final score for logging/diagnostics.
struct ScoredCandidate
{
  geometry_msgs::msg::Pose pose;
  double score{0.0};
  double distance_from_ego{0.0};
  double curvature{0.0};
  double continuity_length{0.0};
};

/// Selects the best feasible shoulder pull-over goal from the live,
/// map-frame shoulder centerline, given the current ego pose and speed.
///
/// Pure logic, no ROS node/publisher/subscriber state -- deliberately kept
/// independent of rclcpp so the selection algorithm itself can be reasoned
/// about (and unit-tested) without a running ROS graph.
class GoalScorer
{
public:
  explicit GoalScorer(const GoalScorerParams & params);

  /// Returns the highest-scoring feasible candidate, or std::nullopt if no
  /// waypoint in `centerline_map` satisfies the hard feasibility gates
  /// (ahead of ego, within lookahead range, past the minimum stopping
  /// distance, with enough confirmed continuity ahead).
  [[nodiscard]] std::optional<ScoredCandidate> selectBestGoal(
    const nav_msgs::msg::Path & centerline_map, const geometry_msgs::msg::Pose & ego_pose,
    double ego_speed_mps) const;

private:
  /// Local curvature at `index` via the standard three-point (Menger)
  /// curvature formula over centerline_map.poses[index-1, index, index+1].
  /// Returns 0.0 at the path endpoints, where no such triplet exists.
  [[nodiscard]] static double estimateCurvature(
    const nav_msgs::msg::Path & path, std::size_t index);

  /// Arclength of the contiguous run of waypoints starting at `index`,
  /// walking forward while consecutive spacing stays within
  /// max_continuity_gap_.
  [[nodiscard]] double continuityRunLength(const nav_msgs::msg::Path & path, std::size_t index)
    const;

  GoalScorerParams params_;
};

}  // namespace autoware::shoulder_pullover_manager
