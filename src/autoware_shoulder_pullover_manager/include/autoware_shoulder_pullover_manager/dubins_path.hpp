#pragma once

#include <geometry_msgs/msg/pose.hpp>

namespace autoware::shoulder_pullover_manager
{

/// Shortest Dubins connector (forward-only, constant-radius) between two 2D
/// poses -- gives PullOverTrajectoryPlanner a path *shape* with curvature
/// exactly bounded by `radius` everywhere by construction, instead of
/// hoping a pair of independently-fit Cartesian quintics happens to stay
/// under the limit (see trajectory_planner.hpp's class docs for why that
/// approach broke down on large-heading-change goals).
///
/// Restricted to the four CSC families (LSL/RSR/LSR/RSL); the CCC families
/// (LRL/RLR) are intentionally not implemented -- they are only ever
/// shorter than the best CSC path for goals requiring close to a full
/// heading reversal at very short range, which this application's forward
/// shoulder-approach goals do not produce. If no CSC path exists, `valid()`
/// is false and the caller should treat the goal as unreachable by this
/// planner rather than guess.
class DubinsPath
{
public:
  struct PathPoint
  {
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
  };

  DubinsPath(
    const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal, double radius);

  [[nodiscard]] bool valid() const { return valid_; }

  /// Total path length (m). Only meaningful if valid().
  [[nodiscard]] double length() const { return length_; }

  /// Position and heading at arc-length `s` along the path, clamped to
  /// [0, length()]. Only meaningful if valid().
  [[nodiscard]] PathPoint pointAt(double s) const;

private:
  bool valid_{false};
  double radius_{0.0};
  double start_x_{0.0};
  double start_y_{0.0};
  double start_yaw_{0.0};
  char mode_[3]{'S', 'S', 'S'};       ///< e.g. {'L','S','L'}.
  double seg_len_[3]{0.0, 0.0, 0.0};  ///< Real arc-length (m) of each segment.
  double length_{0.0};
};

}  // namespace autoware::shoulder_pullover_manager
