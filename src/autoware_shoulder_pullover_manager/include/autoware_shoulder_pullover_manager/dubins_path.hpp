#pragma once

#include "autoware_shoulder_pullover_manager/path_shape.hpp"

#include <geometry_msgs/msg/pose.hpp>

namespace autoware::shoulder_pullover_manager
{

/// Shortest Dubins connector (forward-only, constant-radius) between two 2D
/// poses -- curvature is exactly bounded by `radius` everywhere by
/// construction. **Fallback shape only as of 2026-08-05** -- see
/// CurvatureSpiralPath (preferred: converges to a single, usually much
/// shorter path with no forced constant-radius detour) and
/// trajectory_planner.hpp's class docs for why a second shape generator
/// was needed at all.
///
/// Restricted to the four CSC families (LSL/RSR/LSR/RSL). The class docs
/// used to claim the CCC families (LRL/RLR) are "only ever shorter... for
/// goals requiring close to a full heading reversal" -- **that turned out
/// to be wrong, don't restate it**: a real live-tested goal with nearly
/// *parallel* start/goal headings (a lateral "sidestep", not a reversal at
/// all) produced a CSC solution requiring an absurd 60-100m detour loop at
/// safe radii, and LRL/RLR were tried and did not help either (verified
/// offline) -- the true minimum turning radius for a short path on that
/// exact goal turned out to sit right at this vehicle's kinematic hard
/// limit, which is why CurvatureSpiralPath exists: not to fix a coverage
/// gap in Dubins mode selection, but because a single global radius is
/// itself the wrong model for a maneuver that only needs a tight turn
/// briefly, not throughout. CCC still isn't implemented here (added
/// complexity for a family that would not have fixed that case either),
/// but don't assume its absence is harmless for some *other* untested
/// geometry -- if this fallback path is ever hit in practice and still
/// fails, treat "maybe this needed LRL/RLR" as a live hypothesis, not a
/// closed question. If no CSC path exists, `valid()` is false and the
/// caller should treat the goal as unreachable by this shape rather than
/// guess.
class DubinsPath : public PathShape
{
public:
  DubinsPath(
    const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal, double radius);

  [[nodiscard]] bool valid() const override { return valid_; }

  /// Total path length (m). Only meaningful if valid().
  [[nodiscard]] double length() const override { return length_; }

  /// Position and heading at arc-length `s` along the path, clamped to
  /// [0, length()]. Only meaningful if valid().
  [[nodiscard]] PathPoint pointAt(double s) const override;

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
