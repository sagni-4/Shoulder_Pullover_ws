#pragma once

namespace autoware::shoulder_pullover_manager
{

/// A 2D position + heading at some point along a path.
struct PathPoint
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
};

/// Common interface for the maneuver-shape generators PullOverTrajectoryPlanner
/// samples a speed profile over (see trajectory_planner.hpp's class docs for
/// why there is more than one: CurvatureSpiralPath is preferred when it
/// converges and stays under the curvature cap, with DubinsPath kept as a
/// deterministic fallback since a numerical solver's convergence isn't
/// guaranteed for every geometry).
class PathShape
{
public:
  virtual ~PathShape() = default;

  /// False if this shape could not be constructed for the given start/goal
  /// (e.g. the spiral solver didn't converge, or a Dubins path doesn't
  /// exist at the given radius) -- callers should not call length() or
  /// pointAt() in that case.
  [[nodiscard]] virtual bool valid() const = 0;

  /// Total path length (m). Only meaningful if valid().
  [[nodiscard]] virtual double length() const = 0;

  /// Position and heading at arc-length `s` along the path, clamped to
  /// [0, length()]. Only meaningful if valid().
  [[nodiscard]] virtual PathPoint pointAt(double s) const = 0;
};

}  // namespace autoware::shoulder_pullover_manager
