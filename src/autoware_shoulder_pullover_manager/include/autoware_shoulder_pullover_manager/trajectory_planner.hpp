#pragma once

#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <rclcpp/time.hpp>

#include <optional>
#include <string>

namespace autoware::shoulder_pullover_manager
{

/// Ego kinematic state used as the trajectory's initial boundary condition.
struct KinematicState
{
  geometry_msgs::msg::Pose pose;
  double speed{0.0};  ///< Signed forward speed (m/s), from odometry.
};

/// Tunable limits for PullOverTrajectoryPlanner. Defaults are deliberately
/// tighter than planning_validator's own hard limits (see the pull-over MRM
/// framework report, control/validation contract section) -- this is a slow,
/// precision entry maneuver, not open-road driving, so comfort/precision
/// margins are appropriate on top of the hard safety ceiling the validator
/// enforces regardless.
struct TrajectoryPlannerParams
{
  double dt{0.1};                     ///< Sample spacing (s) of the output Trajectory.
  double min_duration{2.0};           ///< Shortest candidate maneuver duration (s) tried.
  double max_duration{15.0};          ///< Longest candidate maneuver duration (s) tried.
  double duration_step{1.0};          ///< Increment (s) between tried candidate durations.
  double terminal_approach_speed{0.3};  ///< Small nonzero speed (m/s) used only to keep the
                                         ///< heading boundary condition well-defined near the
                                         ///< stop; the actual last sample is force-zeroed.

  double max_curvature{0.8};          ///< 1/m. Conservative vs. validator's ~1.0 (live-tested: a
                                       ///< low-speed final heading adjustment legitimately needs
                                       ///< ~0.53-0.54 1/m -- physically fine, since the
                                       ///< speed-squared lateral-accel check already bounds the
                                       ///< actually-dangerous quantity separately; 0.5 was too
                                       ///< tight a standalone geometric cap for that case, 0.8
                                       ///< still leaves real margin under validator's ~1.0).
  double min_speed_for_curvature_check{0.15};  ///< m/s. Below this speed, curvature/lateral-accel/
                                               ///< lateral-jerk checks are skipped entirely rather
                                               ///< than evaluated -- live-tested finding:
                                               ///< curvature = d(heading)/segment_length blows up
                                               ///< near the very end of any stopping maneuver as
                                               ///< segment_length -> 0, even for a physically
                                               ///< harmless heading adjustment. A near-stationary
                                               ///< vehicle can point its wheels sharply with zero
                                               ///< real dynamic risk (unlike at speed, where the
                                               ///< same geometric curvature really would be
                                               ///< dangerous) -- raising the raw threshold instead
                                               ///< of gating by speed was tried first and only
                                               ///< pushed the same division-by-near-zero artifact
                                               ///< to a later sample, not fixed it.
  double max_lateral_accel{2.5};      ///< m/s^2. Conservative vs. validator's 9.8.
  double max_lateral_jerk{3.0};       ///< m/s^3. Conservative vs. validator's 7.0.
  double max_longitudinal_accel{1.5};   ///< m/s^2.
  double min_longitudinal_accel{-2.5};  ///< m/s^2 (braking).

  double ego_footprint_radius{1.5};   ///< m. Circular over-approximation of the ego footprint.
  double default_object_radius{1.0};  ///< m. Used only if an object has no usable shape info.
  double collision_margin{1.0};       ///< m. Extra clearance required on top of both radii.
};

/// Generates a short-horizon, dynamically-feasible, collision-checked
/// trajectory from the vehicle's current state directly to a chosen goal
/// pose -- independent of the Lanelet2 routing graph.
///
/// This exists because Autoware's own routing (`mission_planner`) can only
/// plan to a goal that lies within the HD map's routable lanelet graph, and
/// this project's Town04 map has zero `road_shoulder`-subtype lanelets (or
/// any other lanelet) covering the perception-detected shoulder area --
/// confirmed by grepping the map's .osm file directly. A shoulder pull-over
/// therefore cannot be expressed as a request to Autoware's router at all;
/// it requires a planner that reasons directly in Cartesian space from live
/// perception (shoulder_centerline's waypoints), not from the HD map.
///
/// Method: independent quintic (minimum-jerk) Hermite polynomials for x(t)
/// and y(t), matching start/end position, velocity, and acceleration
/// boundary conditions -- the same boundary-value-problem structure as the
/// Frenet-frame quintic planner in Werling et al. 2010 (see the framework
/// report), specialized here to a direct Cartesian parameterization rather
/// than a lanelet-centerline-relative Frenet frame, since the destination
/// itself is off that frame entirely. This is a deliberate, documented
/// simplification appropriate for the short (tens-of-meters), low-speed
/// maneuver distances involved -- not a claim that Frenet/lanelet-relative
/// planning is unnecessary in general (the report's math section covers
/// that fuller formulation for the on-road portion of a maneuver).
///
/// A small family of candidate maneuver durations is tried (short to long);
/// for each, the resulting trajectory is checked against configured
/// curvature/jerk/acceleration limits and against tracked objects'
/// predicted paths, and the shortest feasible, collision-free candidate is
/// returned. Called every planning cycle (receding-horizon replanning) by
/// the owning node, not just once at trigger time.
class PullOverTrajectoryPlanner
{
public:
  explicit PullOverTrajectoryPlanner(const TrajectoryPlannerParams & params);

  /// Returns a feasible trajectory from `start` to `goal_pose`, or
  /// std::nullopt if no candidate duration in
  /// [min_duration, max_duration] satisfies both the kinematic constraints
  /// and the collision check against `objects`. `stamp` becomes the
  /// trajectory header's stamp and the time-zero reference for both the
  /// output trajectory's `time_from_start` fields and for aligning against
  /// `objects`' predicted-path time steps.
  ///
  /// If `failure_reason` is non-null and no candidate succeeds, it is set
  /// to a human-readable explanation of why the *last* tried candidate was
  /// rejected (e.g. which constraint, by how much, or which object it
  /// would have collided with) -- callers should log this rather than
  /// guess at the cause.
  [[nodiscard]] std::optional<autoware_planning_msgs::msg::Trajectory> plan(
    const KinematicState & start, const geometry_msgs::msg::Pose & goal_pose,
    const autoware_perception_msgs::msg::PredictedObjects & objects, const rclcpp::Time & stamp,
    std::string * failure_reason = nullptr) const;

private:
  /// Coefficients of a scalar quintic q(t) = c0 + c1 t + c2 t^2 + c3 t^3 +
  /// c4 t^4 + c5 t^5, plus its derivatives -- used independently for x(t)
  /// and y(t).
  struct Quintic
  {
    double c0{0.0}, c1{0.0}, c2{0.0}, c3{0.0}, c4{0.0}, c5{0.0};
    [[nodiscard]] double position(double t) const;
    [[nodiscard]] double velocity(double t) const;
    [[nodiscard]] double acceleration(double t) const;
  };

  /// Solves for a quintic matching position/velocity/acceleration boundary
  /// conditions at t=0 and t=duration.
  [[nodiscard]] static Quintic solveQuintic(
    double q0, double v0, double a0, double qT, double vT, double aT, double duration);

  [[nodiscard]] std::optional<autoware_planning_msgs::msg::Trajectory> buildCandidate(
    const KinematicState & start, const geometry_msgs::msg::Pose & goal_pose, double duration,
    const rclcpp::Time & stamp) const;

  [[nodiscard]] bool satisfiesKinematicConstraints(
    const autoware_planning_msgs::msg::Trajectory & trajectory,
    std::string * failure_reason = nullptr) const;

  [[nodiscard]] bool isCollisionFree(
    const autoware_planning_msgs::msg::Trajectory & trajectory,
    const autoware_perception_msgs::msg::PredictedObjects & objects,
    const rclcpp::Time & trajectory_start_time, std::string * failure_reason = nullptr) const;

  TrajectoryPlannerParams params_;
};

}  // namespace autoware::shoulder_pullover_manager
