#pragma once

#include <rclcpp/rclcpp.hpp>

#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tier4_system_msgs/msg/mrm_behavior_status.hpp>

#include <optional>

namespace autoware::shoulder_pullover_manager
{

/// Arbitrates between the normal planning stack's trajectory and the
/// pull-over manager's trajectory, publishing exactly one of them to
/// planning_validator's input -- the single point where this project's
/// standalone pull-over planner actually reaches real vehicle control.
///
/// Why this exists: `/planning/scenario_planning/velocity_smoother/trajectory`
/// has exactly one publisher (velocity_smoother) and one subscriber
/// (planning_validator) in this project's stack (confirmed via
/// `ros2 topic info --verbose`) -- planning_validator's own output
/// (`/planning/trajectory`) has *nine* subscribers, including
/// control_validator and lane_departure_checker_node, which run safety
/// checks of their own. Injecting there directly would bypass those
/// checks entirely. Injecting here instead means planning_validator keeps
/// validating whatever it receives -- pull-over trajectories included --
/// before anything reaches the controller, and every other consumer of
/// `/planning/trajectory` keeps working exactly as it always has, since
/// planning_validator's own publishing behavior is untouched.
///
/// Getting this node's *output* into planning_validator's *input* requires
/// reloading planning_validator with its `~/input/trajectory` remap
/// pointed at this node's output topic instead of velocity_smoother's --
/// see project memory (pullover_mrm_framework.md) for the exact
/// component-unload/load procedure used, done deliberately as a
/// same-container component reload (not a full stack restart) so nothing
/// else in the running simulation is disturbed.
///
/// State machine, deliberately simple and one-directional (see
/// `latched_` below):
/// - Before pull-over is ever triggered: pure pass-through of the normal
///   trajectory, untouched.
/// - While pull-over is OPERATING: forwards its live, freshly-replanned
///   trajectory every time one arrives.
/// - Once pull-over leaves OPERATING (succeeded *or* failed) after having
///   been engaged at all: latches permanently into publishing a
///   synthesized "hold position" trajectory at ego's current pose,
///   republished periodically with a fresh timestamp so downstream
///   consumers (which check trajectory freshness) don't flag it as
///   stale. Never reverts to forwarding the normal trajectory again --
///   deliberate, matches real MRM semantics (an MRM outcome isn't quietly
///   undone by the vehicle just resuming normal driving on its own).
///
/// **`latched_` is set from *two* independent signals, not just the
/// status topic -- don't simplify this back to one.** Live-tested finding
/// (2026-08-05): pull_over_manager_node's status timer runs at only 5Hz,
/// but its planning timer actually publishes the pull-over trajectory at
/// 10Hz and can go from triggered to arrived in a single cycle when ego
/// is already slow and the selected goal is close. The entire kOperating
/// phase completed in well under one status-timer period in that test --
/// the *first* status message this node ever received already showed
/// post-completion (AVAILABLE), so latching only on the status topic
/// missed the engagement entirely: the real vehicle kept driving its
/// normal route the whole time, obliviously, while the MRM status
/// flickered on the operator's own dashboard faster than anything
/// downstream could react to. Receiving *any* message on the pull-over
/// trajectory topic is independent, lower-latency proof that kOperating
/// or kDecelerating was active the instant it was published (see
/// PullOverManagerNode::onPlanningTimer's early-return guard) -- latching
/// on that too closes the race regardless of how fast a maneuver
/// completes.
class TrajectoryGateNode : public rclcpp::Node
{
public:
  explicit TrajectoryGateNode(const rclcpp::NodeOptions & options);

private:
  void onNormalTrajectory(const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg);
  void onPulloverTrajectory(const autoware_planning_msgs::msg::Trajectory::ConstSharedPtr msg);
  void onPulloverStatus(const tier4_system_msgs::msg::MrmBehaviorStatus::ConstSharedPtr msg);
  void onOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void onHoldTimer();

  /// A minimal, valid Trajectory (a couple of coincident points, v=0/a=0)
  /// at `pose`, timestamped `stamp` -- "stay exactly here" in a form the
  /// rest of the stack (which expects a real Trajectory message, not the
  /// absence of one) can consume like any other planned trajectory.
  [[nodiscard]] autoware_planning_msgs::msg::Trajectory buildHoldTrajectory(
    const geometry_msgs::msg::Pose & pose, const rclcpp::Time & stamp) const;

  rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr normal_trajectory_sub_;
  rclcpp::Subscription<autoware_planning_msgs::msg::Trajectory>::SharedPtr
    pullover_trajectory_sub_;
  rclcpp::Subscription<tier4_system_msgs::msg::MrmBehaviorStatus>::SharedPtr
    pullover_status_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr output_pub_;
  rclcpp::TimerBase::SharedPtr hold_timer_;

  bool latched_{false};  ///< True forever once pull-over has ever gone OPERATING.
  bool holding_{false};  ///< True once latched_ *and* no longer OPERATING (see class docs).
  std::optional<nav_msgs::msg::Odometry> latest_odometry_;
};

}  // namespace autoware::shoulder_pullover_manager
