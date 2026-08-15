#include "autoware_shoulder_pullover_manager/trajectory_gate_node.hpp"

#include <chrono>
#include <vector>

namespace autoware::shoulder_pullover_manager
{

using autoware_planning_msgs::msg::Trajectory;
using autoware_planning_msgs::msg::TrajectoryPoint;
using tier4_system_msgs::msg::MrmBehaviorStatus;

TrajectoryGateNode::TrajectoryGateNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("pullover_trajectory_gate", options)
{
  const std::string normal_trajectory_topic = declare_parameter<std::string>(
    "normal_trajectory_topic", "/planning/scenario_planning/velocity_smoother/trajectory");
  const std::string pullover_trajectory_topic = declare_parameter<std::string>(
    "pullover_trajectory_topic", "/shoulder_pullover_manager/planned_trajectory");
  const std::string pullover_status_topic = declare_parameter<std::string>(
    "pullover_status_topic", "/system/mrm/pull_over_manager/status");
  const std::string odometry_topic =
    declare_parameter<std::string>("odometry_topic", "/localization/kinematic_state");
  const std::string output_trajectory_topic =
    declare_parameter<std::string>("output_trajectory_topic", "/planning/trajectory_gate/output");
  const double hold_publish_rate_hz =
    declare_parameter<double>("hold_publish_rate_hz", 10.0);
  trajectory_staleness_timeout_ =
    declare_parameter<double>("trajectory_staleness_timeout", trajectory_staleness_timeout_);

  output_pub_ = create_publisher<Trajectory>(output_trajectory_topic, rclcpp::QoS(1));

  normal_trajectory_sub_ = create_subscription<Trajectory>(
    normal_trajectory_topic, rclcpp::QoS(1),
    [this](const Trajectory::ConstSharedPtr msg) { onNormalTrajectory(msg); });
  pullover_trajectory_sub_ = create_subscription<Trajectory>(
    pullover_trajectory_topic, rclcpp::QoS(1),
    [this](const Trajectory::ConstSharedPtr msg) { onPulloverTrajectory(msg); });
  pullover_status_sub_ = create_subscription<MrmBehaviorStatus>(
    pullover_status_topic, rclcpp::QoS(1),
    [this](const MrmBehaviorStatus::ConstSharedPtr msg) { onPulloverStatus(msg); });
  odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odometry_topic, rclcpp::QoS(1),
    [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { onOdometry(msg); });

  const auto hold_period =
    std::chrono::duration<double>(1.0 / std::max(hold_publish_rate_hz, 1e-3));
  hold_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(hold_period),
    [this]() { onHoldTimer(); });

  RCLCPP_INFO(
    get_logger(),
    "pullover_trajectory_gate ready. Passing through '%s' to '%s' until pull-over first "
    "engages; will then forward '%s' while operating, and latch into a stationary hold "
    "trajectory once the maneuver ends (succeeded or failed).",
    normal_trajectory_topic.c_str(), output_trajectory_topic.c_str(),
    pullover_trajectory_topic.c_str());
}

void TrajectoryGateNode::onNormalTrajectory(const Trajectory::ConstSharedPtr msg)
{
  if (!latched_) {
    output_pub_->publish(*msg);
  }
  // Once latched_, the normal trajectory is deliberately dropped -- see class docs.
}

void TrajectoryGateNode::onPulloverTrajectory(const Trajectory::ConstSharedPtr msg)
{
  // Latching here too (not just on the status topic) closes a real race,
  // confirmed live (2026-08-05): pull_over_manager_node's status timer
  // runs at only 5Hz, but its planning timer (the thing that actually
  // publishes this trajectory) runs at 10Hz and can go from triggered to
  // arrived in a single cycle when ego is already slow and the selected
  // goal is close -- the entire kOperating phase completed in well under
  // one status-timer period, so the *first* status message this node
  // ever saw already showed post-completion (AVAILABLE), and latched_
  // never got set at all; the real vehicle kept driving its normal route
  // the whole time, obliviously, while the MRM status flickered on/off
  // faster than anything downstream could react to. The mere existence of
  // a message on this topic is proof kOperating or kDecelerating was
  // active the instant it was published (see
  // PullOverManagerNode::onPlanningTimer's early-return guard), which is
  // a strictly more reliable, lower-latency signal than waiting for the
  // status topic to agree.
  latched_ = true;
  holding_ = false;
  last_pullover_trajectory_time_ = now();
  output_pub_->publish(*msg);
}

void TrajectoryGateNode::onPulloverStatus(const MrmBehaviorStatus::ConstSharedPtr msg)
{
  if (msg->state == MrmBehaviorStatus::OPERATING) {
    latched_ = true;
    holding_ = false;
  }
  // Deliberately no else-branch reacting to a non-OPERATING status here -- see class docs for
  // why: onHoldTimer() decides whether to actually hold from pull-over trajectory *staleness*
  // instead, since this status topic can blip non-OPERATING for a single cycle while
  // pull_over_manager_node is still actively publishing real trajectories.
}

void TrajectoryGateNode::onOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  latest_odometry_ = *msg;
}

void TrajectoryGateNode::onHoldTimer()
{
  if (!latched_ || !latest_odometry_.has_value()) {
    return;
  }
  const bool trajectory_is_stale =
    !last_pullover_trajectory_time_.has_value() ||
    (now() - *last_pullover_trajectory_time_).seconds() > trajectory_staleness_timeout_;
  if (!trajectory_is_stale) {
    // pull_over_manager_node is still actively feeding fresh trajectories -- nothing to hold
    // for, regardless of what the (possibly momentarily stale/blipping) status topic last said.
    return;
  }
  holding_ = true;
  output_pub_->publish(buildHoldTrajectory(latest_odometry_->pose.pose, now()));
}

Trajectory TrajectoryGateNode::buildHoldTrajectory(
  const geometry_msgs::msg::Pose & pose, const rclcpp::Time & stamp) const
{
  Trajectory trajectory;
  trajectory.header.stamp = stamp;
  trajectory.header.frame_id = "map";

  // Two coincident, stationary points -- the minimum a real Trajectory
  // consumer (velocity/steering interpolation expects at least a
  // start+end) can be handed while still unambiguously meaning "stay
  // exactly here." Re-published with a fresh stamp every hold_timer_
  // tick, which is what actually keeps downstream freshness checks happy
  // -- the *content* never needs to change once ego is stopped.
  TrajectoryPoint point;
  point.pose = pose;
  point.longitudinal_velocity_mps = 0.0F;
  point.lateral_velocity_mps = 0.0F;
  point.acceleration_mps2 = 0.0F;
  point.time_from_start.sec = 0;
  point.time_from_start.nanosec = 0;
  trajectory.points.push_back(point);

  TrajectoryPoint point_later = point;
  point_later.time_from_start.sec = 1;
  trajectory.points.push_back(point_later);

  return trajectory;
}

}  // namespace autoware::shoulder_pullover_manager
