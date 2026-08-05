#include "autoware_shoulder_pullover_manager/pull_over_manager_node.hpp"

#include <chrono>
#include <cmath>
#include <string>
#include <utility>

namespace autoware::shoulder_pullover_manager
{

using autoware_adapi_v1_msgs::msg::OperationModeState;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_planning_msgs::msg::Trajectory;
using tier4_system_msgs::msg::MrmBehaviorStatus;
using tier4_system_msgs::srv::OperateMrm;
using namespace std::chrono_literals;

namespace
{
/// Absolute names autoware_mrm_handler already calls out to (verified live:
/// `ros2 node info /system/mrm_handler` lists these exact names as a
/// Service Client and a Subscriber, respectively) -- deliberately NOT `~/`
/// relative names, which would resolve against this node's own
/// name/namespace instead and silently fail to be what mrm_handler is
/// actually looking for.
constexpr char kOperateServiceName[] = "/system/mrm/pull_over_manager/operate";
constexpr char kStatusTopicName[] = "/system/mrm/pull_over_manager/status";

nav_msgs::msg::Path toDebugPath(const Trajectory & trajectory)
{
  nav_msgs::msg::Path path;
  path.header = trajectory.header;
  path.poses.reserve(trajectory.points.size());
  for (const auto & point : trajectory.points) {
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header = trajectory.header;
    pose_stamped.pose = point.pose;
    path.poses.push_back(pose_stamped);
  }
  return path;
}

double yawFromQuaternionLocal(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}
}  // namespace

PullOverManagerNode::PullOverManagerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("shoulder_pullover_manager", options)
{
  // --- Parameters: goal scoring --------------------------------------------
  scorer_params_.max_lookahead_distance =
    declare_parameter<double>("max_lookahead_distance", scorer_params_.max_lookahead_distance);
  scorer_params_.comfortable_deceleration = declare_parameter<double>(
    "comfortable_deceleration", scorer_params_.comfortable_deceleration);
  scorer_params_.stopping_margin =
    declare_parameter<double>("stopping_margin", scorer_params_.stopping_margin);
  scorer_params_.max_continuity_gap =
    declare_parameter<double>("max_continuity_gap", scorer_params_.max_continuity_gap);
  scorer_params_.min_continuity_length =
    declare_parameter<double>("min_continuity_length", scorer_params_.min_continuity_length);
  scorer_params_.reference_run_length =
    declare_parameter<double>("reference_run_length", scorer_params_.reference_run_length);
  scorer_params_.max_reference_curvature =
    declare_parameter<double>("max_reference_curvature", scorer_params_.max_reference_curvature);
  scorer_params_.weight_continuity =
    declare_parameter<double>("weight_continuity", scorer_params_.weight_continuity);
  scorer_params_.weight_curvature =
    declare_parameter<double>("weight_curvature", scorer_params_.weight_curvature);

  // --- Parameters: trajectory planning --------------------------------------
  trajectory_params_.dt = declare_parameter<double>("trajectory_dt", trajectory_params_.dt);
  trajectory_params_.min_duration =
    declare_parameter<double>("trajectory_min_duration", trajectory_params_.min_duration);
  trajectory_params_.max_duration =
    declare_parameter<double>("trajectory_max_duration", trajectory_params_.max_duration);
  trajectory_params_.duration_step =
    declare_parameter<double>("trajectory_duration_step", trajectory_params_.duration_step);
  trajectory_params_.terminal_approach_speed = declare_parameter<double>(
    "trajectory_terminal_approach_speed", trajectory_params_.terminal_approach_speed);
  trajectory_params_.max_curvature =
    declare_parameter<double>("trajectory_max_curvature", trajectory_params_.max_curvature);
  trajectory_params_.max_lateral_accel = declare_parameter<double>(
    "trajectory_max_lateral_accel", trajectory_params_.max_lateral_accel);
  trajectory_params_.max_lateral_jerk =
    declare_parameter<double>("trajectory_max_lateral_jerk", trajectory_params_.max_lateral_jerk);
  trajectory_params_.max_longitudinal_accel = declare_parameter<double>(
    "trajectory_max_longitudinal_accel", trajectory_params_.max_longitudinal_accel);
  trajectory_params_.min_longitudinal_accel = declare_parameter<double>(
    "trajectory_min_longitudinal_accel", trajectory_params_.min_longitudinal_accel);
  trajectory_params_.ego_footprint_radius = declare_parameter<double>(
    "trajectory_ego_footprint_radius", trajectory_params_.ego_footprint_radius);
  trajectory_params_.default_object_radius = declare_parameter<double>(
    "trajectory_default_object_radius", trajectory_params_.default_object_radius);
  trajectory_params_.collision_margin =
    declare_parameter<double>("trajectory_collision_margin", trajectory_params_.collision_margin);

  // --- Parameters: node behavior --------------------------------------------
  require_autonomous_mode_ = declare_parameter<bool>("require_autonomous_mode", true);
  status_publish_rate_hz_ = declare_parameter<double>("status_publish_rate_hz", 5.0);
  poll_rate_hz_ = declare_parameter<double>("poll_rate_hz", 20.0);
  planning_rate_hz_ = declare_parameter<double>("planning_rate_hz", 10.0);
  arrival_distance_threshold_ = declare_parameter<double>("arrival_distance_threshold", 1.0);
  arrival_speed_threshold_ = declare_parameter<double>("arrival_speed_threshold", 0.2);
  max_consecutive_planning_failures_ =
    declare_parameter<int>("max_consecutive_planning_failures", 30);
  // Deliberately empty by default: this node's own trajectory is only ever
  // published on its own namespaced topics unless explicitly pointed at the
  // live control-facing topic. Doing so requires planning_validator's own
  // output to be remapped away from that name first (see the class-level
  // docs and the launch file) -- otherwise two publishers would fight over
  // the same topic, which would be genuinely unsafe with the vehicle under
  // autonomous control. This is a deliberate safety default, not an
  // oversight.
  direct_trajectory_output_topic_ =
    declare_parameter<std::string>("direct_trajectory_output_topic", "");

  visualization_width_ = declare_parameter<double>("visualization_width", visualization_width_);
  visualization_alpha_ = declare_parameter<double>("visualization_alpha", visualization_alpha_);

  keyboard_trigger_enabled_ = declare_parameter<bool>("keyboard_trigger_enabled", true);
  const std::string trigger_key_param = declare_parameter<std::string>("keyboard_trigger_key", "p");
  keyboard_trigger_key_ = trigger_key_param.empty() ? 'p' : trigger_key_param.front();

  const std::string centerline_topic = declare_parameter<std::string>(
    "centerline_topic", "/shoulder_centerline_node/shoulder_centerline_path_map");
  const std::string odometry_topic =
    declare_parameter<std::string>("odometry_topic", "/localization/kinematic_state");
  const std::string objects_topic = declare_parameter<std::string>(
    "objects_topic", "/perception/object_recognition/objects");

  // --- Collaborators -------------------------------------------------------
  goal_scorer_ = std::make_unique<GoalScorer>(scorer_params_);
  trajectory_planner_ = std::make_unique<PullOverTrajectoryPlanner>(trajectory_params_);

  if (keyboard_trigger_enabled_) {
    keyboard_trigger_ = std::make_unique<KeyboardTrigger>(
      keyboard_trigger_key_, [this]() { keyboard_trigger_pending_.store(true); });
    keyboard_trigger_->start();
    if (keyboard_trigger_->isActive()) {
      RCLCPP_INFO(
        get_logger(), "Keyboard trigger armed: press '%c' in this terminal to simulate an MRM "
                      "pull-over request.",
        keyboard_trigger_key_);
    } else {
      RCLCPP_WARN(
        get_logger(),
        "Keyboard trigger requested but stdin is not a TTY -- only the OperateMrm service "
        "trigger path is available in this launch context.");
    }
  }

  // --- Subscriptions ---------------------------------------------------
  centerline_sub_ = create_subscription<nav_msgs::msg::Path>(
    centerline_topic, rclcpp::QoS(1),
    [this](const nav_msgs::msg::Path::ConstSharedPtr msg) { onCenterline(msg); });

  odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odometry_topic, rclcpp::QoS(1),
    [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { onOdometry(msg); });

  objects_sub_ = create_subscription<PredictedObjects>(
    objects_topic, rclcpp::QoS(1),
    [this](const PredictedObjects::ConstSharedPtr msg) { onObjects(msg); });

  operation_mode_sub_ = create_subscription<OperationModeState>(
    "/api/operation_mode/state", rclcpp::QoS(1),
    [this](const OperationModeState::ConstSharedPtr msg) { onOperationModeState(msg); });

  // --- Service: fills Autoware's reserved pull_over_manager slot ---------
  operate_service_ = create_service<OperateMrm>(
    kOperateServiceName, [this](
                            const std::shared_ptr<OperateMrm::Request> request,
                            std::shared_ptr<OperateMrm::Response> response) {
      onOperateMrm(request, response);
    });

  status_pub_ = create_publisher<MrmBehaviorStatus>(kStatusTopicName, rclcpp::QoS(1));
  trajectory_debug_pub_ =
    create_publisher<Trajectory>("~/planned_trajectory", rclcpp::QoS(1));
  planned_path_debug_pub_ =
    create_publisher<nav_msgs::msg::Path>("~/planned_path_debug", rclcpp::QoS(1));
  planned_path_marker_pub_ =
    create_publisher<visualization_msgs::msg::Marker>("~/planned_path_marker", rclcpp::QoS(1));
  if (!direct_trajectory_output_topic_.empty()) {
    trajectory_direct_pub_ =
      create_publisher<Trajectory>(direct_trajectory_output_topic_, rclcpp::QoS(1));
    RCLCPP_WARN(
      get_logger(),
      "direct_trajectory_output_topic is set to '%s' -- this node WILL publish directly onto "
      "that topic while operating. Make sure nothing else publishes there concurrently.",
      direct_trajectory_output_topic_.c_str());
  }

  // --- Timers ---------------------------------------------------------
  poll_timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / poll_rate_hz_), [this]() { onPollTimer(); });
  status_timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / status_publish_rate_hz_), [this]() { onStatusTimer(); });
  planning_timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / planning_rate_hz_), [this]() { onPlanningTimer(); });

  RCLCPP_INFO(
    get_logger(),
    "shoulder_pullover_manager ready. Serving %s and %s; centerline topic '%s', odometry topic "
    "'%s', objects topic '%s'. Standalone trajectory planner -- does not depend on Autoware's "
    "routing/lanelet graph for the maneuver itself.",
    kOperateServiceName, kStatusTopicName, centerline_topic.c_str(), odometry_topic.c_str(),
    objects_topic.c_str());
}

PullOverManagerNode::~PullOverManagerNode()
{
  if (keyboard_trigger_) {
    keyboard_trigger_->stop();
  }
}

// --- Subscription callbacks ---------------------------------------------

void PullOverManagerNode::onCenterline(const nav_msgs::msg::Path::ConstSharedPtr msg)
{
  latest_centerline_ = msg;
}

void PullOverManagerNode::onOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  latest_odometry_ = msg;
}

void PullOverManagerNode::onObjects(const PredictedObjects::ConstSharedPtr msg)
{
  latest_objects_ = *msg;
}

void PullOverManagerNode::onOperationModeState(
  const autoware_adapi_v1_msgs::msg::OperationModeState::ConstSharedPtr msg)
{
  autonomous_mode_engaged_ = (msg->mode == OperationModeState::AUTONOMOUS);
}

// --- Service callback -----------------------------------------------------

void PullOverManagerNode::onOperateMrm(
  const std::shared_ptr<tier4_system_msgs::srv::OperateMrm::Request> request,
  std::shared_ptr<tier4_system_msgs::srv::OperateMrm::Response> response)
{
  if (!request->operate) {
    RCLCPP_INFO(get_logger(), "OperateMrm(operate=false) received -- standing down.");
    state_ = ManagerState::kIdle;
    active_goal_.reset();
    response->response.success = true;
    response->response.message = "stood down";
    return;
  }

  const auto [success, message] = triggerPullOver("OperateMrm service");
  response->response.success = success;
  response->response.message = message;
}

// --- Timers ---------------------------------------------------------------

void PullOverManagerNode::onPollTimer()
{
  if (keyboard_trigger_pending_.exchange(false)) {
    triggerPullOver("simulated keyboard trigger ('" + std::string(1, keyboard_trigger_key_) + "')");
  }
}

void PullOverManagerNode::onStatusTimer() { publishStatus(); }

visualization_msgs::msg::Marker PullOverManagerNode::toRibbonMarker(const Trajectory & trajectory) const
{
  visualization_msgs::msg::Marker marker;
  marker.header = trajectory.header;
  marker.ns = "pullover_path";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.r = 1.0F;
  marker.color.g = 1.0F;
  marker.color.b = 0.0F;
  marker.color.a = static_cast<float>(visualization_alpha_);
  // Auto-expires if this node stops publishing (e.g. a crash) instead of
  // leaving a stale shaded path behind in RViz -- republished well within
  // this window every planning cycle in normal operation.
  marker.lifetime = rclcpp::Duration::from_seconds(0.5);

  const double half_width = visualization_width_ / 2.0;

  for (std::size_t i = 0; i + 1 < trajectory.points.size(); ++i) {
    const auto & p0 = trajectory.points[i].pose;
    const auto & p1 = trajectory.points[i + 1].pose;

    const double yaw0 = yawFromQuaternionLocal(p0.orientation);
    const double yaw1 = yawFromQuaternionLocal(p1.orientation);
    const double nx0 = -std::sin(yaw0);
    const double ny0 = std::cos(yaw0);
    const double nx1 = -std::sin(yaw1);
    const double ny1 = std::cos(yaw1);

    geometry_msgs::msg::Point left0;
    left0.x = p0.position.x + half_width * nx0;
    left0.y = p0.position.y + half_width * ny0;
    left0.z = p0.position.z;
    geometry_msgs::msg::Point right0;
    right0.x = p0.position.x - half_width * nx0;
    right0.y = p0.position.y - half_width * ny0;
    right0.z = p0.position.z;
    geometry_msgs::msg::Point left1;
    left1.x = p1.position.x + half_width * nx1;
    left1.y = p1.position.y + half_width * ny1;
    left1.z = p1.position.z;
    geometry_msgs::msg::Point right1;
    right1.x = p1.position.x - half_width * nx1;
    right1.y = p1.position.y - half_width * ny1;
    right1.z = p1.position.z;

    // Two triangles per segment, consistent winding, forming a solid
    // quad/ribbon between consecutive trajectory points.
    marker.points.push_back(left0);
    marker.points.push_back(right0);
    marker.points.push_back(left1);

    marker.points.push_back(right0);
    marker.points.push_back(right1);
    marker.points.push_back(left1);
  }

  return marker;
}

void PullOverManagerNode::publishStatus()
{
  MrmBehaviorStatus status;
  status.stamp = now();

  const bool ready = latest_centerline_ != nullptr && !latest_centerline_->poses.empty() &&
                      latest_odometry_ != nullptr;

  switch (state_) {
    case ManagerState::kOperating:
      status.state = MrmBehaviorStatus::OPERATING;
      break;
    default:
      status.state = ready ? MrmBehaviorStatus::AVAILABLE : MrmBehaviorStatus::NOT_AVAILABLE;
      break;
  }
  status_pub_->publish(status);
}

void PullOverManagerNode::onPlanningTimer()
{
  if (state_ != ManagerState::kOperating || !active_goal_.has_value()) {
    return;
  }
  if (!latest_odometry_) {
    return;  // Can't plan without knowing where we are.
  }

  const auto & ego_pose = latest_odometry_->pose.pose;
  const double ego_speed = std::hypot(
    latest_odometry_->twist.twist.linear.x, latest_odometry_->twist.twist.linear.y);

  const double distance_to_goal = std::hypot(
    active_goal_->position.x - ego_pose.position.x, active_goal_->position.y - ego_pose.position.y);

  if (distance_to_goal <= arrival_distance_threshold_ && ego_speed <= arrival_speed_threshold_) {
    RCLCPP_INFO(
      get_logger(), "Arrived at pull-over goal (distance=%.2fm, speed=%.2fm/s) -- maneuver succeeded.",
      distance_to_goal, ego_speed);
    state_ = ManagerState::kSucceeded;
    active_goal_.reset();
    return;
  }

  KinematicState start;
  start.pose = ego_pose;
  start.speed = ego_speed;

  const auto trajectory =
    trajectory_planner_->plan(start, *active_goal_, latest_objects_, now());

  if (!trajectory.has_value()) {
    ++consecutive_planning_failures_;
    RCLCPP_WARN(
      get_logger(), "No feasible/collision-free trajectory found this cycle (%d/%d consecutive).",
      consecutive_planning_failures_, max_consecutive_planning_failures_);
    if (consecutive_planning_failures_ >= max_consecutive_planning_failures_) {
      RCLCPP_ERROR(
        get_logger(),
        "Giving up: no feasible trajectory for %d consecutive cycles -- likely blocked by an "
        "obstacle or an infeasible goal.",
        consecutive_planning_failures_);
      state_ = ManagerState::kFailed;
      active_goal_.reset();
    }
    return;
  }

  consecutive_planning_failures_ = 0;
  trajectory_debug_pub_->publish(*trajectory);
  planned_path_debug_pub_->publish(toDebugPath(*trajectory));
  planned_path_marker_pub_->publish(toRibbonMarker(*trajectory));
  if (trajectory_direct_pub_) {
    trajectory_direct_pub_->publish(*trajectory);
  }
}

// --- Core maneuver logic ---------------------------------------------------

std::pair<bool, std::string> PullOverManagerNode::triggerPullOver(const std::string & trigger_source)
{
  RCLCPP_INFO(get_logger(), "Pull-over trigger received from: %s", trigger_source.c_str());

  if (state_ == ManagerState::kOperating) {
    const std::string message = "Ignored: a pull-over maneuver is already in progress.";
    RCLCPP_WARN(get_logger(), "%s", message.c_str());
    return {false, message};
  }

  if (require_autonomous_mode_ && !autonomous_mode_engaged_) {
    const std::string message =
      "Refused: vehicle is not currently in AUTONOMOUS operation mode -- engage autonomous "
      "driving before requesting a pull-over.";
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    state_ = ManagerState::kFailed;
    return {false, message};
  }

  if (!latest_centerline_ || latest_centerline_->poses.empty()) {
    const std::string message = "Refused: no shoulder centerline data received yet.";
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    state_ = ManagerState::kFailed;
    return {false, message};
  }

  if (!latest_odometry_) {
    const std::string message = "Refused: no ego odometry received yet.";
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    state_ = ManagerState::kFailed;
    return {false, message};
  }

  const double ego_speed = std::hypot(
    latest_odometry_->twist.twist.linear.x, latest_odometry_->twist.twist.linear.y);

  const auto goal = goal_scorer_->selectBestGoal(
    *latest_centerline_, latest_odometry_->pose.pose, ego_speed);

  if (!goal.has_value()) {
    const std::string message =
      "Refused: no feasible shoulder goal found (nothing within range, ahead of ego, past the "
      "minimum stopping distance, with enough confirmed continuity).";
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    state_ = ManagerState::kFailed;
    return {false, message};
  }

  RCLCPP_INFO(
    get_logger(),
    "Selected shoulder goal at (%.2f, %.2f): score=%.3f distance=%.1fm curvature=%.4f "
    "continuity=%.1fm. Starting standalone trajectory planner (replanning at %.1f Hz).",
    goal->pose.position.x, goal->pose.position.y, goal->score, goal->distance_from_ego,
    goal->curvature, goal->continuity_length, planning_rate_hz_);

  active_goal_ = goal->pose;
  consecutive_planning_failures_ = 0;
  state_ = ManagerState::kOperating;
  return {true, "Pull-over trajectory planning started toward the selected shoulder goal."};
}

}  // namespace autoware::shoulder_pullover_manager
