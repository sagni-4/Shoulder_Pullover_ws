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
  scorer_params_.min_maneuver_forward_progress = declare_parameter<double>(
    "min_maneuver_forward_progress", scorer_params_.min_maneuver_forward_progress);
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
  scorer_params_.weight_maneuver_ease =
    declare_parameter<double>("weight_maneuver_ease", scorer_params_.weight_maneuver_ease);
  scorer_params_.weight_jerk_ease =
    declare_parameter<double>("weight_jerk_ease", scorer_params_.weight_jerk_ease);
  scorer_params_.max_maneuver_curvature =
    declare_parameter<double>("max_maneuver_curvature", scorer_params_.max_maneuver_curvature);
  scorer_params_.max_maneuver_jerk =
    declare_parameter<double>("max_maneuver_jerk", scorer_params_.max_maneuver_jerk);
  scorer_params_.heading_smoothing_distance = declare_parameter<double>(
    "heading_smoothing_distance", scorer_params_.heading_smoothing_distance);

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
  trajectory_params_.min_path_turn_radius = declare_parameter<double>(
    "trajectory_min_path_turn_radius", trajectory_params_.min_path_turn_radius);
  trajectory_params_.max_path_turn_radius = declare_parameter<double>(
    "trajectory_max_path_turn_radius", trajectory_params_.max_path_turn_radius);
  trajectory_params_.path_turn_radius_step = declare_parameter<double>(
    "trajectory_path_turn_radius_step", trajectory_params_.path_turn_radius_step);
  trajectory_params_.max_curvature =
    declare_parameter<double>("trajectory_max_curvature", trajectory_params_.max_curvature);
  trajectory_params_.min_speed_for_curvature_check = declare_parameter<double>(
    "trajectory_min_speed_for_curvature_check", trajectory_params_.min_speed_for_curvature_check);
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
  max_consecutive_traffic_blocked_cycles_ =
    declare_parameter<int>("max_consecutive_traffic_blocked_cycles", 300);
  deceleration_lookahead_distance_ = declare_parameter<double>(
    "deceleration_lookahead_distance", deceleration_lookahead_distance_);
  deceleration_giveup_speed_threshold_ = declare_parameter<double>(
    "deceleration_giveup_speed_threshold", deceleration_giveup_speed_threshold_);
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

  // --- Parameters: controller STOPPED-state departure fix -------------------
  // See pushControllerStopDistOverride()'s docs for the root cause this
  // works around (autoware_pid_longitudinal_controller's departure_condition
  // being unsatisfiable for a receding-horizon planner that legitimately
  // starts each replan at ego's real, near-zero current speed).
  controller_stop_dist_override_enabled_ =
    declare_parameter<bool>("controller_stop_dist_override_enabled", true);
  controller_node_name_ = declare_parameter<std::string>(
    "controller_node_name", controller_node_name_);
  controller_override_drive_state_stop_dist_ = declare_parameter<double>(
    "controller_override_drive_state_stop_dist", controller_override_drive_state_stop_dist_);
  controller_override_stuck_speed_threshold_ = declare_parameter<double>(
    "controller_override_stuck_speed_threshold", controller_override_stuck_speed_threshold_);
  controller_override_stuck_duration_ = declare_parameter<double>(
    "controller_override_stuck_duration", controller_override_stuck_duration_);

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

  if (controller_stop_dist_override_enabled_) {
    controller_param_client_ =
      std::make_shared<rclcpp::AsyncParametersClient>(this, controller_node_name_);
    // Block briefly (constructor time, before spin() starts) until this
    // client's DDS request writer has actually matched the controller's
    // get_parameters service reader. Skipping this and firing get_parameters
    // immediately is a classic silent-drop race: a request sent before
    // endpoint matching completes is dropped with no error and no timeout,
    // which is exactly what happened the first time this was live-tested --
    // the capture callback below simply never fired, neither the success nor
    // the failure branch.
    if (!controller_param_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_WARN(
        get_logger(),
        "Controller node '%s' parameter service not available after 5s at startup -- the "
        "STOPPED-state departure override will stay disabled.",
        controller_node_name_.c_str());
    } else {
      // Capture the live controller's own current drive_state_stop_dist once,
      // at startup, as the value to restore to later -- deliberately not
      // hardcoded to the launch-file default, in case some other override is
      // already in effect for this stack. If the call fails, keep the
      // built-in fallback (0.5, the stock default) and leave
      // have_original_drive_state_stop_dist_ false so the override path logs
      // a warning and refuses to act rather than silently overwriting a value
      // it never actually confirmed.
      controller_param_client_->get_parameters(
        {"drive_state_stop_dist"},
        [this](std::shared_future<std::vector<rclcpp::Parameter>> future) {
          const auto params = future.get();
          if (params.empty()) {
            RCLCPP_WARN(
              get_logger(),
              "Could not read drive_state_stop_dist from '%s' at startup -- the "
              "STOPPED-state departure override will stay disabled until this succeeds.",
              controller_node_name_.c_str());
            return;
          }
          original_drive_state_stop_dist_ = params.front().as_double();
          have_original_drive_state_stop_dist_ = true;
          RCLCPP_INFO(
            get_logger(), "Captured original drive_state_stop_dist=%.3f from '%s'.",
            original_drive_state_stop_dist_, controller_node_name_.c_str());
        });
    }
  }

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

  change_to_stop_client_ = create_client<autoware_adapi_v1_msgs::srv::ChangeOperationMode>(
    "/api/operation_mode/change_to_stop");

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
    deceleration_goal_.reset();
    last_trajectory_.reset();
    zero_speed_since_.reset();
    contingency_stop_goal_.reset();
    consecutive_traffic_blocked_cycles_ = 0;
    restoreControllerStopDistOverride();
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
    case ManagerState::kDecelerating:
    case ManagerState::kOperating:
      // Both report OPERATING to mrm_handler/the trajectory gate -- from
      // their perspective the MRM has been engaged and is actively doing
      // something, whether that is in-lane braking or the curved
      // maneuver itself (see kDecelerating's docs).
      status.state = MrmBehaviorStatus::OPERATING;
      break;
    default:
      status.state = ready ? MrmBehaviorStatus::AVAILABLE : MrmBehaviorStatus::NOT_AVAILABLE;
      break;
  }
  status_pub_->publish(status);
}

void PullOverManagerNode::pushControllerStopDistOverride()
{
  if (!controller_stop_dist_override_enabled_ || controller_override_active_) {
    return;
  }
  if (!have_original_drive_state_stop_dist_) {
    RCLCPP_WARN(
      get_logger(),
      "Skipping controller STOPPED-state departure override: original "
      "drive_state_stop_dist was never successfully read from '%s'. The vehicle may stall "
      "at 0.0 m/s if it comes to a stop mid-maneuver (see project notes on the "
      "autoware_pid_longitudinal_controller STOPPED-state bug).",
      controller_node_name_.c_str());
    return;
  }
  controller_override_active_ = true;
  controller_param_client_->set_parameters(
    {rclcpp::Parameter(
      "drive_state_stop_dist", controller_override_drive_state_stop_dist_)},
    [this](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
      for (const auto & result : future.get()) {
        if (!result.successful) {
          RCLCPP_ERROR(
            get_logger(), "Controller rejected drive_state_stop_dist override: %s",
            result.reason.c_str());
        }
      }
    });
  RCLCPP_WARN(
    get_logger(),
    "Ego has been stuck below %.2f m/s for %.1fs mid-maneuver -- pushing "
    "drive_state_stop_dist=%.3f to '%s' (was %.3f) to break the controller out of a latched "
    "STOPPED state.",
    controller_override_stuck_speed_threshold_, controller_override_stuck_duration_,
    controller_override_drive_state_stop_dist_, controller_node_name_.c_str(),
    original_drive_state_stop_dist_);
}

void PullOverManagerNode::restoreControllerStopDistOverride()
{
  if (!controller_override_active_) {
    return;
  }
  controller_override_active_ = false;
  controller_param_client_->set_parameters(
    {rclcpp::Parameter("drive_state_stop_dist", original_drive_state_stop_dist_)},
    [this](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
      for (const auto & result : future.get()) {
        if (!result.successful) {
          RCLCPP_ERROR(
            get_logger(), "Controller rejected drive_state_stop_dist restore: %s",
            result.reason.c_str());
        }
      }
    });
  RCLCPP_INFO(
    get_logger(), "Restored drive_state_stop_dist=%.3f on '%s'.",
    original_drive_state_stop_dist_, controller_node_name_.c_str());
}

void PullOverManagerNode::requestOperationModeStop()
{
  if (!change_to_stop_client_->service_is_ready()) {
    RCLCPP_WARN(
      get_logger(),
      "/api/operation_mode/change_to_stop not available -- mrm_handler's own MRM state will "
      "stay latched at OPERATING even though the maneuver succeeded (it waits for operation "
      "mode to reach STOP, see requestOperationModeStop()'s docs).");
    return;
  }
  auto request = std::make_shared<autoware_adapi_v1_msgs::srv::ChangeOperationMode::Request>();
  change_to_stop_client_->async_send_request(
    request,
    [this](rclcpp::Client<autoware_adapi_v1_msgs::srv::ChangeOperationMode>::SharedFuture future) {
      const auto response = future.get();
      if (!response->status.success) {
        RCLCPP_WARN(
          get_logger(), "change_to_stop after arrival failed: %s",
          response->status.message.c_str());
      }
    });
}

void PullOverManagerNode::checkControllerStuckAndOverride(
  double ego_speed, const rclcpp::Time & now)
{
  if (!controller_stop_dist_override_enabled_) {
    return;
  }
  if (ego_speed >= controller_override_stuck_speed_threshold_) {
    // Moving again (or never stuck) -- restoreControllerStopDistOverride()
    // is itself a no-op if the override was never pushed, so it's safe to
    // call unconditionally here every cycle.
    zero_speed_since_.reset();
    restoreControllerStopDistOverride();
    return;
  }
  if (!zero_speed_since_.has_value()) {
    zero_speed_since_ = now;
    return;
  }
  const double stuck_duration = (now - *zero_speed_since_).seconds();
  if (stuck_duration >= controller_override_stuck_duration_) {
    pushControllerStopDistOverride();
  }
}

void PullOverManagerNode::onPlanningTimer()
{
  if (state_ != ManagerState::kDecelerating && state_ != ManagerState::kOperating) {
    return;
  }
  if (!latest_odometry_) {
    return;  // Can't plan without knowing where we are.
  }

  const auto & ego_pose = latest_odometry_->pose.pose;
  const double ego_speed = std::hypot(
    latest_odometry_->twist.twist.linear.x, latest_odometry_->twist.twist.linear.y);

  checkControllerStuckAndOverride(ego_speed, now());

  if (state_ == ManagerState::kDecelerating) {
    // Re-check every cycle at the *current* (dropping) speed -- see
    // kDecelerating's docs for why there is no single fixed "slow enough"
    // threshold to wait for instead.
    if (latest_centerline_ && !latest_centerline_->poses.empty()) {
      const auto goal =
        goal_scorer_->selectBestGoal(*latest_centerline_, ego_pose, ego_speed);
      if (goal.has_value()) {
        RCLCPP_INFO(
          get_logger(),
          "Feasible shoulder goal now reachable at %.2f m/s (score=%.3f distance=%.1fm "
          "maneuver_curvature=%.4f maneuver_initial_jerk=%.3f) -- switching from in-lane "
          "braking to the pull-over maneuver.",
          ego_speed, goal->score, goal->distance_from_ego, goal->maneuver_curvature,
          goal->maneuver_initial_jerk);
        active_goal_ = goal->pose;
        deceleration_goal_.reset();
        consecutive_planning_failures_ = 0;
        state_ = ManagerState::kOperating;
        return;  // Let the next cycle (100ms away) build the real pull-over trajectory.
      }
    }

    if (ego_speed <= deceleration_giveup_speed_threshold_) {
      RCLCPP_ERROR(
        get_logger(),
        "Decelerated to %.2f m/s with still no feasible shoulder goal within range -- "
        "giving up (nothing reachable here, not just \"too fast right now\").",
        ego_speed);
      state_ = ManagerState::kFailed;
      deceleration_goal_.reset();
      restoreControllerStopDistOverride();
      return;
    }

    // Not slow enough yet (or centerline data momentarily unavailable) --
    // keep braking straight ahead, in-lane, toward the *fixed* goal picked
    // when kDecelerating began (see deceleration_goal_'s docs for why this
    // must not be resynthesized from ego's current pose every cycle).
    // Deliberately reuses PullOverTrajectoryPlanner itself.
    if (!deceleration_goal_.has_value()) {
      deceleration_goal_ = buildDecelerationGoal(ego_pose);
    }

    const auto plan_stamp = now();
    KinematicState start;
    start.pose = ego_pose;
    start.speed = ego_speed;
    start.accel = seedAccelerationFromLastTrajectory(plan_stamp);

    std::string failure_reason;
    const auto trajectory = trajectory_planner_->plan(
      start, *deceleration_goal_, latest_objects_, plan_stamp, &failure_reason);
    if (!trajectory.has_value()) {
      RCLCPP_WARN(
        get_logger(), "No feasible in-lane braking trajectory found this cycle. Last reason: %s",
        failure_reason.c_str());
      return;  // Transient; try again next cycle, no consecutive-failure counting here --
               // this straight-line case is not expected to fail the way a curved
               // pull-over approach can.
    }
    publishTrajectory(*trajectory);
    return;
  }

  // state_ == kOperating from here on -- unchanged from before kDecelerating existed.
  if (!active_goal_.has_value()) {
    return;
  }

  const double distance_to_goal = std::hypot(
    active_goal_->position.x - ego_pose.position.x, active_goal_->position.y - ego_pose.position.y);

  if (distance_to_goal <= arrival_distance_threshold_ && ego_speed <= arrival_speed_threshold_) {
    RCLCPP_INFO(
      get_logger(), "Arrived at pull-over goal (distance=%.2fm, speed=%.2fm/s) -- maneuver succeeded.",
      distance_to_goal, ego_speed);
    state_ = ManagerState::kSucceeded;
    active_goal_.reset();
    restoreControllerStopDistOverride();
    requestOperationModeStop();
    return;
  }

  const auto plan_stamp = now();
  KinematicState start;
  start.pose = ego_pose;
  start.speed = ego_speed;
  start.accel = seedAccelerationFromLastTrajectory(plan_stamp);

  std::string failure_reason;
  bool blocked_by_traffic = false;
  const auto trajectory = trajectory_planner_->plan(
    start, *active_goal_, latest_objects_, plan_stamp, &failure_reason, &blocked_by_traffic);

  if (trajectory.has_value()) {
    consecutive_planning_failures_ = 0;
    consecutive_traffic_blocked_cycles_ = 0;
    contingency_stop_goal_.reset();
    publishTrajectory(*trajectory);
    return;
  }

  if (blocked_by_traffic) {
    // The real maneuver's shape/timing is fine -- something is just
    // currently in the way. Contingency-brake toward a fixed point ahead
    // (same buildDecelerationGoal()/fix-once pattern kDecelerating uses,
    // see contingency_stop_goal_'s docs for why resynthesizing it fresh
    // every cycle would silently defeat real deceleration) rather than
    // either blindly continuing the last stale goal-directed trajectory or
    // publishing nothing until the tight max_consecutive_planning_failures_
    // threshold gives up on what is not actually an infeasible goal.
    ++consecutive_traffic_blocked_cycles_;
    if (!contingency_stop_goal_.has_value()) {
      contingency_stop_goal_ = buildDecelerationGoal(ego_pose);
      RCLCPP_WARN(
        get_logger(),
        "Pull-over path blocked by traffic (%s) -- contingency-braking in place until it "
        "clears.",
        failure_reason.c_str());
    }
    std::string contingency_reason;
    const auto contingency_trajectory = trajectory_planner_->plan(
      start, *contingency_stop_goal_, latest_objects_, plan_stamp, &contingency_reason);
    if (contingency_trajectory.has_value()) {
      publishTrajectory(*contingency_trajectory);
    } else {
      // Rare (the contingency goal is a trivial straight-line brake) -- e.g.
      // the blocking object is now directly ahead on that line too. Publish
      // nothing this cycle; the gate holds the last trajectory (which was
      // itself already collision-checked when generated) until next cycle.
      RCLCPP_WARN(
        get_logger(), "Contingency brake trajectory also infeasible this cycle. Last reason: %s",
        contingency_reason.c_str());
    }
    if (consecutive_traffic_blocked_cycles_ >= max_consecutive_traffic_blocked_cycles_) {
      RCLCPP_ERROR(
        get_logger(), "Giving up: pull-over path has been blocked by traffic for %d consecutive cycles.",
        consecutive_traffic_blocked_cycles_);
      state_ = ManagerState::kFailed;
      active_goal_.reset();
      contingency_stop_goal_.reset();
      restoreControllerStopDistOverride();
    }
    return;
  }

  // Not traffic -- a genuine kinematic/shape infeasibility for this goal.
  ++consecutive_planning_failures_;
  RCLCPP_WARN(
    get_logger(),
    "No feasible trajectory found this cycle (%d/%d consecutive). Last reason: %s",
    consecutive_planning_failures_, max_consecutive_planning_failures_, failure_reason.c_str());
  if (consecutive_planning_failures_ >= max_consecutive_planning_failures_) {
    RCLCPP_ERROR(
      get_logger(),
      "Giving up: no feasible trajectory for %d consecutive cycles -- infeasible goal.",
      consecutive_planning_failures_);
    state_ = ManagerState::kFailed;
    active_goal_.reset();
    restoreControllerStopDistOverride();
  }
}

void PullOverManagerNode::publishTrajectory(const Trajectory & trajectory)
{
  trajectory_debug_pub_->publish(trajectory);
  planned_path_debug_pub_->publish(toDebugPath(trajectory));
  planned_path_marker_pub_->publish(toRibbonMarker(trajectory));
  if (trajectory_direct_pub_) {
    trajectory_direct_pub_->publish(trajectory);
  }
  // Kept only to warm-start the next cycle's start-acceleration boundary
  // condition -- see seedAccelerationFromLastTrajectory().
  last_trajectory_ = trajectory;
}

geometry_msgs::msg::Pose PullOverManagerNode::buildDecelerationGoal(
  const geometry_msgs::msg::Pose & ego_pose) const
{
  const double yaw = yawFromQuaternionLocal(ego_pose.orientation);
  geometry_msgs::msg::Pose goal = ego_pose;
  goal.position.x += deceleration_lookahead_distance_ * std::cos(yaw);
  goal.position.y += deceleration_lookahead_distance_ * std::sin(yaw);
  return goal;
}

double PullOverManagerNode::seedAccelerationFromLastTrajectory(const rclcpp::Time & now) const
{
  if (!last_trajectory_.has_value() || last_trajectory_->points.empty()) {
    return 0.0;  // Nothing to warm-start from yet (first cycle of a fresh maneuver).
  }

  const double elapsed = (now - rclcpp::Time(last_trajectory_->header.stamp)).seconds();
  if (elapsed < 0.0) {
    return 0.0;  // Clock anomaly -- don't trust it.
  }

  const auto & points = last_trajectory_->points;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double t_i = rclcpp::Duration(points[i].time_from_start).seconds();
    if (t_i < elapsed) {
      continue;
    }
    if (i == 0) {
      return static_cast<double>(points[0].acceleration_mps2);
    }
    const double t_prev = rclcpp::Duration(points[i - 1].time_from_start).seconds();
    const double span = t_i - t_prev;
    if (span <= 1e-6) {
      return static_cast<double>(points[i].acceleration_mps2);
    }
    const double ratio = (elapsed - t_prev) / span;
    const double a_prev = points[i - 1].acceleration_mps2;
    const double a_i = points[i].acceleration_mps2;
    return a_prev + ratio * (a_i - a_prev);
  }

  // elapsed is beyond that trajectory's own horizon entirely -- a missed
  // cycle or genuinely stale state. Don't extrapolate past what that plan
  // ever claimed; fall back to the safe default instead.
  return 0.0;
}

// --- Core maneuver logic ---------------------------------------------------

std::pair<bool, std::string> PullOverManagerNode::triggerPullOver(const std::string & trigger_source)
{
  RCLCPP_INFO(get_logger(), "Pull-over trigger received from: %s", trigger_source.c_str());

  if (state_ == ManagerState::kOperating || state_ == ManagerState::kDecelerating) {
    const std::string message = "Ignored: a pull-over maneuver is already in progress.";
    RCLCPP_WARN(get_logger(), "%s", message.c_str());
    return {false, message};
  }

  // Fresh trigger -- don't warm-start the new maneuver's acceleration
  // boundary condition from a previous, unrelated maneuver's leftover
  // trajectory (see seedAccelerationFromLastTrajectory()).
  last_trajectory_.reset();
  zero_speed_since_.reset();
  contingency_stop_goal_.reset();
  consecutive_traffic_blocked_cycles_ = 0;

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
    // Not a refusal -- see kDecelerating's docs for why "no feasible goal at
    // the current speed" is treated as "not yet", not "never", given
    // feasibility is speed-dependent (jerk scales with speed cubed) rather
    // than being about this goal being permanently unreachable.
    const std::string message =
      "No feasible shoulder goal at current speed (" + std::to_string(ego_speed) +
      " m/s) -- decelerating in-lane until one becomes reachable.";
    RCLCPP_WARN(get_logger(), "%s", message.c_str());
    state_ = ManagerState::kDecelerating;
    return {true, message};
  }

  RCLCPP_INFO(
    get_logger(),
    "Selected shoulder goal at (%.2f, %.2f): score=%.3f distance=%.1fm curvature=%.4f "
    "continuity=%.1fm maneuver_curvature=%.4f maneuver_initial_jerk=%.3f. Starting standalone "
    "trajectory planner (replanning at %.1f Hz).",
    goal->pose.position.x, goal->pose.position.y, goal->score, goal->distance_from_ego,
    goal->curvature, goal->continuity_length, goal->maneuver_curvature,
    goal->maneuver_initial_jerk, planning_rate_hz_);

  active_goal_ = goal->pose;
  consecutive_planning_failures_ = 0;
  state_ = ManagerState::kOperating;
  return {true, "Pull-over trajectory planning started toward the selected shoulder goal."};
}

}  // namespace autoware::shoulder_pullover_manager
