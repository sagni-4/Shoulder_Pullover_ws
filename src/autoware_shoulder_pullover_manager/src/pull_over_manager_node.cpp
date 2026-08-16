#include "autoware_shoulder_pullover_manager/pull_over_manager_node.hpp"

#include <chrono>
#include <cmath>
#include <limits>
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
  // Measured-width gate (added 2026-08-16, see goal_scorer.hpp) -- vehicle_half_width
  // itself is populated below from the same vehicle_info fields the arrival gate uses.
  scorer_params_.goal_width_margin =
    declare_parameter<double>("goal_width_margin", scorer_params_.goal_width_margin);
  scorer_params_.width_measurement_scale = declare_parameter<double>(
    "width_measurement_scale", scorer_params_.width_measurement_scale);
  scorer_params_.width_window_half_length = declare_parameter<double>(
    "width_window_half_length", scorer_params_.width_window_half_length);
  scorer_params_.width_window_quantile = declare_parameter<double>(
    "width_window_quantile", scorer_params_.width_window_quantile);
  scorer_params_.heading_change_curvature_coefficient = declare_parameter<double>(
    "heading_change_curvature_coefficient",
    scorer_params_.heading_change_curvature_coefficient);
  scorer_params_.forward_progress_safety_factor = declare_parameter<double>(
    "forward_progress_safety_factor", scorer_params_.forward_progress_safety_factor);
  scorer_params_.weight_width =
    declare_parameter<double>("weight_width", scorer_params_.weight_width);
  scorer_params_.reference_extra_half_width = declare_parameter<double>(
    "reference_extra_half_width", scorer_params_.reference_extra_half_width);
  scorer_params_.require_width_data =
    declare_parameter<bool>("require_width_data", scorer_params_.require_width_data);

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
  trajectory_params_.min_departure_reference_speed = declare_parameter<double>(
    "trajectory_min_departure_reference_speed", trajectory_params_.min_departure_reference_speed);
  trajectory_params_.final_approach_distance = declare_parameter<double>(
    "trajectory_final_approach_distance", trajectory_params_.final_approach_distance);
  trajectory_params_.final_approach_decel = declare_parameter<double>(
    "trajectory_final_approach_decel", trajectory_params_.final_approach_decel);
  trajectory_params_.max_speed =
    declare_parameter<double>("trajectory_max_speed", trajectory_params_.max_speed);
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
  arrival_heading_threshold_ =
    declare_parameter<double>("arrival_heading_threshold", arrival_heading_threshold_);
  lateral_safety_margin_ =
    declare_parameter<double>("lateral_safety_margin", lateral_safety_margin_);
  require_shoulder_containment_ =
    declare_parameter<bool>("require_shoulder_containment", require_shoulder_containment_);
  // Same vehicle_info fields (wheel_tread, left_overhang, right_overhang) every other node in
  // this launch receives -- see vehicle_half_width_'s docs for the formula. Declared with
  // sample_vehicle's own known values as defaults so a standalone launch (no vehicle_info
  // params passed at all) still gets a real number instead of silently defaulting to 0.
  {
    const double wheel_tread = declare_parameter<double>("wheel_tread", 1.64);
    const double left_overhang = declare_parameter<double>("left_overhang", 0.128);
    const double right_overhang = declare_parameter<double>("right_overhang", 0.128);
    vehicle_half_width_ = 0.5 * wheel_tread + std::max(left_overhang, right_overhang);
    // base_link -> front bumper, for GoalScorer's swept-footprint check. Same vehicle_info
    // fields every other node in this launch receives.
    const double wheel_base = declare_parameter<double>("wheel_base", 2.79);
    const double front_overhang = declare_parameter<double>("front_overhang", 1.0);
    scorer_params_.vehicle_front_length = wheel_base + front_overhang;
  }
  final_approach_engage_distance_ = declare_parameter<double>(
    "final_approach_engage_distance", final_approach_engage_distance_);
  final_approach_heading_tolerance_ = declare_parameter<double>(
    "final_approach_heading_tolerance", final_approach_heading_tolerance_);
  final_approach_reference_speed_ = declare_parameter<double>(
    "final_approach_reference_speed", final_approach_reference_speed_);
  scorer_params_.goal_inboard_bias =
    declare_parameter<double>("goal_inboard_bias", scorer_params_.goal_inboard_bias);
  max_consecutive_planning_failures_ =
    declare_parameter<int>("max_consecutive_planning_failures", 30);
  max_consecutive_traffic_blocked_cycles_ =
    declare_parameter<int>("max_consecutive_traffic_blocked_cycles", 300);
  standdown_grace_period_ =
    declare_parameter<double>("standdown_grace_period", standdown_grace_period_);
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

  // See pushAebImuPathOverride()'s docs for the root cause this works around
  // (AEB's plan-agnostic IMU-path collision fallback firing on an intentional,
  // planned off-lane maneuver).
  aeb_override_enabled_ = declare_parameter<bool>("aeb_override_enabled", true);
  aeb_node_name_ = declare_parameter<std::string>("aeb_node_name", aeb_node_name_);

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
  // Physically derived from trajectory_params_ itself (see max_reachable_distance's docs in
  // goal_scorer.hpp) -- kept computed here, not independently configured, so the two components
  // can never drift out of consistency with each other.
  scorer_params_.max_reachable_distance =
    0.9 * (15.0 / 8.0) * trajectory_params_.max_speed * trajectory_params_.max_duration;
  // Same computed value the arrival containment gate uses -- goal selection and arrival
  // must agree on how wide the vehicle actually is (see goal_scorer.hpp's
  // vehicle_half_width docs).
  scorer_params_.vehicle_half_width = vehicle_half_width_;
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

  if (aeb_override_enabled_) {
    aeb_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, aeb_node_name_);
    // Same endpoint-matching race as controller_param_client_ above -- block
    // briefly at construction time rather than firing immediately.
    if (!aeb_param_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_WARN(
        get_logger(),
        "AEB node '%s' parameter service not available after 5s at startup -- the AEB "
        "IMU-path override will stay disabled (AEB's plan-agnostic collision fallback may "
        "false-positive on an intentional pull-over maneuver, see pushAebImuPathOverride()'s "
        "docs).",
        aeb_node_name_.c_str());
      aeb_param_client_.reset();
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

  // Published by shoulder_centerline_node index-aligned with centerline_topic's poses -- see
  // isConfirmedSafeToStop()'s docs. Naming matches that node's halfwidth_pub_ (its own topic
  // name + "_halfwidth"), not independently configurable here to keep the pairing unambiguous.
  shoulder_halfwidth_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
    centerline_topic + "_halfwidth", rclcpp::QoS(1),
    [this](const std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {
      latest_shoulder_halfwidth_ = msg;
    });

  odometry_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odometry_topic, rclcpp::QoS(1),
    [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { onOdometry(msg); });

  objects_sub_ = create_subscription<PredictedObjects>(
    objects_topic, rclcpp::QoS(1),
    [this](const PredictedObjects::ConstSharedPtr msg) { onObjects(msg); });

  // Publisher (/adapi/node/operation_mode) is TRANSIENT_LOCAL and only
  // republishes on change -- live-verified 2026-08-12: with the default
  // (VOLATILE) subscription QoS used here before, a fresh instance of this
  // node started while AUTONOMOUS mode was already engaged (no new change
  // event to trigger a republish) never received a single message on this
  // topic, so autonomous_mode_engaged_ stayed at its false default forever
  // and every trigger was refused. Match durability to actually receive the
  // latched last-known state immediately on startup.
  operation_mode_sub_ = create_subscription<OperationModeState>(
    "/api/operation_mode/state", rclcpp::QoS(1).transient_local(),
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
    if (state_ == ManagerState::kOperating || state_ == ManagerState::kDecelerating) {
      // See standdown_requested_since_'s docs -- don't throw away an active maneuver on the
      // first cancel; give it standdown_grace_period_ to either be reversed by a fresh
      // operate=true (an upstream blip, not a real decision) or actually elapse
      // (onPlanningTimer() performs the real stand-down then).
      if (!standdown_requested_since_.has_value()) {
        standdown_requested_since_ = now();
        RCLCPP_INFO(
          get_logger(),
          "OperateMrm(operate=false) received while a maneuver is active -- starting a %.1fs "
          "grace period before actually standing down, instead of discarding progress "
          "immediately (see standdown_requested_since_'s docs).",
          standdown_grace_period_);
      }
      response->response.success = true;
      response->response.message = "cancel acknowledged, pending grace period";
      return;
    }
    performStandDown("OperateMrm(operate=false) received");
    response->response.success = true;
    response->response.message = "stood down";
    return;
  }

  if (standdown_requested_since_.has_value()) {
    // A pending cancel is being reversed by a fresh engage before the grace period elapsed --
    // exactly the upstream-blip case standdown_requested_since_ exists to absorb. Resume
    // without touching active_goal_/last_trajectory_/anything else -- there is nothing to
    // re-trigger since, from this node's own state machine's perspective, the maneuver never
    // actually stopped.
    RCLCPP_INFO(
      get_logger(),
      "OperateMrm(operate=true) received %.2fs into a pending grace-period stand-down -- "
      "treating the cancel as a transient upstream blip and resuming without resetting "
      "progress.",
      (now() - *standdown_requested_since_).seconds());
    standdown_requested_since_.reset();
    response->response.success = true;
    response->response.message = "already in progress -- pending stand-down cancelled";
    return;
  }

  const auto [success, message] = triggerPullOver("OperateMrm service");
  response->response.success = success;
  response->response.message = message;
}

bool PullOverManagerNode::reselectGoalInPlace(
  const geometry_msgs::msg::Pose & ego_pose, double ego_speed)
{
  if (!latest_centerline_ || latest_centerline_->poses.empty()) {
    return false;
  }
  // Never swap the goal during final approach -- see final_approach_engage_distance_'s docs.
  // Every candidate GoalScorer can return is at least min_stopping_distance (~5m) ahead, so
  // re-selecting here would move the target further away than it currently is and the vehicle
  // could never close the last metre.
  if (active_goal_.has_value()) {
    const double distance_to_active_goal = std::hypot(
      active_goal_->position.x - ego_pose.position.x,
      active_goal_->position.y - ego_pose.position.y);
    if (distance_to_active_goal <= final_approach_engage_distance_) {
      return false;
    }
  }
  const auto goal =
    goal_scorer_->selectBestGoal(*latest_centerline_, ego_pose, ego_speed, usableHalfWidths());
  if (!goal.has_value()) {
    return false;
  }
  RCLCPP_INFO(
    get_logger(),
    "Re-selected shoulder goal in place at (%.2f, %.2f) without leaving the maneuver "
    "(score=%.3f distance=%.1fm maneuver_curvature=%.4f measured_half_width=%.2fm) -- the "
    "approach keeps converging instead of restarting from a straight-ahead crawl.",
    goal->pose.position.x, goal->pose.position.y, goal->score, goal->distance_from_ego,
    goal->maneuver_curvature, goal->measured_half_width);
  active_goal_ = goal->pose;
  active_goal_shape_hint_.reset();  // Fresh geometry -- last cycle's shape no longer applies.
  contingency_stop_goal_.reset();
  contingency_stop_goal_shape_hint_.reset();
  consecutive_planning_failures_ = 0;
  last_attempted_curvature_ = 0.0;
  consecutive_curvature_increases_ = 0;
  consecutive_traffic_blocked_cycles_ = 0;
  return true;
}

void PullOverManagerNode::performStandDown(const std::string & reason)
{
  RCLCPP_INFO(get_logger(), "%s -- standing down.", reason.c_str());
  state_ = ManagerState::kIdle;
  active_goal_.reset();
  active_goal_shape_hint_.reset();
  deceleration_goal_.reset();
  deceleration_goal_shape_hint_.reset();
  last_trajectory_.reset();
  zero_speed_since_.reset();
  contingency_stop_goal_.reset();
  contingency_stop_goal_shape_hint_.reset();
  consecutive_traffic_blocked_cycles_ = 0;
  standdown_requested_since_.reset();
  restoreControllerStopDistOverride();
  restoreAebImuPathOverride();
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

void PullOverManagerNode::pushAebImuPathOverride()
{
  if (!aeb_override_enabled_ || !aeb_param_client_ || aeb_override_active_) {
    return;
  }
  aeb_override_active_ = true;
  aeb_param_client_->set_parameters(
    {rclcpp::Parameter("use_imu_path", false)},
    [this](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
      for (const auto & result : future.get()) {
        if (!result.successful) {
          RCLCPP_ERROR(
            get_logger(), "AEB rejected use_imu_path override: %s", result.reason.c_str());
        }
      }
    });
  RCLCPP_WARN(
    get_logger(),
    "Pull-over maneuver starting -- disabling '%s' use_imu_path for its duration (its "
    "plan-agnostic IMU-path collision fallback can false-positive on this intentional off-lane "
    "maneuver and trigger an MRM escalation to EMERGENCY_STOP; use_predicted_trajectory, which "
    "correctly reflects the actual planned curve, stays enabled throughout).",
    aeb_node_name_.c_str());
}

void PullOverManagerNode::restoreAebImuPathOverride()
{
  if (!aeb_override_active_) {
    return;
  }
  aeb_override_active_ = false;
  aeb_param_client_->set_parameters(
    {rclcpp::Parameter("use_imu_path", true)},
    [this](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
      for (const auto & result : future.get()) {
        if (!result.successful) {
          RCLCPP_ERROR(
            get_logger(), "AEB rejected use_imu_path restore: %s", result.reason.c_str());
        }
      }
    });
  RCLCPP_INFO(get_logger(), "Restored '%s' use_imu_path=true.", aeb_node_name_.c_str());
}

std::vector<float> PullOverManagerNode::usableHalfWidths() const
{
  if (
    !latest_centerline_ || !latest_shoulder_halfwidth_ ||
    latest_shoulder_halfwidth_->data.size() != latest_centerline_->poses.size()) {
    return {};
  }
  return latest_shoulder_halfwidth_->data;
}

bool PullOverManagerNode::shoulderContainmentAt(
  const geometry_msgs::msg::Pose & ego_pose, double * lateral_offset, double * half_width) const
{
  if (!latest_centerline_ || latest_centerline_->poses.empty()) {
    return false;
  }
  if (
    !latest_shoulder_halfwidth_ ||
    latest_shoulder_halfwidth_->data.size() != latest_centerline_->poses.size()) {
    return false;
  }
  std::size_t nearest_index = 0;
  double nearest_dist_sq = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < latest_centerline_->poses.size(); ++i) {
    const auto & p = latest_centerline_->poses[i].pose.position;
    const double dx = p.x - ego_pose.position.x;
    const double dy = p.y - ego_pose.position.y;
    const double dist_sq = dx * dx + dy * dy;
    if (dist_sq < nearest_dist_sq) {
      nearest_dist_sq = dist_sq;
      nearest_index = i;
    }
  }
  const auto & nearest_point = latest_centerline_->poses[nearest_index].pose.position;
  const double ego_yaw = yawFromQuaternionLocal(ego_pose.orientation);
  const double to_point_x = nearest_point.x - ego_pose.position.x;
  const double to_point_y = nearest_point.y - ego_pose.position.y;
  if (lateral_offset) {
    *lateral_offset =
      std::abs(-std::sin(ego_yaw) * to_point_x + std::cos(ego_yaw) * to_point_y);
  }
  if (half_width) {
    *half_width = scorer_params_.width_measurement_scale *
                  static_cast<double>(latest_shoulder_halfwidth_->data[nearest_index]);
  }
  return true;
}

bool PullOverManagerNode::isConfirmedSafeToStop(
  const geometry_msgs::msg::Pose & ego_pose, double ego_speed, double distance_to_goal,
  double heading_error) const
{
  (void)ego_speed;  // Not itself a gate here -- see docs: this answers "is this position/
                     // heading/width combination the genuine goal", speed is checked
                     // separately by the caller before declaring kSucceeded.
  if (distance_to_goal > arrival_distance_threshold_) {
    return false;
  }
  if (std::abs(heading_error) > arrival_heading_threshold_) {
    return false;
  }
  double lateral_offset = 0.0;
  double half_width = 0.0;
  if (!shoulderContainmentAt(ego_pose, &lateral_offset, &half_width)) {
    // No usable measured width for the centerline as it stands right now -- can't confirm
    // containment this cycle. Fails closed unless explicitly disabled.
    return !require_shoulder_containment_;
  }
  const double required_half_width = lateral_offset + vehicle_half_width_ + lateral_safety_margin_;
  return required_half_width <= half_width;
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

  // See standdown_requested_since_'s docs -- a pending cancel that has held for the full
  // grace period with no re-engage is treated as real, and this is where that actually
  // happens (onOperateMrm() only starts the timer; it never performs the reset itself while
  // a maneuver is active). Checked every planning cycle so it fires as close to the deadline
  // as this timer's own period allows, not just on the next OperateMrm call.
  //
  // Loss of autonomous mode is checked first and independently, and always wins immediately
  // regardless of how much of the grace period remains -- unlike the diagnostic-derived
  // availability flag this grace period exists to absorb noise from, actually leaving
  // AUTONOMOUS is a direct, unambiguous fact about vehicle control, not something that
  // legitimately needs debouncing.
  if (standdown_requested_since_.has_value() && require_autonomous_mode_ && !autonomous_mode_engaged_) {
    performStandDown("Autonomous mode lost while a stand-down was pending");
    return;
  }
  if (
    standdown_requested_since_.has_value() &&
    (now() - *standdown_requested_since_).seconds() >= standdown_grace_period_) {
    performStandDown("Pending stand-down grace period elapsed with no re-engage");
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
        goal_scorer_->selectBestGoal(*latest_centerline_, ego_pose, ego_speed, usableHalfWidths());
      // Live-verified bug, 2026-08-13: GoalScorer has no notion of
      // trajectory_params_.max_speed (a *hard, unconditional* ceiling
      // PullOverTrajectoryPlanner::plan() checks on every candidate's point[0] --
      // see satisfiesKinematicConstraints -- and point[0]'s speed always equals
      // ego's real current speed, not something plan() can choose). A goal that
      // passes GoalScorer's own (curvature/jerk) gates *can* still be found while
      // ego_speed is still above that ceiling (bag-confirmed: goal accepted at
      // 2.51 m/s against a 2.0 m/s trajectory_params_.max_speed), guaranteeing
      // every plan() attempt fails from its very first cycle with "speed exceeds
      // max" -- not a real infeasibility, just switching to kOperating one cycle
      // too early. kDecelerating's own crawl is already actively bringing ego_speed
      // down every cycle, so simply waiting the one or two extra 100ms cycles for
      // it to clear this ceiling too (rather than committing the instant GoalScorer
      // alone is satisfied) avoids ever entering kOperating in a state plan() is
      // guaranteed to reject.
      if (goal.has_value() && ego_speed <= trajectory_params_.max_speed) {
        RCLCPP_INFO(
          get_logger(),
          "Feasible shoulder goal now reachable at %.2f m/s (score=%.3f distance=%.1fm "
          "maneuver_curvature=%.4f maneuver_initial_jerk=%.3f measured_half_width=%.2fm) -- "
          "switching from in-lane braking to the pull-over maneuver.",
          ego_speed, goal->score, goal->distance_from_ego, goal->maneuver_curvature,
          goal->maneuver_initial_jerk, goal->measured_half_width);
        active_goal_ = goal->pose;
        deceleration_goal_.reset();
        deceleration_goal_shape_hint_.reset();
        decelerating_since_.reset();
        consecutive_planning_failures_ = 0;
        last_attempted_curvature_ = 0.0;
        consecutive_curvature_increases_ = 0;
        state_ = ManagerState::kOperating;
        return;  // Let the next cycle (100ms away) build the real pull-over trajectory.
      }
    }

    if (decelerating_since_.has_value() &&
        (now() - *decelerating_since_).seconds() >= kdecelerating_giveup_duration_) {
      RCLCPP_ERROR(
        get_logger(),
        "Searched for %.1fs with still no feasible shoulder goal within range -- giving up "
        "(nothing reachable here, not just \"too fast right now\").",
        kdecelerating_giveup_duration_);
      state_ = ManagerState::kFailed;
      deceleration_goal_.reset();
      deceleration_goal_shape_hint_.reset();
      decelerating_since_.reset();
      restoreControllerStopDistOverride();
      restoreAebImuPathOverride();
      return;
    }

    // Not reachable yet (or centerline data momentarily unavailable) -- keep
    // crawling straight ahead, in-lane, decelerating toward (and then
    // holding at) min_crawl_speed_, recomputed fresh every cycle from
    // ego's current state. Hand-built (buildCrawlTrajectory()), NOT routed
    // through PullOverTrajectoryPlanner::plan() -- see that function's docs
    // for why: plan()'s quintic always force-zeros its terminal sample,
    // which (even with a receding goal) kept converging real vehicle speed
    // to true v=0 and could latch the stock longitudinal controller's
    // STOPPED-state departure logic indefinitely (live-verified 2026-08-13).
    const auto plan_stamp = now();
    publishTrajectory(buildCrawlTrajectory(ego_pose, ego_speed, plan_stamp));
    return;
  }

  // state_ == kOperating from here on -- unchanged from before kDecelerating existed.
  if (!active_goal_.has_value()) {
    return;
  }

  const double distance_to_goal = std::hypot(
    active_goal_->position.x - ego_pose.position.x, active_goal_->position.y - ego_pose.position.y);
  double heading_error = yawFromQuaternionLocal(active_goal_->orientation) -
                          yawFromQuaternionLocal(ego_pose.orientation);
  while (heading_error > M_PI) heading_error -= 2.0 * M_PI;
  while (heading_error < -M_PI) heading_error += 2.0 * M_PI;

  // Strict, ego-pose-based confirmation that the vehicle is genuinely parked correctly --
  // the gate for declaring success. See isConfirmedSafeToStop()'s docs.
  const bool confirmed_safe_to_stop =
    isConfirmedSafeToStop(ego_pose, ego_speed, distance_to_goal, heading_error);

  // Clearance to *decelerate toward* the goal -- a strictly weaker question than "are we
  // parked correctly", and it must not be conflated with it.
  //
  // Live-verified deadlock this breaks, 2026-08-16: allow_full_stop used to be
  // confirmed_safe_to_stop, whose containment term is evaluated at ego's *current* pose. On
  // final approach ego is still ~1m short and ~0.4m off the centerline, so containment fails,
  // so no cycle is allowed to command a stop, so every published sample stays floored at
  // min_departure_reference_speed (2.5 m/s) -- at which speed the vehicle cannot settle onto
  // the goal, so it never centers, so containment never becomes true. Bag
  // pullover_final2_1786911450: the vehicle reached the shoulder with a good heading (3.69
  // deg, passing) and simply could not finish, leaving MRM stuck in MRM_OPERATING (its
  // MRM_SUCCEEDED needs operation mode STOP, which this node only requests on kSucceeded)
  // and the ADAPI motion state stuck at MOVING for the same reason.
  //
  // Deciding this from distance-to-goal is sound rather than a loosening: the goal is not an
  // arbitrary point. GoalScorer only emits goals sitting on the measured centerline that
  // already passed the footprint-width gate, oriented along the shoulder's own tangent, and
  // every PathShape terminates at exactly that pose -- so committing to decelerate onto a
  // validated goal IS the maneuver. The SAE J3016 concern behind the floor (do not linger
  // stopped while still exposed in a travel lane) is satisfied by the *goal* being off-lane,
  // not by ego already being centered on it. Declaring success stays strictly gated by
  // confirmed_safe_to_stop.
  // Committing to a stop also requires the vehicle to have substantially finished turning.
  // Stopping mid-turn is what makes the parked pose look skewed and unnatural, and it is how
  // the vehicle previously came to rest 0.13 rad off the tangent. The tolerance is looser
  // than arrival_heading_threshold_ so the final decel itself still has room to finish the
  // last few degrees -- this gate only rejects a *visibly* crooked stop.
  const bool cleared_to_decelerate_to_stop =
    distance_to_goal <= final_approach_engage_distance_ &&
    std::abs(heading_error) <= final_approach_heading_tolerance_;

  // Precision beats exposure once we are this close and off the travel lane -- see plan()'s
  // departure_speed_floor docs. Applied over the whole final-approach window (not only once
  // committed) so speed is already low while the last of the heading converges.
  const double departure_speed_floor = (distance_to_goal <= final_approach_engage_distance_)
                                          ? final_approach_reference_speed_
                                          : -1.0;

  // Endgame visibility: without this the only observable is "no arrival", with each of the
  // four criteria individually invisible -- which cost several full test cycles of guessing.
  if (distance_to_goal <= final_approach_engage_distance_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Final approach: distance=%.2fm (<=%.2f? %s) |heading_error|=%.3frad (<=%.3f? %s) "
      "speed=%.2fm/s (<=%.2f? %s) containment=%s",
      distance_to_goal, arrival_distance_threshold_,
      distance_to_goal <= arrival_distance_threshold_ ? "yes" : "NO", std::abs(heading_error),
      arrival_heading_threshold_,
      std::abs(heading_error) <= arrival_heading_threshold_ ? "yes" : "NO", ego_speed,
      arrival_speed_threshold_, ego_speed <= arrival_speed_threshold_ ? "yes" : "NO",
      confirmed_safe_to_stop ? "yes" : "NO");
  }

  if (confirmed_safe_to_stop && ego_speed <= arrival_speed_threshold_) {
    RCLCPP_INFO(
      get_logger(),
      "Arrived at pull-over goal (distance=%.2fm, speed=%.2fm/s, heading_error=%.3frad, "
      "shoulder containment confirmed) -- maneuver succeeded.",
      distance_to_goal, ego_speed, heading_error);
    state_ = ManagerState::kSucceeded;
    active_goal_.reset();
    active_goal_shape_hint_.reset();
    restoreControllerStopDistOverride();
    restoreAebImuPathOverride();
    requestOperationModeStop();
    return;
  }

  // Live-verified root cause, 2026-08-13 (found from a recorded rosbag, not guessed): every
  // "no feasible trajectory, gives up after 30 cycles" failure this session traced back to the
  // *same* geometric collapse GoalScorer::min_maneuver_forward_progress already guards against at
  // *selection* time (see its docs in goal_scorer.hpp) -- a candidate with a large lateral offset
  // but vanishing forward room is a near-singularity where required curvature explodes, not a
  // solver bug (confirmed: CurvatureSpiralPath's Newton solve was landing on wildly different,
  // all-individually-correct-for-their-own-near-degenerate-input solutions cycle to cycle, exactly
  // the numerical signature of a near-singular residual, not a defect in the solver itself). Unlike
  // at selection time, nothing here re-checks this as ego keeps creeping *forward* every cycle
  // toward a *fixed* active_goal_ that can require mostly *lateral* movement -- so forward room to
  // it can and did collapse below the same threshold mid-maneuver, live-observed turning e.g. a
  // 0.19 1/m maneuver at selection into a 0.75 1/m impossibility just a few seconds later purely
  // from ego's own forward progress, not from the goal moving. Detected reactively here, the same
  // way blocked_by_traffic already handles "temporarily blocked, not fundamentally infeasible":
  // abandon this goal and drop back to kDecelerating's own search loop (which re-runs GoalScorer
  // every cycle against ego's *current* pose) rather than grinding through
  // max_consecutive_planning_failures_ cycles against a goal that has become geometrically
  // impossible and giving up on the whole maneuver outright.
  // Live-verified BUG, 2026-08-15: forward_progress (a projection) is always <=
  // distance_to_goal (Cauchy-Schwarz), so as ego genuinely converges toward arrival this check
  // unconditionally fires *before* distance_to_goal can ever reach arrival_distance_threshold_
  // -- the goal gets abandoned just short of every genuine arrival, not just near-singular
  // sidesteps. Gated on distance_to_goal also still being outside
  // min_maneuver_forward_progress: a real near-singular sidestep has large residual lateral
  // offset (small forward_progress, *large* distance_to_goal); normal final-approach
  // convergence has both shrink together. Only the former should abandon.
  {
    const double ego_yaw = yawFromQuaternionLocal(ego_pose.orientation);
    const double to_goal_x = active_goal_->position.x - ego_pose.position.x;
    const double to_goal_y = active_goal_->position.y - ego_pose.position.y;
    const double forward_progress =
      to_goal_x * std::cos(ego_yaw) + to_goal_y * std::sin(ego_yaw);
    // A goal that is now BEHIND ego is unreachable outright -- this planner only generates
    // forward paths -- so it must be abandoned regardless of how near it still is. Checked
    // before (and outside) the distance guard below, which exists to protect a genuine final
    // approach and would otherwise suppress this: live-verified 2026-08-16, ego overshot a
    // goal by 1.19m, and because that is inside min_maneuver_forward_progress the abandon
    // never fired, so the planner reported "no damped step improves the residual" for the
    // full 30-cycle budget and the whole maneuver failed instead of simply re-selecting a
    // goal further along the shoulder (which the same session showed works well).
    if (forward_progress < 0.0) {
      RCLCPP_WARN(
        get_logger(),
        "Ego has passed the active goal (now %.2fm behind) -- unreachable by a forward-only "
        "planner. Abandoning it rather than retrying a goal we can never reach.",
        -forward_progress);
      if (reselectGoalInPlace(ego_pose, ego_speed)) {
        return;
      }
      active_goal_.reset();
      active_goal_shape_hint_.reset();
      contingency_stop_goal_.reset();
      contingency_stop_goal_shape_hint_.reset();
      consecutive_planning_failures_ = 0;
      last_attempted_curvature_ = 0.0;
      consecutive_curvature_increases_ = 0;
      consecutive_traffic_blocked_cycles_ = 0;
      state_ = ManagerState::kDecelerating;
      decelerating_since_ = now();
      return;
    }
  }

  if (distance_to_goal > scorer_params_.min_maneuver_forward_progress) {
    const double ego_yaw = yawFromQuaternionLocal(ego_pose.orientation);
    const double to_goal_x = active_goal_->position.x - ego_pose.position.x;
    const double to_goal_y = active_goal_->position.y - ego_pose.position.y;
    const double forward_progress =
      to_goal_x * std::cos(ego_yaw) + to_goal_y * std::sin(ego_yaw);
    // Deliberately the FLAT floor here, not GoalScorer::requiredForwardProgress() -- that
    // relation is only valid from a standing start. It comes from a quintic with zero
    // lateral slope at both ends (y'(0)=0), which describes ego at *selection* time,
    // tracking straight down its lane with no heading yet committed toward the shoulder.
    // Mid-maneuver ego is already turning, so a large part of the lateral shift is being
    // delivered by heading it already has and the from-scratch requirement badly overstates
    // the room still needed. Tried live 2026-08-16 and reverted the same session: applying
    // it here abandoned healthy goals on margins of centimetres ("forward room has collapsed
    // to 11.90m (below the 11.93m ... requires)") on the very cycle after selection, three
    // times in a row, until the search timed out. This check exists only to catch a goal
    // degenerating into a pure sidestep as ego overshoots past it, which the flat floor
    // detects correctly.
    if (forward_progress < scorer_params_.min_maneuver_forward_progress) {
      RCLCPP_WARN(
        get_logger(),
        "Active goal's forward room has collapsed to %.2fm (below the %.2fm minimum) as ego "
        "advanced -- this goal is now a near-singular sidestep, not a temporary planning "
        "failure. Abandoning it and returning to search.",
        forward_progress, scorer_params_.min_maneuver_forward_progress);
      // Prefer swapping in a fresh goal without leaving the maneuver -- see
      // reselectGoalInPlace()'s docs for why falling back to kDecelerating's straight-ahead
      // crawl costs the heading convergence already achieved.
      if (reselectGoalInPlace(ego_pose, ego_speed)) {
        return;
      }
      active_goal_.reset();
      active_goal_shape_hint_.reset();
      contingency_stop_goal_.reset();
      contingency_stop_goal_shape_hint_.reset();
      consecutive_planning_failures_ = 0;
      last_attempted_curvature_ = 0.0;
      consecutive_curvature_increases_ = 0;
      consecutive_traffic_blocked_cycles_ = 0;
      state_ = ManagerState::kDecelerating;
      decelerating_since_ = now();
      return;
    }
  }

  const auto plan_stamp = now();
  KinematicState start;
  start.pose = ego_pose;
  start.speed = ego_speed;
  start.accel = seedAccelerationFromLastTrajectory(plan_stamp);

  std::string failure_reason;
  bool blocked_by_traffic = false;
  PullOverTrajectoryPlanner::ShapeHint used_hint;
  double attempted_curvature = 0.0;
  // allow_full_stop=cleared_to_decelerate_to_stop -- see that flag's docs: once ego is within
  // final_approach_engage_distance_ of an already-validated goal, this cycle's candidate may
  // decelerate onto it for real; farther out every sample, including the terminal one, stays
  // floored at min_departure_reference_speed. Success itself remains gated on the strict,
  // ego-pose-based confirmed_safe_to_stop.
  const auto trajectory = trajectory_planner_->plan(
    start, *active_goal_, latest_objects_, plan_stamp, &failure_reason, &blocked_by_traffic,
    active_goal_shape_hint_ ? &*active_goal_shape_hint_ : nullptr, &used_hint,
    &attempted_curvature, cleared_to_decelerate_to_stop, departure_speed_floor);

  if (trajectory.has_value()) {
    active_goal_shape_hint_ = used_hint;
    consecutive_planning_failures_ = 0;
    last_attempted_curvature_ = 0.0;
    consecutive_curvature_increases_ = 0;
    consecutive_traffic_blocked_cycles_ = 0;
    contingency_stop_goal_.reset();
    contingency_stop_goal_shape_hint_.reset();
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
    PullOverTrajectoryPlanner::ShapeHint contingency_used_hint;
    // allow_full_stop=true here (unlike the main goal-directed plan() call above): a real
    // object is detected in the way, so this genuinely must be able to decelerate all the way
    // to 0 -- not the "still converging, don't stop yet" case allow_full_stop was added for.
    const auto contingency_trajectory = trajectory_planner_->plan(
      start, *contingency_stop_goal_, latest_objects_, plan_stamp, &contingency_reason, nullptr,
      contingency_stop_goal_shape_hint_ ? &*contingency_stop_goal_shape_hint_ : nullptr,
      &contingency_used_hint, nullptr, true);
    if (contingency_trajectory.has_value()) {
      contingency_stop_goal_shape_hint_ = contingency_used_hint;
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
      active_goal_shape_hint_.reset();
      contingency_stop_goal_.reset();
      contingency_stop_goal_shape_hint_.reset();
      restoreControllerStopDistOverride();
      restoreAebImuPathOverride();
    }
    return;
  }

  // Not traffic -- a genuine kinematic/shape infeasibility for this goal.
  //
  // Live-verified 2026-08-13: this branch used to publish nothing on a
  // failed cycle, leaving the gate holding whatever trajectory was last
  // published -- typically still the pre-maneuver in-lane one, since a
  // goal can fail on its very first kOperating cycle, before this planner
  // has ever published anything for it. Ego then just kept coasting on
  // that stale plan for the whole max_consecutive_planning_failures_
  // window instead of holding or backing off, which shrinks
  // distance-to-goal while heading error stays roughly fixed -- exactly
  // the near-singularity distance-vs-curvature relationship
  // GoalScorer::selectBestGoal already guards against at *selection*
  // time (see min_maneuver_forward_progress's docs), just re-encountered
  // here during *execution*. A goal that was feasible (if narrowly) when
  // selected can and did become infeasible a couple seconds later purely
  // from this drift. Mirrors the blocked_by_traffic branch above, which
  // already contingency-brakes in place instead of coasting blind -- but
  // uses buildCrawlTrajectory() (hand-built, recomputed fresh every cycle,
  // never targets v=0), not buildDecelerationGoal()+plan() (fixed once,
  // genuinely decelerates to a stop): live-verified 2026-08-13, routing
  // through plan() at all here (even with a receding goal) kept converging
  // real vehicle speed to true v=0 -- plan()'s quintic always force-zeros
  // its terminal sample, so every fresh replan still computed real
  // deceleration at the current instant regardless of goal distance -- and
  // reaching that v=0 could latch the stock longitudinal controller's
  // STOPPED-state departure logic indefinitely. Crawling at
  // min_crawl_speed_ via a trajectory that structurally never reaches zero
  // sidesteps that whole failure class by never entering it.
  // blocked_by_traffic above is a genuine must-actually-stop case
  // (something is physically in the way) so it deliberately keeps the real
  // braking goal through plan().
  ++consecutive_planning_failures_;

  // Track whether the attempted curvature is trending worse cycle to cycle -- see
  // consecutive_curvature_increases_'s docs for the full mechanism this catches (crawling
  // straight while retrying a fixed goal can only shrink distance without correcting heading,
  // so a marginal goal's required curvature can only climb). Only start counting once already
  // past the early-warning fraction of the cap, so ordinary noise near a comfortably-feasible
  // goal doesn't trip this.
  const double curvature_early_warning_threshold =
    curvature_trend_early_warning_fraction_ * trajectory_params_.max_curvature;
  if (
    attempted_curvature > last_attempted_curvature_ &&
    attempted_curvature > curvature_early_warning_threshold) {
    ++consecutive_curvature_increases_;
  } else {
    consecutive_curvature_increases_ = 0;
  }
  last_attempted_curvature_ = attempted_curvature;

  RCLCPP_WARN(
    get_logger(),
    "No feasible trajectory found this cycle (%d/%d consecutive). Last reason: %s",
    consecutive_planning_failures_, max_consecutive_planning_failures_, failure_reason.c_str());
  publishTrajectory(buildCrawlTrajectory(ego_pose, ego_speed, plan_stamp));

  if (consecutive_curvature_increases_ >= max_consecutive_curvature_increases_) {
    RCLCPP_WARN(
      get_logger(),
      "Abandoning active goal: attempted curvature has worsened for %d consecutive cycles (now "
      "%.4f, past the %.4f early-warning threshold under the %.2f cap) -- crawling straight "
      "while retrying this goal can only make its geometry worse, not better. Returning to "
      "search instead of grinding through the full %d-cycle give-up budget.",
      consecutive_curvature_increases_, attempted_curvature, curvature_early_warning_threshold,
      trajectory_params_.max_curvature, max_consecutive_planning_failures_);
    // Same preference as the forward-room branch above: keep the maneuver alive with a
    // freshly-scored goal rather than restarting from a straight-ahead crawl.
    if (reselectGoalInPlace(ego_pose, ego_speed)) {
      return;
    }
    active_goal_.reset();
    active_goal_shape_hint_.reset();
    contingency_stop_goal_.reset();
    contingency_stop_goal_shape_hint_.reset();
    consecutive_planning_failures_ = 0;
    last_attempted_curvature_ = 0.0;
    consecutive_curvature_increases_ = 0;
    consecutive_traffic_blocked_cycles_ = 0;
    state_ = ManagerState::kDecelerating;
    decelerating_since_ = now();
    return;
  }

  if (consecutive_planning_failures_ >= max_consecutive_planning_failures_) {
    RCLCPP_ERROR(
      get_logger(),
      "Giving up: no feasible trajectory for %d consecutive cycles -- infeasible goal.",
      consecutive_planning_failures_);
    state_ = ManagerState::kFailed;
    active_goal_.reset();
    active_goal_shape_hint_.reset();
    contingency_stop_goal_.reset();
    contingency_stop_goal_shape_hint_.reset();
    restoreControllerStopDistOverride();
    restoreAebImuPathOverride();
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

Trajectory PullOverManagerNode::buildCrawlTrajectory(
  const geometry_msgs::msg::Pose & ego_pose, double ego_speed, const rclcpp::Time & stamp) const
{
  const double yaw = yawFromQuaternionLocal(ego_pose.orientation);
  const double cos_yaw = std::cos(yaw);
  const double sin_yaw = std::sin(yaw);

  // Ramp segment (if above the floor): v(t) = ego_speed - crawl_deceleration_ * t,
  // x(t) = ego_speed * t - 0.5 * crawl_deceleration_ * t^2. Hold segment: constant at
  // min_crawl_speed_. Never reaches v=0, unlike anything routed through plan() -- see
  // this function's docs.
  const double ramp_duration =
    ego_speed > min_crawl_speed_ ? (ego_speed - min_crawl_speed_) / crawl_deceleration_ : 0.0;
  const double ramp_distance =
    ego_speed * ramp_duration - 0.5 * crawl_deceleration_ * ramp_duration * ramp_duration;

  Trajectory trajectory;
  trajectory.header.stamp = stamp;
  trajectory.header.frame_id = "map";

  const double dt = trajectory_params_.dt;
  const int num_samples =
    std::max(2, static_cast<int>(std::round(crawl_trajectory_duration_ / dt)) + 1);
  trajectory.points.reserve(static_cast<std::size_t>(num_samples));

  for (int i = 0; i < num_samples; ++i) {
    const double t = std::min(crawl_trajectory_duration_, i * dt);
    double v;
    double x;
    if (t < ramp_duration) {
      v = ego_speed - crawl_deceleration_ * t;
      x = ego_speed * t - 0.5 * crawl_deceleration_ * t * t;
    } else {
      v = min_crawl_speed_;
      x = ramp_distance + min_crawl_speed_ * (t - ramp_duration);
    }

    autoware_planning_msgs::msg::TrajectoryPoint point;
    point.pose = ego_pose;
    point.pose.position.x += x * cos_yaw;
    point.pose.position.y += x * sin_yaw;
    point.longitudinal_velocity_mps = static_cast<float>(v);
    point.lateral_velocity_mps = 0.0F;
    point.acceleration_mps2 = static_cast<float>(t < ramp_duration ? -crawl_deceleration_ : 0.0);
    point.heading_rate_rps = 0.0F;
    point.front_wheel_angle_rad = 0.0F;
    point.rear_wheel_angle_rad = 0.0F;
    const auto duration = rclcpp::Duration::from_seconds(t);
    point.time_from_start.sec = static_cast<int32_t>(duration.seconds());
    point.time_from_start.nanosec =
      static_cast<uint32_t>(duration.nanoseconds() - static_cast<int64_t>(duration.seconds()) * 1'000'000'000LL);
    trajectory.points.push_back(point);
  }

  return trajectory;
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
    // Live-verified root cause, 2026-08-13 (found from the autoware launch log, not guessed):
    // mrm_handler re-calls this service's operate=true repeatedly as part of its own normal
    // operating cadence while PULL_OVER is the active MRM behavior -- NOT only on a genuinely
    // fresh trigger. Returning success=false here (the old behavior) told mrm_handler every
    // single one of those redundant re-calls was a *failure*, and mrm_handler's own response to
    // "PULL_OVER failed to operate" is to immediately escalate to EMERGENCY_STOP -- bag/log-
    // confirmed live: this fired within ~1-2s of every single trigger this session, aborting an
    // otherwise-correctly-executing maneuver (real trajectory being planned and followed, per
    // RViz) via a completely different path than any of this node's own trajectory/speed logic.
    // From mrm_handler's perspective a maneuver already in progress is success, not failure --
    // pull_over IS actively doing its job -- so report it that way; only genuine refusals below
    // (not autonomous, no centerline, no odometry) should return false.
    const std::string message = "Already in progress -- no-op.";
    return {true, message};
  }

  // Fresh trigger -- don't warm-start the new maneuver's acceleration
  // boundary condition from a previous, unrelated maneuver's leftover
  // trajectory (see seedAccelerationFromLastTrajectory()).
  last_trajectory_.reset();
  zero_speed_since_.reset();
  active_goal_shape_hint_.reset();
  deceleration_goal_shape_hint_.reset();
  contingency_stop_goal_.reset();
  contingency_stop_goal_shape_hint_.reset();
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

  // Past this point, a maneuver (kDecelerating or kOperating) is definitely starting -- see
  // pushAebImuPathOverride()'s docs for why AEB's IMU-path fallback needs to be suppressed for
  // the whole span of an active pull-over maneuver, not just once curved-path execution begins.
  pushAebImuPathOverride();

  const double ego_speed = std::hypot(
    latest_odometry_->twist.twist.linear.x, latest_odometry_->twist.twist.linear.y);

  const auto goal = goal_scorer_->selectBestGoal(
    *latest_centerline_, latest_odometry_->pose.pose, ego_speed, usableHalfWidths());

  // See the matching check in onPlanningTimer()'s kDecelerating branch: a goal
  // GoalScorer accepts is not necessarily executable *yet* if ego_speed is still
  // above trajectory_params_.max_speed, the hard, unconditional ceiling
  // PullOverTrajectoryPlanner::plan() checks on every candidate's point[0] (which
  // always equals ego's real current speed) -- committing straight to kOperating
  // here would guarantee the very first plan() cycle fails. Route this the same
  // way as "no goal found yet": start kDecelerating's crawl-and-recheck loop,
  // which re-runs GoalScorer every cycle as ego_speed drops toward and past that
  // ceiling anyway.
  if (!goal.has_value() || ego_speed > trajectory_params_.max_speed) {
    // Not a refusal -- see kDecelerating's docs for why "no feasible goal at
    // the current speed" is treated as "not yet", not "never", given
    // feasibility is speed-dependent (jerk scales with speed cubed) rather
    // than being about this goal being permanently unreachable.
    const std::string message =
      "No feasible shoulder goal at current speed (" + std::to_string(ego_speed) +
      " m/s) -- decelerating in-lane until one becomes reachable.";
    RCLCPP_WARN(get_logger(), "%s", message.c_str());
    state_ = ManagerState::kDecelerating;
    decelerating_since_ = now();
    return {true, message};
  }

  RCLCPP_INFO(
    get_logger(),
    "Selected shoulder goal at (%.2f, %.2f): score=%.3f distance=%.1fm curvature=%.4f "
    "continuity=%.1fm maneuver_curvature=%.4f maneuver_initial_jerk=%.3f "
    "measured_half_width=%.2fm. Starting standalone trajectory planner (replanning at %.1f Hz).",
    goal->pose.position.x, goal->pose.position.y, goal->score, goal->distance_from_ego,
    goal->curvature, goal->continuity_length, goal->maneuver_curvature,
    goal->maneuver_initial_jerk, goal->measured_half_width, planning_rate_hz_);

  active_goal_ = goal->pose;
  consecutive_planning_failures_ = 0;
  last_attempted_curvature_ = 0.0;
  consecutive_curvature_increases_ = 0;
  state_ = ManagerState::kOperating;
  return {true, "Pull-over trajectory planning started toward the selected shoulder goal."};
}

}  // namespace autoware::shoulder_pullover_manager
