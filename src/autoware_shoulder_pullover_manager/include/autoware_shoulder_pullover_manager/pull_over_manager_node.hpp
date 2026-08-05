#pragma once

#include "autoware_shoulder_pullover_manager/goal_scorer.hpp"
#include "autoware_shoulder_pullover_manager/keyboard_trigger.hpp"
#include "autoware_shoulder_pullover_manager/trajectory_planner.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_adapi_v1_msgs/msg/operation_mode_state.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tier4_system_msgs/msg/mrm_behavior_status.hpp>
#include <tier4_system_msgs/srv/operate_mrm.hpp>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace autoware::shoulder_pullover_manager
{

/// Internal maneuver state. Deliberately explicit (rather than a couple of
/// booleans) so re-entrancy, logging, and status-topic reporting all have a
/// single, unambiguous source of truth.
enum class ManagerState {
  kIdle,        ///< No maneuver requested; ready to accept a new trigger.
  kOperating,   ///< A goal has been selected; replanning and (optionally)
                ///< publishing a trajectory toward it every cycle.
  kSucceeded,   ///< Ego reached the goal, stopped. Transient, reverts to kIdle.
  kFailed,      ///< A precondition failed, or no feasible trajectory could be
                ///< found for too many consecutive cycles. Transient.
};

/// Fills Autoware's reserved-but-unimplemented "pull_over" MRM extension
/// point: serves `/system/mrm/pull_over_manager/operate`
/// (tier4_system_msgs/srv/OperateMrm) and publishes
/// `/system/mrm/pull_over_manager/status`
/// (tier4_system_msgs/msg/MrmBehaviorStatus), exactly the contract
/// `autoware_mrm_handler` already calls out to but which, as of this
/// project's architecture study, no package in the Autoware tree serves.
///
/// Unlike an on-map maneuver (which could ask Autoware's own
/// mission_planner/behavior_path_planner to route there), the target
/// shoulder area has no representation in this project's Lanelet2 map at
/// all -- confirmed by inspecting the map file directly, zero
/// `road_shoulder` lanelets exist. Asking mission_planner to route to an
/// off-map coordinate reliably fails ("The planned route is empty"), so
/// this node does not depend on Autoware's routing/behavior-planning stack
/// for the maneuver itself. Instead, it runs its own standalone,
/// receding-horizon trajectory planner (PullOverTrajectoryPlanner) directly
/// against live perception data (the shoulder centerline and tracked
/// objects), independent of the HD map's routable graph.
///
/// On trigger (either the real OperateMrm service, or -- for now, while no
/// real MRM/diagnostics condition is wired up -- a raw 'P' keypress used
/// purely to simulate one), this node:
///   1. Scores candidate goals along the live, map-frame shoulder
///      centerline (GoalScorer) and fixes the best one as the maneuver's
///      target.
///   2. Every planning cycle thereafter, replans a fresh, collision-checked
///      trajectory from the *current* ego state to that fixed goal
///      (PullOverTrajectoryPlanner), publishing it for visualization and,
///      optionally, direct consumption (see `direct_trajectory_output_topic`).
///   3. Declares success once ego is stopped at the goal.
///
/// All ROS calls (service requests, publishing) happen exclusively on the
/// single-threaded executor callback thread; the keyboard-trigger's
/// background thread only ever touches an `std::atomic<bool>`.
class PullOverManagerNode : public rclcpp::Node
{
public:
  explicit PullOverManagerNode(const rclcpp::NodeOptions & options);
  ~PullOverManagerNode() override;

private:
  // --- Subscription callbacks -------------------------------------------
  void onCenterline(const nav_msgs::msg::Path::ConstSharedPtr msg);
  void onOdometry(const nav_msgs::msg::Odometry::ConstSharedPtr msg);
  void onObjects(const autoware_perception_msgs::msg::PredictedObjects::ConstSharedPtr msg);
  void onOperationModeState(
    const autoware_adapi_v1_msgs::msg::OperationModeState::ConstSharedPtr msg);

  // --- Service callback ---------------------------------------------------
  void onOperateMrm(
    const std::shared_ptr<tier4_system_msgs::srv::OperateMrm::Request> request,
    std::shared_ptr<tier4_system_msgs::srv::OperateMrm::Response> response);

  // --- Timers ---------------------------------------------------------
  /// Drains the keyboard-trigger flag on the executor thread; this is the
  /// only place the simulated keyboard trigger is allowed to actually act.
  void onPollTimer();
  /// Periodically publishes MrmBehaviorStatus so mrm_handler (or any future
  /// caller) can see this behavior is alive and whether it is currently
  /// operating.
  void onStatusTimer();
  /// While kOperating, replans and republishes the trajectory toward the
  /// fixed goal every cycle, and checks for arrival/failure.
  void onPlanningTimer();

  // --- Core logic, shared by both the OperateMrm service and the keyboard
  // trigger -------------------------------------------------------------
  /// Attempts to start a pull-over maneuver. Returns {success, message}.
  /// `trigger_source` is purely for logging (e.g. "OperateMrm service" vs.
  /// "simulated keyboard trigger").
  std::pair<bool, std::string> triggerPullOver(const std::string & trigger_source);

  void publishStatus();

  // --- Parameters ---------------------------------------------------------
  GoalScorerParams scorer_params_;
  TrajectoryPlannerParams trajectory_params_;
  bool require_autonomous_mode_{true};
  double status_publish_rate_hz_{5.0};
  double poll_rate_hz_{20.0};
  double planning_rate_hz_{10.0};
  double arrival_distance_threshold_{1.0};
  double arrival_speed_threshold_{0.2};
  int max_consecutive_planning_failures_{30};
  std::string direct_trajectory_output_topic_;  ///< Empty (default) = debug-only, see docs above.
  char keyboard_trigger_key_{'p'};
  bool keyboard_trigger_enabled_{true};

  // --- Collaborators ---------------------------------------------------
  std::unique_ptr<GoalScorer> goal_scorer_;
  std::unique_ptr<PullOverTrajectoryPlanner> trajectory_planner_;
  std::unique_ptr<KeyboardTrigger> keyboard_trigger_;

  // --- Latest observed state (executor thread only, except the atomic) ---
  nav_msgs::msg::Path::ConstSharedPtr latest_centerline_;
  nav_msgs::msg::Odometry::ConstSharedPtr latest_odometry_;
  autoware_perception_msgs::msg::PredictedObjects latest_objects_;
  bool autonomous_mode_engaged_{false};
  std::atomic<bool> keyboard_trigger_pending_{false};

  ManagerState state_{ManagerState::kIdle};
  std::optional<geometry_msgs::msg::Pose> active_goal_;
  int consecutive_planning_failures_{0};

  // --- ROS interfaces ---------------------------------------------------
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr centerline_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<autoware_perception_msgs::msg::PredictedObjects>::SharedPtr objects_sub_;
  rclcpp::Subscription<autoware_adapi_v1_msgs::msg::OperationModeState>::SharedPtr
    operation_mode_sub_;
  rclcpp::Service<tier4_system_msgs::srv::OperateMrm>::SharedPtr operate_service_;
  rclcpp::Publisher<tier4_system_msgs::msg::MrmBehaviorStatus>::SharedPtr status_pub_;
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr trajectory_debug_pub_;
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr trajectory_direct_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_path_debug_pub_;
  rclcpp::TimerBase::SharedPtr poll_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr planning_timer_;
};

}  // namespace autoware::shoulder_pullover_manager
