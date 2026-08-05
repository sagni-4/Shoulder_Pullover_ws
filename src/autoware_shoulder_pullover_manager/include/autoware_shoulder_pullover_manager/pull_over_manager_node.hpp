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
#include <visualization_msgs/msg/marker.hpp>

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
  kIdle,          ///< No maneuver requested; ready to accept a new trigger.
  kDecelerating,  ///< Triggered, but no shoulder goal is feasible *yet* at
                  ///< ego's current speed (see GoalScorer's speed-dependent
                  ///< jerk gate) -- replanning a straight-ahead, in-lane
                  ///< braking trajectory every cycle while re-checking
                  ///< GoalScorer at the current (dropping) speed, until a
                  ///< real goal becomes reachable. Added 2026-08-05: without
                  ///< this, triggering above roughly 3 m/s reliably refused
                  ///< outright, since jerk feasibility scales with speed
                  ///< cubed -- see project memory.
  kOperating,     ///< A shoulder goal has been selected; replanning and
                  ///< (optionally) publishing the curved pull-over
                  ///< trajectory toward it every cycle.
  kSucceeded,     ///< Ego reached the goal, stopped. Transient, reverts to kIdle.
  kFailed,        ///< A precondition failed, ego decelerated to a near-stop
                  ///< with still no feasible shoulder goal in range, or no
                  ///< feasible trajectory could be found for too many
                  ///< consecutive cycles. Transient.
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
///      centerline (GoalScorer) at ego's *current* speed.
///   2. If none are feasible yet (see kDecelerating's docs -- live-tested
///      finding, 2026-08-05: at typical cruising speed, roughly 4+ m/s on
///      this map, nothing nearby passes GoalScorer's speed-dependent jerk
///      gate at all), enters kDecelerating: replans a straight-ahead,
///      in-lane braking trajectory every cycle (reusing
///      PullOverTrajectoryPlanner itself with a synthetic straight-line
///      goal, see buildDecelerationGoal), *while re-running GoalScorer
///      every cycle at the current, dropping speed* -- there is no single
///      correct fixed "slow enough now" threshold, since feasibility
///      depends on the specific candidate goal's own geometry, not just
///      speed, so this reactively waits for GoalScorer itself to succeed
///      rather than guessing a target. The moment it does, switches to
///      kOperating using that goal.
///   3. Once a goal is fixed (whether found immediately or only after
///      decelerating), every planning cycle thereafter replans a fresh,
///      collision-checked trajectory from the *current* ego state to that
///      fixed goal (PullOverTrajectoryPlanner), publishing it for
///      visualization and, optionally, direct consumption (see
///      `direct_trajectory_output_topic`).
///   4. Declares success once ego is stopped at the goal.
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

  /// Shared by both kDecelerating and kOperating -- publishes `trajectory`
  /// on every configured output (debug trajectory/path/marker, and
  /// direct_trajectory_output_topic_ if set).
  void publishTrajectory(const autoware_planning_msgs::msg::Trajectory & trajectory);

  /// A synthetic goal directly ahead of `ego_pose` along its *own current
  /// heading* (same orientation, position offset by
  /// deceleration_lookahead_distance_) -- deliberately not a real shoulder
  /// goal, just a target for PullOverTrajectoryPlanner to decelerate
  /// straight toward while staying in-lane. Reuses the exact same planner
  /// as the real maneuver rather than new bespoke braking logic: a
  /// zero-heading-change goal is the easiest possible case for it (the
  /// spiral shape converges in a single iteration with curvature exactly
  /// 0, verified offline), so this is a safe, minimal-risk reuse, not a
  /// new code path to independently trust.
  [[nodiscard]] geometry_msgs::msg::Pose buildDecelerationGoal(
    const geometry_msgs::msg::Pose & ego_pose) const;

  /// Builds a solid, translucent yellow ribbon (visualization_msgs::Marker,
  /// TRIANGLE_LIST) of `visualization_width_` running along `trajectory`,
  /// deliberately styled to match Autoware's own default RViz config, which
  /// shades its planned path green -- yellow visually marks this as a
  /// distinct, non-standard maneuver at a glance. Baked into the message
  /// itself (fixed color/width) rather than left to per-user RViz display
  /// configuration, so it looks correct for any viewer without manual setup.
  [[nodiscard]] visualization_msgs::msg::Marker toRibbonMarker(
    const autoware_planning_msgs::msg::Trajectory & trajectory) const;

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
  double deceleration_lookahead_distance_{40.0};  ///< m. How far ahead (along ego's current
                                                   ///< heading) the synthetic straight-line
                                                   ///< braking goal is placed each cycle.
  double deceleration_giveup_speed_threshold_{0.3};  ///< m/s. If ego decelerates to at/below this
                                                      ///< with still no feasible shoulder goal in
                                                      ///< range, that is a real "nothing reachable
                                                      ///< here" case (distinct from "too fast
                                                      ///< right now") -- give up rather than
                                                      ///< brake to a dead stop and sit there
                                                      ///< indefinitely re-checking every cycle.
  std::string direct_trajectory_output_topic_;  ///< Empty (default) = debug-only, see docs above.
  char keyboard_trigger_key_{'p'};
  bool keyboard_trigger_enabled_{true};
  double visualization_width_{3.0};   ///< Ribbon width (m) of the RViz shading, roughly lane-width
                                       ///< by default to visually match Autoware's own planned-path
                                       ///< shading (see the framework's screenshot reference).
  double visualization_alpha_{0.6};

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
  /// Fixed once, the instant kDecelerating begins -- **not** resynthesized
  /// from ego's current pose every cycle. Live-tested finding (2026-08-05):
  /// a receding-horizon quintic that resets v0 to ego's *current* actual
  /// speed every 100ms always reports a near-term target speed equal to
  /// whatever the vehicle is already doing (since v0 is defined to match
  /// it), so the controller never actually sees an authoritative "go
  /// slower now" -- the real slow-down was always deferred to a later
  /// part of a trajectory that got discarded and replaced before the
  /// vehicle ever reached it (observed: 115 real seconds to go from
  /// 4.36 m/s to 2.59 m/s, an average of ~0.015 m/s^2 -- imperceptible).
  /// Fixing the goal once, the same way active_goal_ already is for the
  /// real maneuver, means the remaining distance genuinely shrinks each
  /// cycle as ego approaches it, so urgency (and therefore commanded
  /// deceleration) actually increases over time instead of resetting.
  std::optional<geometry_msgs::msg::Pose> deceleration_goal_;
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
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr planned_path_marker_pub_;
  rclcpp::TimerBase::SharedPtr poll_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr planning_timer_;
};

}  // namespace autoware::shoulder_pullover_manager
