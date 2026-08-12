#pragma once

#include "autoware_shoulder_pullover_manager/goal_scorer.hpp"
#include "autoware_shoulder_pullover_manager/keyboard_trigger.hpp"
#include "autoware_shoulder_pullover_manager/trajectory_planner.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>

#include <autoware_adapi_v1_msgs/msg/operation_mode_state.hpp>
#include <autoware_adapi_v1_msgs/srv/change_operation_mode.hpp>
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

  /// Pushes a relaxed `drive_state_stop_dist` to the stock
  /// autoware_pid_longitudinal_controller. Root cause (found live,
  /// 2026-08-10): that controller's STOPPED->DRIVE departure condition is
  /// `stop_dist > drive_state_stop_dist`, where `stop_dist` is the signed
  /// arc-length from ego to the trajectory's own nearest zero-velocity
  /// point. Because this project's planner is receding-horizon, point[0]'s
  /// velocity always equals ego's *actual* current speed -- so the instant
  /// ego is genuinely near-stopped mid maneuver, that nearest zero-velocity
  /// point is trivially point[0] itself and stop_dist collapses to ~0,
  /// permanently failing the default `drive_state_stop_dist=0.5` departure
  /// check no matter how aggressively the trajectory ramps up a few samples
  /// later (this is what pinned the vehicle at 0.0 m/s for 480+ seconds in
  /// live testing).
  ///
  /// CRITICAL, live-verified 2026-08-12 -- only call this reactively, once
  /// checkControllerStuckAndOverride() has confirmed ego has been genuinely
  /// stuck at near-zero speed for controller_override_stuck_duration_,
  /// never proactively at maneuver start: `drive_state_stop_dist` also
  /// gates a *second*, unrelated transition, `departure_condition_from_stopping
  /// = stop_dist > drive_state_stop_dist + drive_state_offset_stop_dist`
  /// (STOPPING->DRIVE, normally requiring stop_dist > ~1.5, essentially
  /// never true during an ordinary controlled braking-to-a-stop). Pushing
  /// this override for the *entire* kDecelerating/kOperating span (the
  /// first live attempt) collapses that threshold to ~-1.0 too, which is
  /// almost always true -- so every ordinary entry into STOPPING while
  /// braking toward any stop (not just a stuck one) immediately bounced
  /// back to DRIVE, and that transition's own code
  /// (`m_prev_raw_ctrl_cmd.acc = std::max(0.0, ...)`, PID reset) floors the
  /// previous command non-negative on every bounce. Live-observed
  /// consequence: the vehicle accelerated to ~38.6 m/s instead of
  /// decelerating from a 4.2 m/s cruise. `drive_state_offset_stop_dist` is
  /// declared at controller construction only, NOT covered by its
  /// paramCallback (verified by grep), so it cannot be live-compensated to
  /// cancel this out -- the only safe fix is to never be active while the
  /// controller is in STOPPING, which this reactive, stuck-triggered-only
  /// design guarantees (by the time controller_override_stuck_duration_ of
  /// near-zero speed has elapsed, the controller is long past any STOPPING
  /// transient and genuinely latched in STOPPED, a state whose own
  /// STOPPED->DRIVE code does not floor acceleration the way STOPPING's
  /// does).
  void pushControllerStopDistOverride();

  /// Restores the controller's `drive_state_stop_dist` to the value observed
  /// at startup. Called both reactively (the instant ego speed climbs back
  /// above controller_override_stuck_speed_threshold_, confirming departure)
  /// and whenever a maneuver concludes for any reason (success, failure, or
  /// explicit stand-down), so normal stock hysteresis applies again as soon
  /// as this node is no longer actively relying on the override -- see
  /// pushControllerStopDistOverride()'s docs for why leaving it active any
  /// longer than necessary is unsafe, not just unnecessary.
  void restoreControllerStopDistOverride();

  /// Called once per planning cycle (both kDecelerating and kOperating)
  /// with ego's current speed. Tracks how long ego has been continuously
  /// below controller_override_stuck_speed_threshold_; once that exceeds
  /// controller_override_stuck_duration_ (deliberately well above the
  /// controller's own stopped_state_entry_duration_time and any ordinary
  /// STOPPING-phase transient), pushes the override. The instant speed
  /// rises back above the threshold, restores it immediately. See
  /// pushControllerStopDistOverride()'s docs for why this must stay purely
  /// reactive rather than proactive.
  void checkControllerStuckAndOverride(double ego_speed, const rclcpp::Time & now);

  /// Called once, the instant a maneuver reaches kSucceeded. mrm_handler's
  /// own MRM_OPERATING->MRM_SUCCEEDED transition for the PULL_OVER behavior
  /// (see autoware_mrm_handler's updateMrmState()) requires
  /// isArrivedAtGoal(), which checks `/api/operation_mode/state == STOP` --
  /// not this node's own arrival check, and not anything this node used to
  /// touch. Since our maneuver isn't set via the normal routing/route-
  /// completion path, nothing ever transitioned operation mode to STOP on
  /// its own, so mrm_handler's MRM state stayed latched at OPERATING forever
  /// even once the vehicle had genuinely stopped at the goal -- live-observed
  /// 2026-08-12, RViz's MRM State/Behavior panel showed "Operating" /
  /// "Pull Over" indefinitely. Requesting STOP here closes that loop.
  void requestOperationModeStop();

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

  /// Warm-start value for KinematicState::accel: interpolates
  /// last_trajectory_'s own `acceleration_mps2` at however much real time
  /// has elapsed since it was generated (last_trajectory_'s header.stamp).
  /// Returns 0.0 if there is no prior trajectory, the elapsed time is
  /// negative (clock anomaly), or elapsed time has run past that
  /// trajectory's own horizon entirely (stale -- a missed cycle or a fresh
  /// maneuver just starting, not something to extrapolate past). This is
  /// the fix for the 2026-08-10 live-verified bug where re-imposing a0=0 on
  /// every single 100ms replan meant the vehicle could never accumulate real
  /// acceleration -- see trajectory_planner.cpp's buildCandidate() docs for
  /// the full mechanism.
  [[nodiscard]] double seedAccelerationFromLastTrajectory(const rclcpp::Time & now) const;

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
  int max_consecutive_traffic_blocked_cycles_{300};  ///< See consecutive_traffic_blocked_cycles_.
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
  bool controller_stop_dist_override_enabled_{true};
  std::string controller_node_name_{"/control/trajectory_follower/controller_node_exe"};
  double controller_override_drive_state_stop_dist_{-2.0};  ///< See pushControllerStopDistOverride().
  double controller_override_stuck_speed_threshold_{0.05};  ///< m/s. See checkControllerStuckAndOverride().
  double controller_override_stuck_duration_{2.0};  ///< s. See checkControllerStuckAndOverride().
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
  /// Fixed the instant kOperating's plan() first reports blocked_by_traffic
  /// (see that flag's docs in trajectory_planner.hpp), the same way
  /// deceleration_goal_ is fixed for kDecelerating and for the identical
  /// reason: resynthesizing a "brake from here" goal fresh from ego's
  /// current pose every cycle would mean v0 always equals ego's *current*
  /// speed, so the commanded near-term speed never visibly drops -- see
  /// deceleration_goal_'s own docs for the live-verified failure mode this
  /// caused before that fix. Cleared the instant the real active_goal_ plan
  /// succeeds again (traffic cleared) or the maneuver ends.
  std::optional<geometry_msgs::msg::Pose> contingency_stop_goal_;
  int consecutive_planning_failures_{0};
  /// Cycles the *real* pull-over maneuver has spent blocked by traffic
  /// (contingency-braking instead of progressing toward active_goal_) --
  /// separate from consecutive_planning_failures_ so genuinely waiting for
  /// another vehicle/pedestrian to clear the path (which can reasonably
  /// take much longer than a few seconds) doesn't trigger the same tight
  /// give-up threshold meant for "no valid maneuver shape exists at all."
  /// Reset to 0 the instant active_goal_'s own plan succeeds again.
  int consecutive_traffic_blocked_cycles_{0};
  /// The most recently *published* trajectory (kDecelerating or kOperating,
  /// whichever was active), kept only to seed the next planning cycle's
  /// start-acceleration boundary condition -- see
  /// seedAccelerationFromLastTrajectory(). Reset (along with active_goal_
  /// and deceleration_goal_) whenever a fresh maneuver begins, so a new
  /// trigger never warm-starts from an unrelated prior maneuver's leftover
  /// acceleration state.
  std::optional<autoware_planning_msgs::msg::Trajectory> last_trajectory_;

  /// True once original_drive_state_stop_dist_ has been successfully read
  /// from the live controller at startup -- pushControllerStopDistOverride()
  /// refuses to act (and logs a warning) until this is true, since without a
  /// known-good original value there would be nothing safe to restore to.
  bool have_original_drive_state_stop_dist_{false};
  double original_drive_state_stop_dist_{0.5};
  /// True for exactly the span between a successful
  /// pushControllerStopDistOverride() and its matching
  /// restoreControllerStopDistOverride() -- guards against a redundant
  /// restore call (e.g. an explicit stand-down arriving after a maneuver has
  /// already concluded and already restored) issuing a pointless extra RPC.
  bool controller_override_active_{false};
  /// Set the first cycle ego speed drops below
  /// controller_override_stuck_speed_threshold_, cleared the instant it
  /// rises back above that threshold or a maneuver ends -- see
  /// checkControllerStuckAndOverride().
  std::optional<rclcpp::Time> zero_speed_since_;
  rclcpp::AsyncParametersClient::SharedPtr controller_param_client_;
  rclcpp::Client<autoware_adapi_v1_msgs::srv::ChangeOperationMode>::SharedPtr
    change_to_stop_client_;

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
