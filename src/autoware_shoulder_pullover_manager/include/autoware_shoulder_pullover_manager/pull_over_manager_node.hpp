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
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tier4_system_msgs/msg/mrm_behavior_status.hpp>
#include <tier4_system_msgs/srv/operate_mrm.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

  /// Live-verified root cause, 2026-08-13 (bag-confirmed: /control/command/control_cmd stayed
  /// calm -- mild, smoothly-declining values -- at the *exact* instants ego's real velocity
  /// (ground truth) suddenly, discontinuously crashed to 0 twice in one live test, an implied
  /// ~7-9 m/s^2 step nothing this node's own trajectories ever command; RViz screenshots from the
  /// same test show the pull-over maneuver's curved path correctly planned and being followed
  /// right up to each crash, and /api/fail_safe/mrm_state ultimately escalated to
  /// EMERGENCY_STOP). Root cause: `autoware_autonomous_emergency_braking` (AEB) checks collision
  /// against *two* independently-generated ego paths (see its own use_predicted_trajectory and
  /// use_imu_path params) -- the MPC-based predicted path correctly reflects whatever trajectory
  /// this node is actually feeding the controller (including a curved pull-over maneuver), but
  /// the IMU-path is a naive, plan-agnostic extrapolation from raw current heading/yaw-rate/speed
  /// alone. A pull-over maneuver *deliberately* steers off the original lane toward
  /// shoulder-adjacent objects (curbs, poles, the shoulder boundary itself) that this project's
  /// own collision check (PullOverTrajectoryPlanner::isCollisionFree, checked against the *real*
  /// curved path) already clears -- but AEB's IMU-path fallback, blind to that real path, can
  /// still see those same nearby points as an imminent collision along its own straight-ish
  /// extrapolation and fire independently of anything this node commands, invisible to
  /// /control/command/control_cmd because AEB's stop is not routed through the normal
  /// trajectory-follower at all (it raises a diagnostic that this project's own live testing
  /// confirmed downstream escalates the whole MRM decision to EMERGENCY_STOP, overriding
  /// pull_over entirely).
  ///
  /// Fix: temporarily disable *only* AEB's use_imu_path for the span of an active maneuver via
  /// the same reactive, restore-on-exit AsyncParametersClient pattern already used for
  /// drive_state_stop_dist above -- use_predicted_trajectory stays on throughout, so AEB still
  /// provides real, path-aware collision protection against the pull-over trajectory this node
  /// actually plans and PullOverTrajectoryPlanner's own object-avoidance check remains the
  /// primary collision guarantee either way; only the plan-agnostic fallback that cannot tell an
  /// intentional curve from an imminent collision is suppressed, and only while this node is
  /// actively driving the vehicle off-lane on purpose.
  void pushAebImuPathOverride();
  void restoreAebImuPathOverride();

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

  /// True iff `ego_pose` (at `ego_speed`) is confirmed, by every independent criterion
  /// available, to be the genuine final pull-over position -- not merely "close to
  /// active_goal_'s point". Checks, all required:
  ///  1. distance_to_goal <= arrival_distance_threshold_ (existing).
  ///  2. |heading_error| <= arrival_heading_threshold_ (existing) -- ego's heading aligned
  ///     with the shoulder's own local direction (active_goal_'s orientation is
  ///     GoalScorer::smoothedHeading() from the live centerline at selection time, i.e.
  ///     already the shoulder's tangent, not an arbitrary pose).
  ///  3. Lateral containment -- ego's actual footprint, not just its reference point, fits
  ///     within the *measured* shoulder half-width (see latest_shoulder_halfwidth_'s docs) at
  ///     the nearest live centerline point to ego_pose, with vehicle_half_width_ +
  ///     lateral_safety_margin_ of clearance. Fails closed (returns false) if no measured
  ///     width data is available and require_shoulder_containment_ is true (the default).
  /// This is deliberately a *stricter, independent* gate than the plain arrival check used to
  /// be -- see plan()'s allow_full_stop docs for why: this function's result is what decides
  /// whether the current planning cycle is even allowed to command the vehicle toward a real
  /// stop at all, not just whether to declare kSucceeded after the fact.
  [[nodiscard]] bool isConfirmedSafeToStop(
    const geometry_msgs::msg::Pose & ego_pose, double ego_speed, double distance_to_goal,
    double heading_error) const;

  /// Lateral offset from `ego_pose` to the nearest live centerline point, and the (scaled)
  /// measured half-width there. Returns false, leaving the outputs untouched, when no
  /// index-aligned width data exists. Shared by the arrival gate and the recorded
  /// diagnostics so the two can never disagree about the same instant.
  [[nodiscard]] bool shoulderContainmentAt(
    const geometry_msgs::msg::Pose & ego_pose, double * lateral_offset,
    double * half_width) const;

  /// latest_shoulder_halfwidth_'s data as a vector iff it is present AND index-aligned
  /// (size-matched) with latest_centerline_ as it stands right now; empty otherwise --
  /// the exact "usable width data" condition both GoalScorer::selectBestGoal (goal gate)
  /// and isConfirmedSafeToStop (arrival gate) need, factored here so the two can never
  /// disagree about whether data exists.
  [[nodiscard]] std::vector<float> usableHalfWidths() const;

  /// Re-scores the live centerline from ego's *current* pose and, if anything is feasible,
  /// swaps it in as active_goal_ while STAYING in kOperating; returns false (leaving state
  /// untouched) when nothing is. Used wherever a goal is abandoned mid-maneuver.
  ///
  /// Why this exists: dropping to kDecelerating on every abandon was actively harmful once
  /// ego was already converging onto the shoulder. kDecelerating publishes
  /// buildCrawlTrajectory(), which by design drives *straight along ego's current heading*
  /// -- so every abandon threw away the turn in progress and drove the vehicle straight
  /// ahead at whatever heading it had mid-curve, then re-selected a goal from that worse
  /// pose. Live-observed 2026-08-16: the vehicle reached the shoulder but ended up parked
  /// 0.197 rad off its tangent, because the approach was repeatedly interrupted this way
  /// and never ran to completion. Re-selecting in place keeps a real, goal-directed curved
  /// trajectory published every cycle, so heading keeps converging across the swap.
  [[nodiscard]] bool reselectGoalInPlace(
    const geometry_msgs::msg::Pose & ego_pose, double ego_speed);

  /// The actual, unconditional stand-down: resets state_ to kIdle and clears every piece of
  /// in-progress maneuver state (active_goal_, last_trajectory_, override state, etc). Shared
  /// by onOperateMrm()'s immediate path (nothing was in progress to protect) and
  /// onPlanningTimer()'s grace-period-elapsed path (something *was* in progress, but the
  /// cancel held for standdown_grace_period_ with no re-engage, so it's treated as real) --
  /// see standdown_requested_since_'s docs for why those are no longer the same code path
  /// they used to be. `reason` is purely for the log line.
  void performStandDown(const std::string & reason);

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

  /// Live-verified 2026-08-13: buildDecelerationGoal() (fixed, cached, genuinely-
  /// decelerate-to-a-stop) is the wrong tool for "still searching for a shoulder goal, not
  /// actually blocked by anything." An earlier attempt fed a *receding* goal (recomputed fresh
  /// each cycle, always some distance ahead) into trajectory_planner_->plan() instead, but that
  /// still converged real vehicle speed to true v=0: plan()'s quintic *always* force-zeros its
  /// terminal sample (a hard invariant of that solver, needed for the real curved maneuver's
  /// actual arrival-and-park case), so every fresh replan kept computing real deceleration at
  /// the *current* instant to satisfy "reach v=0 by the goal," regardless of how far away that
  /// goal nominally was -- a receding goal only prevents actually *reaching* the stop point, it
  /// doesn't stop the moment-to-moment commanded speed from converging toward zero. Once
  /// genuinely at v=0, autoware_pid_longitudinal_controller's STOPPED-state departure logic
  /// (multiple independent gates: drive_state_stop_dist's departure_condition_from_stopped,
  /// enable_keep_stopped_until_steer_convergence's keep_stopped_condition, and possibly others
  /// not yet fully enumerated) can then latch the vehicle there indefinitely -- confirmed live
  /// even with the steer-convergence gate explicitly disabled.
  ///
  /// The robust fix: bypass plan() and its zero-terminal invariant entirely. Hand-builds a
  /// straight-line trajectory (ego's current heading, zero curvature) that decelerates at
  /// crawl_deceleration_ from ego_speed down to min_crawl_speed_ and then holds constant at
  /// min_crawl_speed_ for the rest of the horizon -- never zero, by construction, unlike
  /// anything routed through plan(). Recomputed fresh every single planning cycle from ego's
  /// current state. Used only for the search/holding case (kDecelerating, and kOperating's
  /// genuine-infeasibility contingency); NOT a replacement for buildDecelerationGoal() in
  /// genuinely-must-stop cases (e.g. blocked_by_traffic in onPlanningTimer()), and the real
  /// curved maneuver toward active_goal_ is untouched, still uses trajectory_planner_->plan(),
  /// which SHOULD end in a real stop once actually parked.
  [[nodiscard]] autoware_planning_msgs::msg::Trajectory buildCrawlTrajectory(
    const geometry_msgs::msg::Pose & ego_pose, double ego_speed, const rclcpp::Time & stamp) const;

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
  double arrival_heading_threshold_{0.1};  ///< rad (~5.7deg). Live-verified bug, 2026-08-13: the
                                            ///< arrival check used to test only position+speed,
                                            ///< never heading -- since this is a receding-horizon
                                            ///< planner that refits a fresh curve toward
                                            ///< active_goal_ every cycle, ego's real driven path
                                            ///< can pass within arrival_distance_threshold_ of the
                                            ///< goal *position* mid-turn, well before its heading
                                            ///< has converged to the goal's, especially for a
                                            ///< goal requiring a large heading change (e.g.
                                            ///< merging onto a shoulder that runs parallel to the
                                            ///< road from a lane a large heading-delta away).
                                            ///< Live-observed: RViz showed the vehicle declared
                                            ///< "succeeded" and parked roughly perpendicular
                                            ///< (~75deg off) to the shoulder instead of parallel.
                                            ///< Every PathShape (CurvatureSpiralPath/DubinsPath,
                                            ///< see path_shape.hpp) is constructed to terminate
                                            ///< at active_goal_'s exact pose, position AND
                                            ///< orientation, so requiring heading convergence too
                                            ///< does not change what the vehicle is being asked
                                            ///< to do -- it only stops declaring success before
                                            ///< that ask has actually been met. 0.1 rad is tight
                                            ///< enough to guarantee a visibly-parallel final
                                            ///< heading while staying comfortably above any
                                            ///< residual per-cycle replanning/tracking noise.
  /// User-directed 2026-08-15, added alongside distance/speed/heading above as a fourth,
  /// independent arrival criterion: is the vehicle's actual footprint, not just its reference
  /// point, laterally contained within the *measured* shoulder width at ego's current position?
  /// Grounded in the same automated-parking-assist practice as arrival_heading_threshold_
  /// (checking the vehicle's actual extent against the target space, not just a point-pose
  /// match) plus the practical concern that "close to the goal point" alone says nothing about
  /// whether part of the vehicle still overhangs back into the travel lane. Uses
  /// shoulder_centerline_node's measured half-width (see isConfirmedSafeToStop()'s docs) -- not
  /// an assumed constant -- so this reflects the *actual* shoulder at this specific spot.
  double lateral_safety_margin_{0.15};  ///< m. Extra clearance required beyond the vehicle's own
                                        ///< half-width before the shoulder is considered wide
                                        ///< enough -- absorbs localization/perception error and
                                        ///< leaves a visible, non-hairline gap from the shoulder
                                        ///< edge, similar in spirit to standard parking-assist
                                        ///< clearance margins.
                                        ///<
                                        ///< Lowered 0.3 -> 0.15 on 2026-08-16 against live
                                        ///< measurements. This budget is spent on *centering*,
                                        ///< not on width: the check requires lateral_offset +
                                        ///< vehicle_half_width + this <= measured half-width, so
                                        ///< with the 1.32m median half-width now measured on this
                                        ///< shoulder and a 0.948m vehicle half-width, 0.3 left
                                        ///< only 0.07m of allowable offset from the centerline --
                                        ///< tighter than the stock controller's own tracking
                                        ///< error, so arrival could never confirm no matter how
                                        ///< well the vehicle parked. 0.15 leaves 0.22m, which the
                                        ///< maneuver can actually hit. Real clearance is larger
                                        ///< than this arithmetic suggests, since the measured
                                        ///< half-width is itself a conservative view of the
                                        ///< shoulder (1.32m measured vs 1.75m CARLA ground
                                        ///< truth -- the mask does not reach the very edge).
                                        ///< Note goal_width_margin (GoalScorer, 0.3) is a
                                        ///< separate, deliberately stricter budget: it decides
                                        ///< whether a stretch is wide enough to aim at in the
                                        ///< first place, where being conservative costs only a
                                        ///< candidate, not a completed maneuver.
  /// If true (default) and no measured shoulder-width data is available (topic never received,
  /// or shoulder_centerline_node running in "median" mode, which has no extent to measure --
  /// see lateralHalfWidth()'s docs in that package), containment can never be confirmed and the
  /// vehicle keeps crawling at min_crawl_speed_/min_departure_reference_speed_ instead of
  /// stopping -- fails closed rather than silently skipping the check, consistent with this
  /// being a safety-relevant criterion the user explicitly asked for. Set false only for
  /// environments/tests that intentionally don't run shoulder_centerline_node with width
  /// publishing (falls back to distance+speed+heading only, the pre-2026-08-15 behavior).
  bool require_shoulder_containment_{true};
  /// Half of the vehicle's real width, computed at startup from wheel_tread/left_overhang/
  /// right_overhang (the same vehicle_info fields every other Autoware node in this launch
  /// receives) as 0.5*wheel_tread + max(left_overhang, right_overhang) -- the standard
  /// Autoware vehicle_info width formula, not an assumed constant. Used by
  /// isConfirmedSafeToStop()'s containment check.
  double vehicle_half_width_{0.95};
  /// m. Distance to the active goal below which the maneuver is committed to finishing on it:
  /// this cycle's trajectory may decelerate to a genuine stop (plan()'s allow_full_stop), and
  /// the goal is no longer swapped out by reselectGoalInPlace(). Deliberately larger than
  /// arrival_distance_threshold_ (1.0m) so the vehicle has room to actually decelerate into
  /// arrival rather than needing to already be there.
  ///
  /// Added 2026-08-16 to break two coupled endgame failures seen live (bag
  /// pullover_final2_1786911450): (a) allow_full_stop was gated on containment measured at
  /// ego's current pose, which cannot become true while the vehicle is still short of the
  /// goal and off-center, so speed stayed floored at 2.5 m/s and it could never settle; and
  /// (b) every re-selection picks a goal at least GoalScorer's min_stopping_distance (~5m)
  /// ahead, so a re-selection triggered during final approach makes the target recede and the
  /// vehicle can never close the last metre -- live it stopped 1.11m short of a goal that had
  /// just moved from y=203.7 to y=208.5.
  double final_approach_engage_distance_{4.0};
  /// rad. How straight the vehicle must already be before it is allowed to commit to a real
  /// stop. Deliberately looser than arrival_heading_threshold_ (0.1) -- the final decel still
  /// has to finish the last few degrees -- but tight enough to refuse a visibly crooked stop.
  /// Without it the vehicle could halt mid-turn, which is exactly how it came to rest 0.13
  /// rad off the shoulder tangent, looking parked at an angle rather than pulled over.
  double final_approach_heading_tolerance_{0.20};
  /// m/s. Speed floor used inside final_approach_engage_distance_ instead of the planner's
  /// min_departure_reference_speed (2.5). That value is an *exposure* limit for crossing the
  /// travel lane; here the binding concern is *precision*, and 2.5 m/s is too fast for the
  /// lateral controller to converge heading over the last few metres of a curving entry --
  /// live-verified as a 7.5 deg residual at rest. 1.0 m/s still clears stock Autoware's
  /// zero-velocity deadlock epsilons (1e-3 / 1e-2) by two orders of magnitude, so it does not
  /// reintroduce the STOPPED-state latch this floor originally existed to avoid.
  double final_approach_reference_speed_{1.0};
  int max_consecutive_planning_failures_{30};
  int max_consecutive_traffic_blocked_cycles_{300};  ///< See consecutive_traffic_blocked_cycles_.
  double deceleration_lookahead_distance_{40.0};  ///< m. How far ahead (along ego's current
                                                   ///< heading) the synthetic straight-line
                                                   ///< braking goal is placed each cycle -- only
                                                   ///< used for genuine must-actually-stop cases
                                                   ///< (blocked_by_traffic) now; see
                                                   ///< buildCrawlTrajectory() for the
                                                   ///< search/holding case, which never targets
                                                   ///< a real stop.
  double deceleration_giveup_speed_threshold_{0.3};  ///< m/s. Unused now that kDecelerating
                                                      ///< crawls rather than brakes to a stop
                                                      ///< (speed floors at min_crawl_speed_,
                                                      ///< always above this) -- superseded by
                                                      ///< kdecelerating_giveup_duration_. Left
                                                      ///< declared for config-file compatibility.
  double min_crawl_speed_{2.5};  ///< m/s (~9 km/h). Live-verified 2026-08-13: the floor below
                                  ///< which autoware_pid_longitudinal_controller's STOPPED-state
                                  ///< departure logic can latch indefinitely -- comfortably
                                  ///< clears every zero-velocity epsilon found in that stock
                                  ///< controller (vel_epsilon=1e-3, stopped_state_entry_vel=0.01,
                                  ///< MPC's stop_state_entry_target_speed=0.001) by orders of
                                  ///< magnitude. Never let actual commanded speed drop below
                                  ///< this while kDecelerating or contingency-holding during
                                  ///< kOperating -- see buildCrawlTrajectory().
                                  ///<
                                  ///< Raised from 1.0 to 2.5 m/s 2026-08-15, user-directed: on a
                                  ///< highway with moving traffic, crawling near-zero (or briefly
                                  ///< re-stopping) anywhere before the vehicle is safely off the
                                  ///< active lane is itself a hazard, not just an inconvenience --
                                  ///< SAE J3016's minimal risk condition guidance explicitly
                                  ///< favors stopping *outside* active lanes over stopping within
                                  ///< them, and rear-end collisions are consistently one of the
                                  ///< largest categories of highway crashes in general collision
                                  ///< statistics. 2.5 m/s matches the ~10 km/h low-speed "creep"
                                  ///< convention already standard for automotive low-speed
                                  ///< maneuvering (parking, automatic-transmission creep torque
                                  ///< caps) and now equals TrajectoryPlannerParams::
                                  ///< min_departure_reference_speed so the search phase (this
                                  ///< field) and the in-maneuver phase (that one, via plan()'s
                                  ///< allow_full_stop) agree on one minimum.
  double crawl_deceleration_{0.5};  ///< m/s^2. Deceleration rate buildCrawlTrajectory() ramps
                                     ///< down at, from ego_speed to min_crawl_speed_, before
                                     ///< holding constant -- see that function's docs.
                                     ///<
                                     ///< Live/bag-verified bug, 2026-08-13: was 1.5, and even
                                     ///< though the *published* reference correctly floors at
                                     ///< min_crawl_speed_ (bag-confirmed: /control/command/
                                     ///< control_cmd's commanded velocity genuinely settled at
                                     ///< exactly 1.000), *real* ego velocity (from
                                     ///< /vehicle/status/velocity_status, effectively CARLA
                                     ///< ground truth) blew straight through that floor down to
                                     ///< a genuine 0.000 m/s roughly 0.6s *before* the commanded
                                     ///< reference itself even finished converging to the floor.
                                     ///< This is a classic closed-loop overshoot: this node reads
                                     ///< ego_speed from /localization/kinematic_state (EKF-fused,
                                     ///< not raw ground truth), and this project's EKF instance
                                     ///< has a persistent, live-observed twist-queue backlog
                                     ///< ("Twist queue size (3) is exceeding max_queue_size (2)"),
                                     ///< i.e. a real fusion lag -- every cycle, buildCrawlTrajectory()
                                     ///< recommits to decelerating at crawl_deceleration_ from
                                     ///< whichever (lagged, so effectively stale-high) ego_speed
                                     ///< it was just handed, so the *true*, un-lagged vehicle
                                     ///< keeps decelerating past the intended floor for the
                                     ///< duration of that lag before the loop "notices" and stops
                                     ///< commanding further braking. For a fixed sensing/control
                                     ///< latency Δt, overshoot past the floor is approximately
                                     ///< crawl_deceleration_ * Δt -- halving the deceleration rate
                                     ///< halves the overshoot for the same Δt. This is a real,
                                     ///< reusable control-systems relationship (see e.g. Åström &
                                     ///< Murray, *Feedback Systems*, ch. 3 on the effect of loop
                                     ///< delay on step-response overshoot), not a magic number --
                                     ///< 0.5 m/s^2 gives roughly 3x the margin the old 1.5 m/s^2
                                     ///< did against the same observed lag, which live-testing
                                     ///< confirmed keeps real ego speed from ever crashing through
                                     ///< min_crawl_speed_ during the search phase.
  double crawl_trajectory_duration_{8.0};  ///< s. Total horizon of buildCrawlTrajectory()'s
                                            ///< hand-built trajectory -- decelerates at
                                            ///< crawl_deceleration_ from ego_speed to
                                            ///< min_crawl_speed_ (if above it), then holds
                                            ///< constant at min_crawl_speed_ for the remainder.
                                            ///< Raised from 5.0 alongside crawl_deceleration_'s
                                            ///< reduction so the horizon still comfortably covers
                                            ///< a full ramp from a typical cruise speed (e.g.
                                            ///< ~4.5 m/s: ramp_duration=(4.5-1.0)/0.5=7.0s) with
                                            ///< some hold-phase margin left over, rather than the
                                            ///< published trajectory ending mid-ramp.
  double kdecelerating_giveup_duration_{40.0};  ///< s. If kDecelerating has been searching this
                                                 ///< long with still no feasible shoulder goal,
                                                 ///< give up -- replaces the old speed-threshold
                                                 ///< give-up, which no longer fires now that
                                                 ///< speed floors at min_crawl_speed_ instead of
                                                 ///< decaying toward 0. Raised 20->40 2026-08-16
                                                 ///< alongside the measured-width goal gate
                                                 ///< (goal_scorer.hpp): wide-enough shoulder
                                                 ///< pockets are sparser than "any shoulder", so
                                                 ///< the crawl may legitimately need to cover
                                                 ///< more ground (~100m at min_crawl_speed_)
                                                 ///< before one enters lookahead range.
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
  /// Index-aligned with latest_centerline_->poses (data[i] is the measured half-width, meters,
  /// at poses[i]) -- see shoulder_centerline_node's halfwidth_pub_ docs for how it's measured.
  /// Independently timestamped/updated from latest_centerline_ itself (separate topic), so a
  /// size mismatch against the *current* latest_centerline_ is possible for one cycle right
  /// after either updates -- isConfirmedSafeToStop() guards against that explicitly rather than
  /// assuming the two always agree in length.
  std_msgs::msg::Float32MultiArray::ConstSharedPtr latest_shoulder_halfwidth_;
  nav_msgs::msg::Odometry::ConstSharedPtr latest_odometry_;
  autoware_perception_msgs::msg::PredictedObjects latest_objects_;
  bool autonomous_mode_engaged_{false};
  std::atomic<bool> keyboard_trigger_pending_{false};

  ManagerState state_{ManagerState::kIdle};
  std::optional<geometry_msgs::msg::Pose> active_goal_;
  /// Which shape+duration successfully planned toward active_goal_ last
  /// cycle, fed back in as PullOverTrajectoryPlanner::plan()'s
  /// preferred_hint next cycle -- see ShapeHint's docs (trajectory_planner.hpp)
  /// for why this matters: without it, a receding-horizon replan from
  /// scratch every ~100ms could land on a different duration/shape/radius
  /// near any feasibility boundary even though the goal never moved, so the
  /// real driven path was a patchwork of many different short-lived plans
  /// instead of one stable curve -- live-observed 2026-08-12 as the real
  /// path visibly diverging from the visualized planned path. Reset
  /// whenever active_goal_ itself resets.
  std::optional<PullOverTrajectoryPlanner::ShapeHint> active_goal_shape_hint_;
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
  /// Same purpose as active_goal_shape_hint_, for deceleration_goal_'s own
  /// plan() calls. Reset whenever deceleration_goal_ itself resets.
  std::optional<PullOverTrajectoryPlanner::ShapeHint> deceleration_goal_shape_hint_;
  /// Set the instant kDecelerating begins, cleared on exit (either a real
  /// goal becomes reachable, or giving up) -- drives
  /// kdecelerating_giveup_duration_'s time-based give-up, which replaced
  /// deceleration_giveup_speed_threshold_'s speed-based one now that
  /// kDecelerating crawls at min_crawl_speed_ instead of braking to a stop.
  std::optional<rclcpp::Time> decelerating_since_;
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
  /// Same purpose as active_goal_shape_hint_, for contingency_stop_goal_'s
  /// own plan() calls. Reset whenever contingency_stop_goal_ itself resets.
  std::optional<PullOverTrajectoryPlanner::ShapeHint> contingency_stop_goal_shape_hint_;
  int consecutive_planning_failures_{0};
  /// Peak curvature the trajectory planner's spiral attempt converged to on the most recent
  /// genuine (non-traffic-blocked) planning failure against active_goal_ -- see
  /// consecutive_curvature_increases_'s docs for why this is tracked cycle to cycle instead of
  /// just discarded like the rest of plan()'s per-cycle output. Reset to 0.0 alongside
  /// consecutive_planning_failures_ (fresh goal committed, or a cycle succeeds again).
  double last_attempted_curvature_{0.0};
  /// Consecutive genuine-infeasibility cycles in which the attempted spiral curvature both (a)
  /// increased from the previous cycle and (b) already exceeds
  /// curvature_trend_early_warning_fraction_ * trajectory_params_.max_curvature. Reset to 0
  /// alongside consecutive_planning_failures_.
  ///
  /// Why this exists: buildCrawlTrajectory() (used while retrying a genuinely-infeasible goal,
  /// see onPlanningTimer()) deliberately moves ego in a straight line along its *current*
  /// heading rather than curving toward active_goal_ -- necessary to avoid re-triggering the
  /// stock longitudinal controller's STOPPED-state departure deadlock (see that function's
  /// docs), but as an unavoidable geometric consequence it cannot reduce the heading error to a
  /// goal it isn't steering toward. Every cycle spent crawling while retrying the *same* fixed
  /// goal therefore shrinks the remaining distance without correcting heading, which can only
  /// make the curvature required to still reach that goal increase -- confirmed live 2026-08-14:
  /// a goal that scored comfortably feasible (curvature 0.1865) at selection time climbed to
  /// 0.237, then past the 0.27 cap up to 0.618, over the standard 30-cycle
  /// max_consecutive_planning_failures_ give-up budget (~10.5s), while ego kept crawling forward
  /// the whole time instead of holding position. min_maneuver_forward_progress (see
  /// GoalScorer) exists to catch exactly this kind of geometric collapse but does not reliably
  /// trip in time for every such case, since "forward room" and "curvature margin" are related
  /// but not the same signal -- this is the direct, solver-reported version of the same check,
  /// answering "is this specific attempt getting worse" rather than approximating it
  /// geometrically. This is a *detection* mechanism only, not a substitute for
  /// min_maneuver_forward_progress or for max_curvature itself -- do not raise max_curvature to
  /// chase this failure mode (see its own docs for why that was already tried and reverted).
  int consecutive_curvature_increases_{0};
  double curvature_trend_early_warning_fraction_{0.8};  ///< Fraction of
                                                          ///< trajectory_params_.max_curvature
                                                          ///< above which an increasing attempt
                                                          ///< starts counting toward
                                                          ///< max_consecutive_curvature_increases_
                                                          ///< -- below this fraction, an
                                                          ///< increasing-but-still-comfortable
                                                          ///< curvature is not worth abandoning
                                                          ///< over (e.g. ordinary cycle-to-cycle
                                                          ///< search noise near a well-scored
                                                          ///< goal).
  int max_consecutive_curvature_increases_{3};  ///< Abandon active_goal_ (return to kDecelerating
                                                 ///< / search) once this many consecutive cycles
                                                 ///< show a worsening curvature trend past
                                                 ///< curvature_trend_early_warning_fraction_,
                                                 ///< instead of grinding through the full
                                                 ///< max_consecutive_planning_failures_ budget
                                                 ///< while crawling straight into a geometry that
                                                 ///< can only get worse. Deliberately much smaller
                                                 ///< than max_consecutive_planning_failures_ (30):
                                                 ///< that budget still exists for goals whose
                                                 ///< failure reason isn't curvature-trending (e.g.
                                                 ///< jerk/accel limits at a roughly-constant
                                                 ///< curvature), where retrying as-is can still
                                                 ///< recover.
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

  /// Set the instant an OperateMrm(operate=false) arrives while a maneuver is actively
  /// in progress (kOperating/kDecelerating); cleared either by a fresh operate=true
  /// arriving before standdown_grace_period_ elapses (treated as a false alarm -- resume
  /// with no state lost) or by the grace period actually elapsing (a real stand-down,
  /// performed then). See onOperateMrm()'s and standdown_grace_period_'s docs for why this
  /// exists: mrm_handler cancels this node's OperateMrm service on every upstream
  /// availability blip, not only on a genuine operator decision to abandon pull-over --
  /// live rosbag-confirmed 2026-08-15 (root-caused via /diagnostics_graph/struct+/status,
  /// see project memory) that an intermittent ekf_localizer WARN under
  /// /autoware/localization/sensor_fusion_status, with no hysteresis applied to it,
  /// propagates through mrm_handler's own AND-gated availability check and cancels+re-calls
  /// this service every ~1-2s for well over a minute at a stretch. Before this field
  /// existed, every single one of those cancels immediately threw away the active goal and
  /// all trajectory/warm-start state (see the old unconditional reset this replaces), so the
  /// vehicle was repeatedly restarting its approach from scratch -- accelerating toward a
  /// freshly-reselected goal, then getting cancelled again before traveling more than a few
  /// centimeters -- and never made net progress. This node has no ability to fix the
  /// upstream diagnostic or mrm_handler's own re-evaluation logic (not this project's code,
  /// see project scope notes) -- the only lever available here is not overreacting to a
  /// same-service cancel/re-call pair that arrives faster than a maneuver could plausibly
  /// have been abandoned and reconsidered for real.
  std::optional<rclcpp::Time> standdown_requested_since_;
  /// Seconds an OperateMrm(operate=false) must persist, with no intervening operate=true,
  /// before it's treated as a genuine stand-down rather than an upstream availability blip
  /// -- see standdown_requested_since_'s docs.
  ///
  /// Raised from an initial 3.0s to 90.0s, 2026-08-15, after live data showed 3.0s was still
  /// far too short: /diagnostics_graph/status (see project memory for the capture/analysis
  /// technique) showed individual ekf_localizer WARN episodes lasting up to 70.76s, and a
  /// live re-test with the 3.0s grace period still recorded 6 full stand-downs (each
  /// discarding the selected goal and restarting the approach) in under 2 minutes -- the fix
  /// was directionally correct but sized against the *typical* blip, not the *worst observed*
  /// one. 90s gives >19s of margin over the worst episode seen so far.
  ///
  /// This is deliberately NOT a small, "safe-because-it's-short" number, and that is
  /// intentional, not an oversight: standing down early is not actually the safer choice
  /// here. autoware_vehicle_cmd_gate independently enforces real vehicle safety off of
  /// `/api/fail_safe/mrm_state` directly (`onMrmState()`/`is_system_emergency_`, live-
  /// confirmed 2026-08-15 via rosbag: whenever mrm_handler reports EMERGENCY_STOP, the real
  /// published `/control/command/control_cmd` visibly switches to mrm_emergency_stop_
  /// operator's own commands, not this node's, regardless of what this node internally
  /// believes or publishes) -- so a long pending-standdown window on *this* node's side does
  /// not, on its own, let the vehicle do anything unsafe: whatever this node keeps computing
  /// is moot while the real emergency behavior is active, and simply resumes cleanly, with
  /// its goal and warm-start state intact, the moment mrm_handler hands PULL_OVER back. What
  /// a short grace period actually bought was the opposite of safety: needless full resets of
  /// an otherwise-fine maneuver, each one restarting the approach and burning through the
  /// *overall* MRM's limited patience (mrm_handler's own escalation-to-EMERGENCY_STOP-for-
  /// good, see project memory) without ever making net progress -- a textbook case of
  /// oscillating behavior selection needing a commitment/hysteresis threshold rather than a
  /// fast reaction, a well-established pattern in autonomous-vehicle behavior arbitration
  /// literature for exactly this "cost/availability signal flickers, selected behavior
  /// shouldn't" failure mode.
  ///
  /// Still bounded, not infinite, for two independent reasons: `autonomous_mode_engaged_`
  /// dropping is checked separately (see onOperateMrm()) and forces an immediate real stand-
  /// down regardless of this timer -- a genuine loss of autonomous control is a strictly
  /// stronger, much less noisy signal than a diagnostic-derived availability flag, and always
  /// wins outright. And even absent that, 90s is still a firm ceiling: if availability truly
  /// never recovers, this node gives up for real at the deadline rather than holding a
  /// half-triggered state forever.
  double standdown_grace_period_{90.0};

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

  /// See pushAebImuPathOverride()/restoreAebImuPathOverride(). Same
  /// active-flag pattern as controller_override_active_, guarding against a
  /// redundant restore RPC.
  bool aeb_override_enabled_{true};
  std::string aeb_node_name_{"/control/autonomous_emergency_braking"};
  bool aeb_override_active_{false};
  rclcpp::AsyncParametersClient::SharedPtr aeb_param_client_;

  // --- ROS interfaces ---------------------------------------------------
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr centerline_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr shoulder_halfwidth_sub_;
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
