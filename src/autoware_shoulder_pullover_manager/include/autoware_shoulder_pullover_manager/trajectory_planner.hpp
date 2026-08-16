#pragma once

#include "autoware_shoulder_pullover_manager/curvature_spiral_path.hpp"
#include "autoware_shoulder_pullover_manager/dubins_path.hpp"
#include "autoware_shoulder_pullover_manager/path_shape.hpp"

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
  /// Seed for the *start* acceleration boundary condition (a0) of the next
  /// quintic solve, m/s^2. Deliberately NOT a live sensor reading (see
  /// buildCandidate's docs for why that was rejected) -- callers should
  /// derive this from the previously-*commanded* trajectory's own predicted
  /// acceleration at the current time, not from odometry/IMU. Defaults to
  /// 0.0, which reproduces the original always-zero behavior for any caller
  /// that doesn't set it (e.g. a first-ever call with no prior trajectory to
  /// warm-start from).
  double accel{0.0};
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
  double min_departure_reference_speed{2.5};  ///< m/s (~9 km/h) floor applied to published
                                               ///< speed -- never the true v0, which stays the
                                               ///< quintic's exact boundary condition.
                                               ///<
                                               ///< Raised from 1.0 to 2.5 m/s 2026-08-15, user-
                                               ///< directed: live-testing showed the vehicle
                                               ///< repeatedly decelerating toward 0 and
                                               ///< re-accelerating mid-maneuver while still
                                               ///< exposed on or crossing into the shoulder next
                                               ///< to moving highway traffic -- explicitly the
                                               ///< condition SAE J3016's minimal risk condition
                                               ///< guidance steers away from (its own worked
                                               ///< examples are "stopped outside active lanes",
                                               ///< not stopped within/crossing them), and general
                                               ///< rear-end-collision-risk literature attributes a
                                               ///< large share of highway crashes to exactly this
                                               ///< class of low-speed/stopped exposure. 2.5 m/s
                                               ///< matches the ~10 km/h low-speed "creep"
                                               ///< convention already standard in automotive
                                               ///< automatic-transmission/parking-maneuver control
                                               ///< and equals PullOverManagerNode::min_crawl_speed_
                                               ///< so the pre-maneuver search phase and the
                                               ///< in-maneuver phase agree on one minimum.
                                               ///<
                                               ///< As of 2026-08-15 this floor is no longer
                                               ///< conditional on final_approach_distance alone --
                                               ///< see allow_full_stop's docs on plan(): whether
                                               ///< the terminal sample is allowed below this
                                               ///< floor (down to a genuine stop) is decided
                                               ///< explicitly by the caller, not implicitly by
                                               ///< remaining distance, so a goal that simply
                                               ///< looks geometrically close hasn't yet been
                                               ///< *confirmed* safe to fully stop at can never
                                               ///< coast down to 0 no matter how close it is.

  double final_approach_distance{2.0};  ///< m. Once a candidate is built with allow_full_stop
                                         ///< true (see plan()'s docs), the remaining arc length
                                         ///< below which min_departure_reference_speed's floor
                                         ///< stops being applied, letting the quintic's own true
                                         ///< (possibly near-zero) velocity through so the vehicle
                                         ///< can actually decelerate smoothly to a stop at the
                                         ///< goal instead of the floor holding speed up right to
                                         ///< the last sample. Irrelevant when allow_full_stop is
                                         ///< false -- the floor then applies to every sample
                                         ///< regardless of remaining distance. Physically
                                         ///< derived, not arbitrary: must be at least the
                                         ///< kinematic stopping distance from the highest speed
                                         ///< that could be commanded when this zone is entered,
                                         ///< d = v^2 / (2*|min_longitudinal_accel|). Worst case
                                         ///< here is v=trajectory_max_speed=2.0 m/s decelerating
                                         ///< at min_longitudinal_accel=-2.5 m/s^2: d = 4.0/5.0 =
                                         ///< 0.8m. The 2.0m default keeps a >2x margin over that
                                         ///< floor (comfortable, not max-braking, deceleration)
                                         ///< and stays comfortably inside arrival_distance_threshold_'s
                                         ///< typical 1.0m in PullOverManagerNode, so the
                                         ///< arrival check always evaluates speed *after* this
                                         ///< zone's genuine decel has had room to act.
  double max_speed{3.0};  ///< m/s. Hard ceiling on every sample's *published* speed, checked
                           ///< unconditionally in satisfiesKinematicConstraints() (unlike the
                           ///< curvature/lateral checks below, this is never speed-gated -- it
                           ///< exists specifically to catch runaway speed itself). Added
                           ///< 2026-08-12 after a live near-miss: satisfiesKinematicConstraints()
                           ///< previously checked curvature/lateral-accel/lateral-jerk/
                           ///< longitudinal-accel but never bounded raw velocity magnitude at
                           ///< all. A receding-horizon replan starting from a *growing* v0 (e.g.
                           ///< while recovering from a stuck/STOPPED state) toward a fixed
                           ///< remaining goal distance can produce a minimum-jerk quintic whose
                           ///< *middle* samples overshoot well above both boundary speeds even
                           ///< though every other per-sample check passes -- observed live as
                           ///< unchecked acceleration to ~9.3 m/s. This is a slow, precision
                           ///< entry maneuver (see the struct-level comment above), so this is
                           ///< a deliberately low ceiling, comfortably under any reasonable
                           ///< external safety-monitor threshold -- this check should trip
                           ///< first, not the external monitor.
                           ///<
                           ///< Raised from 2.0 to 3.0 m/s 2026-08-15: live-verified BUG --
                           ///< min_departure_reference_speed was independently raised to 2.5 m/s
                           ///< the same day without reconciling it against this ceiling, and
                           ///< since that floor now applies to *every* sample when
                           ///< allow_full_stop is false (not just outside some distance gate,
                           ///< see that field's docs), every single candidate was rejected here
                           ///< unconditionally -- "speed 2.5 m/s exceeds max 2 m/s at point 0",
                           ///< every cycle, regardless of goal geometry, confirmed live via
                           ///< /shoulder_pullover_manager's own log. 3.0 m/s keeps a real margin
                           ///< above the 2.5 m/s floor for the quintic's own natural profile to
                           ///< vary within without immediately tripping this ceiling, while
                           ///< still comfortably below any external safety-monitor threshold.
                           ///< Keep this and min_departure_reference_speed in the same file in
                           ///< sync from now on -- they are not independent constants.

  double min_path_turn_radius{4.0};   ///< m. Tightest DubinsPath turning radius tried (see
                                       ///< buildCandidate/plan) -- curvature=0.25 1/m, under the
                                       ///< 0.27 checked cap, itself under the 0.302 hard
                                       ///< kinematic wall (see max_curvature below).
  double max_path_turn_radius{15.0};  ///< m. Widest turning radius tried before giving up on a
                                       ///< goal entirely. Still short relative to
                                       ///< GoalScorer's 60m max_lookahead_distance, so even the
                                       ///< widest candidate stays a local maneuver.
  double path_turn_radius_step{0.25};  ///< m. Increment between tried radii (see plan()).
                                        ///< **Radius is searched, not fixed** -- live-tested
                                        ///< finding (2026-08-05): a DubinsPath's arc<->straight
                                        ///< segment junction has genuinely discontinuous
                                        ///< curvature (0 <-> 1/radius instantaneously), which
                                        ///< shows up as a lateral-jerk spike in the discretized
                                        ///< check at that one sample. This spike does *not*
                                        ///< shrink monotonically with a wider radius -- widening
                                        ///< the radius also lengthens the path, which raises
                                        ///< speed at the junction for a fixed duration, and the
                                        ///< two effects compete non-monotonically (confirmed
                                        ///< offline: for one real live-tested goal, radius 5.5-6.0
                                        ///< passed cleanly, but 6.25+ produced an even *worse*
                                        ///< jerk spike at a different junction than the one at
                                        ///< radius 4.0-5.0). A single hardcoded "radius that
                                        ///< happened to work for one test goal" would be exactly
                                        ///< the same fragile-tuning mistake as the earlier
                                        ///< max_curvature escalation, just moved one level
                                        ///< removed -- searching a range and keeping the first
                                        ///< fully-feasible one (same pattern already used for
                                        ///< duration below) is what actually generalizes across
                                        ///< goals.

  double max_curvature{0.27};         ///< 1/m. **Not** tuned against planning_validator's own
                                       ///< curvature threshold (1.0 1/m) -- that figure is a
                                       ///< generic sanity/corruption check (same family as a
                                       ///< 100.0-threshold check next to it in
                                       ///< trajectory_checker.param.yaml), not a claim that this
                                       ///< vehicle can actually steer that sharply. The real,
                                       ///< speed-independent ceiling is kinematic: for the
                                       ///< sample_vehicle model actually running in this project
                                       ///< (wheel_base=2.79m, max_steer_angle=0.70rad, from
                                       ///< sample_vehicle_description/config/vehicle_info.param.yaml),
                                       ///< curvature = tan(steer_angle)/wheel_base caps out at
                                       ///< tan(0.70)/2.79 ~ 0.302 1/m (min turn radius ~3.3m) --
                                       ///< the front wheels physically cannot turn any sharper
                                       ///< than that, at any speed. 0.27 leaves ~10% actuator
                                       ///< margin under that hard wall. An earlier version of this
                                       ///< cap was raised repeatedly (0.5 -> 0.7 -> 0.8) chasing
                                       ///< live-tested near-goal violations (0.53 -> 0.736 -> 0.835
                                       ///< 1/m) that looked "physically fine" because lateral_accel
                                       ///< stayed trivial at the low speeds involved --
                                       ///< lateral_accel is speed-gated and doesn't capture this;
                                       ///< a slow vehicle still can't out-steer its own wheelbase.
                                       ///< Do not raise this again without changing the vehicle
                                       ///< model or re-deriving from real steering kinematics.
                                       ///< This check is now a redundant safety net rather than
                                       ///< the primary curvature guarantee -- since the DubinsPath
                                       ///< shape change (see buildCandidate / path_turn_radius
                                       ///< above), curvature is bounded *by construction*
                                       ///< everywhere except within a handful of
                                       ///< finite-difference samples straddling a segment
                                       ///< junction (where curvature is genuinely discontinuous,
                                       ///< e.g. arc-to-straight); if this check still fails after
                                       ///< that change, suspect the DubinsPath implementation or
                                       ///< path_turn_radius margin before suspecting the
                                       ///< duration/speed profile.
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
/// Method (revised twice -- see the two "why this changed" notes below,
/// read them in order, don't redo either mistake): the maneuver's *shape*
/// is generated by trying, in order, two independent `PathShape`
/// implementations (see path_shape.hpp) connecting start and goal poses:
///
/// 1. `CurvatureSpiralPath` (preferred) -- variable curvature, a smooth
///    cubic polynomial of arc length, solved via Newton shooting so it
///    fits the exact goal with usually the shortest reasonable path and no
///    forced constant-radius detour. Tried first because when it converges
///    it is almost always both shorter and gentler than the fallback.
/// 2. `DubinsPath` (fallback) -- constant-radius, searched over a range of
///    radii (see path_turn_radius_step's docs). Deterministic and always
///    finds *some* path if one exists at any tried radius, unlike the
///    spiral's numerical solve, which is not guaranteed to converge for
///    every geometry -- kept specifically as the thing to fall back on
///    when the spiral doesn't converge, not because it's otherwise
///    preferred.
///
/// Whichever shape is used, *speed along* it is a single scalar quintic
/// (minimum-jerk) profile s(t) over arc length, matching start speed and a
/// near-zero terminal approach speed -- the same boundary-value-problem
/// structure as the Frenet-frame quintic planner in Werling et al. 2010
/// (see the framework report), applied to a 1D longitudinal coordinate
/// along whichever shape was chosen.
///
/// Why this changed (round 1, 2026-08-05 morning): an earlier version fit
/// two *independent* Cartesian quintics for x(t) and y(t) directly. That
/// works fine for short, nearly-straight goals, but was live-tested
/// against a real shoulder goal requiring a ~90-degree heading change over
/// ~11.6m and found to require ~0.28 1/m curvature *no matter how the
/// duration or terminal speed were tuned* (verified by sweeping duration up
/// to 90s and terminal speed up to 5 m/s offline) -- the peak curvature
/// converged to a floor well above the true geometric minimum for that
/// exact pair (~0.12 1/m, achievable by a constant-curvature arc), because
/// independently-fit per-axis polynomials don't distribute a large heading
/// change evenly across the path. This is a genuine shape limitation, not
/// a numerical artifact -- hence the Dubins-path rewrite.
///
/// Why this changed (round 2, 2026-08-05 evening): a *different* real live
/// goal -- moving ego, start/goal headings nearly *parallel* but offset
/// diagonally, a lateral "sidestep" shape, not a big-heading-change shape
/// at all -- broke the pure-Dubins approach in the opposite direction: the
/// short, sensible CSC solution (mode RSL) turned out to only be
/// mathematically valid below roughly the vehicle's own kinematic radius
/// limit (verified offline, including checking the LRL/RLR family too --
/// didn't help), so every *safe* fixed radius produced only a wildly long
/// same-direction loop (60-100+ meters, verified offline) instead. A
/// single global turning radius is the wrong model for a maneuver that
/// only needs a tight turn briefly, not throughout -- hence
/// `CurvatureSpiralPath`, which lets curvature vary along the path instead
/// of committing to one value for the whole thing. (Also verified offline:
/// this specific sidestep goal remains only barely, if at all, achievable
/// even by the spiral within the safety-margined curvature cap -- it's a
/// goal right at the edge of what this vehicle can physically do in a
/// short direct maneuver, not a remaining algorithm gap. If the spiral
/// also reports invalid for a goal like this, that is probably the correct
/// answer, not a bug to keep chasing.)
///
/// This remains a local, from-live-perception maneuver -- not a claim that
/// Frenet/lanelet-relative planning is unnecessary in general (the report's
/// math section covers that fuller formulation for the on-road portion of
/// a maneuver); it's still independent of the Lanelet2 routing graph for
/// the reasons above.
///
/// For whichever shape is in use, a family of candidate maneuver durations
/// is tried (short to long) -- for the Dubins fallback, this is nested
/// inside its own radius search (tight to wide); for the spiral, there is
/// no shape-level search (see CurvatureSpiralPath's class docs for why
/// one isn't needed there). For each combination, the resulting trajectory
/// is checked against configured curvature/jerk/acceleration limits and
/// against tracked objects' predicted paths, and the first feasible,
/// collision-free candidate is returned. Called every planning cycle
/// (receding-horizon replanning) by the owning node, not just once at
/// trigger time.
class PullOverTrajectoryPlanner
{
public:
  /// Identifies exactly which shape+parameters a successful plan() call
  /// used, so the caller (PullOverManagerNode) can ask for the *same* one
  /// again next cycle -- see plan()'s `preferred_hint`/`used_hint` docs for
  /// why this exists: added 2026-08-12 after live testing found the real
  /// driven path visibly diverging from the visualized planned path. Root
  /// cause: this is a receding-horizon planner that rebuilds the entire
  /// trajectory from scratch every ~100ms, and `initial_guess` (the
  /// duration search's own starting point) shifts every single cycle as
  /// distance-to-goal and speed change -- so "first feasible candidate"
  /// could land on a different duration, or even a different shape/radius
  /// entirely, cycle to cycle near any feasibility boundary, even though
  /// the goal itself never moved. The real vehicle was never actually
  /// tracking one smooth curve; it was following a patchwork of many
  /// different short-lived plans. This also plausibly explains why MPC's
  /// own steer-convergence check never settled (see project memory) -- it
  /// was being fed a constantly-shifting reference, not a fixed target to
  /// converge to.
  struct ShapeHint
  {
    bool use_spiral{true};    ///< true: CurvatureSpiralPath; false: DubinsPath with turn_radius.
    double turn_radius{0.0};  ///< Only meaningful when !use_spiral.
    double duration{0.0};
  };

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
  ///
  /// If `blocked_by_traffic` is non-null, it is set to true iff at least one
  /// tried candidate satisfied every kinematic constraint and was rejected
  /// *only* by the collision check -- i.e. the maneuver's geometry is fine,
  /// something is just currently in the way. Added 2026-08-12: this is the
  /// signal PullOverManagerNode uses to distinguish "temporarily blocked by
  /// traffic, brake and wait for it to clear" from "no valid shape exists
  /// for this goal at all" -- those two failure modes look identical from
  /// nullopt alone but call for very different responses (the first should
  /// actively contingency-brake and keep retrying indefinitely; the second
  /// should count toward giving up). Left false (not merely unset) when the
  /// function succeeds, so callers can check it unconditionally.
  ///
  /// If `preferred_hint` is non-null, that *exact* shape+duration is tried
  /// FIRST, as a single candidate, before falling back to the normal fresh
  /// search below -- see ShapeHint's docs for why (cycle-to-cycle stability
  /// of the reference trajectory, not just of the goal). If it's still
  /// feasible and collision-free, it wins immediately, keeping the plan
  /// identical to last cycle's. Only if it's no longer feasible does this
  /// fall through to the full search, which behaves exactly as it did
  /// before this parameter existed.
  ///
  /// If `used_hint` is non-null, it is set to whichever shape+duration the
  /// returned trajectory actually used (whether that came from
  /// `preferred_hint` succeeding again or a fresh search finding something
  /// new) -- callers should feed this back in as next cycle's
  /// `preferred_hint`. Left unmodified if plan() returns std::nullopt.
  ///
  /// If `out_attempted_curvature` is non-null and plan() fails (returns std::nullopt), it is
  /// set to the preferred spiral shape's CurvatureSpiralPath::attemptedPeakCurvature() for this
  /// call -- 0.0 if that shape's Newton solve never converged at all. Lets a caller retrying the
  /// *same* goal across consecutive cycles (e.g. PullOverManagerNode's kOperating loop) detect a
  /// worsening-curvature trend and abandon early, rather than only seeing a pass/fail bit each
  /// cycle -- see that loop's forward-progress-collapse abandon check for the sibling geometric
  /// signal this complements.
  /// `allow_full_stop` gates whether the returned trajectory is allowed to command a genuine
  /// v=0 anywhere at all, including its own terminal point. Default false: every published
  /// sample (including the terminal one) is floored at
  /// `params_.min_departure_reference_speed`, so this candidate can move the vehicle but can
  /// never itself bring it to a real stop. Only pass true once the caller has independently
  /// confirmed, against ego's actual current pose (not a future trajectory point), that this
  /// goal is the genuine, validated final stopping point (see
  /// PullOverManagerNode::isConfirmedSafeToStop's docs) -- e.g. via a heading-alignment and
  /// shoulder-containment check, not merely "close to some goal". Grounded in SAE J3016's
  /// minimal risk condition guidance (examples given are explicitly "stopped outside active
  /// lanes", not stopped within them) and the general collision-risk literature on vehicles
  /// stopped at very low speed while still exposed to moving traffic.
  /// `departure_speed_floor` overrides `params_.min_departure_reference_speed` for this call
  /// when >= 0. It exists because that floor serves two different purposes at two different
  /// phases and they want different values: crossing the travel lane it is an *exposure*
  /// limit (do not dawdle where traffic is), but on final approach it becomes a *precision*
  /// limit -- at 2.5 m/s the lateral controller cannot converge heading over the last few
  /// metres of a curving entry, which live-verified 2026-08-16 left the vehicle parked
  /// 0.13 rad (7.5 deg) off the shoulder tangent, just outside the arrival gate, with a
  /// visibly unnatural skewed stop. A lower floor near the goal buys tracking accuracy while
  /// still clearing stock Autoware's zero-velocity deadlock epsilons (vel_epsilon=1e-3,
  /// stopped_state_entry_vel=0.01) by two orders of magnitude.
  [[nodiscard]] std::optional<autoware_planning_msgs::msg::Trajectory> plan(
    const KinematicState & start, const geometry_msgs::msg::Pose & goal_pose,
    const autoware_perception_msgs::msg::PredictedObjects & objects, const rclcpp::Time & stamp,
    std::string * failure_reason = nullptr, bool * blocked_by_traffic = nullptr,
    const ShapeHint * preferred_hint = nullptr, ShapeHint * used_hint = nullptr,
    double * out_attempted_curvature = nullptr, bool allow_full_stop = false,
    double departure_speed_floor = -1.0) const;

private:
  /// Coefficients of a scalar quintic q(t) = c0 + c1 t + c2 t^2 + c3 t^3 +
  /// c4 t^4 + c5 t^5, plus its derivatives -- used for the single scalar
  /// arc-length profile s(t) along whichever PathShape was chosen (see
  /// class docs).
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

  /// `shape` must be valid() -- caller's responsibility to check before
  /// calling. Samples a scalar arc-length quintic speed profile over
  /// `shape` at `params_.dt` spacing for `duration` seconds. `goal_pose` is
  /// used only for its z (ground height); `shape` is 2D+heading only.
  [[nodiscard]] std::optional<autoware_planning_msgs::msg::Trajectory> buildCandidate(
    const KinematicState & start, const PathShape & shape,
    const geometry_msgs::msg::Pose & goal_pose, double duration, const rclcpp::Time & stamp,
    bool allow_full_stop, double departure_speed_floor) const;

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
