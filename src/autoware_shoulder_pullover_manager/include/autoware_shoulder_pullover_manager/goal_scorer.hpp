#pragma once

#include "autoware_shoulder_pullover_manager/curvature_spiral_path.hpp"

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/path.hpp>

#include <optional>
#include <vector>

namespace autoware::shoulder_pullover_manager
{

/// Tunable parameters for candidate shoulder-goal scoring.
///
/// This is a deliberately *classical* (point-estimate, no probability)
/// implementation of the goal-scoring formula from the project's pull-over
/// MRM framework report. The report's "lateral clearance" term, originally
/// omitted because no topic exposed shoulder width, was added 2026-08-16 (see
/// vehicle_half_width/goal_width_margin below) once shoulder_centerline_node
/// began publishing its measured per-point half-width (halfwidth_pub_, added
/// 2026-08-15). The "maturity" (per-point sample-confidence) term remains
/// intentionally omitted rather than faked -- still no topic exposes it.
///
/// **Added 2026-08-05: maneuver feasibility from ego's actual current
/// pose.** The original `curvature` term below only measures the shoulder
/// centerline's own local shape at the candidate point -- it says nothing
/// about whether *reaching* that point from ego's current position and
/// heading is itself a reasonable maneuver. Live testing found this was
/// the dominant real-world failure mode: GoalScorer kept selecting
/// near-top-scored candidates (0.99+) that PullOverTrajectoryPlanner then
/// correctly refused, over and over, because they required a lateral
/// "sidestep" maneuver at or beyond the vehicle's kinematic limit -- not a
/// planner bug, a goal-selection blind spot. `selectBestGoal` now
/// constructs a `CurvatureSpiralPath` from ego to each surviving candidate
/// (the same shape generator PullOverTrajectoryPlanner itself prefers, see
/// trajectory_planner.hpp) and uses it both as a hard feasibility gate and
/// a soft preference for gentler maneuvers -- see max_maneuver_curvature
/// and weight_maneuver_ease below.
struct GoalScorerParams
{
  /// Candidates farther than this (straight-line, meters) from the current
  /// ego position are not considered -- keeps the search local to what the
  /// vehicle can actually reach soon, and avoids picking a stale waypoint
  /// from a completely different part of the accumulated map-frame trail.
  double max_lookahead_distance{60.0};

  /// Minimum forward-projected distance (meters, along ego's *current*
  /// heading) a candidate must offer -- **not** the same as the
  /// straight-line `distance` used for max_lookahead_distance above; this
  /// is that same vector's component specifically along ego's forward
  /// direction. Live-tested root cause (2026-08-10): a candidate can have a
  /// perfectly reasonable straight-line distance (e.g. 6.6m) while being
  /// almost entirely *lateral* offset with only a sliver of forward room
  /// (e.g. 0.2-0.4m) -- CurvatureSpiralPath's required curvature blows up
  /// as forward distance shrinks toward zero for a fixed lateral offset (a
  /// genuine geometric near-singularity, not a solver bug), so such a goal
  /// can read as comfortably feasible (e.g. curvature 0.25, under the 0.27
  /// cap) at the exact instant GoalScorer checks it, then become wildly
  /// infeasible (curvature 0.7+) just 1-3 planning cycles later once ego
  /// has moved even slightly -- because `PullOverTrajectoryPlanner::plan()`
  /// re-derives the maneuver shape from ego's *current* pose every single
  /// cycle, but nothing ever re-validates that a goal fixed into
  /// `active_goal_` stays reachable as that pose evolves. Rejecting
  /// near-zero-forward-progress candidates outright (rather than trying to
  /// out-guess the singularity with a curvature safety margin, which would
  /// still be fragile right up to wherever the new margin's edge is) closes
  /// this at the source: a goal needing 6.5m of lateral movement in under 2m
  /// of forward room is a fragile "almost pure sidestep" no matter how
  /// gently it scores at any one instant.
  double min_maneuver_forward_progress{2.0};

  /// Comfortable deceleration (m/s^2) used to compute the minimum feasible
  /// stopping distance at the vehicle's current speed. Matches the order of
  /// magnitude Autoware's own comfortable_stop MRM behavior targets.
  double comfortable_deceleration{2.0};

  /// Extra fixed margin (meters) added on top of the physical stopping
  /// distance, so the maneuver has room to actually execute the lane
  /// change(s) and shoulder entry, not just brake to a halt exactly at the
  /// goal.
  double stopping_margin{5.0};

  /// A run of consecutive centerline waypoints ahead of a candidate is only
  /// considered "continuous" while consecutive spacing stays below this
  /// (meters). A gap larger than this usually means the accumulated trail's
  /// ragged growing tip, or a genuine break in shoulder detection -- see
  /// project memory on why the tip of the accumulated trail is the noisiest
  /// part.
  double max_continuity_gap{3.0};

  /// Hard minimum continuous run length (meters) ahead of a candidate for it
  /// to be considered at all. Below this, there is not demonstrably enough
  /// confirmed shoulder ahead to actually park in.
  double min_continuity_length{8.0};

  /// Run length (meters) at or above which the continuity score term
  /// saturates to 1.0.
  double reference_run_length{20.0};

  /// Local centerline curvature (1/m) at or above which the curvature
  /// penalty term saturates to 1.0. Chosen well below planning_validator's
  /// hard curvature limit so scoring prefers gentle entry points long before
  /// validator rejection would ever become a concern.
  double max_reference_curvature{0.3};

  /// Score weight for the continuity/run-length term.
  double weight_continuity{0.4};

  /// Score weight for the (inverted) centerline-shape curvature term.
  double weight_curvature{0.2};

  /// Score weight for the (inverted) required-maneuver-curvature term --
  /// see the class docs above. Weighted comparably to (here, more than)
  /// the other two terms since live testing found maneuver feasibility to
  /// be the dominant practical concern, not a minor tiebreaker.
  double weight_maneuver_ease{0.2};

  /// Score weight for the (inverted) required-initial-jerk term -- see
  /// max_maneuver_jerk below. A *separate* term from weight_maneuver_ease
  /// on purpose: a goal can have a perfectly gentle peak curvature yet
  /// still demand a steep initial curvature ramp that only becomes a
  /// problem at real driving speed (jerk scales with speed cubed) --
  /// live-tested, see project memory. One is a geometry property, the
  /// other is a geometry-times-current-speed property; conflating them
  /// into one term would hide exactly the failure mode this was added
  /// for.
  double weight_jerk_ease{0.2};

  /// Hard feasibility gate (1/m), matching
  /// TrajectoryPlannerParams::max_curvature in trajectory_planner.hpp --
  /// **keep these two in sync**, they're both derived from the same real
  /// vehicle steering kinematics (wheel_base/max_steer_angle), not
  /// independently tunable. Duplicated here rather than shared via a
  /// single param because GoalScorer and PullOverTrajectoryPlanner are
  /// deliberately separate, independently-testable components (see class
  /// docs) each populated by pull_over_manager_node.cpp -- if the vehicle
  /// model ever changes, both this and trajectory_planner's max_curvature
  /// need updating together.
  double max_maneuver_curvature{0.27};

  /// Hard feasibility gate (m/s^3) on *estimated initial* lateral jerk
  /// (ego_speed_mps^3 * CurvatureSpiralPath::initialCurvatureRate(), see
  /// that method's docs), matching
  /// TrajectoryPlannerParams::max_lateral_jerk -- keep these in sync for
  /// the same reason as max_maneuver_curvature above. This is an
  /// *approximation* (assumes speed is still close to ego_speed_mps at
  /// the maneuver's very start, true given the a0=0 boundary condition
  /// PullOverTrajectoryPlanner uses -- not an exact replica of its
  /// discretized dt-sampled check), good enough to screen out a bad
  /// candidate before ever attempting it, not meant to replace that
  /// check.
  double max_maneuver_jerk{3.0};

  /// --- Measured-shoulder-width gate + preference, added 2026-08-16 ---
  /// Live-verified root cause (bag `pullover_widthdata_1786901356`, 2026-08-16, plus
  /// fullfix2/3 the day before): every goal this scorer committed sat on a stretch whose
  /// *measured* half-width (shoulder_centerline_node's halfwidth topic) was 0.41-1.20m,
  /// under the ~1.25m the vehicle's own footprint needs -- the vehicle turned onto a
  /// ~0.8m-wide paved strip and physically wedged against the shoulder-edge barrier at
  /// 25% throttle for minutes. Width data existed the whole time; only the *arrival*
  /// containment gate consumed it, never selection. This is the same footprint-in-corridor
  /// test stock Autoware's goal_planner already runs at goal-search time
  /// (goal_searcher.cpp: vehicle footprint + margin_from_boundary=0.75 must fit within the
  /// pull-over lane polygon) -- applied here against the perception-measured corridor
  /// instead of HD-map lane polygons, since this project's map has no shoulder lanelets at
  /// all (the founding constraint, see PullOverManagerNode's class docs).
  ///
  /// Same value as PullOverManagerNode::vehicle_half_width_ (0.5*wheel_tread +
  /// max(left_overhang, right_overhang)) -- populated by the owning node from the same
  /// vehicle_info params, duplicated per this file's existing convention for
  /// max_maneuver_curvature (independently-testable components, owning node keeps them
  /// consistent).
  double vehicle_half_width{0.948};
  /// m. Extra clearance beyond vehicle_half_width required of the (scaled, windowed)
  /// measured half-width before a candidate is eligible at all. Same spirit as
  /// goal_planner's margin_from_boundary (0.75 stock) but smaller: that margin is against
  /// exact HD-map boundaries, while ours is against a measurement that is already
  /// conservative by construction (see width_measurement_scale) *and* gated by a windowed
  /// minimum (see width_window_half_length), both of which under- rather than over-state
  /// the usable corridor.
  double goal_width_margin{0.3};
  /// Multiplier applied to the published half-width before gating. 1.0 (identity) since
  /// 2026-08-16, when shoulder_centerline_node began measuring half-width *directly* --
  /// back-projecting the shoulder mask run's own left/right edge pixels to the ground
  /// plane, the same IPM the centerline position itself uses -- so the published value is
  /// already the physical half-width and needs no correction.
  ///
  /// A 1.25 factor briefly lived here to invert the old trimmed-extent estimator's
  /// (1-2*trim) bias. That was the wrong fix for the wrong model: live data showed the old
  /// value wasn't a scaled-down width at all but the midpoint estimator's jitter (a
  /// near-constant ~0.40m regardless of a shoulder whose CARLA ground truth is 1.75m
  /// half-width), which no scalar can correct. Kept as a parameter rather than deleted so a
  /// future sensor/estimator with a known, characterized bias can be corrected here without
  /// touching the gate logic -- but do not reintroduce a nonzero correction without
  /// evidence the underlying measurement is genuinely proportional to truth.
  double width_measurement_scale{1.0};
  /// m. Half-length of the arclength window, centered on the candidate, over which the
  /// MINIMUM measured half-width is taken for the gate -- the parked vehicle occupies
  /// roughly its own length of shoulder, not one centerline point, so a single wide bin
  /// inside a narrow stretch must not pass (and, symmetrically, via the windowed minimum
  /// a single dropout bin -- observed live: an isolated 0.01m reading amid 0.4m
  /// neighbors -- inside a genuinely wide stretch WILL conservatively fail it; that
  /// costs a candidate, never safety). Default ~half the sample_vehicle's ~4.8m length.
  double width_window_half_length{2.5};
  /// Score weight for preferring wider (scaled, windowed-min) shoulder beyond the bare
  /// gate -- pushes goals toward the deepest/widest part of a pocket rather than its
  /// marginal entrance, complementing the hard gate the same way weight_maneuver_ease
  /// complements max_maneuver_curvature.
  double weight_width{0.3};
  /// m. Extra half-width (beyond the gate's requirement) at which the width score term
  /// saturates to 1.0.
  double reference_extra_half_width{0.5};
  /// If true (default), candidates are ineligible when no measured width data is
  /// available at all (empty/mismatched halfwidth array) -- fail closed, consistent with
  /// PullOverManagerNode::require_shoulder_containment_'s arrival-side behavior. The
  /// owning node passes an empty vector to selectBestGoal when its subscription has
  /// nothing usable; set false only for setups intentionally not publishing width.
  bool require_width_data{true};

  /// How far (meters, each direction along the centerline) to look when
  /// computing the candidate goal's *heading* via a length-weighted average
  /// of local tangent segments, instead of trusting that single waypoint's
  /// own orientation field directly. Added 2026-08-12: shoulder_centerline_node
  /// publishes per-point headings derived from live PointPainting fusion +
  /// spline fitting, which carries real point-to-point jitter -- using one
  /// noisy sample directly as the trajectory's terminal heading produced a
  /// final parked pose visibly skewed from parallel-to-shoulder, not a
  /// planner bug but a goal-definition one. Averaging tangent segments
  /// across a short window (bounded by max_continuity_gap_ same as
  /// continuityRunLength, so it never reaches across a real discontinuity)
  /// gives a heading that reflects the shoulder's actual local direction
  /// instead of one sample's noise.
  double heading_smoothing_distance{3.0};

  /// Live-verified bug, 2026-08-13: candidates farther than this were selected and then failed
  /// PullOverTrajectoryPlanner::plan() every single cycle up to max_consecutive_planning_failures_,
  /// not from a search gap but a hard mathematical ceiling -- a minimum-jerk quintic position
  /// profile connecting near-zero boundary speeds at both ends has PEAK velocity approximately
  /// (15/8) * (distance/duration) (standard result for a 5th-order polynomial with zero
  /// velocity+acceleration at both endpoints). Bag-confirmed live: a 20.2m goal, searched up to
  /// trajectory_duration's own cap (15s), peaked at ~2.0-2.03 m/s against a 2.0 m/s
  /// trajectory_max_speed ceiling -- no duration up to that cap can ever bring a candidate this
  /// far under the speed ceiling, so PullOverTrajectoryPlanner::plan()'s whole duration search was
  /// guaranteed to fail before ever being tried. Must be populated by the owning node from
  /// PullOverTrajectoryPlanner's own params (max_speed, max_duration) so the two stay physically
  /// consistent -- see pull_over_manager_node.cpp's constructor. Defaults to a value consistent
  /// with this class's own stock defaults (2.0 m/s, 15s) with a ~10% safety margin below the
  /// exact analytic ceiling, since the real check is a discretized per-sample one, not the exact
  /// continuous peak.
  double max_reachable_distance{14.4};
};

/// A single scored candidate shoulder-goal, with the raw measurements kept
/// alongside the final score for logging/diagnostics.
struct ScoredCandidate
{
  geometry_msgs::msg::Pose pose;
  double score{0.0};
  double distance_from_ego{0.0};
  double curvature{0.0};            ///< Shoulder centerline's own local shape at this point.
  double continuity_length{0.0};
  double maneuver_curvature{0.0};  ///< Peak curvature of the ego->candidate CurvatureSpiralPath.
  double maneuver_initial_jerk{0.0};  ///< Estimated initial lateral jerk at ego's current speed.
  double measured_half_width{0.0};  ///< Scaled, windowed-min half-width (m) the width gate saw
                                     ///< at this candidate -- 0.0 if width gating was inactive
                                     ///< (no data + require_width_data false). For logging.
};

/// Selects the best feasible shoulder pull-over goal from the live,
/// map-frame shoulder centerline, given the current ego pose and speed.
///
/// Pure logic, no ROS node/publisher/subscriber state -- deliberately kept
/// independent of rclcpp so the selection algorithm itself can be reasoned
/// about (and unit-tested) without a running ROS graph.
class GoalScorer
{
public:
  explicit GoalScorer(const GoalScorerParams & params);

  /// Returns the highest-scoring feasible candidate, or std::nullopt if no
  /// waypoint in `centerline_map` satisfies the hard feasibility gates
  /// (ahead of ego, within lookahead range, past the minimum stopping
  /// distance, with enough confirmed continuity ahead, wide enough for the
  /// vehicle's actual footprint per the measured width gate -- see
  /// vehicle_half_width's docs -- and, see class docs, reachable from ego's
  /// current pose via a CurvatureSpiralPath whose peak curvature stays
  /// under max_maneuver_curvature). The spiral gate is checked only for
  /// candidates that already pass the cheaper ones, since it costs a full
  /// Newton solve (occasionally several, if the goal-curvature sweep kicks
  /// in -- see CurvatureSpiralPath's class docs) per candidate.
  ///
  /// `half_widths` is shoulder_centerline_node's measured half-width array,
  /// index-aligned with centerline_map.poses (see that node's halfwidth_pub_
  /// docs). Pass an empty vector when no usable (present, size-matched) data
  /// exists -- candidates then fail the width gate outright when
  /// require_width_data is true (the default), or skip it when false.
  [[nodiscard]] std::optional<ScoredCandidate> selectBestGoal(
    const nav_msgs::msg::Path & centerline_map, const geometry_msgs::msg::Pose & ego_pose,
    double ego_speed_mps, const std::vector<float> & half_widths = {}) const;

private:
  /// Local curvature at `index` via the standard three-point (Menger)
  /// curvature formula over centerline_map.poses[index-1, index, index+1].
  /// Returns 0.0 at the path endpoints, where no such triplet exists.
  [[nodiscard]] static double estimateCurvature(
    const nav_msgs::msg::Path & path, std::size_t index);

  /// Arclength of the contiguous run of waypoints starting at `index`,
  /// walking forward while consecutive spacing stays within
  /// max_continuity_gap_.
  [[nodiscard]] double continuityRunLength(const nav_msgs::msg::Path & path, std::size_t index)
    const;

  /// Minimum raw measured half-width within width_window_half_length (arclength, each
  /// direction, stopping at a gap larger than max_continuity_gap -- same walk rule as
  /// continuityRunLength/smoothedHeading) of `index`. Windowed because the parked
  /// vehicle occupies its own length of shoulder, not a point -- see
  /// width_window_half_length's docs. Returns the raw (unscaled) minimum; the caller
  /// applies width_measurement_scale.
  [[nodiscard]] double windowedMinHalfWidth(
    const nav_msgs::msg::Path & path, const std::vector<float> & half_widths,
    std::size_t index) const;

  /// Length-weighted average tangent direction of the centerline segments
  /// within heading_smoothing_distance_ of `index` in either direction
  /// (stopping early at a gap larger than max_continuity_gap_, same rule
  /// continuityRunLength uses) -- see heading_smoothing_distance's docs for
  /// why this replaces trusting that one waypoint's own orientation field.
  /// Falls back to that raw field only if no usable neighbor segment exists
  /// (e.g. `index` is the path's only point).
  [[nodiscard]] geometry_msgs::msg::Quaternion smoothedHeading(
    const nav_msgs::msg::Path & path, std::size_t index) const;

  GoalScorerParams params_;
};

}  // namespace autoware::shoulder_pullover_manager
