#pragma once

#include "autoware_shoulder_pullover_manager/path_shape.hpp"

#include <geometry_msgs/msg/pose.hpp>

#include <string>
#include <vector>

namespace autoware::shoulder_pullover_manager
{

/// Variable-curvature connector between two 2D poses -- the *preferred*
/// maneuver shape as of 2026-08-05 (see trajectory_planner.hpp's class
/// docs for the full story of why DubinsPath's single global turning
/// radius turned out to be the wrong model for some real goals: it forces
/// a turn tight enough for the *hardest* part of the maneuver everywhere
/// along the whole path, even where a much gentler curve would do, which
/// on at least one live-tested "lateral sidestep" goal produced an absurd
/// 60-100m detour loop at any radius wide enough to stay safe).
///
/// Method: curvature is a cubic polynomial of normalized arc length,
/// kappa(u) = k0 + k1*u + k2*u^2 + k3*u^3 for u in [0,1] (s = u * S) --
/// smooth and continuous by construction, unlike a Dubins path's
/// discontinuous jump at an arc<->straight segment junction (the source of
/// the lateral-jerk problems documented in trajectory_planner.hpp). Start
/// and goal curvature (k0 and kappa(1)) are fixed at 0 -- both boundary
/// poses are treated as "currently going straight relative to their own
/// heading", the same simplifying spirit as this planner's other
/// deliberately-zeroed boundary conditions (see trajectory_planner.hpp).
/// That leaves exactly 3 free unknowns (k1, k2, and total length S) for
/// exactly 3 hard constraints (goal x, goal y, goal heading) -- solved via
/// Newton-Raphson shooting with a numerically (Simpson's-rule) integrated
/// residual and a finite-difference Jacobian, reusing the same 3x3
/// closed-form Cramer's-rule solve already used elsewhere in this package
/// for the same reason (small and fixed in size, not worth an Eigen
/// dependency).
///
/// **This system has a single dominant solution for a given start/goal
/// pair** (verified offline: varying the Newton solver's initial guess for
/// S across a 10x range, and varying (k1,k2) seeds, converged to the same
/// answer every time for every geometry tested) -- so unlike DubinsPath
/// there is no radius/duration-style shape search to run; either the one
/// solution converges and satisfies `max_curvature`, or it doesn't.
///
/// **The solve is damped (backtracking line search), not raw Newton --
/// don't simplify this back out.** A raw full-step Newton iteration is not
/// globally convergent for this residual: a real live-tested goal only
/// ~58 degrees off the chord bearing (nowhere near the most extreme
/// geometry tested) diverged to nonsense (S ranging from 0.2 to 700+
/// depending on initial guess) with an undamped step, yet converged
/// cleanly in ~7 iterations once steps were halved on any iteration that
/// didn't actually shrink the residual. A case that looked at first like
/// "genuine infeasibility caught cleanly" (a near-180-degree apparent
/// heading reversal) turned out to just be the *same* undamped-Newton
/// fragility, not a real structural limit -- it converges fine too once
/// damped, with an unremarkable curvature. Don't assume a convergence
/// failure means the goal is truly unreachable by this shape without
/// first checking whether it's actually a solver robustness gap; on the
/// other hand, don't assume damping makes convergence guaranteed either --
/// `valid()` is still the caller's only signal, and a diverged or
/// numerically-invalid attempt (iteration cap, residual-norm check,
/// curvature check against `max_curvature` passed to the constructor) is
/// caught internally and reported as invalid, never returned as if it were
/// a real path.
class CurvatureSpiralPath : public PathShape
{
public:
  /// `max_curvature` gates validity directly -- a converged solution whose
  /// peak |curvature| exceeds it is reported invalid rather than returned.
  /// Tries goal curvature fixed at 0 first (the common case, and the only
  /// case worth documenting in the class docs above); if that converges
  /// but the cap is exceeded, sweeps goal curvature over a small range as
  /// a second pass and keeps whichever converging value minimizes peak
  /// curvature. **Don't rip this sweep back out as "just closes a small
  /// gap, not worth it"** -- earlier offline testing on a genuinely
  /// near-the-vehicle's-kinematic-limit goal found it only closed a small
  /// fraction of a large gap there, which looked like a good reason to
  /// skip it entirely; a live-tested near-miss goal (0.2745 against the
  /// 0.27 cap, only 1.8% over) later showed the sweep reliably closes
  /// *small* gaps (down to 0.2546 there) even though it can't close large
  /// ones -- both things are true, and the second one is the common case
  /// in practice (see project memory, 2026-08-05 evening entries).
  ///
  /// If `diagnostic` is non-null, it is always set to a short summary of
  /// what happened (converged or not, iteration count, resulting length
  /// and peak curvature if it did converge) -- same "log the actual
  /// reason, don't guess" pattern already used by
  /// PullOverTrajectoryPlanner's own constraint checks. Cheap, and this is
  /// exactly the kind of thing that turned out to matter: whether a
  /// caller sees "spiral failed" or "spiral converged to curvature=0.58,
  /// over the 0.27 cap" is the difference between diagnosing an actual
  /// live failure in seconds versus re-deriving it offline from scratch.
  CurvatureSpiralPath(
    const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal,
    double max_curvature, std::string * diagnostic = nullptr);

  [[nodiscard]] bool valid() const override { return valid_; }
  [[nodiscard]] double length() const override { return length_; }
  [[nodiscard]] PathPoint pointAt(double s) const override;

  /// Peak |curvature| (1/m) actually achieved by the solved path. Only
  /// meaningful if valid() -- exposed (unlike DubinsPath, where it's just
  /// 1/radius and not worth a method) so callers that need to compare
  /// *how hard* a maneuver is, not just whether it's feasible, don't have
  /// to re-derive it (see GoalScorer, which prefers goals with a gentler
  /// required maneuver among otherwise-feasible candidates).
  [[nodiscard]] double peakCurvature() const { return peak_curvature_; }

  /// Peak |curvature| (1/m) the Newton solve actually converged to, even when it exceeds
  /// max_curvature and valid() is therefore false -- 0.0 if the solve never converged at all
  /// (see give_up_reason in that case; there is no curvature estimate to report). Added so a
  /// caller retrying against the same goal across consecutive receding-horizon cycles (e.g.
  /// PullOverManagerNode's kOperating loop) can track *how much* an infeasible attempt is
  /// missing by and detect a worsening trend, instead of only getting a pass/fail bit each
  /// cycle -- see PullOverManagerNode's min_maneuver_forward_progress-adjacent abandon logic.
  [[nodiscard]] double attemptedPeakCurvature() const { return attempted_peak_curvature_; }

  /// d(curvature)/d(arc length) at s=0 (1/m^2), signed. Only meaningful if
  /// valid(). Exposed for exactly one reason: curvature is forced to
  /// *exactly* 0 at s=0 (see class docs), so it always ramps up from
  /// there, and near a fixed start speed v0 (a0=0 boundary condition, see
  /// PullOverTrajectoryPlanner) the resulting lateral jerk right at the
  /// maneuver's start is approximately v0^3 * initialCurvatureRate() --
  /// live-tested finding (2026-08-05): this can violate a real jerk limit
  /// even when peakCurvature() is comfortably under the curvature cap,
  /// specifically at real driving speeds (v0 a few m/s), because jerk
  /// scales with speed *cubed*. Neither raising the goal-curvature sweep
  /// range nor raising duration fixes this -- verified offline, duration
  /// barely changes speed this close to t=0 given the fixed v0/a0=0 start
  /// boundary, and the sweep trades curvature against jerk without a
  /// combination that satisfies both for at least one live-tested case.
  /// This exists so GoalScorer can reject (or penalize) goals that would
  /// violate a real vehicle's steering-rate limit at ego's *current*
  /// speed, before PullOverTrajectoryPlanner ever tries and fails on them
  /// -- the same rationale as peakCurvature(), just for the
  /// speed-dependent half of the problem.
  [[nodiscard]] double initialCurvatureRate() const { return initial_curvature_rate_; }

private:
  bool valid_{false};
  double length_{0.0};
  double peak_curvature_{0.0};
  double attempted_peak_curvature_{0.0};
  double initial_curvature_rate_{0.0};

  /// Dense, evenly-spaced-in-arc-length sample table built once at
  /// construction (trapezoidal integration, fine-grained) -- pointAt()
  /// linearly interpolates between the two nearest entries. Built because,
  /// unlike DubinsPath's circular arcs, this curve has no closed form for
  /// position given arc length.
  struct Sample
  {
    double x, y, yaw;
  };
  std::vector<Sample> samples_;
};

}  // namespace autoware::shoulder_pullover_manager
