#include "autoware_shoulder_pullover_manager/curvature_spiral_path.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

namespace autoware::shoulder_pullover_manager
{

namespace
{
double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double wrapPi(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

/// Solves A*x=b for a 3x3 system via Cramer's rule -- same rationale as
/// trajectory_planner.cpp's solve3x3 (small and fixed in size, not worth
/// an Eigen dependency); duplicated locally rather than shared since it's
/// a dozen lines and this package has no existing shared-utility header.
bool solve3x3(const double A[3][3], const double b[3], double x[3])
{
  const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                      A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                      A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  if (std::abs(det) < 1e-12) {
    return false;
  }
  auto detWithColumn = [&](int col) {
    double M[3][3];
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        M[r][c] = (c == col) ? b[r] : A[r][c];
      }
    }
    return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
           M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  };
  x[0] = detWithColumn(0) / det;
  x[1] = detWithColumn(1) / det;
  x[2] = detWithColumn(2) / det;
  return true;
}

/// theta(u) = theta0 + S*(k0*u + k1*u^2/2 + k2*u^3/3 + k3*u^4/4).
double thetaAt(double theta0, double k0, double k1, double k2, double k3, double S, double u)
{
  return theta0 + S * (k0 * u + k1 * u * u / 2.0 + k2 * u * u * u / 3.0 +
                        k3 * u * u * u * u / 4.0);
}

struct EndpointResult
{
  double x, y, yaw;
};

/// Simpson's-rule integration of the spiral from u=0 to u=1, i.e. the full
/// path -- used only inside the Newton solve's residual/Jacobian
/// evaluation (n_steps=40 is far more than that shooting loop needs;
/// pointAt()'s dense table is built separately, see buildSampleTable).
EndpointResult integrateEndpoint(
  double x0, double y0, double theta0, double k0, double k1, double k2, double k3, double S,
  int n_steps)
{
  const double h = 1.0 / n_steps;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double last_theta = theta0;
  for (int i = 0; i <= n_steps; ++i) {
    const double u = i * h;
    const double theta = thetaAt(theta0, k0, k1, k2, k3, S, u);
    const double w = (i == 0 || i == n_steps) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
    sum_x += w * std::cos(theta);
    sum_y += w * std::sin(theta);
    last_theta = theta;
  }
  const double dx = S * (h / 3.0) * sum_x;
  const double dy = S * (h / 3.0) * sum_y;
  return {x0 + dx, y0 + dy, last_theta};
}

constexpr double kK0 = 0.0;  // Start curvature fixed at 0 -- see class docs.
constexpr int kResidualIntegrationSteps = 40;  // must stay even (Simpson's rule).
constexpr int kMaxNewtonIterations = 50;
constexpr double kResidualTolerance = 1e-6;

struct SolveResult
{
  bool converged{false};
  double k1{0.0}, k2{0.0}, k3{0.0}, S{0.0};
  double max_curvature_found{0.0};
  int iterations_used{0};
  const char * give_up_reason{"did not converge within iteration cap"};
};

/// Damped Newton shooting for a *given, fixed* goal curvature -- see class
/// docs for why goal curvature itself is swept by the caller rather than
/// baked in here as always 0.
SolveResult solveForGoalCurvature(
  double x0, double y0, double theta0, double x1, double y1, double theta1,
  double goal_curvature, double S_init)
{
  SolveResult result;
  double params[3] = {0.0, 0.0, S_init};

  auto residual = [&](const double p[3]) {
    const double k1 = p[0];
    const double k2 = p[1];
    const double S = p[2];
    const double k3 = goal_curvature - kK0 - k1 - k2;
    const EndpointResult end =
      integrateEndpoint(x0, y0, theta0, kK0, k1, k2, k3, S, kResidualIntegrationSteps);
    return std::array<double, 3>{end.x - x1, end.y - y1, wrapPi(end.yaw - theta1)};
  };

  for (int iter = 0; iter < kMaxNewtonIterations; ++iter) {
    result.iterations_used = iter;
    const auto F = residual(params);
    const double norm = std::sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
    if (norm < kResidualTolerance) {
      result.converged = true;
      break;
    }

    double J[3][3];
    for (int j = 0; j < 3; ++j) {
      double p2[3] = {params[0], params[1], params[2]};
      const double step = 1e-6 * std::max(1.0, std::abs(params[j]));
      p2[j] += step;
      const auto F2 = residual(p2);
      for (int r = 0; r < 3; ++r) {
        J[r][j] = (F2[r] - F[r]) / step;
      }
    }

    const double negF[3] = {-F[0], -F[1], -F[2]};
    double delta[3];
    if (!solve3x3(J, negF, delta)) {
      result.give_up_reason = "singular Jacobian";
      break;
    }

    // Backtracking line search: an undamped full Newton step is not
    // globally convergent for this residual (verified offline -- a real
    // live-tested goal only ~58 degrees off the chord bearing, nowhere
    // near the most extreme case tested, diverged to a nonsense S ranging
    // from 0.2 to 700+ depending on initial guess with a raw full step).
    // Halving the step until the residual norm actually decreases is a
    // standard, simple fix and was sufficient in every offline test case,
    // including ones the undamped version could not solve at all.
    bool accepted = false;
    double step_scale = 1.0;
    constexpr int kMaxBacktracks = 20;
    for (int bt = 0; bt < kMaxBacktracks; ++bt) {
      double candidate[3] = {
        params[0] + step_scale * delta[0], params[1] + step_scale * delta[1],
        params[2] + step_scale * delta[2]};
      if (candidate[2] <= 0.05) {
        step_scale *= 0.5;
        continue;
      }
      const auto F_candidate = residual(candidate);
      const double candidate_norm = std::sqrt(
        F_candidate[0] * F_candidate[0] + F_candidate[1] * F_candidate[1] +
        F_candidate[2] * F_candidate[2]);
      if (candidate_norm < norm) {
        params[0] = candidate[0];
        params[1] = candidate[1];
        params[2] = candidate[2];
        accepted = true;
        break;
      }
      step_scale *= 0.5;
    }
    if (!accepted) {
      result.give_up_reason = "no damped step improves the residual";
      break;
    }
  }

  if (!result.converged) {
    return result;
  }
  result.k1 = params[0];
  result.k2 = params[1];
  result.S = params[2];
  result.k3 = goal_curvature - kK0 - result.k1 - result.k2;
  if (result.S <= 0.0) {
    result.converged = false;
    result.give_up_reason = "converged but S<=0 (nonsense solution)";
    return result;
  }

  // Peak curvature is a cubic in u -- dense sampling is simple and cheap
  // enough (this runs a handful of times per candidate shape, not per
  // trajectory sample).
  constexpr int kCurvatureSamples = 100;
  for (int i = 0; i <= kCurvatureSamples; ++i) {
    const double u = static_cast<double>(i) / kCurvatureSamples;
    const double kappa = kK0 + result.k1 * u + result.k2 * u * u + result.k3 * u * u * u;
    result.max_curvature_found = std::max(result.max_curvature_found, std::abs(kappa));
  }
  return result;
}
}  // namespace

CurvatureSpiralPath::CurvatureSpiralPath(
  const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal,
  double max_curvature, std::string * diagnostic)
{
  const double x0 = start.position.x;
  const double y0 = start.position.y;
  const double theta0 = yawFromQuaternion(start.orientation);
  const double x1 = goal.position.x;
  const double y1 = goal.position.y;
  const double theta1 = yawFromQuaternion(goal.orientation);

  const double dist = std::hypot(x1 - x0, y1 - y0);
  if (dist < 1e-6) {
    valid_ = false;
    if (diagnostic) {
      *diagnostic = "spiral: start and goal positions coincide";
    }
    return;
  }
  const double S_init = 1.3 * dist + 0.1;

  // Fast path: goal curvature fixed at 0 (the common, canonical case).
  SolveResult best = solveForGoalCurvature(x0, y0, theta0, x1, y1, theta1, 0.0, S_init);
  double best_goal_curvature = 0.0;

  // Second pass, only if the fast path didn't produce a usable result:
  // sweep goal curvature over a small range and keep whichever converging
  // value minimizes peak curvature -- see class docs for why this is
  // worth the extra solves (closes small near-miss gaps reliably, even
  // though it can't close large ones).
  if (!best.converged || best.max_curvature_found > max_curvature) {
    constexpr double kSweepRange = 0.3;
    constexpr double kSweepStep = 0.02;
    for (double kg = -kSweepRange; kg <= kSweepRange; kg += kSweepStep) {
      if (kg == 0.0) continue;  // already tried above.
      const SolveResult candidate =
        solveForGoalCurvature(x0, y0, theta0, x1, y1, theta1, kg, S_init);
      if (!candidate.converged) continue;
      const bool best_is_usable = best.converged && best.max_curvature_found <= max_curvature;
      if (best_is_usable && candidate.max_curvature_found >= best.max_curvature_found) {
        continue;  // Already have something at least as good.
      }
      if (candidate.max_curvature_found < best.max_curvature_found || !best.converged) {
        best = candidate;
        best_goal_curvature = kg;
      }
    }
  }

  if (!best.converged) {
    valid_ = false;
    if (diagnostic) {
      std::ostringstream oss;
      oss << "spiral: " << best.give_up_reason << " after " << best.iterations_used
          << " iterations (including goal-curvature sweep)";
      *diagnostic = oss.str();
    }
    return;
  }
  // Record the converged attempt's peak curvature even when it exceeds max_curvature below --
  // see attemptedPeakCurvature()'s docs for why this is exposed despite valid_ becoming false.
  attempted_peak_curvature_ = best.max_curvature_found;

  if (best.max_curvature_found > max_curvature) {
    valid_ = false;
    if (diagnostic) {
      std::ostringstream oss;
      oss << "spiral: converged (S=" << best.S << "m, best goal_curvature=" << best_goal_curvature
          << ") but peak curvature " << best.max_curvature_found << " exceeds max "
          << max_curvature << " even after sweeping goal curvature";
      *diagnostic = oss.str();
    }
    return;
  }

  const double k1 = best.k1;
  const double k2 = best.k2;
  const double k3 = best.k3;
  const double S = best.S;

  // Build the dense sample table for pointAt() -- trapezoidal integration,
  // fine-grained (400 steps) since there's no closed form for position
  // given arc length on this curve.
  constexpr int kTableSteps = 400;
  samples_.reserve(kTableSteps + 1);
  double x = x0;
  double y = y0;
  double prev_theta = theta0;
  samples_.push_back({x, y, prev_theta});
  const double du = 1.0 / kTableSteps;
  for (int i = 1; i <= kTableSteps; ++i) {
    const double u = i * du;
    const double theta = thetaAt(theta0, kK0, k1, k2, k3, S, u);
    x += S * du * 0.5 * (std::cos(prev_theta) + std::cos(theta));
    y += S * du * 0.5 * (std::sin(prev_theta) + std::sin(theta));
    samples_.push_back({x, y, theta});
    prev_theta = theta;
  }

  length_ = S;
  peak_curvature_ = best.max_curvature_found;
  // kappa(u) = k0 + k1*u + k2*u^2 + k3*u^3 -- d(kappa)/du at u=0 is k1;
  // d(kappa)/ds = d(kappa)/du * du/ds = k1 * (1/S).
  initial_curvature_rate_ = k1 / S;
  valid_ = true;
  if (diagnostic) {
    std::ostringstream oss;
    oss << "spiral: converged (S=" << S << "m, goal_curvature=" << best_goal_curvature
        << ", " << best.iterations_used << " iterations, peak curvature="
        << best.max_curvature_found << ")";
    *diagnostic = oss.str();
  }
}

PathPoint CurvatureSpiralPath::pointAt(double s) const
{
  s = std::clamp(s, 0.0, length_);
  const double u = (length_ > 1e-9) ? (s / length_) : 0.0;
  const double idx_f = u * static_cast<double>(samples_.size() - 1);
  const std::size_t idx0 =
    std::min(static_cast<std::size_t>(std::floor(idx_f)), samples_.size() - 2);
  const std::size_t idx1 = idx0 + 1;
  const double frac = idx_f - static_cast<double>(idx0);

  const Sample & a = samples_[idx0];
  const Sample & b = samples_[idx1];
  return {
    a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac, a.yaw + (b.yaw - a.yaw) * frac};
}

}  // namespace autoware::shoulder_pullover_manager
