#include "autoware_shoulder_pullover_manager/dubins_path.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

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

double mod2pi(double theta)
{
  double v = std::fmod(theta, 2.0 * M_PI);
  if (v < 0.0) {
    v += 2.0 * M_PI;
  }
  return v;
}

/// Normalized (radius=1) segment lengths for each Dubins CSC family --
/// standard reduced-Dubins formulas (Shkel & Lumelsky 2001 / LaValle
/// "Planning Algorithms" ch.15). Verified numerically offline (endpoint
/// reconstruction, multiple start/goal/heading combinations including this
/// project's live-tested 90-degree-turn goal) before porting here -- do not
/// hand-edit these without re-verifying the same way, sign errors here are
/// easy to make and would silently produce a wrong-shaped path.
struct Lengths
{
  double t, p, q;
};

std::optional<Lengths> dubinsLsl(double alpha, double beta, double d)
{
  const double p_sq =
    2 + d * d - 2 * std::cos(alpha - beta) + 2 * d * (std::sin(alpha) - std::sin(beta));
  if (p_sq < 0) return std::nullopt;
  const double tmp1 =
    std::atan2(std::cos(beta) - std::cos(alpha), d + std::sin(alpha) - std::sin(beta));
  return Lengths{mod2pi(-alpha + tmp1), std::sqrt(p_sq), mod2pi(beta - tmp1)};
}

std::optional<Lengths> dubinsRsr(double alpha, double beta, double d)
{
  const double p_sq =
    2 + d * d - 2 * std::cos(alpha - beta) + 2 * d * (std::sin(beta) - std::sin(alpha));
  if (p_sq < 0) return std::nullopt;
  const double tmp1 =
    std::atan2(std::cos(alpha) - std::cos(beta), d - std::sin(alpha) + std::sin(beta));
  return Lengths{mod2pi(alpha - tmp1), std::sqrt(p_sq), mod2pi(-beta + tmp1)};
}

std::optional<Lengths> dubinsLsr(double alpha, double beta, double d)
{
  const double p_sq =
    -2 + d * d + 2 * std::cos(alpha - beta) + 2 * d * (std::sin(alpha) + std::sin(beta));
  if (p_sq < 0) return std::nullopt;
  const double p = std::sqrt(p_sq);
  const double tmp2 = std::atan2(-std::cos(alpha) - std::cos(beta), d + std::sin(alpha) + std::sin(beta)) -
                       std::atan2(-2.0, p);
  return Lengths{mod2pi(-alpha + tmp2), p, mod2pi(-mod2pi(beta) + tmp2)};
}

std::optional<Lengths> dubinsRsl(double alpha, double beta, double d)
{
  const double p_sq =
    d * d - 2 + 2 * std::cos(alpha - beta) - 2 * d * (std::sin(alpha) + std::sin(beta));
  if (p_sq < 0) return std::nullopt;
  const double p = std::sqrt(p_sq);
  const double tmp2 = std::atan2(std::cos(alpha) + std::cos(beta), d - std::sin(alpha) - std::sin(beta)) -
                       std::atan2(2.0, p);
  return Lengths{mod2pi(alpha - tmp2), p, mod2pi(beta - tmp2)};
}
}  // namespace

DubinsPath::DubinsPath(
  const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal, double radius)
: radius_(radius),
  start_x_(start.position.x),
  start_y_(start.position.y),
  start_yaw_(yawFromQuaternion(start.orientation))
{
  const double goal_yaw = yawFromQuaternion(goal.orientation);
  const double dx = goal.position.x - start_x_;
  const double dy = goal.position.y - start_y_;
  const double dist = std::hypot(dx, dy);
  if (dist < 1e-6 || radius <= 1e-6) {
    valid_ = false;
    return;
  }

  const double d = dist / radius_;
  const double theta = std::atan2(dy, dx);
  const double alpha = mod2pi(start_yaw_ - theta);
  const double beta = mod2pi(goal_yaw - theta);

  struct Candidate
  {
    const char * mode;
    std::optional<Lengths> lengths;
  };
  const std::array<Candidate, 4> candidates{
    Candidate{"LSL", dubinsLsl(alpha, beta, d)}, Candidate{"RSR", dubinsRsr(alpha, beta, d)},
    Candidate{"LSR", dubinsLsr(alpha, beta, d)}, Candidate{"RSL", dubinsRsl(alpha, beta, d)}};

  double best_total = std::numeric_limits<double>::infinity();
  for (const auto & c : candidates) {
    if (!c.lengths) continue;
    const double total = c.lengths->t + c.lengths->p + c.lengths->q;
    if (total < best_total) {
      best_total = total;
      mode_[0] = c.mode[0];
      mode_[1] = c.mode[1];
      mode_[2] = c.mode[2];
      seg_len_[0] = c.lengths->t * radius_;
      seg_len_[1] = c.lengths->p * radius_;
      seg_len_[2] = c.lengths->q * radius_;
    }
  }

  if (std::isinf(best_total)) {
    valid_ = false;
    return;
  }
  length_ = seg_len_[0] + seg_len_[1] + seg_len_[2];
  valid_ = true;
}

DubinsPath::PathPoint DubinsPath::pointAt(double s) const
{
  s = std::clamp(s, 0.0, length_);
  double x = start_x_;
  double y = start_y_;
  double yaw = start_yaw_;
  double remaining = s;

  for (int i = 0; i < 3 && remaining > 1e-9; ++i) {
    const double seg = std::min(remaining, seg_len_[i]);
    if (mode_[i] == 'S') {
      x += seg * std::cos(yaw);
      y += seg * std::sin(yaw);
    } else if (mode_[i] == 'L') {
      const double phi = seg / radius_;
      const double nx = x + radius_ * (std::sin(yaw + phi) - std::sin(yaw));
      const double ny = y - radius_ * (std::cos(yaw + phi) - std::cos(yaw));
      x = nx;
      y = ny;
      yaw += phi;
    } else {  // 'R'
      const double phi = seg / radius_;
      const double nx = x + radius_ * (std::sin(yaw) - std::sin(yaw - phi));
      const double ny = y + radius_ * (std::cos(yaw - phi) - std::cos(yaw));
      x = nx;
      y = ny;
      yaw -= phi;
    }
    remaining -= seg;
  }
  return {x, y, yaw};
}

}  // namespace autoware::shoulder_pullover_manager
