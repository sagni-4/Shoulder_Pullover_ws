#pragma once
// Natural cubic spline over uniformly-spaced knots (t = 0, 1, ..., n-1),
// evaluated with the standard tridiagonal (Thomas algorithm) second-derivative
// solve -- the textbook "spline"/"splint" formulation (Numerical Recipes-style),
// not a research technique of its own. Used to fit x(t), y(t), z(t) through
// the shoulder centerline's binned centroids, which are already uniformly
// spaced in forward distance (bin_size), so unit knot spacing matches the
// bin indices directly with no extra chord-length computation needed.
#include <algorithm>
#include <cstddef>
#include <vector>

namespace shoulder_centerline {

class CubicSpline1D {
public:
  explicit CubicSpline1D(const std::vector<double> & y) : y_(y)
  {
    const std::size_t n = y_.size();
    m_.assign(n, 0.0);
    if (n < 3) {
      return;  // fewer than 3 knots: eval() degrades to linear/constant below
    }

    std::vector<double> a(n, 0.0), b(n, 0.0), c(n, 0.0), d(n, 0.0);
    b[0] = 1.0;
    d[0] = 0.0;  // natural boundary: M[0] = 0
    for (std::size_t i = 1; i + 1 < n; ++i) {
      a[i] = 1.0;
      b[i] = 4.0;
      c[i] = 1.0;
      d[i] = 6.0 * (y_[i + 1] - 2.0 * y_[i] + y_[i - 1]);
    }
    b[n - 1] = 1.0;
    d[n - 1] = 0.0;  // natural boundary: M[n-1] = 0

    for (std::size_t i = 1; i < n; ++i) {
      const double w = a[i] / b[i - 1];
      b[i] -= w * c[i - 1];
      d[i] -= w * d[i - 1];
    }
    m_[n - 1] = d[n - 1] / b[n - 1];
    for (std::size_t i = n - 1; i-- > 0; ) {
      m_[i] = (d[i] - c[i] * m_[i + 1]) / b[i];
    }
  }

  // deriv: 0 = value, 1 = first derivative, 2 = second derivative (all wrt t).
  double eval(double t, int deriv = 0) const
  {
    const std::size_t n = y_.size();
    if (n == 0) {
      return 0.0;
    }
    if (n == 1) {
      return deriv == 0 ? y_[0] : 0.0;
    }
    t = std::clamp(t, 0.0, static_cast<double>(n - 1));
    std::size_t i = std::min(static_cast<std::size_t>(t), n - 2);
    const double s = t - static_cast<double>(i);
    const double yi = y_[i], yi1 = y_[i + 1], Mi = m_[i], Mi1 = m_[i + 1];
    const double s1 = 1.0 - s;
    switch (deriv) {
      case 0:
        return yi * s1 + yi1 * s + (s1 * s1 * s1 - s1) * Mi / 6.0 + (s * s * s - s) * Mi1 / 6.0;
      case 1:
        return (yi1 - yi) + (-3.0 * s1 * s1 + 1.0) * Mi / 6.0 + (3.0 * s * s - 1.0) * Mi1 / 6.0;
      case 2:
        return Mi * s1 + Mi1 * s;
      default:
        return 0.0;
    }
  }

private:
  std::vector<double> y_;
  std::vector<double> m_;  // second derivatives at each knot
};

}  // namespace shoulder_centerline
