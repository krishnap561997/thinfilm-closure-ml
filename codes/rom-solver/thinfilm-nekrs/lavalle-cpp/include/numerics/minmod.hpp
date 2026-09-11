#pragma once
#include <algorithm>
#include <cmath>
inline double minmod(double a, double b) {
    if (a*b <= 0.0) return 0.0;
    return std::copysign(std::min(std::abs(a),std::abs(b)),a);
}
