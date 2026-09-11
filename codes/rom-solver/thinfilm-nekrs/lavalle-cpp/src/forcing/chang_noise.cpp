#include "forcing/chang_noise.hpp"
#include <cmath>

namespace { constexpr double pi = 3.141592653589793238462643383279502884; }

ChangNoise::ChangNoise(int modes, double fcut, double amplitude, unsigned seed)
: modes_(modes), fcut_(fcut), amplitude_(amplitude), phase_(modes) {
    // Same stochastic construction as MATLAB, but std::mt19937 uniform draws are
    // not guaranteed to be bitwise identical to MATLAB rng(seed,'twister').
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unif(0.0,1.0);
    for (double& ph : phase_) ph = 2.0*pi*unif(gen);
}

double ChangNoise::value(double t) const {
    const double df = fcut_/static_cast<double>(modes_);
    double sum = 0.0;
    for (int k=1; k<=modes_; ++k)
        sum += std::cos(2.0*pi*(k*df)*t + phase_[static_cast<std::size_t>(k-1)]);
    return amplitude_*std::sqrt(2.0/static_cast<double>(modes_))*sum;
}
