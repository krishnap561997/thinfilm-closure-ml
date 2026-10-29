#pragma once
#include <random>
#include <vector>

class ChangNoise {
public:
    ChangNoise(int modes, double fcut, double amplitude, unsigned seed = 1);
    double value(double t) const;
private:
    int modes_;
    double fcut_, amplitude_;
    std::vector<double> phase_;
};
