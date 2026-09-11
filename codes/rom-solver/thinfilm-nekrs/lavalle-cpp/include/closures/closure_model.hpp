#pragma once
#include "config/parameters.hpp"

// Quantities supplied by constitutive closure at a point.
// dphi_* are exposed because a learned shape-factor flux changes the
// characteristic Jacobian and therefore the Rusanov wavespeed.
struct ClosurePoint {
    double phi_sf = 1.0;
    double dphi_sf_dh = 0.0;
    double dphi_sf_dq = 0.0;
    double tau_w = 0.0;
    double phi_i = 0.0;
    double A0 = 0.0;
    double A1 = 0.0;
};

class ClosureModel {
public:
    virtual ~ClosureModel() = default;
    virtual ClosurePoint evaluate(double h, double q, const Parameters& p) const = 0;
};
