#include "physics/pressure_law.hpp"
#include <cmath>

double LavallePressureLaw::lambda(const Parameters& p) { return std::sin(p.theta)*p.Re/p.F2; }
double LavallePressureLaw::P(double h, const Parameters& p) {
    const double L=lambda(p);
    return std::cos(p.theta)*h*h/(2.0*p.F2) + 2.0*L*L*h*h*h*h*h/225.0;
}
double LavallePressureLaw::dPdh(double h, const Parameters& p) {
    const double L=lambda(p);
    return std::cos(p.theta)*h/p.F2 + (2.0/45.0)*L*L*h*h*h*h;
}
