#include "closures/analytical_lavalle.hpp"

ClosurePoint AnalyticalLavalleClosure::evaluate(double h, double q, const Parameters&) const {
    ClosurePoint c;
    // Lavalle's reformulated augmented solver has canonical convective flux q^2/h.
    c.phi_sf = 1.0;
    c.dphi_sf_dh = 0.0;
    c.dphi_sf_dq = 0.0;
    c.tau_w = 3.0*q/(h*h); // source term 3*u/h = 3*q/h^2
    return c;
}
