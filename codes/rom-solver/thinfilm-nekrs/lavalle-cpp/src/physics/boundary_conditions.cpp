#include "physics/boundary_conditions.hpp"

void apply_white_noise_bc(State& s, int ng, const ChangNoise& noise, double t) {
    const int nx = static_cast<int>(s.size());
    const double inlet = 1.0 + noise.value(t);
    for (int k=0; k<ng; ++k) {
        s.h[k]=1.0; s.q[k]=inlet; s.r[k]=0.0;
    }
    // MATLAB also imposes the inlet state on the first physical cell.
    s.h[ng]=1.0; s.q[ng]=inlet; s.r[ng]=0.0;

    const int src = nx-ng-1; // last physical cell
    for (int k=0; k<ng; ++k) {
        const int dst = nx-1-k;
        s.h[dst]=s.h[src]; s.q[dst]=s.q[src]; s.r[dst]=s.r[src];
    }
}
