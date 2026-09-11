#pragma once
#include "core/state.hpp"
#include "closures/closure_model.hpp"
#include "config/parameters.hpp"

struct FluxDiagnostics {
    double min_discriminant = 1e300;
    double max_abs_lambda = 0.0;
};

struct NumericalFlux {
    Field h, q, r;
    explicit NumericalFlux(std::size_t nfaces=0):h(nfaces,0.0),q(nfaces,0.0),r(nfaces,0.0){}
};

NumericalFlux compute_rusanov_fluxes(const State& s, int ng, double dx,
                                     const Parameters& p, const ClosureModel& closure,
                                     FluxDiagnostics* diagnostics=nullptr);
