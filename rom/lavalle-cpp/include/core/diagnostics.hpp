#pragma once
#include "core/state.hpp"
#include "config/parameters.hpp"
#include "closures/closure_model.hpp"
#include "physics/hyperbolic_flux.hpp"
struct Diagnostics {
    double h_min{}, h_max{}, max_abs_hx{}, min_hyperbolic_discriminant{}, max_abs_lambda{};
};
Diagnostics compute_diagnostics(const State& s, const Parameters& p, const ClosureModel& closure);
