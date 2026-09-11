#pragma once
#include "core/state.hpp"
#include "config/parameters.hpp"
#include "closures/closure_model.hpp"
#include "forcing/chang_noise.hpp"
#include "physics/hyperbolic_flux.hpp"

RHS evaluate_rhs(const State& input, double t, const Parameters& p,
                 const ClosureModel& closure, const ChangNoise& noise,
                 FluxDiagnostics* diagnostics=nullptr);
