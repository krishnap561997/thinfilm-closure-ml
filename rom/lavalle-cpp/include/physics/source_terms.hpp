#pragma once
#include "core/state.hpp"
#include "closures/closure_model.hpp"
#include "config/parameters.hpp"
Field compute_momentum_source(const State& s, const Parameters& p, const ClosureModel& closure);
Field compute_viscous_diffusion(const State& s, const Parameters& p);
