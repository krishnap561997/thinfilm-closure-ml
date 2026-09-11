#pragma once
#include "core/state.hpp"
#include "config/parameters.hpp"
#include "closures/closure_model.hpp"
#include "forcing/chang_noise.hpp"

void rk4_step(State& s, double t, double dt, const Parameters& p,
              const ClosureModel& closure, const ChangNoise& noise);
