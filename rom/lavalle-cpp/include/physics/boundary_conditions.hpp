#pragma once
#include "core/state.hpp"
#include "forcing/chang_noise.hpp"

void apply_white_noise_bc(State& s, int ng, const ChangNoise& noise, double t);
