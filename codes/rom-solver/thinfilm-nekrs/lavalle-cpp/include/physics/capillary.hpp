#pragma once
#include "core/state.hpp"
#include "config/parameters.hpp"
struct CapillaryRHS { Field q,r; explicit CapillaryRHS(std::size_t n=0):q(n,0.0),r(n,0.0){} };
CapillaryRHS compute_augmented_capillary(const State& s, const Parameters& p);
