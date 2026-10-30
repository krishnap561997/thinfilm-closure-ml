#pragma once
#include "core/fields.hpp"

struct State {
    Field h, q, r; // r = h*w, Noble--Vila augmented momentum

    explicit State(std::size_t n = 0) : h(n), q(n), r(n) {}
    std::size_t size() const { return h.size(); }
};

struct RHS {
    Field h, q, r;
    explicit RHS(std::size_t n = 0) : h(n,0.0), q(n,0.0), r(n,0.0) {}
};
