#pragma once
#include "closures/closure_model.hpp"

class AnalyticalLavalleClosure final : public ClosureModel {
public:
    ClosurePoint evaluate(double h, double q, const Parameters& p) const override;
};
