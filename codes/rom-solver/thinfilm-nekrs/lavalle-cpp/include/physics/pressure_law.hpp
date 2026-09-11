#pragma once
#include "config/parameters.hpp"

struct LavallePressureLaw {
    static double lambda(const Parameters& p);
    static double P(double h, const Parameters& p);
    static double dPdh(double h, const Parameters& p);
};
