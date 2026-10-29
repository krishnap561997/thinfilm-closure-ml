#pragma once

class InletForcing
{
public:
    virtual double value(double t) const = 0;

    virtual ~InletForcing() = default;
};
