#pragma once
#include "core/fields.hpp"
#include "config/parameters.hpp"
#include <H5Cpp.h>
#include <memory>
#include <string>

class H5TrainingWriter {
public:
    H5TrainingWriter(const std::string& filename, const Field& x, const Parameters& p);
    void append(double t, const Field& h, const Field& q, const Field& r,
                const Field& dhdt, const Field& dqdt, double F_inlet, double q_inlet);
private:
    std::unique_ptr<H5::H5File> file_;
    std::size_t nx_, index_=0;
    void create_1d_unlimited(const std::string& name);
    void create_2d_unlimited(const std::string& name);
    void append_scalar(const std::string& name,double value);
    void append_row(const std::string& name,const Field& row);
    void write_attributes(const Parameters& p);
};
