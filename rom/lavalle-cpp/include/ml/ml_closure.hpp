#pragma once
// Learned closure: ΔM_x + ΔT_w (one model) or ΔM_x and ΔT_w (two models) as local MLPs of
// (h, q, h_x, q_x, Re, Ca, angle) evaluated on the ROM state at every RK4 stage. Added to
// the q-equation as -ΔM_x - ΔT_w (same sign convention as the oracle module).
// Self-contained; to remove delete include/ml, src/ml and every line tagged "// MLCLOSURE".
//
// Run: ./thinfilm input.txt oracle.txt ml.txt   (oracle.txt supplies the DNS initial state
// and inlet; use "terms = none" so only the ML source acts). ml.txt keys:
//   sum_model = <file>             ΔM_x+ΔT_w model   (export: ml/functional_mlp/export_for_cpp.py)
//   mx_model  = <file>             ΔM_x model  \ either sum_model, or both of these
//   tw_model  = <file>             ΔT_w model  /
//   hx_smooth = 0                  optional: Gaussian sigma [cells] applied to h, q before d/dx
#include "core/fields.hpp"
#include "config/parameters.hpp"
#include <memory>
#include <string>
#include <vector>

class MLP {
public:
    explicit MLP(const std::string& file);
    // Fields by name: values for h, q, h_x, q_x; Re, Ca, angle taken from p.
    double eval(double h, double q, double hx, double qx, const Parameters& p) const;
    const std::vector<std::string>& inputs() const { return names_; }
private:
    std::vector<std::string> names_;
    std::vector<double> xm_, xs_;
    double ym_ = 0, ys_ = 1;
    struct Layer { int rows, cols; std::vector<double> W, b; };
    std::vector<Layer> layers_;
};

class MLClosure {
public:
    MLClosure(const std::string& config_file, const Parameters& p);
    std::string summary() const;
    // Adds the learned source to interior cells of rhs_q, given the state fields h, q (with ghosts).
    void add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const;
    static const MLClosure* active() { return active_; }
    static void set_active(const MLClosure* m) { active_ = m; }
private:
    std::unique_ptr<MLP> sum_, mx_, tw_;
    double smooth_ = 0.0;
    std::string desc_;
    static inline const MLClosure* active_ = nullptr;
};
