#pragma once
// Learned closures evaluated on the ROM state at every RK4 stage and added to the q-equation
// with the oracle sign convention:  -ΔM_x - ΔT_w + ΔP_x.
//   * ΔM_x + ΔT_w: one model (sum_model) or two (mx_model, tw_model).
//   * ΔP_x (v2 target, docs/results/dpx-closure.md): px_model, optional.
// Self-contained; to remove delete include/ml, src/ml and every line tagged "// MLCLOSURE".
//
// Run: ./thinfilm input.txt oracle.txt ml.txt   (oracle.txt supplies the DNS initial state
// and inlet; use "terms = none" so only the ML source acts). ml.txt keys:
//   sum_model = <file>             ΔM_x+ΔT_w model   (export: ml/functional_mlp/export_for_cpp.py)
//   mx_model  = <file>             ΔM_x model  \ either sum_model, or both of these, or neither
//   tw_model  = <file>             ΔT_w model  /
//   px_model  = <file>             optional ΔP_x v2 model
//   hx_smooth = 0                  Gaussian sigma [cells] applied to h, q before the central
//                                  differences that give the plain inputs h_x, q_x
//   px_curvature_fix = 1           with px_model: also add h_s (κ_x,s - h_xxx,s)/We, because
//                                  ΔP_x v2 is defined against exact curvature while the ROM's
//                                  Noble–Vila capillarity is the linearised h h_xxx/We
//   px_sigma  = 0.5                filter width [h_N] for that term
//
// Model inputs: h, q, h_x, q_x (central differences), Re, Ca, angle, and filtered fields
// <f>_s<sigma> with f in {h_x, h_xx, h_xxx, q_x, q_xx, kappa, kappa_x, kappa_x_We} and sigma in
// h_N written with 'p' for '.', e.g. h_xxx_s0p5. Filtered fields use a Gaussian of width sigma
// with analytic derivatives, truncated at 4 sigma, edges clamped: the same operator as
// scipy.ndimage.gaussian_filter1d(mode='nearest') in ml/functional_mlp/prepare_data.py.
#include "core/fields.hpp"
#include "config/parameters.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

class MLP {
public:
    explicit MLP(const std::string& file);
    // cols[k] points to the field of input k (nullptr for Re, Ca, angle, taken from p).
    double eval(const std::vector<const double*>& cols, int i, const Parameters& p) const;
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
    // Adds the learned sources to interior cells of rhs_q, given the state fields h, q (with ghosts).
    void add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const;
    // Input fields by name (exposed for checking against the Python features).
    std::map<std::string, Field> features(const Field& h, const Field& q, const Parameters& p,
                                          const std::vector<std::string>& names) const;
    // Individual model outputs on interior cells (keys "sum", "mx", "tw", "px", "px_fix").
    std::map<std::string, Field> outputs(const Field& h, const Field& q, const Parameters& p) const;
    static const MLClosure* active() { return active_; }
    static void set_active(const MLClosure* m) { active_ = m; }
private:
    std::unique_ptr<MLP> sum_, mx_, tw_, px_;
    double smooth_ = 0.0, px_sigma_ = 0.5;
    bool px_fix_ = true;
    std::vector<std::string> needed_;
    std::string desc_;
    static inline const MLClosure* active_ = nullptr;
};
