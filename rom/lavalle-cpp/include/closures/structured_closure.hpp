#pragma once
// Structured closure V1 (docs/plans/structured-closure-os-plan.md sec. 3.2, tasks C.4-C.9).  // STRUCTCLOSURE
//   d_t q + d_x[ Phi(h) q^2/h + P_Lav(h) + dP(h) ] = kappa(h) (Q - q)/h^2 + capillary + viscous
//   Q = h^3 (1 + g1(h) h_x + g2(h) q_x + g3(h) h_xx),  gradients D_G(sigma) (Gaussian, as in training)
// Flux part: through ClosurePoint (phi_sf, dphi_sf_dh, dP, ddP_dh) into the Rusanov flux and its
// wave speed.  Relaxation part: Lavalle's tau_w stays in compute_momentum_source and add_source adds
//   [kappa (Q - q) - (3/Re)(h^3 - q)] / h^2.
// Coefficients are 1D tables in h for the run's (Re, Ca, beta), exported by
// ml/structured_closure/export_tables.py (columns h Phi Phi_h dP dP_h kappa g1 g2 g3, uniform h).
// Run: ./thinfilm input.txt <oracle.txt|none> sc.txt   with sc.txt keys
//   structured_table = <file>      required (its presence selects this closure instead of MLClosure)
//   sigma = 0.5                    gradient filter width [h_N]; must match training (checked against table header)
//   relax = 1                      0: flux part only (diagnostic)
#include "closures/closure_model.hpp"
#include "core/fields.hpp"
#include <memory>
#include <string>
#include <vector>

class StructuredClosure : public ClosureModel {
public:
    StructuredClosure(const std::string& config_file, const Parameters& p);
    ClosurePoint evaluate(double h, double q, const Parameters& p) const override;
    void add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const;
    std::string summary() const { return desc_; }
    static bool is_structured_config(const std::string& file);
    static const StructuredClosure* active() { return active_; }
    static void set_active(const StructuredClosure* s) { active_ = s; }
private:
    enum Col { PHI = 0, PHI_H, DP, DP_H, KAP, G1, G2, G3, NCOL };
    double lookup(Col c, double h) const;
    std::vector<double> tab_[NCOL];
    double h0_ = 0, dh_ = 1, sigma_ = 0.5;
    int n_ = 0;
    bool relax_ = true;
    std::string desc_;
    static inline const StructuredClosure* active_ = nullptr;
};

// Flat-film-exact source MLP (progress tracker "Next options", option 2).  // STRUCTCLOSURE
//   S = N(h, q, h_x, q_x, h_xx, Re, Ca, angle) - N(h, h^3, 0, 0, 0, Re, Ca, angle), added to the q-equation as -S.
// Gradients: zero-sum D_G(sigma) (gauss_deriv_hN), as in training.  The flux stays Lavalle's.
// Config (third argument): flat_mlp = <file in MLP format>, sigma = <h_N>.  Export: ml/structured_closure/export_mlp.py
class MLP;
class FlatMLPClosure {
public:
    FlatMLPClosure(const std::string& config_file, const Parameters& p);
    ~FlatMLPClosure();
    void add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const;
    std::string summary() const { return desc_; }
    static bool is_config(const std::string& file);
    static const FlatMLPClosure* active() { return active_; }
    static void set_active(const FlatMLPClosure* s) { active_ = s; }
private:
    std::unique_ptr<MLP> mlp_;
    double sigma_ = 0.5;
    std::string desc_;
    // N0(h) = N(h, h^3, 0, 0, 0; p) tabulated once per run (p is fixed), linear interpolation; h = 1 is a node,
    // so the uniform film stays exact. Outside [t0_, t0_ + (nt_-1) dt_] the network is evaluated directly.
    std::vector<double> n0_; double t0_ = 0, tdh_ = 1e-4; int nt_ = 0;
    static inline const FlatMLPClosure* active_ = nullptr;
};
