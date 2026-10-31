#pragma once
// Oracle a-posteriori forcing: exact DNS closure corrections prescribed as functions of
// (x, t), plus a DNS initial state and inlet. Self-contained module.
//
// To remove: delete include/oracle/, src/oracle/ and every line tagged "// ORACLE"
// (grep -rn ORACLE src include).
//
// Run: ./thinfilm input.txt oracle.txt   with oracle.txt keys
//   forcing_file = <rom_forcing.h5>     (pipeline/oracle/make_rom_forcing.py)
//   terms        = Tw,Mx,Px,Dx           any subset, or none
// The q-equation gets  -dMx - dTw + dPx + dDx  (cell averages, linear in time).
#include "core/state.hpp"
#include "config/parameters.hpp"
#include <H5Cpp.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

class OracleForcing {
public:
    OracleForcing(const std::string& config_file, const Parameters& p);

    double t_start() const { return t_.front(); }
    double t_end() const { return t_.back(); }
    std::string summary() const;

    // DNS state at t_start: h, q and r = sqrt(h/We) h_x on all cells (ghosts copied).
    void set_initial_state(State& s, const Parameters& p) const;
    // Overwrites the inlet ghost cells and first physical cell with the DNS inlet values.
    void apply_inlet(State& s, double t, const Parameters& p) const;
    // Adds the selected corrections to the interior cells of rhs_q.
    void add_source(Field& rhs_q, double t, const Parameters& p) const;

    // The forcing used by evaluate_rhs (nullptr: no oracle).
    static const OracleForcing* active() { return active_; }
    static void set_active(const OracleForcing* o) { active_ = o; }

private:
    static constexpr int kTerms = 4;  // Mx, Tw, Px, Dx
    static constexpr std::array<const char*, kTerms> kNames{"dMx", "dTw", "dPx", "dDx"};
    static constexpr std::array<double, kTerms> kSign{-1.0, -1.0, 1.0, 1.0};

    std::string file_name_;
    std::unique_ptr<H5::H5File> file_;
    std::vector<double> t_;
    std::array<bool, kTerms> use_{};
    std::size_t n_ = 0;
    Field h0_, q0_, hx0_;
    Field h_in_, q_in_, hx_in_;

    // Two-level cache of the correction rows bracketing the current time.
    mutable std::size_t cached_k_ = static_cast<std::size_t>(-1);
    mutable std::array<Field, kTerms> lo_, hi_;

    std::size_t bracket(double t, double& w) const;
    void load(std::size_t k) const;
    void read_row(const char* name, std::size_t k, Field& row) const;

    static inline const OracleForcing* active_ = nullptr;
};
