#include "closures/structured_closure.hpp"  // STRUCTCLOSURE
#include "ml/ml_closure.hpp"                // gauss_deriv_hN
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
std::map<std::string, std::string> read_kv(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("StructuredClosure: cannot open " + file);
    std::map<std::string, std::string> kv; std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#'); if (hash != std::string::npos) line = line.substr(0, hash);
        const auto eq = line.find('='); if (eq == std::string::npos) continue;
        auto trim = [](std::string s) { s.erase(0, s.find_first_not_of(" \t")); s.erase(s.find_last_not_of(" \t\r") + 1); return s; };
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return kv;
}
}

bool StructuredClosure::is_structured_config(const std::string& file) {
    try { return read_kv(file).count("structured_table") > 0; } catch (...) { return false; }
}

StructuredClosure::StructuredClosure(const std::string& config_file, const Parameters& p) {
    auto kv = read_kv(config_file);
    const std::string tf = kv.at("structured_table");
    if (kv.count("sigma")) sigma_ = std::stod(kv["sigma"]);
    if (kv.count("relax")) relax_ = std::stoi(kv["relax"]) != 0;
    std::ifstream in(tf);
    if (!in) throw std::runtime_error("StructuredClosure: cannot open table " + tf);
    std::string line; std::vector<double> hs; double tsig = -1, tRe = -1;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::istringstream ss(line.substr(1)); std::string key; ss >> key;
            if (key == "sigma") ss >> tsig;
            if (key == "Re") ss >> tRe;
            continue;
        }
        std::istringstream ss(line); double h; ss >> h; hs.push_back(h);
        for (int c = 0; c < NCOL; ++c) { double v; ss >> v; tab_[c].push_back(v); }
    }
    n_ = int(hs.size());
    if (n_ < 2) throw std::runtime_error("StructuredClosure: empty table " + tf);
    h0_ = hs[0]; dh_ = (hs.back() - hs[0]) / (n_ - 1);
    if (tsig > 0 && std::abs(tsig - sigma_) > 1e-9)
        throw std::runtime_error("StructuredClosure: sigma in " + config_file + " differs from the table's");
    if (tRe > 0 && std::abs(tRe - p.Re) > 1e-6 * p.Re)
        throw std::runtime_error("StructuredClosure: table exported for another Re");
    // stiffness monitor (C.9): max kappa dt / h^2 over h in [0.2, 4]
    double stiff = 0;
    for (double h = 0.2; h <= 4.0; h += 0.01) stiff = std::max(stiff, lookup(KAP, h) * p.dt / (h * h));
    std::ostringstream d;
    d << "Structured closure V1: table " << tf << " (" << n_ << " points, h " << h0_ << "-" << hs.back()
      << "), sigma " << sigma_ << (relax_ ? "" : ", relaxation OFF") << "; max kappa dt/h^2 (h 0.2-4) = " << stiff
      << (stiff > 0.5 ? "  WARNING: > 0.5, consider implicit relaxation (C.9)" : "") << "\n";
    desc_ = d.str();
}

double StructuredClosure::lookup(Col c, double h) const {
    const double s = std::clamp((h - h0_) / dh_, 0.0, double(n_ - 1) - 1e-12);
    const int i = int(s); const double w = s - i;
    return (1 - w) * tab_[c][i] + w * tab_[c][i + 1];
}

ClosurePoint StructuredClosure::evaluate(double h, double q, const Parameters&) const {
    ClosurePoint c;
    c.phi_sf = lookup(PHI, h);
    c.dphi_sf_dh = lookup(PHI_H, h);
    c.dphi_sf_dq = 0.0;
    c.dP = lookup(DP, h);
    c.ddP_dh = lookup(DP_H, h);
    c.tau_w = 3.0 * q / (h * h);   // Lavalle; the relaxation correction is added in add_source
    return c;
}

void StructuredClosure::add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const {
    if (!relax_) return;
    const Field hx = gauss_deriv_hN(h, sigma_, p.dx, 1), qx = gauss_deriv_hN(q, sigma_, p.dx, 1),
                hxx = gauss_deriv_hN(h, sigma_, p.dx, 2);
    const int nx = int(h.size());
    for (int i = p.ng; i < nx - p.ng; ++i) {
        const double hh = std::max(h[i], 1e-12), h3 = hh * hh * hh;
        const double Q = h3 * (1.0 + lookup(G1, hh) * hx[i] + lookup(G2, hh) * qx[i] + lookup(G3, hh) * hxx[i]);
        rhs_q[i] += (lookup(KAP, hh) * (Q - q[i]) - 3.0 / p.Re * (h3 - q[i])) / (hh * hh);
    }
}

// ------------------------------------------------------------------------------------------------ FlatMLPClosure
bool FlatMLPClosure::is_config(const std::string& file) {
    try { return read_kv(file).count("flat_mlp") > 0; } catch (...) { return false; }
}

FlatMLPClosure::FlatMLPClosure(const std::string& config_file, const Parameters&) {
    auto kv = read_kv(config_file);
    mlp_ = std::make_unique<MLP>(kv.at("flat_mlp"));
    if (kv.count("sigma")) sigma_ = std::stod(kv["sigma"]);
    for (const auto& n : mlp_->inputs())
        if (n != "h" && n != "q" && n != "h_x" && n != "q_x" && n != "h_xx" && n != "Re" && n != "Ca" && n != "angle")
            throw std::runtime_error("FlatMLPClosure: unsupported input " + n);
    desc_ = "Flat-film-exact source MLP: " + kv.at("flat_mlp") + ", sigma " + std::to_string(sigma_) +
            " (S = N(state) - N(h, h^3, 0, 0, 0))\n";
}

FlatMLPClosure::~FlatMLPClosure() = default;

void FlatMLPClosure::add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const {
    const Field hx = gauss_deriv_hN(h, sigma_, p.dx, 1), qx = gauss_deriv_hN(q, sigma_, p.dx, 1),
                hxx = gauss_deriv_hN(h, sigma_, p.dx, 2);
    const int nx = int(h.size());
    Field h3(nx), zero(nx, 0.0);
    for (int i = 0; i < nx; ++i) h3[i] = h[i] * h[i] * h[i];
    std::vector<const double*> c1, c0;
    for (const auto& n : mlp_->inputs()) {
        if (n == "h") { c1.push_back(h.data()); c0.push_back(h.data()); }
        else if (n == "q") { c1.push_back(q.data()); c0.push_back(h3.data()); }
        else if (n == "h_x") { c1.push_back(hx.data()); c0.push_back(zero.data()); }
        else if (n == "q_x") { c1.push_back(qx.data()); c0.push_back(zero.data()); }
        else if (n == "h_xx") { c1.push_back(hxx.data()); c0.push_back(zero.data()); }
        else { c1.push_back(nullptr); c0.push_back(nullptr); }
    }
    for (int i = p.ng; i < nx - p.ng; ++i)
        rhs_q[i] -= mlp_->eval(c1, i, p) - mlp_->eval(c0, i, p);
}
