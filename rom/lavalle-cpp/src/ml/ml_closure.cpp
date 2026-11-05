#include "ml/ml_closure.hpp"
#include "numerics/derivatives.hpp"
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

MLP::MLP(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("ml: cannot open " + file);
    std::string tok;
    int L = 0;
    auto vec = [&](std::vector<double>& v, std::size_t n) { v.resize(n); for (auto& x : v) in >> x; };
    while (in >> tok) {
        if (tok == "targets") { std::string l; std::getline(in, l); }
        else if (tok == "inputs") { int n; in >> n; names_.resize(n); for (auto& s : names_) in >> s; }
        else if (tok == "x_mean") vec(xm_, names_.size());
        else if (tok == "x_std") vec(xs_, names_.size());
        else if (tok == "y_mean") in >> ym_;
        else if (tok == "y_std") in >> ys_;
        else if (tok == "layers") { in >> L; }
        else if (tok == "W") { Layer l; in >> l.rows >> l.cols; vec(l.W, std::size_t(l.rows) * l.cols); layers_.push_back(std::move(l)); }
        else if (tok == "b") { int n; in >> n; vec(layers_.back().b, n); }
        else throw std::runtime_error("ml: bad token '" + tok + "' in " + file);
    }
    if (L == 0 || int(layers_.size()) != L || xm_.size() != names_.size())
        throw std::runtime_error("ml: incomplete model file " + file);
}

double MLP::eval(double h, double q, double hx, double qx, const Parameters& p) const {
    std::vector<double> a(names_.size());
    for (std::size_t i = 0; i < names_.size(); ++i) {
        const std::string& n = names_[i];
        double v;
        if (n == "h") v = h; else if (n == "q") v = q; else if (n == "h_x") v = hx; else if (n == "q_x") v = qx;
        else if (n == "Re") v = p.Re; else if (n == "Ca") v = p.Ca; else if (n == "angle") v = p.theta;
        else throw std::runtime_error("ml: unsupported input " + n);
        a[i] = (v - xm_[i]) / xs_[i];
    }
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        const Layer& l = layers_[k];
        std::vector<double> o(l.rows);
        for (int r = 0; r < l.rows; ++r) {
            double s = l.b[r];
            for (int c = 0; c < l.cols; ++c) s += l.W[std::size_t(r) * l.cols + c] * a[c];
            o[r] = (k + 1 < layers_.size()) ? std::tanh(s) : s;
        }
        a = std::move(o);
    }
    return a[0] * ys_ + ym_;
}

MLClosure::MLClosure(const std::string& config_file, const Parameters&) {
    std::ifstream in(config_file);
    if (!in) throw std::runtime_error("ml: cannot open " + config_file);
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = line.substr(0, line.find('#'));
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) { s.erase(0, s.find_first_not_of(" \t\r")); s.erase(s.find_last_not_of(" \t\r") + 1); return s; };
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    const bool s = kv.count("sum_model"), m = kv.count("mx_model"), t = kv.count("tw_model");
    if (s == (m || t) || (m != t)) throw std::runtime_error("ml: give sum_model, or both mx_model and tw_model");
    if (s) { sum_ = std::make_unique<MLP>(kv["sum_model"]); desc_ = "sum model " + kv["sum_model"]; }
    else { mx_ = std::make_unique<MLP>(kv["mx_model"]); tw_ = std::make_unique<MLP>(kv["tw_model"]);
           desc_ = "Mx model " + kv["mx_model"] + "; Tw model " + kv["tw_model"]; }
    if (kv.count("hx_smooth")) smooth_ = std::stod(kv["hx_smooth"]);
}

std::string MLClosure::summary() const { return "ML closure: " + desc_ + "\n"; }

namespace {
Field gauss(const Field& f, double sigma) {
    if (sigma <= 0) return f;
    const int w = int(std::ceil(4 * sigma));
    std::vector<double> k(2 * w + 1); double sum = 0;
    for (int j = -w; j <= w; ++j) { k[j + w] = std::exp(-0.5 * j * j / (sigma * sigma)); sum += k[j + w]; }
    Field o(f.size());
    const int n = int(f.size());
    for (int i = 0; i < n; ++i) { double a = 0;
        for (int j = -w; j <= w; ++j) { int ii = std::min(n - 1, std::max(0, i + j)); a += k[j + w] * f[ii]; }
        o[i] = a / sum; }
    return o;
}
}

void MLClosure::add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const {
    const Field hs = gauss(h, smooth_), qs = gauss(q, smooth_);
    const Field hx = ddx_1d(hs, p.dx), qx = ddx_1d(qs, p.dx);
    const int n = int(h.size());
    for (int i = p.ng; i < n - p.ng; ++i) {
        const double src = sum_ ? sum_->eval(h[i], q[i], hx[i], qx[i], p)
                                : mx_->eval(h[i], q[i], hx[i], qx[i], p) + tw_->eval(h[i], q[i], hx[i], qx[i], p);
        rhs_q[i] -= src;
    }
}
