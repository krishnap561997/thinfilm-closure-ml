#include "ml/ml_closure.hpp"
#include "numerics/derivatives.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
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

double MLP::eval(const std::vector<const double*>& cols, int i, const Parameters& p) const {
    thread_local std::vector<double> a, o;
    a.resize(names_.size());
    for (std::size_t k = 0; k < names_.size(); ++k) {
        double v;
        if (cols[k]) v = cols[k][i];
        else if (names_[k] == "Re") v = p.Re;
        else if (names_[k] == "Ca") v = p.Ca;
        else v = p.theta;  // "angle" (checked in MLClosure)
        a[k] = (v - xm_[k]) / xs_[k];
    }
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        const Layer& l = layers_[k];
        o.resize(l.rows);
        const bool hidden = k + 1 < layers_.size();
        for (int r = 0; r < l.rows; ++r) {
            const double* w = &l.W[std::size_t(r) * l.cols];
            double s = l.b[r];
            for (int c = 0; c < l.cols; ++c) s += w[c] * a[c];
            o[r] = hidden ? std::tanh(s) : s;
        }
        std::swap(a, o);
    }
    return a[0] * ys_ + ym_;
}

namespace {
Field gauss(const Field& f, double sigma) {  // plain smoothing for hx_smooth (sigma in cells)
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

// scipy.ndimage.gaussian_filter1d(f, s, order=m, mode='nearest', truncate=4) / dx^m, s = sigma/dx.
Field gauss_deriv(const Field& f, double sigma_hN, double dx, int m) {
    const double s = sigma_hN / dx, s2 = s * s;
    const int r = int(4.0 * s + 0.5);
    std::vector<double> phi(2 * r + 1);
    double sum = 0;
    for (int j = -r; j <= r; ++j) { phi[j + r] = std::exp(-0.5 * j * j / s2); sum += phi[j + r]; }
    for (auto& v : phi) v /= sum;
    std::vector<double> c(m + 1, 0.0); c[0] = 1;  // polynomial coefficients of the m-th derivative
    for (int it = 0; it < m; ++it) {
        std::vector<double> nc(m + 1, 0.0);
        for (int k = 0; k <= m; ++k) {
            if (k + 1 <= m) nc[k] += (k + 1) * c[k + 1];
            if (k >= 1) nc[k] += -c[k - 1] / s2;
        }
        c = nc;
    }
    std::vector<double> K(2 * r + 1);
    for (int j = -r; j <= r; ++j) {
        double poly = 0, xp = 1;
        for (int k = 0; k <= m; ++k) { poly += c[k] * xp; xp *= j; }
        K[j + r] = poly * phi[j + r];
    }
    const int n = int(f.size());
    const double scale = std::pow(dx, -m);
    Field o(n);
    for (int i = 0; i < n; ++i) {
        double a = 0;
        for (int j = -r; j <= r; ++j) a += K[j + r] * f[std::min(n - 1, std::max(0, i - j))];
        o[i] = a * scale;
    }
    return o;
}

bool split_filtered(const std::string& name, std::string& base, double& sigma) {
    const auto pos = name.rfind("_s");
    if (pos == std::string::npos) return false;
    std::string sg = name.substr(pos + 2);
    if (sg.empty() || !std::isdigit(static_cast<unsigned char>(sg[0]))) return false;
    std::replace(sg.begin(), sg.end(), 'p', '.');
    base = name.substr(0, pos);
    sigma = std::stod(sg);
    return true;
}

bool is_param(const std::string& n) { return n == "Re" || n == "Ca" || n == "angle"; }
}  // namespace

MLClosure::MLClosure(const std::string& config_file, const Parameters& p) {
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
    if ((s && (m || t)) || (m != t)) throw std::runtime_error("ml: give sum_model, or both mx_model and tw_model, or neither");
    if (s) { sum_ = std::make_unique<MLP>(kv["sum_model"]); desc_ = "sum model " + kv["sum_model"]; }
    if (m) { mx_ = std::make_unique<MLP>(kv["mx_model"]); tw_ = std::make_unique<MLP>(kv["tw_model"]);
             desc_ = "Mx model " + kv["mx_model"] + "; Tw model " + kv["tw_model"]; }
    if (kv.count("px_model")) {
        px_ = std::make_unique<MLP>(kv["px_model"]);
        desc_ += (desc_.empty() ? "" : "; ") + std::string("Px model ") + kv["px_model"];
        if (kv.count("px_curvature_fix")) px_fix_ = std::stoi(kv["px_curvature_fix"]) != 0;
        if (kv.count("px_sigma")) px_sigma_ = std::stod(kv["px_sigma"]);
        desc_ += px_fix_ ? " (+ h(kappa_x - h_xxx)/We, sigma " + std::to_string(px_sigma_) + ")" : " (no curvature fix)";
    }
    if (!sum_ && !mx_ && !px_) throw std::runtime_error("ml: no model given");
    if (kv.count("hx_smooth")) smooth_ = std::stod(kv["hx_smooth"]);
    std::set<std::string> need;
    for (const MLP* mod : {sum_.get(), mx_.get(), tw_.get(), px_.get()}) {
        if (!mod) continue;
        for (const auto& n : mod->inputs()) if (!is_param(n)) need.insert(n);
    }
    needed_.assign(need.begin(), need.end());
    features(Field(10 * p.ng + 64, 1.0), Field(10 * p.ng + 64, 1.0), p, needed_);  // validates names
}

std::string MLClosure::summary() const { return "ML closure: " + desc_ + "\n"; }

std::map<std::string, Field> MLClosure::features(const Field& h, const Field& q, const Parameters& p,
                                                 const std::vector<std::string>& names) const {
    std::map<std::string, Field> out;
    std::map<std::string, Field> cache;  // filtered derivatives by "<var><order>_<sigma>"
    auto gd = [&](const Field& f, const char* var, int order, double sg) -> const Field& {
        const std::string key = std::string(var) + std::to_string(order) + "_" + std::to_string(sg);
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, gauss_deriv(f, sg, p.dx, order)).first;
        return it->second;
    };
    for (const auto& n : names) {
        if (n == "h") { out[n] = h; continue; }
        if (n == "q") { out[n] = q; continue; }
        if (n == "h_x") { out[n] = ddx_1d(gauss(h, smooth_), p.dx); continue; }
        if (n == "q_x") { out[n] = ddx_1d(gauss(q, smooth_), p.dx); continue; }
        std::string base; double sg;
        if (!split_filtered(n, base, sg)) throw std::runtime_error("ml: unsupported input " + n);
        if (base == "h_x") out[n] = gd(h, "h", 1, sg);
        else if (base == "h_xx") out[n] = gd(h, "h", 2, sg);
        else if (base == "h_xxx") out[n] = gd(h, "h", 3, sg);
        else if (base == "q_x") out[n] = gd(q, "q", 1, sg);
        else if (base == "q_xx") out[n] = gd(q, "q", 2, sg);
        else if (base == "kappa" || base == "kappa_x" || base == "kappa_x_We") {
            const Field &hx = gd(h, "h", 1, sg), &hxx = gd(h, "h", 2, sg), &hxxx = gd(h, "h", 3, sg);
            Field v(h.size());
            for (std::size_t i = 0; i < h.size(); ++i) {
                const double g = 1.0 + hx[i] * hx[i];
                if (base == "kappa") v[i] = hxx[i] / std::pow(g, 1.5);
                else v[i] = hxxx[i] / std::pow(g, 1.5) - 3.0 * hx[i] * hxx[i] * hxx[i] / std::pow(g, 2.5);
                if (base == "kappa_x_We") v[i] /= p.We;
            }
            out[n] = std::move(v);
        }
        else throw std::runtime_error("ml: unsupported input " + n);
    }
    return out;
}

std::map<std::string, Field> MLClosure::outputs(const Field& h, const Field& q, const Parameters& p) const {
    const auto F = features(h, q, p, needed_);
    auto cols = [&](const MLP& mod) {
        std::vector<const double*> c;
        for (const auto& n : mod.inputs()) c.push_back(is_param(n) ? nullptr : F.at(n).data());
        return c;
    };
    const int n = int(h.size());
    std::map<std::string, Field> out;
    auto run = [&](const char* key, const MLP* mod) {
        if (!mod) return;
        const auto c = cols(*mod);
        Field v(n, 0.0);
        for (int i = p.ng; i < n - p.ng; ++i) v[i] = mod->eval(c, i, p);
        out[key] = std::move(v);
    };
    run("sum", sum_.get()); run("mx", mx_.get()); run("tw", tw_.get()); run("px", px_.get());
    if (px_ && px_fix_) {
        const Field hs = gauss_deriv(h, px_sigma_, p.dx, 0), hx = gauss_deriv(h, px_sigma_, p.dx, 1),
                    hxx = gauss_deriv(h, px_sigma_, p.dx, 2), hxxx = gauss_deriv(h, px_sigma_, p.dx, 3);
        Field v(n, 0.0);
        for (int i = p.ng; i < n - p.ng; ++i) {
            const double g = 1.0 + hx[i] * hx[i];
            const double kx = hxxx[i] / std::pow(g, 1.5) - 3.0 * hx[i] * hxx[i] * hxx[i] / std::pow(g, 2.5);
            v[i] = hs[i] * (kx - hxxx[i]) / p.We;
        }
        out["px_fix"] = std::move(v);
    }
    return out;
}

void MLClosure::add_source(Field& rhs_q, const Field& h, const Field& q, const Parameters& p) const {
    const auto o = outputs(h, q, p);
    const int n = int(h.size());
    for (const auto& [k, v] : o) {
        const double sgn = (k == "px" || k == "px_fix") ? 1.0 : -1.0;  // -ΔM_x - ΔT_w + ΔP_x
        for (int i = p.ng; i < n - p.ng; ++i) rhs_q[i] += sgn * v[i];
    }
}
