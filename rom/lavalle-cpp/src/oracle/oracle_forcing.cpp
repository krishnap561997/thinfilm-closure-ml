#include "oracle/oracle_forcing.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
std::map<std::string, std::string> read_keys(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("oracle: cannot open " + path);
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
        line = line.substr(0, line.find('#'));
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) {
            s.erase(0, s.find_first_not_of(" \t\r"));
            s.erase(s.find_last_not_of(" \t\r") + 1);
            return s;
        };
        kv[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return kv;
}

Field read_1d(H5::H5File& f, const std::string& name) {
    auto ds = f.openDataSet(name);
    hsize_t n = 0;
    ds.getSpace().getSimpleExtentDims(&n);
    Field v(n);
    ds.read(v.data(), H5::PredType::NATIVE_DOUBLE);
    return v;
}

double read_attr(H5::H5File& f, const std::string& name) {
    double v = 0.0;
    f.openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &v);
    return v;
}
}

OracleForcing::OracleForcing(const std::string& config_file, const Parameters& p) {
    auto kv = read_keys(config_file);
    if (!kv.count("forcing_file")) throw std::runtime_error("oracle: forcing_file missing in " + config_file);
    file_name_ = kv["forcing_file"];
    const std::string terms = kv.count("terms") ? kv["terms"] : "Tw,Mx,Px,Dx";
    for (int k = 0; k < kTerms; ++k)
        use_[k] = terms.find(std::string(kNames[k]).substr(1)) != std::string::npos;

    file_ = std::make_unique<H5::H5File>(file_name_, H5F_ACC_RDONLY);
    t_ = read_1d(*file_, "/t");
    const Field x = read_1d(*file_, "/x");
    n_ = x.size();
    if (static_cast<int>(n_) != p.in.N)
        throw std::runtime_error("oracle: forcing has " + std::to_string(n_) + " cells, input.txt N = " + std::to_string(p.in.N));
    if (std::abs(read_attr(*file_, "L") - p.L) > 1e-9 * p.L)
        throw std::runtime_error("oracle: forcing L differs from input.txt L");
    for (const char* g : {"Re", "Ca"}) {
        const double v = read_attr(*file_, g), ref = (std::string(g) == "Re") ? p.Re : p.Ca;
        if (std::abs(v - ref) > 1e-6 * ref)
            throw std::runtime_error(std::string("oracle: forcing ") + g + " differs from input.txt");
    }
    if (std::abs(read_attr(*file_, "angle") - p.theta) > 1e-9)
        throw std::runtime_error("oracle: forcing angle differs from input.txt");
    h0_ = read_1d(*file_, "/init/h"); q0_ = read_1d(*file_, "/init/q"); hx0_ = read_1d(*file_, "/init/h_x");
    h_in_ = read_1d(*file_, "/inlet/h"); q_in_ = read_1d(*file_, "/inlet/q"); hx_in_ = read_1d(*file_, "/inlet/h_x");
    for (auto& f : lo_) f.resize(n_);
    for (auto& f : hi_) f.resize(n_);
}

std::string OracleForcing::summary() const {
    std::ostringstream o;
    o << "\n--- Oracle forcing ---\nfile        = " << file_name_
      << "\ntimes       = " << t_.size() << " in [" << t_start() << ", " << t_end() << "]\nterms       =";
    for (int k = 0; k < kTerms; ++k) if (use_[k]) o << ' ' << kNames[k];
    if (std::none_of(use_.begin(), use_.end(), [](bool b) { return b; })) o << " none";
    o << "\n\n";
    return o.str();
}

void OracleForcing::set_initial_state(State& s, const Parameters& p) const {
    for (std::size_t i = 0; i < n_; ++i) {
        const std::size_t I = p.ng + i;
        s.h[I] = h0_[i]; s.q[I] = q0_[i]; s.r[I] = std::sqrt(h0_[i] / p.We) * hx0_[i];
    }
    for (int g = 0; g < p.ng; ++g) {
        const std::size_t L = p.ng - 1 - g, R = p.ng + n_ + g;
        s.h[L] = s.h[p.ng]; s.q[L] = s.q[p.ng]; s.r[L] = s.r[p.ng];
        s.h[R] = s.h[R - 1 - 2 * g]; s.q[R] = s.q[R - 1 - 2 * g]; s.r[R] = s.r[R - 1 - 2 * g];
    }
}

std::size_t OracleForcing::bracket(double t, double& w) const {
    const double eps = 1e-9 * std::max(1.0, std::abs(t));
    if (t < t_.front() - eps || t > t_.back() + eps)
        throw std::runtime_error("oracle: t = " + std::to_string(t) + " outside forcing range");
    std::size_t k = std::upper_bound(t_.begin(), t_.end(), t) - t_.begin();
    k = std::clamp<std::size_t>(k, 1, t_.size() - 1) - 1;
    w = std::clamp((t - t_[k]) / (t_[k + 1] - t_[k]), 0.0, 1.0);
    return k;
}

void OracleForcing::apply_inlet(State& s, double t, const Parameters& p) const {
    double w = 0.0;
    const std::size_t k = bracket(t, w);
    const double h = (1 - w) * h_in_[k] + w * h_in_[k + 1];
    const double q = (1 - w) * q_in_[k] + w * q_in_[k + 1];
    const double hx = (1 - w) * hx_in_[k] + w * hx_in_[k + 1];
    for (int i = 0; i <= p.ng; ++i) { s.h[i] = h; s.q[i] = q; s.r[i] = std::sqrt(h / p.We) * hx; }
}

void OracleForcing::read_row(const char* name, std::size_t k, Field& row) const {
    auto ds = file_->openDataSet(name);
    auto space = ds.getSpace();
    hsize_t start[2] = {k, 0}, count[2] = {1, n_};
    space.selectHyperslab(H5S_SELECT_SET, count, start);
    hsize_t mdims[1] = {n_};
    H5::DataSpace mem(1, mdims);
    ds.read(row.data(), H5::PredType::NATIVE_DOUBLE, mem, space);
}

void OracleForcing::load(std::size_t k) const {
    if (k == cached_k_) return;
    const bool shift = cached_k_ != static_cast<std::size_t>(-1) && k == cached_k_ + 1;
    for (int m = 0; m < kTerms; ++m) {
        if (!use_[m]) continue;
        if (shift) std::swap(lo_[m], hi_[m]);
        else read_row(kNames[m], k, lo_[m]);
        read_row(kNames[m], k + 1, hi_[m]);
    }
    cached_k_ = k;
}

void OracleForcing::add_source(Field& rhs_q, double t, const Parameters& p) const {
    double w = 0.0;
    const std::size_t k = bracket(t, w);
    load(k);
    for (int m = 0; m < kTerms; ++m) {
        if (!use_[m]) continue;
        for (std::size_t i = 0; i < n_; ++i)
            rhs_q[p.ng + i] += kSign[m] * ((1 - w) * lo_[m][i] + w * hi_[m][i]);
    }
}
