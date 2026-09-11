#include "config/parameters.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace { constexpr double pi = 3.141592653589793238462643383279502884; }

Parameters Parameters::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open input file: " + path);
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    if (v.size() < 13) throw std::runtime_error("input.txt must contain at least 13 numeric values");

    Parameters p;
    p.in.fcut_dim=v[0]; p.in.t_end_dim=v[1]; p.in.Ldim=v[2]; p.in.N=static_cast<int>(std::llround(v[3]));
    p.in.MNOISE=static_cast<int>(std::llround(v[4])); p.in.F0_noise=v[5]; p.in.Re_input=v[6];
    p.in.theta_deg=v[7]; p.in.rho=v[8]; p.in.mu=v[9]; p.in.Ca_input=v[10]; p.in.t_save=v[11]; p.in.dt_dim=v[12];

    p.theta = p.in.theta_deg*pi/180.0;
    p.nu = p.in.mu/p.in.rho;
    p.We = p.in.Ca_input*p.in.Re_input;
    p.hN = std::cbrt(3.0*p.in.Re_input*p.nu*p.nu/(p.g*std::sin(p.theta)));
    p.uN = p.nu*p.in.Re_input/p.hN;
    p.TN = p.hN/p.uN;
    p.Re = p.uN*p.hN/p.nu;
    p.F2 = p.uN*p.uN/(p.g*p.hN);
    p.Ca = p.We/p.Re;
    p.gamma = p.in.rho*p.hN*p.uN*p.uN/p.We;
    p.fcut_noise = p.in.fcut_dim*p.TN;

    p.L = p.in.Ldim/p.hN;
    p.Nx = p.in.N + 2*p.ng;
    p.dx = p.L/p.in.N;

    p.t_end = p.in.t_end_dim/p.TN;
    const double dt_guess = p.in.dt_dim/p.TN;
    p.nsteps = static_cast<int>(std::ceil(p.t_end/dt_guess));
    p.dt = p.t_end/p.nsteps;
    p.dt_dim_actual = p.dt*p.TN;
    p.save_every = std::max(1, static_cast<int>(std::llround(p.in.t_save/p.dt_dim_actual)));
    p.progress_every = std::max(1, p.nsteps/250);
    return p;
}
