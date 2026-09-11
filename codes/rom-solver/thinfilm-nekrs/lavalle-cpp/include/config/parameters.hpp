#pragma once
#include <string>

struct InputParameters {
    double fcut_dim{};
    double t_end_dim{};
    double Ldim{};
    int N{};
    int MNOISE{};
    double F0_noise{};
    double Re_input{};
    double theta_deg{};
    double rho{};
    double mu{};
    double Ca_input{};
    double t_save{};
    double dt_dim{};
};

struct Parameters {
    InputParameters in;
    double g = 9.81;
    double theta{}, nu{}, We{}, hN{}, uN{}, TN{}, Re{}, F2{}, Ca{}, gamma{};
    double fcut_noise{}, L{}, dx{}, t_end{}, dt{}, dt_dim_actual{};
    int ng = 1;
    int Nx{}, nsteps{}, save_every{}, progress_every{};
    double h_floor = 1e-7;

    static Parameters from_file(const std::string& path);
};
