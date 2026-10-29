#pragma once

#include <string>

struct InputParameters
{
    // Forcing
    double fcut_dim = 0.0;
    int MNOISE = 0;
    double F0_noise = 0.0;

    // Grid
    int N = 0;

    // Physical parameters
    double Re_input = 0.0;
    double theta_deg = 0.0;
    double rho = 0.0;
    double mu = 0.0;
    double Ca_input = 0.0;
};


struct Parameters
{
    InputParameters in;

    // ---------------------------------------------------------
    // Physical constants/scales
    // ---------------------------------------------------------

    double g = 9.81;

    double theta = 0.0;
    double nu = 0.0;

    double hN = 0.0;
    double uN = 0.0;
    double TN = 0.0;

    double Re = 0.0;
    double F2 = 0.0;
    double We = 0.0;
    double Ca = 0.0;
    double gamma = 0.0;

    double fcut_noise = 0.0;

    // ---------------------------------------------------------
    // Canonical NONDIMENSIONAL simulation quantities
    // ---------------------------------------------------------

    double L = 0.0;
    double t_end = 0.0;
    double dt = 0.0;
    double t_save = 0.0;

    // ---------------------------------------------------------
    // Corresponding dimensional values
    // Useful for output/metadata
    // ---------------------------------------------------------

    double Ldim = 0.0;
    double t_end_dim = 0.0;
    double dt_dim = 0.0;
    double t_save_dim = 0.0;

    // ---------------------------------------------------------
    // Numerical parameters
    // ---------------------------------------------------------

    int ng = 1;
    int Nx = 0;

    double dx = 0.0;

    int nsteps = 0;
    int save_every = 1;
    int progress_every = 1;

    double h_floor = 1.0e-7;

    static Parameters from_file(const std::string& filename);
};
