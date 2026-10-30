# Lavalle / Noble--Vila thin-film solver (C++17)

This is a structured C++ translation of the supplied MATLAB solver. The current executable reproduces the *analytical Lavalle reformulated augmented system*: MUSCL + minmod reconstruction, Rusanov hyperbolic flux, Lavalle pressure law, Noble--Vila augmented capillary operator, viscous diffusion, wall-shear relaxation, Chang-style inlet forcing, and classical RK4.

## Architecture

- `closures/`: constitutive quantities. `AnalyticalLavalleClosure` is the current model; future `MLClosure` / `HybridClosure` implementations should derive from `ClosureModel`.
- `physics/hyperbolic_flux.*`: owns the conservative flux and characteristic analysis. A learned `phi_sf(h,q)` must supply derivatives so the code can recompute the true flux Jacobian and Rusanov wavespeed.
- `physics/source_terms.*`: wall shear and viscous/source terms.
- `physics/capillary.*`: Noble--Vila skew cross-diffusion operator for the augmented variable `r=h*w`.
- `physics/rhs.*`: assembles operators but does not own closure mathematics.
- `integrators/rk4.*`: time integration only; closures are reevaluated at every stage through `evaluate_rhs`.
- `numerics/`: minmod, derivatives, diagnostics.
- `io/`: HDF5 training-data writer matching the MATLAB dataset names.

## Build

### Simple UF/HPC-style build
```bash
module load gcc/14.2.0 hdf5
make
```
The Makefile uses `h5c++`. On HiPerGator the `hdf5/1.10.5` wrapper defaults to a conda compiler and ships no static libraries, so the Makefile sets `HDF5_CXX=g++` and links with `-shlib`. A short smoke test (`t_end = 5`) runs in about 2 s.

### CMake
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run
```bash
./thinfilm input.txt   # run in a scratch/run directory, not in the repo
```
Outputs:
- `thinfilm_training_data.h5`
- `inlet_signal.csv`
- `final_state.csv`

## Important parity notes

1. MATLAB uses `rng(1,'twister')`. C++ uses `std::mt19937(1)`. The stochastic construction is equivalent but the generated phases are not promised to be bitwise identical. For strict trajectory comparison, export MATLAB phases and load them in C++.
2. The MATLAB main loop advances the state from `t` to `t+dt` but calls `append_thinfilm_h5(..., t, updated_state, ...)`. This C++ code writes `t+dt`, which is the physically consistent timestamp.
3. The code retains the MATLAB positivity floor `h >= 1e-7` at every RK intermediate state and after the final update.
4. The current closure has `phi_sf=1` because the supplied Lavalle solver has already performed the 6/5-to-1 + pressure-law reformulation. `tau_w=3q/h^2` is kept as an explicit closure source.

## Oracle closure (planned, Nov 2026)

The oracle *a posteriori* run will add exact DNS corrections taken from `profiles_to_closures.py` output on the ROM grid. ΔF = ∫u² − (q²/h + Λ²h⁵/45) goes in as a face-flux correction, and ΔT_w and ΔP_x as cell sources, interpolated linearly in time at every RK4 stage. The inlet and initial state come from the DNS, with r = √(h/We) h_x. A prescribed ΔF(x,t) does not change the flux Jacobian. A learned ΔF(h,q,…) would, and must then supply its derivatives as described below. `pipeline/extract_closures/check_momentum_budget.py` contains a Python replica of this solver's q-RHS.

## ML extension

Do not make an ML model return the RHS. Implement `ClosureModel::evaluate()` instead. The solver remains responsible for whether a learned quantity appears in a flux, source, or differentiated operator.

For a learned shape factor `phi_sf`, return `phi_sf`, `dphi_sf_dh`, and `dphi_sf_dq`; the hyperbolic module then evaluates the actual learned-flux discriminant and refuses states with loss of hyperbolicity.

For a learned wall shear `tau_w`, return it as a source closure. A hybrid model can preserve the analytical equilibrium/dissipative part and learn a residual.

The placeholders `phi_i`, `A0`, and `A1` are already present in `ClosurePoint`, but the supplied MATLAB solver does not contain the user's newer higher-order PDE terms. Those operators should be added in a separate `higher_order_terms` module rather than inserted into the ML class.
