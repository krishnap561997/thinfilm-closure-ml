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
module load gcc hdf5   # module names may vary
make
```
The Makefile uses `h5c++` by default.

### CMake
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run
```bash
cp input.example.txt input.txt
./thinfilm input.txt
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

## ML extension

Do not make an ML model return the RHS. Implement `ClosureModel::evaluate()` instead. The solver remains responsible for whether a learned quantity appears in a flux, source, or differentiated operator.

For a learned shape factor `phi_sf`, return `phi_sf`, `dphi_sf_dh`, and `dphi_sf_dq`; the hyperbolic module then evaluates the actual learned-flux discriminant and refuses states with loss of hyperbolicity.

For a learned wall shear `tau_w`, return it as a source closure. A hybrid model can preserve the analytical equilibrium/dissipative part and learn a residual.

The placeholders `phi_i`, `A0`, and `A1` are already present in `ClosurePoint`, but the supplied MATLAB solver does not contain the user's newer higher-order PDE terms. Those operators should be added in a separate `higher_order_terms` module rather than inserted into the ML class.
