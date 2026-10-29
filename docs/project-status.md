# Project status

_Last updated: 2026-10-29 from the ONR technical report (27 Oct 2026) and the Paper 1 draft._
Update this file in the same commit as any change that moves the science forward.

## Numerics

**ROM solver (`rom/lavalle-cpp`)**

- Noble–Vila augmented variable for capillarity.
- Rusanov flux with MUSCL reconstruction.
- RK4 time integration.
- A run takes about 15 min, versus about 30 h for DNS (roughly 120× faster).

**DNS (`dns/production`)**

- Basilisk C, CLS-VOF with the integral surface-tension model of Abu-Al-Saud et al. (2018).
- Domain 800 h_N × 6 h_N.
- MAXLEVEL 14, giving Δ/h_N = 0.0488 (about 20 cells per h_N). This was chosen from a grid study at levels 14 and 15.
- The last ~5% of the domain is excluded from analysis because of outlet effects (domain study at 400 vs 800 h_N).

**Inlet forcing (Chang et al. 1996)**

- Broadband: $u(0,z,t) = U_N(z)[1+F(t)]$ with $F = F_0\sqrt{2/M}\sum_m\cos(2\pi f_m t+\phi_m)$.
- F₀ = 0.025, M = 1000, f_cut = 28 Hz (dimensional), seed 12345.

**Database**

- 85 DNS case setups in `datasets/MANIFEST_dns.csv`.
- At the last file count, 29 cases had HDF5 output and 28 had closure-extraction configs.
- The ONR report quotes 25 production cases spanning Re 10–200, Ca 0.005–0.1, and β = 6.4°, 45°, 90°.
- **TODO:** fill the `status` column of the manifest (production / incomplete / test).

## Findings to date

**Correction sizes**

- ΔD_x is negligible.
- ΔM_x and ΔT_w concentrate near wave fronts and capillary ripples.

**Backflow**

- Local backflow (τ_w < 0) appears even at Re = 10, Ca = 0.01, β = 45° under broadband forcing.
- Dhopeshwar et al. (2025) report recirculation only for Re > 30, but they used monochromatic forcing.

**Baseline Lavalle ROM**

- Good at Re = 19.33, β = 6.4°, f = 1.5 Hz.
- At Re = 120, β = 45°, f = 10 Hz it underpredicts the amplitude and speed of the nonlinear waves.

**ΔM_x and ΔT_w**

- Neighbour-stencil MLPs reach a best validation NRMSE of about 0.089 and 0.084.
- Models given only local inputs fail.
- Supplying the first derivatives (h_x, q_x) is sufficient.
- A single-hidden-layer MLP with 16 neurons performs well (`ml/functional_mlp`).

**ΔP_x**

- NRMSE is about 0.54 even with neighbour stencils.
- Correlates better with curvature κ = h_xx/(1+h_x²)^{3/2} (`ml/thinfilm_mlp/augment_curvature.py`).
- May need non-Markovian (Mori–Zwanzig) memory.

## Next steps

1. Find compact functional forms for ΔM_x and ΔT_w and test how they generalise across Re, Ca, and β.
2. Model ΔP_x using curvature and other geometric features, plus temporal-memory models.
3. Embed the corrections in `rom/lavalle-cpp` (`include/closures/closure_model.hpp`) and compare a posteriori against DNS:
   - wave amplitude, wave speed, and ripples;
   - stability;
   - grid independence (Δx-aware MLP inputs, fixed-physical-length CNN stencils);
   - linear stability of the augmented model.

## Paper 1 (draft: "A machine-learning augmented reduced model for thin liquid films down an inclined plane")

**Written:**

- Introduction
- DNS methodology
- IBL and Lavalle closures
- DNS extraction

**Still to do:**

- Validation against Liu & Gollub (1994) and Denner et al. (2018).
- Noble–Vila text.
- Results: a priori, a posteriori, and linear stability.
- Conclusions.
- Missing figures.
- Finish Appendix A.6.

**Known fixes needed:**

- Eq. (5): replace "f ρ_l" with α ρ_l.
- §III.B says "3 groups" but lists four.
- The hydrostatic factor in Eq. (A22).
- Typesetting of Eq. (20).
- The Ca range in the draft (0.001–0.06) differs from the report (0.005–0.1).
