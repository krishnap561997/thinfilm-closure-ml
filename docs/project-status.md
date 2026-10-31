# Project status

_Last updated: 2026-10-31: oracle a posteriori results (Re120 Ca0.01 β45, 10 Hz)._
Update this file in the same commit as any change that moves the science forward.

## Numerics

**ROM solver (`rom/lavalle-cpp`)**

- Noble–Vila augmented variable for capillarity.
- Rusanov flux with MUSCL reconstruction.
- RK4 time integration.
- A run takes about 15 min, versus about 30 h for DNS (roughly 120× faster).
- Builds from a clean clone since 30 Oct (`include/core/` had been hidden by a `core*` ignore rule): `module load gcc/14.2.0 hdf5 && make`.
- No closure hook yet: `ClosureModel::evaluate(h, q)` works point by point. The oracle needs a field- and time-dependent correction added in `evaluate_rhs` (see `rom/lavalle-cpp/README.md`).

**DNS (`dns/production`)**

- Basilisk C, CLS-VOF with the integral surface-tension model of Abu-Al-Saud et al. (2018).
- Domain 800 h_N × 6 h_N.
- MAXLEVEL 14, giving Δ/h_N = 0.0488 (about 20 cells per h_N). This was chosen from a grid study at levels 14 and 15.
- The last ~5% of the domain is excluded from analysis because of outlet effects (domain study at 400 vs 800 h_N).
- In-situ 1D output (`dns/include/column_profiles.h`) writes per-column h, q, ∫u², ∫₀ʰ∂ₓp, ∂_z u|₀ and the interface velocity at t and t−dt. That is ~1 MB per output against 133 MB for a 2D snapshot, and after `profiles_to_closures.py` it matches the 2D extraction to round-off.

**Inlet forcing (Chang et al. 1996)**

- Broadband: $u(0,z,t) = U_N(z)[1+F(t)]$ with $F = F_0\sqrt{2/M}\sum_m\cos(2\pi f_m t+\phi_m)$.
- F₀ = 0.025, M = 1000, f_cut = 28 Hz (dimensional), seed 12345.
- Monochromatic (oracle) runs: u = U_N(z)[1 + 0.03 sin(2π f t)] with f = 10 Hz. For Re120 β45 that is f = 0.0214 nondimensionally, a period of about 47 time units.

**Database**

- 85 DNS case setups in `datasets/MANIFEST_dns.csv`.
- Manifest refresh (30 Oct): 18 complete, 10 partial, 57 not started; 22 cases have extracted closure files in `BasC/DNS/param_space/ClosureData/`.
- Since 30 Oct, DNS cases live in `$THINFILM_STORE/datasets/BasC/DNS/param_space/` (moved from `BasC/param_space`).
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
- A single-hidden-layer MLP with 16 neurons performs well (`ml/functional_mlp`). Approximate validation NRMSE (√val_loss) for ΔM_x / ΔT_w:
  - local (h, q) only: 0.95 / 0.95;
  - + h_x, q_x: 0.16 / 0.21;
  - + second derivatives: 0.15 / 0.21;
  - q_x matters more than h_x.
- The deep stencil MLP (4×64, `ml/thinfilm_mlp` sweep_01, best at J=3, s=5, half-width 0.73 h_N) reaches pooled NRMSE 0.089 / 0.083 (R² 0.992 / 0.993), case-averaged 0.101 / 0.087.
- **Caveat:** the two experiments use different configs (15 vs 19 development cases, 80/10/10 vs 65/15/20 chronological split, flat tolerance 0.05 vs 0.1). All metrics so far are on validation data from the development cases; there is no held-out evaluation yet.

**ΔP_x**

- NRMSE is about 0.54 even with neighbour stencils (best stencil: J=6, s=3, half-width 0.88 h_N; pooled R² 0.70).
- Predictability depends on Ca. R² is 0.93–0.98 for Ca ≥ 0.06 (Re ≤ 40), but about 0 for Re = 10 with Ca ≤ 0.01 and for Re = 40 with Ca = 0.002. Across the 20 development cases, R² correlates with log Ca at 0.65 and with log Re at 0.31. This points to capillarity and supports curvature as an input.
- Correlates better with curvature κ = h_xx/(1+h_x²)^{3/2} (`ml/thinfilm_mlp/augment_curvature.py`).
- May need non-Markovian (Mori–Zwanzig) memory.

**Budget closure check (30 Oct 2026).** Full write-up: `docs/results/budget-check.md`.

- The stored deltas match the raw integrals to machine precision, so C_Lav + ΔC = C_DNS exactly.
- Over all 22 closure files (t = 1000–2000 every 100), S_ML closes the lavalle-cpp discretisation to ≤ 4.5% of d_t q for 19 cases (1 h_N smoothing). The exceptions are low-We cases (Re10 Ca0.01 β45, Re80 Ca0.005 β6.4, Re20 Ca0.01 β45), where the one-step ∂ₜ is unreliable at the snapshots where the mass equation also fails. The per-snapshot mass residual is a cheap data-quality flag.

## Next steps

**Oracle *a posteriori* (31 Oct 2026).** Full write-up: `docs/results/oracle-a-posteriori.md`. Re120 Ca0.01 β45, 10 Hz, 100 time units.

- Lavalle alone: h error ≈ 0.6 (relative to std h) after 5 time units, waves underpredicted (max h 1.45 vs 2.1).
- ΔT_w + ΔM_x: error ≈ 0.2 for 50 time units, wave height restored, stable for 100.
- ΔT_w alone and all four terms are unstable (ΔT_w and ΔM_x largely cancel; ΔP_x is the suspect for the all-terms blow-up).
- Next: ΔM_x/ΔT_w correlation, smoothed ΔP_x, ROM re-initialised from the DNS every 5–10 time units.

See the schedule at the end of this file. Phase 0 (2–8 Nov) covers the oracle runs, the held-out evaluation on one shared data configuration, and the closure hook in `rom/lavalle-cpp/include/closures/closure_model.hpp` (the oracle module in `rom/lavalle-cpp/{include,src}/oracle/` is separate and already done; the learned-closure hook is still to do).

## Paper 1

Title: "A machine-learning augmented reduced model for thin liquid films down an inclined plane" (Krishnaprasad, Moghaddam, Balachandar).

**Where the source lives.** Overleaf, cloned on the laptop at `D:\Papers\thinfilm-paper-01` (git remote git.overleaf.com). It is not on HiPerGator.

**Targets.**
- Journal: IJHMT or AI Thermal Fluids. Switching from revtex/PRFluids to the journal template is still to do, and the introduction should be framed around heat-transfer applications.
- Submission: 15 Dec 2026.
- APS DFD talk on 22 Nov 2026; talk results freeze on 15 Nov.

**Scope.** Markovian closures for ΔM_x, ΔT_w and ΔP_x, with ΔP_x taking curvature as an input. Paper 1 includes:
- *a priori* evaluation on held-out cases;
- an oracle *a posteriori* run;
- the ROM coupled with the learned closures;
- a hyperbolicity check.

Memory models and new architectures go in a Paper 2 only if Paper 1 leaves a gap.

**Planned structure and status (29 Oct 2026):**

| Section | Status |
|---|---|
| 1 Introduction | partly drafted (missing references, heat-transfer framing) |
| 2.1 Governing equations | drafted |
| 2.2 Setup | partly drafted (parameter-space figure, case counts) |
| 2.3 Validation | to write (grid and domain studies from the ONR report, Liu & Gollub) |
| 3.1–3.3 Depth-integrated model, Lavalle closures, extraction | drafted (Noble–Vila text and the fixes below still needed) |
| 4 ML closure models (data protocol, inputs, networks) | to write; currently inside IV.A |
| 5.1 *A priori* (Sec. IV.A) | drafted 29 Oct with red TODOs: held-out eval, same-config comparison, J=0 baseline on the same samples, curvature sweep |
| 5.2 Oracle *a posteriori* | first results 31 Oct (ΔT_w+ΔM_x stable, error 0.6 → 0.2; ΔT_w alone and all-terms unstable) |
| 5.3 ML-augmented ROM (monochromatic + broadband statistics, grid study, cost) | needs new results |
| 5.4 Stability (flux-Jacobian eigenvalues; dispersion relation if time) | needs new results |
| 6 Conclusions | to write |
| Appendix A | partly drafted (remove review notes, finish A.6) |

**Known fixes needed:**

- Eq. (5): replace "f ρ_l" with α ρ_l.
- §III.B says "3 groups" but lists four.
- Eq. (A22): the hydrostatic factor.
- Eq. (20): the 1/Re multiplies only 3q/h².
- The draft's Ca range (0.001–0.06) differs from the report (0.005–0.1).

## Schedule to 15 Dec 2026

| Dates | Work |
|---|---|
| 2–8 Nov | Closure hook in `rom/lavalle-cpp`; oracle runs; held-out evaluation (all three terms, one shared data config); launch burst runs for the partial DNS cases (4–6 at a time, held-out gaps first) |
| 9–15 Nov | Retrain on ROM-consistent derivative inputs (ΔF flux target, ΔT_w pointwise, ΔP_x with curvature); first coupled ROM run (ΔT_w, then ΔM_x). **Talk results freeze 15 Nov** |
| 16–21 Nov | APS DFD slides and practice talk (talk is 22 Nov) |
| 23 Nov–6 Dec | *A posteriori* with all three terms; validation section; hyperbolicity check. Fallback if not stable by 30 Nov: couple only ΔT_w and show the rest *a priori* |
| 7 Dec | Draft to co-authors |
| 7–14 Dec | Revisions; journal template |
| 15 Dec | Submit |

Plan document (claude.ai): https://claude.ai/artifact/5nVWtUcjG4BFfRooFs7grk
