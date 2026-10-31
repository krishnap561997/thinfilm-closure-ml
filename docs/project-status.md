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

**Budget closure check (30 Oct 2026, `pipeline/extract_closures/check_momentum_budget.py`)**

- Uses the (t, t−dt) pair in each snapshot for ∂ₜ. The stored deltas match the raw integrals to machine precision, so C_Lav + ΔC = C_DNS exactly.
- Restricting the DNS state to the ROM grid (N = 4000) and evaluating a Python replica of the lavalle-cpp q-RHS, d_t q − RHS_ROM ≈ S_ML:
  - inertial cases close well; at 1 h_N smoothing E1/E0 = |d_t q − RHS_ROM − S_ML| / |d_t q − RHS_ROM| is 0.04 (Re120 Ca0.01 β45, the oracle target, corr 0.999) and 0.10 (Re200 Ca0.01 β6.4);
  - Re40 Ca0.02 β6.4: 0.35, but the Lavalle misfit itself is only ~7% of d_t q there;
  - **Re10 Ca0.01 β45 (We = 0.1) does not close** (E1/E0 ≈ 0.9). Mass closes over 1 h_N windows (2%), but the one-step ∂ₜh has grid-scale jitter (15% of its variance below 1 h_N), and the DNS momentum budget leaves 13% of d_t q unexplained even at 1 h_N. Suspect ∫∂ₓp near the interface at large σ, or interfacial terms omitted by the passive-gas form. Treat strongly capillary low-Re cases with caution in training and in the oracle.
- **Full sweep (all 22 closure files, t = 1000–2000 every 100; results in `ClosureData/budget_check/`):**
  - at 1 h_N smoothing, the unexplained remainder E1 is ≤ 4.5% of d_t q for 19 of 22 cases; corr(E0, S_ML) ≥ 0.99 for 15. The exceptions are Re10 Ca0.01 β45 (16%), Re80 Ca0.005 β6.4 (9%) and Re20 Ca0.01 β45 (7%), all low We (0.1–0.4);
  - where it fails, it fails at the same snapshots where the mass equation fails (e.g. Re80 Ca0.005 β6.4 at t = 1500–1600: mass 0.24, E1/E0 0.92; at t = 1000: mass 0.01, E1/E0 0.28). Mass involves no closure, so the one-step ∂ₜ is unreliable there. The per-snapshot mass residual is a cheap data-quality flag;
  - β = 6.4° cases show E1/E0 ≈ 0.1–0.45 only because Lavalle is already close there (E0 ≈ 7–12% of d_t q); E1 itself is ≤ 3% of d_t q.
- Rusanov numerical dissipation is ~2% of d_t q for Re120; Noble–Vila capillarity with r = √(h/We) h_x matches h h_xxx/We to ~2–7%.

## Next steps

**Oracle *a posteriori* (31 Oct 2026), Re120 Ca0.01 β45, 10 Hz, t = 1800–1900.** Data in `$THINFILM_DATA/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/`: `run_t1800/` (DNS), `closures/`, `rom/` (`summary.txt`, `summary.csv`, `runs/`).

- **DNS restart** (`dns/oracle/filmflow_restart_mono.c`, job 48362419, 1 h 38 min): 10⁴ profiles every 0.01. It reproduces the original L800 run, with an h difference of 0.3%, 0.6% and 0.9% of std(h) at t = 1810, 1820 and 1830.
- **Time resolution:** linear interpolation over 0.02 has ≤ 0.06% error for ΔM_x, ΔT_w, h and q. ΔP_x is the exception: 14% on the DNS grid and 5% on the ROM grid, so it carries high-frequency temporal content.
- **ROM oracle** (`rom/lavalle-cpp` oracle module, N = 4000, dt = 0.005, DNS initial state and inlet; corrections are cell averages interpolated linearly in time). h error relative to std(h_DNS), 20 < x < 760:

  | elapsed | Lavalle | +ΔT_w | +ΔT_w+ΔM_x | all four |
  |---|---|---|---|---|
  | 1 | 0.115 | 0.283 | 0.060 | 0.014 |
  | 5 | 0.536 | 1.63 | 0.150 | 0.164 |
  | 20 | 0.590 | crashed at 7.9 | 0.199 | 0.555 |
  | 50 | 0.585 | – | 0.192 | crashed at ≈30 |
  | 99.9 | 1.07 | – | 0.550 | – |

  - Lavalle alone underpredicts the waves: max h is 1.45 against 2.1 for the DNS.
  - ΔT_w + ΔM_x restores the wave height (max h 2.10 vs 2.12 at elapsed time 50) and stays stable for 100 units; the error rises again near the end (cause not yet identified, possibly phase drift).
  - ΔT_w alone crashes: h → 0 and the hyperbolicity check fails. ΔT_w and ΔM_x appear to largely cancel, so they should be applied (and possibly learned) together.
  - All four terms is the most accurate early on but grows faster and crashes at t ≈ 1830. ΔP_x is the suspect, given its high-frequency content (not yet confirmed).
- **Next:**
  1. correlation of ΔM_x and ΔT_w (train on the sum?);
  2. rerun "all" with ΔP_x smoothed, and ΔT_w+ΔM_x+ΔP_x without ΔD_x;
  3. ROM re-initialised from the DNS every 5–10 units, to separate closure error from phase drift.

See the schedule at the end of this file. Phase 0 (2–8 Nov) covers the oracle runs, the held-out evaluation on one shared data configuration, and the closure hook in `rom/lavalle-cpp/include/closures/closure_model.hpp`.

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
