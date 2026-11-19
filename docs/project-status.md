# Project status

_Last updated: 2026-11-19: broadband held-out a posteriori results (seed 0): MLP σ 0.5 L10 best on 4 of 6 completed cases (tied with V1 on Re 20), 2 MLP runs blew up (loss of hyperbolicity); seed-1 ROM tests confirm σ 0.5 > σ 0.25. Previous: 2026-11-18: all seed-0 flat-film-exact MLP runs analysed (σ 0.5 L10 best on the oracle), ROM speed-up (6.8×), broadband held-out a posteriori runs and seed-1 training submitted; priority set to the most accurate ML closure, first draft mid-December. Previous: 2026-11-17 (evening), flat-film-exact MLP oracle a posteriori and phase diagnostic; earlier on 17 Nov: comb-forced DNS results, cache v2 built, structured closure V1 trained, implemented in the ROM and run a posteriori on the oracle case (branch `structured-closure`). Previous: 2026-11-16, Orr–Sommerfeld reference and gate GB, Phase A; 2026-11-13, functional-form plan v2; submission moved to 15 Jan 2027._
Update this file in the same commit as any change that moves the science forward.

## Numerics

**ROM solver (`rom/lavalle-cpp`)**

- Noble–Vila augmented variable for capillarity.
- Rusanov flux with MUSCL reconstruction.
- RK4 time integration.
- A run takes about 15 min, versus about 30 h for DNS (roughly 120× faster).
- Builds from a clean clone since 30 Oct (`include/core/` had been hidden by a `core*` ignore rule): `module load gcc/14.2.0 hdf5 && make`.
- Learned-closure hook since 5 Nov: `{include,src}/ml/`, `./thinfilm input.txt oracle.txt ml.txt`; MLPs exported with `ml/functional_mlp/export_for_cpp.py` (see `rom/lavalle-cpp/README.md`). The oracle module (`{include,src}/oracle/`) supplies prescribed DNS corrections, initial state and inlet.

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
- Details and a per-case table: `docs/results/dpx-closure.md`.
- Correlates better with curvature κ = h_xx/(1+h_x²)^{3/2} (`ml/thinfilm_mlp/augment_curvature.py`).
- Extraction fixed on 10 Nov: the stored ΔP_x was dominated by Savitzky–Golay h_xxx noise (× 1/We) and by the interface pressure jump; new target `delta_Px_ML_v2` (σ = 0.5 h_N, liquid-pressure Leibniz form) closes the budget about 2× better for We ≥ 0.4. For We ≤ 0.1 no extraction closes the budget; those cases are out of ΔP_x training for now (`docs/results/dpx-closure.md`).
- In progress (10 Nov): (i) dense-profile budget check for We <= 0.1 (restarts with profiles every 0.01, jobs 50644567–70; `pipeline/extract_closures/profile_budget.py`, results in `$THINFILM_DATA/BasC/DNS/dense/<case>/run_*_budget/budget.csv`); (ii) ΔP_x v2 ablation, 4×64, 19 input groups (`ml/functional_mlp/sweep_config_dpx.json`, cache `cache_dpx_v2`, sweep `runs/dpx_v2_sweep01`; chain jobs 50645426–27 → 50645475 → 50645476; training array resubmitted 2 at a time as 50649593); evaluate with `ml/functional_mlp/EVALUATE.md`. **Result (12 Nov): the Markovian gate fails.** Best input sets: G_s0p5 = A + h_xxx + q_xx, and H_s0p5 = A + κ_x/We. Held-out R² is 0.81 (NRMSE 0.43); R² is 0.94–0.98 for We ≥ 0.8 but only 0.72 and 0.33 for We 0.2 and 0.4. The low-We cases also fail in validation (R² 0.29 and 0.50), so the next steps are the dense budget check, then memory (`docs/results/dpx-closure.md`). Figures (12 Nov) show that most of the low-We error is at x < 250: in Re80 Ca0.005 β6.4, 96% of the squared error is there, and R² is 0.92 for x ≥ 250. The error takes the form of short-wave ΔP_x spikes in the nearly flat upstream film. A possible cause is extraction near cell faces; test this before memory models (`docs/results/figures/dpx-closure/`). (iii) The broadband restart bug was fixed and the affected restarts resubmitted (50644571, 50645032–46); redo closure files for Re050, Re10 Ca0.01, Re10 Ca0.1, Re40 Ca0.1, Re10 Ca0.04 β90 when they finish (`docs/decisions.md`).
- May need non-Markovian (Mori–Zwanzig) memory; that is within Paper 1's scope (decision of 10 Nov). Plan of 10 Nov: robust extraction first, then a curvature ablation (4×64 MLP), then dense-in-time restarts (T_PROF = 0.05) for 5 cases if memory is needed.

**Budget closure check (30 Oct 2026).** Full write-up: `docs/results/budget-check.md`.

- The stored deltas match the raw integrals to machine precision, so C_Lav + ΔC = C_DNS exactly.
- Over all 22 closure files (t = 1000–2000 every 100), S_ML closes the lavalle-cpp discretisation to ≤ 4.5% of d_t q for 19 cases (1 h_N smoothing). The exceptions are low-We cases (Re10 Ca0.01 β45, Re80 Ca0.005 β6.4, Re20 Ca0.01 β45), where the one-step ∂ₜ is unreliable at the snapshots where the mass equation also fails. The per-snapshot mass residual is a cheap data-quality flag.

**Held-out evaluation of the 1×16 MLP (1 Nov 2026).** Full write-up: `docs/results/mlp-heldout-eval.md`.

- On the 7 held-out cases pooled NRMSE is 0.109 (ΔM_x) and 0.100 (ΔT_w); in the nonlinear region (x > 100, h > 1.1 or h < 0.9) it is 0.130 / 0.121 against ≈ 1.2 for Lavalle alone; peak amplitudes within 1.5–3% (ΔT_w Re200 β6.4: 8%).
- Weakest: Re80 Ca0.005 β6.4 and the other β = 6.4° case. The error tracks low Fr² and small correction size, not We (`docs/results/functional-form-structure.md`). Next: functional form.

**Sum-target MLPs for the *a posteriori* ablation (5 Nov 2026).** Full write-up: section "Sum target" in `docs/results/mlp-heldout-eval.md`. Two 1×16 MLPs predict ΔM_x + ΔT_w directly: inputs (h, h_x, q) and (h, h_x), each plus Re, Ca, β.

- Held-out nonlinear-region NRMSE (normalised by the std of the sum; Lavalle alone 1.235): 0.308 for (h, h_x, q) and 0.287 for (h, h_x); R² 0.938 / 0.946. Validation pooled R² 0.899 / 0.906.
- The sum is about twice as hard as either component (0.25 vs 0.11 pooled held-out) because ΔM_x and ΔT_w partly cancel. (h, h_x) is slightly better overall, within single-seed noise; q helps at Re80 Ca0.005 β6.4 (0.557 vs 0.672) and Re20 Ca0.01 β45.
- Both are weak for β = 6.4° (NRMSE 0.56–0.67).

**Functional-form search for ΔM_x and ΔM_x + ΔT_w (1 Nov 2026).** Full write-up: `docs/results/functional-form-structure.md`.
- **Phase 0 of plan v2 (13 Nov, preliminary, SG derivatives on the old cache):** a 3-term Tier-1 form, S ≈ 0.071 ∂ₓ[(q − h³)h²] + 0.088 q q_x/h + 0.120 ∂ₓ[(q − h³)q/h], matches the old 19-term library (LOCO 0.33, held-out 0.29) but fails a β = 6.4° holdout (1.30). Per-case Tier-1 fits reach 0.22 on average, so the gap is the case-to-case variation of the coefficients. Theory coefficients transfer only at low δ and low Ca, and the relaxation term has the opposite sign to theory (unresolved). Most cases sit in ordering-matrix row 2 (δ ≈ 3–30). Full write-up: `docs/results/functional-form-v2.md`.

- ΔT_w ≈ −0.64 ΔM_x (corr −0.975): the two cancel only partly, so forms are developed for the sum ΔM_x + ΔT_w. ΔM_x = ∂ₓΔF with ΔF ≈ F(h, q) (R² ≈ 0.997); it is nearly linear in h_x, q_x.
- Pooled sparse regression (t > 1000): ΔM_x has a 5-term form with no Re/Ca/β dependence (leave-one-case-out NRMSE 0.129, held-out 0.093, β = 6.4° holdout 0.127). The sum has no adequate sparse form yet (LOCO ≈ 0.35–0.4). Not yet compared with the MLP on the same data. These fits used t > 1000; since 10 Nov training uses the entire record (`docs/decisions.md`).

**Mesh sensitivity, MAXLEVEL 14 vs 15 (3 Nov 2026).** Full write-up: `docs/results/mesh-sensitivity.md`. Re120 Ca0.01 β45, 10 Hz, t = 550–650 (level 15 stops at t = 650), crest-aligned.

- Level 15 waves are 1.2% faster (celerity 1.937 vs 1.913); crest height and wavelength agree.
- ΔM_x and ΔT_w agree (peaks within 1%, same position relative to the crest, correlation 0.998). Raw ΔP_x and ΔD_x differ because of grid-scale extraction noise; after a common Gaussian smoothing S_ML agrees to about 10%.
- Caveats: transient (unsaturated) window; DT = 0.003 cap at level 14, CFL-limited at level 15.

**New DNS runs (3 Nov 2026).** 8 white-noise cases farthest (in log Re, log Ca, log sinβ) from existing data were submitted on `bala1s-b` (4 days, 16 CPUs): Re020_Ca0_001_theta6_4, Re010_Ca0_001_theta90, Re020_Ca0_005_theta6_4, Re020_Ca0_002_theta90, Re040_Ca0_002_theta6_4, Re010_Ca0_040_theta90, Re080_Ca0_060_theta45, Re020_Ca0_020_theta6_4. Re10 β6.4° was excluded (linearly stable); the Re20 β6.4° cases are close to marginal and may not saturate. Refresh the manifest when they finish.

**Orr–Sommerfeld reference solver (16 Nov 2026).** Full write-up: `docs/results/orr-sommerfeld.md`; code `analysis/orr_sommerfeld/` (plan `docs/plans/structured-closure-os-plan.md`, B.1–B.3).

- Verified against the Benney limit (error O(k²)), Re_c = (5/6) cot β, and the published OS saddle points of Brevdo et al. (1999) (Kalliadasis et al. Table 7.1), which are matched to the printed digits.
- Re120 Ca0.01 β45: Lavalle's phase speed is within 3% of OS. Its temporal growth is 39% low at the OS maximum (k ≈ 0.3). Its spatial growth at the 10 Hz forcing (ω = 0.134) is only 3% low, but 37% low at ω = 0.45.
- B.4 (all 85 manifest cases). Lavalle's linear growth error depends on δ_lin = k_max Re (OS): it is 7–18% too high for δ < 5 and 25–45% too low for δ > 30. The 28 Hz forcing cut-off lies below the most amplified spatial frequency in 13 of the 36 DNS cases, and below the OS cut-off f_c in 31 of them.

**OS B.5–B.8 and gate GB (16 Nov 2026).** See `docs/results/orr-sommerfeld.md`.

- DNS against OS (oracle case, 10 Hz): spatial growth agrees to 1% and phase speed to 1.5% for x = 60–130. The linear region is x ≈ 60–150. The 20 Hz harmonic is bound, not free.
- Extracted ΔM_x and ΔT_w match the exact OS-mode values to 3–6% in the linear region, which validates the extraction pipeline. ΔM_x and ΔT_w are each mostly in quadrature with h and cancel to 10–40%.
- ROM linear test (frequency-comb inlet): the numerics reproduce analytic Lavalle to about 1%. The current ML closures are 20–28% off OS where they stay linear, and at Re 120 and 200 they go nonlinear through excess growth (separate models) or a shifted base state plus a spurious uniform equilibrium at h ≈ 2.9 (sum model). Both MLPs have spurious uniform equilibria at h ≈ 2–3. It is not an inlet effect: the drifting first cell in ROM outputs is a write-out artefact (the stored inlet cell is integrated but never used).
- Four linear coefficients (a, b, c, d), with D = D_OS imposed and fitted over the forced band, match OS growth to within 10% in all 85 cases. Three are not enough for δ > 5. Proposed D2: L2 with (a, b, c, d).

**Structured closure, Phase A diagnostics (16 Nov 2026).** Full write-up: `docs/results/structured-closure.md`; scripts `scripts/structured-closure/`.

- A.1: neither current MLP preserves the uniform film. S₀ ≠ 0 (up to 0.5 × 3/Re), and the uniform-film slope shifts the kinematic speed by O(1): +0.44 for the sum model and −0.72 for the separate models in the oracle case. The cause is that `flat_mask` removes |h − 1| < 0.05 from training. The separate models' linear part doubles the most-amplified growth.
- A.2: the sum model lies on the DNS crest c(h_max) curve, so its celerity error follows from its amplitude error. The separate models are 3–4% fast at equal height.
- A.3: in all seven cases (δ 2–65), F = −S is negative on wave fronts and positive on backs. Its projection on q′_x has the same sign everywhere and is dominated by the front. R² is only 0.1–0.26.
- A.4: in the linear region of the oracle case, the fitted DNS correction moves Lavalle onto OS at 10 Hz (phase speed 1.84 against 1.85, growth within 5%). The individual (a, b, c) are not identifiable from a single frequency.

**Comb-forced DNS and cache v2 (17 Nov 2026).** See `docs/results/orr-sommerfeld.md` and `docs/decisions.md`.

- **Comb DNS against OS.** At Re 80, 120 and 200 (β 45°, Ca 0.01) the DNS matches OS over the whole forced band: growth median 2–3%, phase speed 1–1.5% below OS at every frequency, and extracted ΔM_x and ΔT_w per frequency within 1–3%. Re 40 Ca 0.04 agrees in x = 30–100. At Re 20 the needed correction is smaller than the measurement scatter. Mean h is 0.3–0.55% below Nusselt (unexplained).
- **Cache v2:** 36 cases, full record, D_G(σ) derivatives for σ = 0.25, 0.5 and 1. Three partial cases fail the ∂ₓΔF check and are excluded from training.

**Structured closure V1 (17 Nov 2026).** Full write-up: `docs/results/structured-closure.md`, "Phase C"; code `ml/structured_closure/`, ROM `// STRUCTCLOSURE`.

- **The design properties hold.** The flat film is exactly preserved, the equations are hyperbolic for every state, there is no stiffness, and the ROM reproduces the trained linear dispersion to 3–4%.
- **The OS constraint (on the dispersion relation, at 85 points, λ_lin = 10) works:**
  - growth error against OS 8–11% in the ROM, against Lavalle's 27–36% (Re 120–200);
  - without it, up to 10× the OS peak;
  - no cost in held-out NRMSE (0.26 at σ 0.25, 0.30 at σ 0.5).
- **Oracle a posteriori:**
  - as good as `ml_sum_hhx` for about 5 time units, then drifts faster;
  - best structured: σ 0.25, h error 0.60 against 0.47 at t = 50;
  - the error is dominated by wave phase drift.
- **Diagnosis.**
  - Pointwise fit is not the issue (NRMSE about 0.12 on the oracle case for all closures). The closure's net power into the waves is a ~0.1% residual of large front and back terms, and every learned closure gets its sign wrong.
  - A power penalty fixes the a priori power but not the a posteriori error.
- **Flat-film-exact MLP (evening, σ 0.25, λ_lin 0/1/10, one seed):** S = N(h, q, h_x, q_x, h_xx) − N(h, h³, 0, 0, 0) with the OS constraint.
  - A priori on the oracle case: NRMSE 0.05 (half of `ml_sum_hhx` and V1) and the right sign of net power (about half the DNS damping) for λ_lin 0 and 10.
  - A posteriori: no crest overgrowth (L10 max h within 4% to t = 100); h error 0.59 at t = 50 (L10; `ml_sum_hhx` 0.47), but 0.58–0.70 at t = 100 against 1.25–1.54 for `ml_sum_hhx` and V1.
  - New failure: waves 3–6% too slow from the start (wave-speed error). After removing the shift, every learned closure keeps a shape error of about 0.3 above the exact closure (wave backs, substrate/ripples).
- **Flat-film-exact MLP, all seed-0 runs (18 Nov):**
  - σ 0.5 L10 is the best learned closure on the oracle so far: h error 0.39 at t = 50 and 0.62 at t = 100 (`ml_sum_hhx` 0.47 / 1.26, exact 0.19 / 0.55); crests 6–10% high from t ≈ 50.
  - Comb: linear growth within 7–11% of OS (median) at Re 120–200 for λ_lin 1 and 10 (Lavalle 27–39%). Without the constraint (L0) the Re 200 waves are damped (47% error), so the constraint is needed.
  - Dense broadband a posteriori (10 time units from the DNS state, DNS inlet): the new MLPs have the lowest error on both held-out cases (Re 20 Ca 0.01 β 45: 0.11–0.14 at t = 10 against 0.22 for `ml_sum_hhx` and 0.28 for Lavalle; Re 120: 0.25–0.30 against 0.42 and 0.59). `ml_sum_hhx` is better on Re 40 Ca 0.002 (training case, We 0.08).
  - ROM speed-up: the flat-film term N(h, h³, 0, 0, 0) is tabulated in h and the cell loop uses OpenMP; 0.164 → 0.024 s per step (4 threads), identical results to 1.5e-9.
- **Broadband held-out a posteriori, seed 0 (19 Nov; gate GD):** 7 held-out cases, ROM from a flat film with its own inlet noise, statistics over t 1000–2000.
  - MLP σ 0.5 L10 is best on 4 of its 6 completed cases and ties with V1 on Re 20: h-PDF distance 0.04–0.07 on those five (Lavalle 0.11–0.29), crest ratio 0.95–1.00 (Lavalle 0.72–0.94).
  - Failures: 2 MLP runs blew up with loss of hyperbolicity (σ 0.25 on Re40 Ca0.04 β45, σ 0.5 on Re80 Ca0.01 β90); σ 0.5 is worse than Lavalle on Re40 Ca0.04 (crests 16% high); at Re 200 β 6.4 its waves are twice too long.
  - Seed 1 (ROM tests): comb growth as seed 0; oracle h error 0.37 / 0.75 at t = 50 / 100 for σ 0.5 (seed 0 0.39 / 0.62).
- **Bug fixed:** the truncated Gaussian second derivative is not zero on constants (cache-v2 h_xx carries −3.5e-3·h at σ 0.5).

## Next steps

**Priority (18 Nov 2026, Krishna):** first draft of Paper 1 by mid-December. Make the most accurate ML closure work, judged a posteriori (closed-loop ROM against the DNS and the exact closure); minimise work on functional forms and symbolic regression (plan Phase E). Slides on hold.

**Running (19 Nov):** seed-1 broadband runs of the two λ_lin 10 MLPs (job 51936778, 2 at a time). Seed-0 broadband, seed-1 training and seed-1 ROM tests are done.

**Current work (17 Nov 2026): structured-closure plan.** The plan is `docs/plans/structured-closure-os-plan.md`. The task tracker is `docs/plans/structured-closure-progress.md`. The explainers for Krishna are `docs/explainers/2026-11-16-structured-closure-explainer.md`, `2026-11-17-structured-closure-explainer.md` and `2026-11-18-flat-mlp-explainer.md`.

- **Done:**
  - the OS reference and linear tests (B.1–B.8, gate GB, D2 = L2);
  - Phase A;
  - the comb DNS (D3);
  - cache v2;
  - Phase C V1 training (one seed), the ROM implementation (C.4–C.10) and the first oracle a posteriori runs.
- **Open questions:**
  - robustness: 2 of 14 MLP broadband runs lose hyperbolicity (whether this is seed-dependent comes from seed 1);
  - seed spread at long times (oracle t = 100 error varies by about 0.15 between seeds);
  - V2 (coefficients depending on q/h³), or more data (11 extra cases), for wave speed.
- **The functional-form plan is secondary.** It serves as the monomial baseline. Phase E closed forms are de-emphasised (18 Nov). It shares cache v2 and the closed-loop harness.


**Oracle *a posteriori* (31 Oct 2026).** Full write-up: `docs/results/oracle-a-posteriori.md`. Re120 Ca0.01 β45, 10 Hz, 100 time units.

- Lavalle alone: h error ≈ 0.6 (relative to std h) after 5 time units, waves underpredicted (max h 1.45 vs 2.1).
- ΔT_w + ΔM_x: error ≈ 0.2 for 50 time units, wave height restored, stable for 100.
- ΔT_w alone diverges (ΔT_w and ΔM_x largely cancel). All four terms stop at elapsed time 30 with h = 0 in the outlet cell (x = 800) while the interior still matches the DNS; the cause is not identified (the "all terms" error is also worse than ΔT_w + ΔM_x from elapsed 10). ΔT_w + ΔM_x drifts late (error 0.19 at 50, 0.55 at 99.9). See "Where it fails" in the write-up.
- Next: ΔM_x/ΔT_w correlation, smoothed ΔP_x, ROM re-initialised from the DNS every 5–10 time units.

**Learned-closure *a posteriori* (5 Nov 2026).** Full write-up: `docs/results/ml-a-posteriori.md`. Same case and window as the oracle; DNS initial state and inlet.

- Two closures: one MLP for ΔM_x + ΔT_w on (h, h_x), and separate ΔM_x, ΔT_w MLPs on (h, q, h_x, q_x). Both stable for 100 time units.
- For the first 10 time units both match the oracle (h error 0.06 at 1, 0.22 at 10; oracle 0.06 / 0.17; Lavalle 0.12 / 0.61) and restore the wave height.
- Then they drift. The sum model over-amplifies the crests (max h 2.38 at 50 and 2.50 at 99.9 against 2.1) and leads the DNS; error 0.47 at 50 (Lavalle 0.59), 1.26 at 99.9 (Lavalle 1.07). The separate models fail sooner (1.03 at 50, 2.05 at 99.9).
- Next: smoothed derivatives (`hx_smooth`), DNS re-initialisation every 5–10 time units, closed-loop-aware training.
- **Adding the ΔP_x v2 MLP (12 Nov)** (G_s0p5, and H_s0p5 with the sum model, plus the exact-curvature term h(κ_x − h_xxx)/We; `rom/runs/ml_*_px*`): **halves the short-time error but destabilises the waves.** h error at elapsed 1 is 0.027 against 0.057 (sum model alone) and 0.060 (oracle). From elapsed 5 the crests overshoot at the steep fronts; from elapsed 10 the runs are worse than without ΔP_x and from 30 worse than Lavalle (1.97–2.40 at 99.9). The waves become flat-topped humps (h up to 3.25) on a thin substrate. G and H behave alike. Next: rerun without the curvature term and with ΔP_x limited outside the training range.

See the schedule at the end of this file. Phase 0 (2–8 Nov) covers the oracle runs, the held-out evaluation on one shared data configuration, and the closure hook in `rom/lavalle-cpp/include/closures/closure_model.hpp` (the oracle module in `rom/lavalle-cpp/{include,src}/oracle/` and the learned-closure hook in `{include,src}/ml/` are done).

## Paper 1

Title: "A machine-learning augmented reduced model for thin liquid films down an inclined plane" (Krishnaprasad, Moghaddam, Balachandar).

**Where the source lives.** Overleaf, cloned on the laptop at `D:\Papers\thinfilm-paper-01` (git remote git.overleaf.com). It is not on HiPerGator.

**Targets.**
- Journal: IJHMT or AI Thermal Fluids. Switching from revtex/PRFluids to the journal template is still to do, and the introduction should be framed around heat-transfer applications.
- First draft: mid-December 2026 (set 18 Nov). Submission: target Fri 15 Jan 2027 (moved from 15 Dec on 13 Nov; draft to co-authors about 4 Jan).
- APS DFD talk on 22 Nov 2026; soft results freeze on Sun 15 Nov.

**Scope.** A comprehensive ML-augmented reduced model for falling films with a passive gas: closures for ΔM_x, ΔT_w and ΔP_x (ΔP_x with curvature inputs, and non-Markovian memory closures if needed). Paper 1 includes:
- *a priori* evaluation on held-out cases;
- an oracle *a posteriori* run;
- the ROM coupled with the learned closures;
- a hyperbolicity check.

Paper 2: gas-sheared films. Paper 3: heated films (heat transfer).

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

## Schedule to 15 Jan 2027 (structured-closure plan §4; FF plan in parallel as baseline)

| Dates | Work |
|---|---|
| 16–22 Nov | Plan intake; OS solver and B.1–B.8; Phase A; gate GB (D2 = L2); comb DNS; cache v2. APS DFD talk 22 Nov |
| 23–29 Nov | Comb DNS write-up; Phase C training (C.1–C.3) on cache v2 |
| 30 Nov–6 Dec | Solver integration and tests (C.4–C.10); Phase D on the oracle case; **gate GD (~6 Dec): which closure Paper 1 carries** |
| 7–20 Dec | Phase D broadband; first draft mid-December; Phase F rollout if approved (Phase E closed forms de-emphasised, 18 Nov) |
| From 4 Jan | Draft to co-authors; submission 15 Jan 2027 |


### Functional-form plan thread (secondary since 16 Nov; runs in parallel)

From `docs/plans/functional-form-plan.md` §5 (13 Nov 2026). The other threads (ΔP_x, memory models) run alongside.

| Dates | Work |
|---|---|
| 13–15 Nov | Functional-form Phase 0 (talk slice); gate G0 Sun 15 Nov |
| 16–21 Nov | Phase 1 infrastructure (cache v2, analytic closure in `lavalle-cpp`, closed-loop harness; jobs queued before the talk); APS DFD slides |
| 22 Nov | APS DFD talk |
| 23 Nov–6 Dec | Phase 2 core results; gate G2 ~2 Dec |
| From 7 Dec | Phase 3, iterating with the writing (lighter over the holidays); no hard results freeze |
| ~4 Jan | Draft to co-authors |
| 4–14 Jan | Revisions; journal template |
| 15 Jan 2027 | Submit |

Functional-form plan (current): `docs/plans/functional-form-plan.md`. Older plan document (claude.ai, 29 Oct): https://claude.ai/artifact/5nVWtUcjG4BFfRooFs7grk
