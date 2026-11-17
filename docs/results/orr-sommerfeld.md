# Orr–Sommerfeld reference solver (tasks B.1–B.3)

_16 Nov 2026. Plan: `docs/plans/structured-closure-os-plan.md` Part I. Code: `analysis/orr_sommerfeld/`. Numbers: `python analysis/orr_sommerfeld/verify.py` (≈5 s), output in `$THINFILM_STORE/analysis/orr_sommerfeld/verification.json`._

Status: B.1–B.8 are done (16 Nov 2026). Gate GB is ready for review; see the summary at the end.

## Solver

- The equations are those of plan §2.2 (passive gas, project Nusselt scaling), with unknowns (φ(z_j), η̂).
- **Temporal problem:** a Chebyshev collocation generalised eigenproblem. The rows are balanced, which leaves the eigenvalues unchanged; without balancing, QZ round-off grows with N and the N = 140 result is *worse* than N = 60 (it agrees only to 1e-4 at Re 120, Ca 0.01). Spurious modes are removed by requiring agreement between N = 48 and N = 64 (to 1e-5).
- **Polish:** each selected eigenvalue is refined by shooting. Two solutions are integrated from the wall with DOP853 (rtol 1e-12), and a secant iteration is run on the 2×2 surface-condition determinant, multiplied by (c − U_s) to remove the pole. Collocation and shooting are independent methods, and they agree to ≤ 8e-8.
- **Spatial problem:** Newton on k c(k) = ω for complex k. It starts from the Gaster estimate k_i ≈ −ω_i/c_g and is continued in ω. The mode is followed with `c_guess`. The residual is at machine precision.
- **Saddle points:** Newton on dω/dk = V, used for the published comparison below.
- **Linearised Lavalle model:** analytic. The temporal form is the quadratic in c from plan Appendix A. The spatial form is the quartic in k obtained with c = ω/k: (i/We)k⁴ + (2ω/Re + iG)k² + (2iω − 9/Re)k + 3ω/Re − iω² = 0.

## Verification (B.3)

| Check | Result |
|---|---|
| 1. Benney limit, Re 0.5–10, Ca 0.1, β 45° | Relative growth error at k = 0.0025: 2e-5 (Re 0.5) to 1.6e-3 (Re 10). The error falls by a factor of 3.9–4.0 for each halving of k, i.e. O(k²) as expected. c_r → 3. |
| 2. Re_c = (5/6) cot β, at k = 1e-3 | β 10°: 4.726085 vs 4.726068; β 45°: 0.833336 vs 0.833333; β 80°: 0.146940 vs 0.146939. The remaining offset is O(k²). |
| 3. Published OS: Brevdo et al. (1999), via Kalliadasis et al. (2012) Table 7.1 (Re 133, β 4.6°, Γ 769.8) | All four tabulated saddle points (V = 0.575, 0.580; branches I, II) are reproduced to the printed digits; see the table below. |
| 4. Convergence | The polished c at k = 0.3 is the same for N = 32 to 80 to 1e-15. The plan §0 table is reproduced to every printed digit. |

**Scaling conversion for check 3.** The book's Nusselt scaling uses the velocity scale 3u_N (U = y − y²/2, kinematic speed 1). Converting to ours:
- k is unchanged;
- ω, c and V are 3 × the book values;
- Re and Ct = cot β are identical;
- We_book = Γ/(3Re)^{2/3} = σ/(ρ g sinβ h_N²) = 1/(3 Ca_ours). This reproduces the book's We = 41.46 at Re = 26.7.

At Re 133 this gives We_book = 14.20, i.e. Ca = 0.02347. The tabulated ω is in the moving frame, ω − kV. The minus signs of k_i are lost in the PDF text and were restored from the book's convention.

| V (book) | Branch | Ours: k_r, k_i, ω_r, ω_i | Book (Brevdo et al. OS) |
|---|---|---|---|
| 0.575 | I | 0.1696, −0.1785, 0.0092, 0.0031 | 0.170, −0.179, 0.0092, 0.0031 |
| 0.575 | II | 0.0404, −0.0459, 0.0053, 0.0037 | 0.040, −0.046, 0.0053, 0.0037 |
| 0.580 | I | 0.1772, −0.1648, 0.0083, 0.0039 | 0.177, −0.165, 0.0083, 0.0039 |
| 0.580 | II | 0.0431, −0.0475, 0.0051, 0.0039 | 0.0431, −0.0475, 0.0051, 0.0039 |

These saddle points use complex k far from the real axis (|k_i| ≈ k_r), so the check covers the complex-k operator that the spatial solver also uses, not only the temporal problem.

## First numbers for the oracle case (Re 120, Ca 0.01, β 45°)

The plan §0 table (temporal) is now confirmed. Lavalle's c_r is within 3% of OS for k = 0.04–0.5. Its growth is too low by 8% at k = 0.08 and by 39% at k = 0.3 (the OS maximum).

Spatial growth −k_i (OS / Lavalle):

| ω | 0.05 | 0.10 | 0.134 (10 Hz) | 0.20 | 0.30 | 0.45 |
|---|---|---|---|---|---|---|
| OS | 0.0058 | 0.0107 | 0.0129 | 0.0157 | 0.0181 | 0.0193 |
| Lavalle | 0.0064 | 0.0110 | 0.0125 | 0.0136 | 0.0135 | 0.0122 |

At the forced frequency Lavalle's spatial growth is only 3% low. The gap opens at higher ω and reaches 37% at ω = 0.45. Below ω ≈ 0.1, Lavalle's growth is slightly *high*. For the monochromatic 10 Hz oracle run, the linear-region amplitude error of Lavalle is therefore small. The larger errors sit in the harmonics and in the broadband runs.

## Case table (B.4)

_Data: all 85 cases of `datasets/MANIFEST_dns.csv`, 36 of which have DNS data. Script: `analysis/orr_sommerfeld/case_table.py`, run as SLURM job 51587062 (2.4 min on 8 CPUs). Table: `$THINFILM_STORE/analysis/orr_sommerfeld/case_table.csv`. Summary: `python analysis/orr_sommerfeld/summarize_case_table.py`._

- **What each row holds:**
  - temporal OS: k_max, maximum growth, c_r(k_max), cut-off k_c, ω_c;
  - spatial OS: ω_max, −k_i, and k_r at ω_max;
  - the same quantities for the linearised Lavalle model;
  - ε_lin = k_max (OS) and δ_lin = ε_lin Re.
- **Conversion to Hz:** T0 = h_N/u_N from each case's `params.in`, as in `filmflow0.c`. Seven cases have no RHO_L or MU_L there and use the `filmflow0.c` defaults (1072 kg/m³, 0.00673 Pa s). For the oracle case these give T0 = 2.14 ms, which reproduces the documented 10 Hz → 0.0214.
- No case is linearly stable, and none failed.

**Forcing cut-off against the unstable band** (this closes the open item "check f_cut against 2f₀" in `decisions.md`):

| | All 85 | With DNS data (36) |
|---|---|---|
| 28 Hz below the OS cut-off f_c | 71 | 31 |
| 28 Hz below the most amplified spatial frequency f_max | 22 | 13 |
| 28 Hz ≥ 2 f_max (first harmonic forced) | 22 | 7 |
| 28 Hz / f_c, median (range) | 0.42 (0.09–7.1) | 0.32 (0.09–2.4) |

- At β = 45° and 90° with Re ≥ 40, the broadband forcing covers only 10–30% of the unstable band. In 13 of the 36 DNS cases it stops below the most amplified frequency. Examples: Re120 Ca0.01 β45 has f_max = 33.5 Hz and f_c = 143 Hz; Re200 Ca0.01 β45 has f_max = 70 Hz.
- These waves therefore grow from forced frequencies below f_max and from harmonics generated nonlinearly, not from the linearly most amplified mode.
- At β = 6.4° the opposite holds: f_c is 11–78 Hz, and at Re 20–40 the forcing extends well into the stable band.

**Lavalle against OS, binned by δ_lin** (ratio of maximum spatial growth rates, Lavalle/OS, all 85 cases):

| δ_lin | n | Lavalle / OS |
|---|---|---|
| < 5 | 33 | 1.07–1.18 (Lavalle 7–18% too high) |
| 5–15 | 22 | 0.90–1.06 |
| 15–30 | 17 | 0.76–0.93 |
| 30–70 | 13 | 0.55–0.76 |

- The error of the first-order model in the linear regime has a sign that depends on δ: it over-amplifies at small δ and under-amplifies at large δ, and the crossover is near δ_lin ≈ 10.
- Lavalle's k_max is up to 2.4 times smaller than OS's (median 1.35 over the DNS cases). Its cut-off k_c is close to OS (0.96–1.27).
- The Benney long-wave cut-off √(3Ca(1.2Re − cot β)) overestimates k_c by up to a factor of 3 at high Re.
- **ε_lin and δ_lin for FF plan D2:**
  - δ_lin (OS) is 1.0–66 over the DNS cases (median 15).
  - δ from Lavalle's k_max is up to 2.3 times smaller (for example 36.7 against 20.1 for the oracle case). The two agree for δ ≲ 5.
  - Which one to use as the long-wave parameter is decision D2.

## DNS in the linear region (B.5)

_Script `analysis/orr_sommerfeld/b5_dns_linear.py`. Oracle run: Re120 Ca0.01 β45, 10 Hz, amplitude 0.03, t = 1800–1900, h every 0.01 on the ROM grid. The complex amplitude A_n(x) of harmonic n is projected over two forcing periods (93.5 time units). Fits use fixed x-windows. Output: `b5_dns_linear.csv`._

| x | −k_i (n = 1) | c_ph (n = 1) | −k_i (n = 2) | c_ph (n = 2) |
|---|---|---|---|---|
| 5–20 | 0.046 | 1.32 | 0.070 | 1.24 |
| 20–40 | 0.0114 | 1.66 | 0.031 | 1.72 |
| 40–60 | 0.0117 | 1.82 | 0.035 | 1.85 |
| 60–80 | 0.0129 | 1.82 | 0.034 | 1.81 |
| 80–100 | 0.0130 | 1.82 | 0.032 | 1.80 |
| 100–130 | 0.0128 | 1.81 | 0.029 | 1.79 |
| 130–160 | 0.0123 | 1.80 | 0.026 | 1.77 |
| 160–200 | 0.0107 | 1.76 | 0.020 | 1.74 |
| **OS** | 0.0129 | 1.847 | 0.0175 | 1.736 |
| Lavalle | 0.0125 | 1.780 | 0.0136 | 1.700 |

- **Fundamental:** after an inlet adjustment zone (x < 20, not modal), the growth rises to the OS value. From x = 60 to 130 it matches OS to 1%.
  - The phase speed is 1.5% below OS. The mesh study found level 15 waves 1.2% faster than level 14, which explains most of this. A gas-phase effect would add to it.
  - The local growth first drops below 80% of its 20 < x < 60 value at x ≈ 198, where |A₁| = 0.27.
  - **Linear region of this case:** x ≈ 60–150 (growth within 5% of OS). The earlier guess of x < 60–100 was partly inside the inlet adjustment zone.
- **The 20 and 30 Hz harmonics are bound, not free.** From x ≈ 20 the 20 Hz harmonic grows 2.75 times faster than the fundamental and travels at the fundamental's speed (OS free-wave speed 1.74). It is generated nonlinearly from the fundamental, so OS does not apply to it.
- **No other data were usable.** The dense broadband restarts (`BasC/DNS/dense/`) cover only 10 time units, a frequency resolution of about 47 Hz at Re 120, so they cannot resolve the 0–28 Hz forced band. The Liu & Gollub monochromatic runs (4.5 and 7 Hz) are still spinning up; for a single frequency, 2D snapshots are enough (spatial envelope). The broadband cases need D3.

## ROM in the linear region (B.6)

_Run generator and analysis: `analysis/orr_sommerfeld/b6_rom/` (`make_runs.py`, `run_array.sh`, `analyse.py`), SLURM array 51588756. Runs: `$THINFILM_STORE/analysis/orr_sommerfeld/b6_rom/<case>/<model>/`._

- **Method.** A frequency-comb inlet replaces separate monochromatic runs. The existing Chang inlet with `MNOISE = M` gives q_in = 1 + Σ A cos(k Δω t + φ_k), k = 1..M.
  - `fcut_dim` is chosen so that 2π/Δω = 100 exactly, and each mode has amplitude 1e-5. Each mode is separated exactly by projection over the last 100 time units.
  - Flat initial state, dx = 0.2, dt = 0.005, L ≈ 7 e-folds of the OS maximum.
  - `lavalle-cpp` now accepts `none` as the oracle argument, so an ML closure can run without DNS forcing. Default behaviour is unchanged.
- **Cases:** Re 20, 120 and 200 at Ca 0.01, β 45° (δ_lin 4.4, 36.7, 64.5). Models: Lavalle, `ml_sum_hhx`, `ml_mx_tw`.
- **Reliable modes:** wavelength under half the fit window (0.1L–0.8L) and R² of ln|A| > 0.95. That is 9–17 modes per run, ω ≈ 0.13–1.1.

Maximum growth error over the reliable modes (relative to the maximum OS growth); maximum phase-speed error in brackets:

| | Re 20 | Re 120 | Re 200 |
|---|---|---|---|
| Lavalle, ROM | 0.10 (0.026) | 0.42 (0.037) | 0.55 (0.029) |
| Lavalle, analytic | 0.10 (0.025) | 0.41 (0.037) | 0.53 (0.029) |
| ML sum, ROM | 0.27 (0.037) | 0.28 (0.034) | not linear (see below) |
| ML sum, analytic (A.1 coefficients) | 0.27 (0.047) | 0.18 (0.022) | 1.19 (0.069) |
| ML separate, ROM | 0.20 (0.034) | not linear | not linear |
| ML separate, analytic | 0.20 (0.032) | 1.11 (0.11) | 1.56 (0.16) |

- **Numerics check.** The ROM reproduces analytic linearised Lavalle to within about 0.02 in these error measures, and mode by mode to about 1% in the middle of the band. MUSCL–minmod at 1e-5 amplitude does not add visible damping on the resolved modes. So the method measures the model, not the scheme.
- **Linearisation check.** Where the ML runs stay linear, their measured dispersion agrees with the analytic dispersion of their A.1 linear coefficients (Re 20: 0.27/0.27 and 0.20/0.20). The A.1 analysis therefore predicts the augmented ROM's linear behaviour.
- **Concern 1 (linear regime).** Neither current ML closure reproduces the OS linear behaviour.
  - The sum model is 27–28% off at Re 20 and 120. Its base film settles at h = 1.019–1.020 instead of 1 because S₀ ≠ 0.
  - The separate models are 20% off at Re 20. Analytically they double the growth at Re 120.
- **ML runs that left the linear regime.** The separate models at Re 120, and both models at Re 200.
  - **It is not an inlet problem** (corrected 16 Nov).
    - The first cell does drift in the output: h reaches 1.37–2.44, and q goes to −0.02–1.50. This is a write-out artefact.
    - `apply_white_noise_bc` (and the oracle `apply_inlet`) clamps the ghost cells *and the first physical cell* to the inlet state, but only in the copy used to build the RHS.
    - `rk4_step` still integrates the stored cell 0, whose RHS is never used again, so it drifts linearly in time (about 8e-4 per time unit with the ML closures, negligible with Lavalle). The dynamics always see the clamped inlet.
    - From cell 1 onward the solution is smooth: the interior relaxes to the model's own uniform state over about 40 units.
    - Fix: re-apply the inlet state to the stored cell after each step, or skip it in the output. This affects every ROM output's first cell, including the oracle runs.
  - **Separate models, Re 120 and 200: excess linear growth.** Their analytic maximum spatial growth is 0.041 (OS 0.019 and 0.016). From 1e-5 that amplifies to about 1e2–1e3 over the domain, which was sized for OS growth, so the comb saturates nonlinearly. It is the closure's linear error, the same one found in A.1.
  - **Sum model, Re 200: a shifted base state and a spurious equilibrium.**
    - The uniform state with q = 1 is h_eq = 1.108 (S₀ ≠ 0), and the interior settles there.
    - The start-up step from 1.108 to the initial h = 1 travels downstream as a front.
    - The model also has spurious uniform equilibria at h = 2.05 and 2.92, and by t = 500 the solution locks onto h = 2.924 locally.
  - **Spurious uniform equilibria exist for both MLPs at every Re tested** (roots of (3/Re)(h − 1/h²) = S(h, 1, 0, 0)):
    - sum model: 2.72 and 2.78 at Re 120, 2.05 and 2.92 at Re 200;
    - separate models: 2.87, 2.49 and 2.95 at Re 20, 120 and 200.
    - Hypothesis, not tested: the sum model's closed-loop crest over-amplification at Re 120 (max h 2.5 against the DNS's 2.1, `ml-a-posteriori.md`) is pulled toward its spurious states near h ≈ 2.7.
    - The structured closure removes both mechanisms by construction: S(h, h³, 0, 0) = 0, and relaxation toward Q = h³ with κ > 0 leaves q = h³ as the only uniform equilibrium.

## Linear closure targets (B.7)

_Code: `linear_targets.py` (SLURM job 51588978, all 85 cases) and `b7_dns_check.py` (DNS comparison). Output: `linear_targets.csv`; summary `summarize_linear_targets.py`._

- **Method.** Along the OS mode with η̂ = 1:
  - q′ = φ(1) + U_s = c;
  - M_x′ = ik[∫2Uφ′ + U_s²];
  - T_w′ = φ″(0)/Re;
  - P_x′ = −ik∫p̂, with p̂ from the x-momentum equation;
  - D_x′ = −k²φ(1)/Re.
  - The linearised Lavalle closures are subtracted to give ΔC′/η̂, complex: the real part is in phase with h, the imaginary part is in quadrature.
- **Momentum-balance check.** The OS terms close the linear depth-integrated momentum balance to ≤ 1.6e-9 over all cases, once the surface viscous term (1/Re)u_z|_h = −k²φ(1)/Re is included.
  - `docs/equations.md` §2 drops that term by assuming u_z|_h = 0, but the passive-gas condition is u_z + w_x = 0.
  - The term is O(k²/Re). It is 1.8e-4 at k = 0.3 for the oracle case, small but not zero, and it belongs to the unclosed remainder of the budget.
- **At the OS k_max** (medians per δ_lin bin of all 85 cases):

| δ_lin | < 5 | 5–15 | 15–30 | 30–70 |
|---|---|---|---|---|
| ΔM_x′ | −0.022 − 0.044i | −0.034 − 0.125i | −0.028 − 0.154i | −0.027 − 0.202i |
| ΔT_w′ | +0.022 + 0.062i | +0.017 + 0.144i | +0.012 + 0.159i | +0.010 + 0.191i |
| S′ = ΔM_x′ + ΔT_w′ | +0.001 + 0.018i | −0.014 + 0.015i | −0.017 + 0.006i | −0.018 − 0.005i |
| \|S′\| / \|Im ΔM_x′\| | 0.42 | 0.19 | 0.12 | 0.10 |
| ΔP_x′, ΔD_x′ | ≤ 0.003 | ≤ 0.003 | ≤ 0.002 | ≤ 0.002 |

  - ΔM_x and ΔT_w are each mostly in quadrature with h and cancel to 10–40%. This is the linear-regime origin of the ΔM_x/ΔT_w cancellation seen in the oracle runs. A closure that learns them separately has to get two large numbers right to get one small one.
  - The net S′ moves from mostly quadrature at small δ to mostly in phase at large δ. ΔP_x′ and ΔD_x′ are negligible in the linear regime.
- **Against the DNS (oracle case, ω = 0.1344, OS spatial mode, linear region of B.5):**

| | ΔM_x/η | ΔT_w/η | (ΔM_x + ΔT_w)/η | ΔP_x/η |
|---|---|---|---|---|
| OS target | −0.0178 − 0.0354i | +0.0124 + 0.0408i | −0.0053 + 0.0054i | 0.0000 |
| DNS x 20–60 | −0.0156 − 0.0399i | +0.0118 + 0.0423i | −0.0038 + 0.0024i | −0.0001 + 0.0001i |
| DNS x 60–100 | −0.0188 − 0.0365i | +0.0132 + 0.0411i | −0.0057 + 0.0046i | −0.0001 + 0.0001i |
| DNS x 100–130 | −0.0218 − 0.0384i | +0.0149 + 0.0422i | −0.0069 + 0.0038i | −0.0001 + 0.0001i |

  - In the linear region (x 60–100), the extracted ΔM_x and ΔT_w match the exact OS values to 3–6% in both components, and their small sum to about 10–15%.
  - The extraction pipeline (Savitzky–Golay derivatives, budget) is therefore accurate where the truth is known. The drift at x > 100 is the start of nonlinearity.

## Linear closure coefficients (B.8)

_Same script and job as B.7._

- **Form fitted:** S_lin = a(ξ − 3η) + bη_x + cξ_x [+ dη_xx]. The uniform film is preserved by construction, and the long-wave D is forced to equal the OS value 1.2Re − cot β, which fixes c from (a, b).
- **Target:** the OS spatial branch at 20 frequencies, either over the full unstable band (0.05–0.95 ω_c) or over the forced and most-amplified band (0.05 ω_c to min(2 ω_max, 0.95 ω_c)).
- **Errors:** maximum growth error relative to the maximum OS growth, and maximum relative k_r error.

Medians per δ_lin bin:

| | δ < 5 | 5–15 | 15–30 | 30–70 |
|---|---|---|---|---|
| Lavalle, growth (full band) | 0.11 | 0.15 | 0.32 | 0.46 |
| (a, b, c), full band | 0.034 | 0.12 | 0.20 | 0.26 |
| (a, b, c, d), full band | 0.030 | 0.063 | 0.13 | 0.17 |
| (a, b, c), forced band | 0.033 | 0.12 | 0.20 | 0.26 |
| **(a, b, c, d), forced band** | **0.017** | **0.015** | **0.041** | **0.079** |
| k_r error: Lavalle / (a, b, c, d) forced band | 0.014 / 0.012 | 0.034 / 0.019 | 0.038 / 0.026 | 0.038 / 0.037 |

- **Four coefficients over the forced band are enough.** Growth error is below 10% in all 85 cases (maximum 0.098), with k_r within 1–4%. Over the full band up to ω_c four coefficients are not enough at high δ (52 of 85 cases below 10%); that is risk 2 of the plan, confined to the short capillary waves.
- **(a, b, c) alone does not reduce the error at δ > 5.** The η_xx coefficient d is needed: it is a dispersive and damping term at finite k.
- **Coefficients, forced band (a, b, c, d) medians:**
  - a = −0.006 to −0.020, i.e. slightly weaker relaxation than Lavalle;
  - b = −0.45 to −1.15 and c = +0.26 to +0.68, both growing in magnitude with δ;
  - d = +0.06 to +0.15.
  - Oracle case: a = −0.0056, b = −1.04, c = 0.61, d = 0.087.
  - The single-frequency DNS fit of A.4 (x < 60: a −0.016, b −0.79, c +0.50) has the same signs. Its coefficients are not identifiable, though (in the 60–130 window they come out completely different), so this is not a validation of the individual coefficients.
- These are the L2 targets for Phase C (`linear_targets.csv`, columns `band_abcd_*`).

## Comb-forced DNS across Re (D3: B.5, B.7, B.8 from the DNS)

_17 Nov 2026. Runs: `dns/comb/` (jobs 51592372–76, `bala1s-b`), data in `$THINFILM_STORE/datasets/BasC/DNS/comb/<case>/`. Analysis: `analysis/orr_sommerfeld/comb_dns/analyse_comb.py` (job 51787456), output `comb_fits.csv`, `comb_modes_<case>.csv`, `comb_growth_<case>.png` in `$THINFILM_STORE/analysis/orr_sommerfeld/comb_dns/`. Instructions: `docs/plans/comb-dns-instructions.md`._

- **Setup.**
  - Five cases at β 45°: Re 20, 80, 120, 200 at Ca 0.01, and Re 40 at Ca 0.04.
  - Production physics and mesh (MAXLEVEL 12, dx 0.049) on a short domain, LX = 200. Flat start.
  - Inlet comb: 21–31 modes of 3e-4 each, period exactly 100, up to 0.97 ω_c.
  - Spin-up to t = 150, then profiles every 0.25 for t = 155–255. Each mode is projected over exactly one period.
  - Fit window x = 60–150.
  - Reliable modes: R² of ln|A| > 0.95 and wavelength < 2 × window, which excludes ω ≲ 0.1.
- **Errors over the reliable modes:**
  - growth error = |−k_i,DNS − (−k_i,OS)| / max −k_i,OS;
  - phase speed = c_DNS/c_OS − 1;
  - ΔM_x, ΔT_w error = |DNS − OS| / |OS|, complex, per unit η.

| | Re 20 | Re 40 (Ca 0.04) | Re 80 | Re 120 | Re 200 |
|---|---|---|---|---|---|
| reliable modes (ω range) | 10 (0.25–0.88) | 16 (0.13–1.07) | 17 (0.13–1.13) | 18 (0.13–1.19) | 22 (0.13–1.45) |
| growth, DNS vs OS: median / max | 0.069 / 0.145 | 0.098 / 0.288 | **0.027** / 0.084 | **0.017** / 0.080 | **0.028** / 0.172 |
| growth, Lavalle vs OS: median / max | 0.027 / 0.097 | 0.150 / 0.276 | 0.244 / 0.303 | 0.317 / 0.410 | 0.374 / 0.531 |
| phase speed, DNS vs OS: median (range) | −0.9% (−2.3, +0.8) | −1.8% (−16.6, +1.2) | −1.0% (−2.0, −0.8) | −1.1% (−1.6, −0.9) | −1.5% (−1.8, −1.1) |
| ΔM_x, DNS vs OS: median / max | 0.036 / 0.046 | 0.046 / 2.5 | 0.012 / 0.072 | 0.010 / 0.072 | 0.033 / 0.051 |
| ΔT_w, DNS vs OS: median / max | 0.032 / 0.059 | 0.042 / 1.3 | 0.017 / 0.050 | 0.013 / 0.052 | 0.010 / 0.033 |
| total wave height in window, max\|h − ⟨h⟩\| | 0.051 | **0.28** | 0.031 | 0.025 | 0.021 |
| drift of ⟨h⟩ between window halves | 3.5e-3 | 4.5e-3 | 1.8e-3 | 1.7e-3 | 1.6e-3 |

**B.5 across Re: the DNS follows OS over the whole forced band at Re 80–200.**
- Growth matches OS to a median of 2–3%.
- The phase speed is 1–1.5% below OS at every frequency, the same offset as the oracle case. It is systematic, so it is consistent with the mesh explanation (level 15 waves ran 1.2% faster than level 14), though still not proven.
- The worst modes (8–17%) are scattered single modes near the top of the reliable band, not a trend.
- Lavalle's growth error rises from 24% to 37% (median), and 30% to 53% (max), over Re 80–200, as in B.6 and B.8.

**B.7 across Re: the extracted closure terms match OS.**
- At Re 80–200, ΔM_x and ΔT_w per frequency agree with the exact OS values to a median of 1–3% (max 5–7%), in amplitude and phase.
- At Re 20 the agreement is 3–4%.
- This extends the single-frequency check of the oracle case to the whole band and to three more Re. The extraction pipeline is accurate in the linear regime wherever it has been tested.

**B.8 from the DNS:** (a, b, c, d) fitted to the DNS k(ω) and, for comparison, to OS at the same frequencies, both with D = 1.2 Re − cot β imposed.

| | a | b | c | d | fit error k_i / k_r |
|---|---|---|---|---|---|
| Re 80, DNS / OS | −0.0054 / −0.0070 | −0.715 / −0.896 | +0.408 / +0.521 | +0.075 / +0.090 | 0.077 / 0.028 vs 0.085 / 0.032 |
| Re 120, DNS / OS | −0.0046 / −0.0056 | −0.914 / −1.072 | +0.524 / +0.626 | +0.055 / +0.063 | 0.119 / 0.037 vs 0.115 / 0.039 |
| Re 200, DNS / OS | −0.0030 / −0.0037 | −1.008 / −1.187 | +0.573 / +0.693 | +0.027 / +0.032 | 0.210 / 0.037 vs 0.150 / 0.043 |
| Re 20, DNS / OS | +0.025 / −0.020 | +1.46 / −0.59 | −0.67 / +0.35 | +0.10 / +0.17 | 0.147 / 0.017 vs 0.019 / 0.017 |
| Re 40, DNS / OS | +0.012 / −0.013 | +0.61 / −0.75 | −0.39 / +0.45 | +0.08 / +0.12 | 0.266 / 0.150 vs 0.019 / 0.020 |

- **Re 80–200:** the DNS coefficients have the OS signs and trend with Re, and are 15–25% smaller in magnitude. The two fits match their data equally well.
  - With D fixed, b and c (and a) trade off along one direction. The difference is mostly along it, so it is a weakly constrained combination, not a different linear physics.
  - The OS values are the L2 targets. The DNS confirms them to this level.
- **The coefficients depend on the frequency set.** The OS fit at the comb frequencies gives d = 0.063 for the oracle case, against 0.087 from the B.8 forced band. Compare DNS and OS only at the same frequencies, as here.

**Re 20 and Re 40: not clean linear tests in the standard window** (Re 40 is usable in x = 30–100, see below).
- **Re 20 (δ_lin 4.4).**
  - The closure terms match OS (3–4%), but the growth error (median 7%) is larger than Lavalle's (3%).
  - The linear correction Lavalle needs at this δ is smaller than the DNS measurement scatter, so the B.8 fit to the DNS has no information. It returns coefficients of the opposite sign with a fit error 8× that of OS.
  - Only 10 modes pass the filter. The window and settling are marginal: ⟨h⟩ drift 3.5e-3.
- **Re 40, Ca 0.04.**
  - Each mode stays below 0.044, but the 31 modes add up to a total wave height of 0.28 inside the window, so the window is not linear.
  - A few modes are badly off: phase speed −17%, closure terms off by 1–2.5× at ω 0.19 and 1.07. The ⟨h⟩ drift is the largest, 4.5e-3.
  - The median errors are still 4–10%, but the case does not test linear theory cleanly in this window.
  - **Rescued by an upstream window** (job 51789519, output `comb_dns/win30_100/` and `win45_110/`):

    | Re 40 Ca 0.04 | x 30–100 | x 45–110 | x 60–150 |
    |---|---|---|---|
    | reliable modes (ω range) | 25 (0.13–1.76) | 22 (0.13–1.57) | 16 (0.13–1.07) |
    | total wave height max\|h − ⟨h⟩\| | 0.053 | 0.072 | 0.28 |
    | growth DNS vs OS, median / max | **0.031** / 0.20 | 0.041 / 0.23 | 0.098 / 0.29 |
    | growth Lavalle vs OS, median | 0.24 | 0.23 | 0.15 |
    | phase speed, median (range) | −1.1% (−2.7, −0.2) | −1.2% (−4.2, −0.2) | −1.8% (−16.6, +1.2) |
    | ΔM_x / ΔT_w vs OS, median | 0.042 / 0.033 | 0.053 / 0.039 | 0.046 / 0.042 |
    | (a, b, c, d) DNS | −0.008, −0.54, +0.30, +0.072 | −0.006, −0.41, +0.23, +0.079 | wrong signs |
    | (a, b, c, d) OS, same ω | −0.013, −0.82, +0.48, +0.085 | −0.014, −0.86, +0.50, +0.089 | |

    - In x = 30–100 the case behaves like Re 80–200: growth median 3%, systematic phase-speed offset −1.1%, closure terms 3–4%, and DNS (a, b, c, d) with the OS signs.
    - The DNS coefficients are about 35% smaller than OS, a larger gap than at Re 80–200 (15–25%).
    - The largest growth errors are a cluster at ω 1.13–1.26, where the DNS grows 15–30% (of the peak) slower than OS, plus ω 1.51, where it grows faster. They are not scattered single modes.
    - The window includes x = 30–60, the inlet-adjustment zone of the oracle case. There the forcing was 100× larger and the adjustment was visible; here the medians show no sign of it.
    - The ⟨h⟩ drift (4.3e-3) is the same in every window, so it is not caused by the nonlinearity.
    - **Use x = 30–100 for this case.** No rerun is needed.

**Mean film thickness.** In every run ⟨h⟩ ≈ 0.9946–0.9974 in the window, 0.3–0.55% below Nusselt. It is not yet explained (candidates: the mean of the inlet flux, the gas phase, mesh mass error). It is small next to the waves and does not affect the mode-by-mode comparisons, which use h − ⟨h⟩_t.

## Gate GB summary

1. **B.3 passes** (Benney limit, Re_c, Brevdo et al. saddle points, convergence).
2. **The DNS agrees with OS in the linear region of the oracle case:**
   - growth to 1% and phase speed to 1.5% (B.5);
   - the extracted ΔM_x and ΔT_w to 3–6% (B.7).
   - The prototype table of plan §0 is confirmed, so nothing here supports L1 over L2.
   - Comb DNS (17 Nov): the same holds over the whole forced band at Re 80, 120 and 200 (growth median 2–3%, phase speed −1 to −1.5%, ΔM_x and ΔT_w 1–3%).
3. **The ROM numerics are accurate in the linear regime** (B.6, Lavalle to about 1%). The current ML closures are not:
   - growth is 20–28% off where they stay linear;
   - at Re 120 and 200 they leave the linear regime, through twice the OS growth (separate models) or a shifted base state plus a spurious uniform equilibrium (sum model). It is not an inlet effect.
4. **Four linear coefficients reproduce OS over the forced band in every case** (B.8).

Proposal for D2: **L2 with (a, b, c, d) fitted over the forced band.** The η_xx coefficient is needed, so Phase C's relaxation part should include g₃ (or an equivalent h_xx dependence) by default, not as an ablation.

## Where it fails

- **Passive gas only.** The DNS has a real gas phase, so differences of a few percent from OS are possible (B.5 will measure them).
- **Parallel base flow, linear, 2D.** The solver says nothing beyond the linear region.
- **The published check is a single case:** saddle points at Re 133, β 4.6°. No published spatial growth-rate curve at our Re 120, β 45° has been compared yet. The Benney and Re_c checks only test small k.
- **Mode selection with complex k relies on continuation.** A large jump in parameters can lock onto another mode. `temporal` raises if the polish moves c by more than 1e-3, but a continuation that jumps onto another converged mode would not be caught. The saddle-point check found a spurious far-field saddle (k ≈ 0.34 + 11.3i) when started on the wrong side of the real axis.
- **`lavalle_spatial` picks the root nearest ω/3.** Checked only for ω ≤ 0.45 at Re 120. It needs a check near the cut-off and at high Re.
- **B.5 rested on one case at one frequency** (its harmonics are bound). The comb DNS extend it to the forced band at Re 80, 120 and 200, all at β 45° and Ca 0.01.
  - Re 40 Ca 0.04 agrees too, in the window x = 30–100. It is nonlinear in x = 60–150.
  - Not tested: other inclinations, Ca other than 0.01 and 0.04, low δ (at Re 20 the correction is below the measurement scatter).
  - The systematic 1–1.5% phase-speed gap is attributed to mesh resolution but not proven.
  - The 0.3–0.55% deficit in mean h is unexplained.
- **The B.6 explanations are diagnostic, not tested by reruns.** Rerunning the nonlinear ML cases on a shorter domain would confirm the excess-growth explanation. The first ROM cell in every output is not physical (see B.6). The reliable-mode filter excludes the lowest and highest frequencies.
- **B.7 targets are for the passive-gas linear mode** and do not include the surface viscous term in the closure split. Against the DNS they were compared at one frequency only.
- **B.8 coefficients are fitted to OS.** At Re 80–200 the DNS-fitted values have the same signs but are 15–25% smaller, along the weakly constrained direction left once D is fixed. The forced-band choice (ω ≤ 2 ω_max) is a judgement. The root picked is the one nearest to OS, which could hide a spurious branch.
- **B.4 is temporal-maximum based.** ε_lin = k_max of the *temporal* problem. The spatial f_max is reported alongside but is not used for δ_lin.
- **Fluid properties of seven cases are defaults** (see B.4). If those runs actually used other properties, their Hz values are wrong.
- **The book PDF text loses minus signs.** The k_i signs in the table above were restored by convention; the magnitudes match.
