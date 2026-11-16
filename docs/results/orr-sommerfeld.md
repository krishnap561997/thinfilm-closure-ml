# Orr–Sommerfeld reference solver (tasks B.1–B.3)

_16 Nov 2026. Plan: `docs/plans/structured-closure-os-plan.md` Part I. Code: `analysis/orr_sommerfeld/`. Numbers: `python analysis/orr_sommerfeld/verify.py` (≈5 s), output in `$THINFILM_STORE/analysis/orr_sommerfeld/verification.json`._

Status: B.1 (temporal), B.2 (spatial), B.3 (verification) and B.4 (case table) are done. B.5–B.8 have not started.

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

## Where it fails

- **Passive gas only.** The DNS has a real gas phase, so differences of a few percent from OS are possible (B.5 will measure them).
- **Parallel base flow, linear, 2D.** The solver says nothing beyond the linear region.
- **The published check is a single case:** saddle points at Re 133, β 4.6°. No published spatial growth-rate curve at our Re 120, β 45° has been compared yet. The Benney and Re_c checks only test small k.
- **Mode selection with complex k relies on continuation.** A large jump in parameters can lock onto another mode. `temporal` raises if the polish moves c by more than 1e-3, but a continuation that jumps onto another converged mode would not be caught. The saddle-point check found a spurious far-field saddle (k ≈ 0.34 + 11.3i) when started on the wrong side of the real axis.
- **`lavalle_spatial` picks the root nearest ω/3.** Checked only for ω ≤ 0.45 at Re 120. It needs a check near the cut-off and at high Re.
- **B.4 is temporal-maximum based.** ε_lin = k_max of the *temporal* problem. The spatial f_max is reported alongside but is not used for δ_lin.
- **Fluid properties of seven cases are defaults** (see B.4). If those runs actually used other properties, their Hz values are wrong.
- **The book PDF text loses minus signs.** The k_i signs in the table above were restored by convention; the magnitudes match.
