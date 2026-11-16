# Plan: a functional form for ΔM_x + ΔT_w (Paper 1)

_Written 13 Nov 2026 in Cowork with Krishna, from repo `main` @ 719b7da, for Claude Code on HiPerGator to carry out. Put this file in the repo at `docs/plans/functional-form-plan.md` in the first commit (task 0.0)._

> **Status (16 Nov 2026): secondary for now.** The active plan is `docs/plans/structured-closure-os-plan.md`. This plan is kept as the monomial/theory-library baseline (decision D1 of that plan). Cache v2 (task 1.1) and the closed-loop harness (task 1.3) are shared with it. Phase 0 results: `docs/results/functional-form-v2.md` (merged into `main` on 16 Nov).

Read first: `CLAUDE.md`, `docs/project-status.md`, `docs/results/functional-form-structure.md`, `docs/results/ordering-matrix.md` (§3, §9), `docs/results/richard-comparison.md` (§6), `docs/results/ml-a-posteriori.md`.

---

## 0. Decisions made on 13 Nov (record them in `docs/decisions.md`, task 0.0)

1. **Submission moves from 15 Dec 2026 to the second week of January 2027 (target Fri 15 Jan 2027).** There is no hard results freeze: results and writing iterate until submission, and the draft to co-authors (about 4 Jan) uses the current results. The APS DFD talk is still 22 Nov, with a soft results freeze on Sun 15 Nov.
2. **The functional form for S = ΔM_x + ΔT_w is part of Paper 1.** If it beats the ML sum model in closed loop (definition: §2.5), it **becomes Paper 1's closure for ΔM_x + ΔT_w**.
3. **Role of the MLPs in Paper 1:** the simple 1×16 MLPs and their input ablation show *a priori* that local information (h, q, h_x, q_x) is all that is needed to predict the two terms. They are the comparison for the form, not the main closure.
4. **Target structure:** route **S** (one library, fitted directly on the sum) comes first. Route **J** (joint fit with two libraries and component penalties) and route **C** (separate fits, then added) are the backups. In Phase 3 all three are compared head-to-head.
5. **Parameter dependence:** run **both** routes and compare them: (P1) per-case fits, then coefficient trends against δ, then a parametric form; (P2) pooled parametric sparse regression with fixed parameter factors.
6. **Model selection uses accuracy plus the closed-loop ROM test only.** Hyperbolicity and the low-δ theory-coefficient check are **reported diagnostics, not gates**.
7. Unchanged: train and score on the **entire time record** (decision of 10 Nov); ΔP_x is **not** part of this fit; same development / held-out split as the MLPs (`ml/functional_mlp/data_config.json`: 15 dev, 7 held-out).

## 1. Where things stand (short)

**A posteriori**, Re120 Ca0.01 β45, 10 Hz, from t = 1800 (held-out case). E_h = rms(h_ROM − h_DNS)/std(h_DNS) over 20 < x < 760:

| elapsed | Lavalle | oracle ΔT_w+ΔM_x | ML sum (h, h_x) `ml_sum_hhx` | ML separate `ml_mx_tw` |
|---|---|---|---|---|
| 1 | 0.115 | 0.060 | 0.057 | 0.058 |
| 10 | 0.606 | 0.172 | 0.219 | 0.218 |
| 50 | 0.585 | 0.192 | 0.467 | 1.029 |
| 99.9 | 1.066 | 0.550 | 1.258 | 2.045 |

- Max h at elapsed 99.9: ML sum 2.50, DNS 2.08. The ML sum over-amplifies the crests from elapsed ≈ 20.
- Applying ΔT_w alone crashes; ΔT_w ≈ −0.64 ΔM_x (corr −0.975), so S keeps about 36% of ΔM_x.

**Functional forms so far** (`ml/functional_forms/fit_sparse.py`, t > 1000):

- ΔM_x: a 5-term form, LOCO 0.129.
- S: no adequate sparse form (LOCO 0.35–0.40). The old library was base(h, q) × gradient × parameter, so **every term carried a derivative**. It had no non-derivative disequilibrium term in (q − h³), which ordering-matrix §3.6 and §9 identify as the common driver of both Lavalle errors.

**Why we expect better behaviour in closed loop.** The Tier-1 term (h³ − q)/(2Re h²) is a linear relaxation of q toward equilibrium, i.e. dissipative. The best ML sum model takes (h, h_x) only and cannot represent it. This is a hypothesis to test, not a result.

## 2. Definitions

### 2.1 Targets and variables

- **S** = `delta_Mx_ML + delta_Tw_ML`. The ROM adds −S to the q-equation.
- **ΔF** = ∫u² − q²/h − h⁵/5 (Λ = 3, so Λ²/45 = 1/5), from `int_u2_dy` in the cache. It is used for route J/C and as a check: ∂ₓΔF should reproduce `delta_Mx_ML`.
- **Conditioned variables:** r = q/h³ − 1 and g = h⁴∂ₓ(q/h³) = h q_x − 3q h_x. Both functions of (h, q, h_x, q_x).
- **Parameter groups:** Re, We = Re·Ca, cot β, Fr² = Re sin β/3, cos β/Fr² = 3 cot β/Re.
- **Diagnostic only (never a ROM input):** φ = (∫u² − q²/h)/h³, Nusselt value h²/5.

### 2.2 Derivative operators

- **D_SG:** Savitzky–Golay, window 21, order 3, as in `fit_sparse.py`. Used **only for the talk slice** (Phase 0).
- **D_G(σ):** a Gaussian filter with analytic derivatives, the same operator as `scipy.ndimage.gaussian_filter1d(mode='nearest')` and the ROM's `<f>_s<σ>` fields. Default σ = 0.5 h_N; the sensitivity study uses 0.25 and 1.0. Everything from Phase 1 on uses D_G, so training and ROM inputs match.

### 2.3 A priori metrics

- **M1:** per-case NRMSE = RMSE/std(truth) over x0 < x < 0.95L, averaged over cases; pooled version too.
- **M2:** the nonlinear-region NRMSE of `scripts/mlp_wave_region_summary.py` (x > 100 and h > 1.1 or h < 0.9), normalised by std over the whole case. This is what makes the comparison with `mlp-heldout-eval.md` fair.
- **Splits:**
  - LOCO over the 15 development cases;
  - the 7 held-out cases;
  - a β-holdout, with all β = 6.4° development cases out;
  - an extrapolation set: the 3 Nov cases, if finished (§6, D3).
- Lavalle alone (S = 0) is the reference row in every table.

### 2.4 Closed-loop metrics

- E_h at elapsed 1, 2, 5, 10, 20, 30, 50, 75 and 99.9 (`scripts/oracle_summary.py`).
- **Ē_h:** the time-mean of E_h over 0–100.
- Max h against the DNS; h_min; the minimum hyperbolicity discriminant.

### 2.5 "Beats the ML sum" (proposed; Krishna to confirm, D1)

On the oracle case, a candidate beats the ML sum if all of these hold:

- it is stable for 100 time units;
- its Ē_h is below that of `ml_sum_hhx`;
- its max-h error at elapsed 50 and 99.9 is no larger than that of `ml_sum_hhx`;
- it is not worse than `ml_sum_hhx` on Ē_h on the additional closed-loop cases (§4, task 1.4), once those exist.

## 3. Library for route S

Each flux term Θ_F enters S as ∂ₓΘ_F. That derivative is **expanded analytically** with the chain rule into h, q, h_x, q_x, h_xx, q_xx; data products are not differentiated numerically. The expected coefficients come from the row-1 q-gauge theory (ordering matrix §9), valid for δ = εRe ≲ 1, and are checked in `scripts/ordering-matrix/`. Build the library in **two bases**: raw monomials, and the conditioned (r, g) basis. Standardise columns with the per-case 1/rms weighting of `fit_sparse.py`.

### 3.1 Tier 1

| id | column in S | factor | expected coefficient |
|---|---|---|---|
| T1 | (h³ − q)/h² | 1/Re | +1/2 |
| T2 | h⁴h_x | 1 | +3/5 |
| T3 | q q_x/h | 1 | +1/35 |
| T4 | q²h_x/h² | 1 | −3/35 |
| T5 | h h_x | cot β/Re | −1/2 |
| F1 | ∂ₓ[(q − h³)q/h] | 1 | 8/35 |
| F2 | ∂ₓ[(q − h³)h²] | 1 | 1/5 |
| F3 | ∂ₓ[q²q_x − 3q³h_x/h] | Re | −2/735 |
| F4 | ∂ₓ[q h²h_x] | cot β | 1/35 |

### 3.2 Tier 2

- Source terms: h⁵h_x; (cot β/Re) h²h_x; (q − h³)h_x/h and (q − h³)q_x/h², each with factor 1/Re or 1.
- Flux terms: ∂ₓ[h⁸h_x]·Re; ∂ₓ[h⁵h_x]·cot β.

### 3.3 Tier 3

- Source terms: (1/We) h h_xxx; Re·{h⁵h_xx, h⁴h_x²}; (1/Re)·{h²h_xx, h h_x²}.
- Flux term: ∂ₓ[q h²h_xxx]/Ca, expected coefficient −1/105.

### 3.4 Diagnostic column (not a ROM input)

(1/Re)(q/h⁴)(φ − h²/5). Richard et al. (2016) τ_w(φ), linearised, gives the coefficient −35/2 (`richard-comparison.md` §6.2). Fit it with and without the slaved terms. The share of residual variance it explains measures **how much any Markovian (h, q) closure misses**.

### 3.5 Identifiability

T- and F-columns overlap: for example, ∂ₓF2 contains h²q_x and h h_x(q − h³) pieces. In route S only their combination is identifiable. Report the combined coefficient of each monomial next to the theory value, and leave the split between flux and wall shear to route J.

## 4. Phases and tasks

Work on a branch `functional-form-v2`, one PR per phase. Do not modify `fit_sparse.py` (its numbers are cited); put the new code in `ml/functional_forms/` (`library.py`, `fit_form.py`, …). Each results section gets a "Where it fails" subsection, as in the other write-ups.

### Phase 0: talk slice (Fri 13 – Sun 15 Nov; hard stop Sunday evening)

Goal: two or three presentable, clearly labelled *preliminary* results.

- **0.0** Add this plan to `docs/plans/`. Add the decisions in §0 to `docs/decisions.md`, and update the schedule and Paper 1 dates in `docs/project-status.md`.
- **0.1** Implement the Tier-1 library (§3.1, raw and (r, g) bases) on the **existing** `functional_mlp/cache_dir`, with D_SG derivatives (h_xx and q_xx are needed for F3 and F4).
- **0.2 Data hygiene.**
  - Re050 Ca0.01 β45: drop t > 1050. Its record after the restart was unforced (`decisions.md`, 10 Nov), and the cache predates the deletion.
  - For Re10 Ca0.01, Re10 Ca0.1 and Re40 Ca0.1, check that the cache was built before the 10 Nov overwrites. If any cached snapshot is one of the overwritten times listed in `decisions.md`, drop it.
  - Record what was dropped.
- **0.3 Fits.**
  - (a) Fixed Tier-1 least squares **per case** (all 22 cases, no selection): coefficients with 95% block-bootstrap intervals (blocks = snapshots) and the M1 error.
  - (b) **Pooled** forward selection on Tier 1 × parameter factors: LOCO, held-out and β-holdout. Report it twice:
    - at t > 1000 and x0 = 100, to compare like for like with `sparse_01` (LOCO 0.35–0.40);
    - on the full record.
  - (c) **Case placement.** Compute ε_k = k_peak h_N from the time-averaged h spectrum over x > 100, and the alternative ε_g = rms(h_x)/mean(h). Then δ = εRe. Produce a table of every case with its ordering-matrix row.
- **0.4 Figures** (scripts in `scripts/figures/functional-form-v2/`):
  1. LOCO NRMSE against the number of terms, old library vs Tier 1;
  2. per-case Tier-1 coefficients against δ, with the theory values as horizontal lines;
  3. per-case NRMSE of the best pooled form against Lavalle.
- **0.5 Stretch, only if 0.1–0.4 are done by Saturday evening:** task 1.2 in its simplest form, plus one closed-loop run of the best pooled Tier-1 form on the oracle case, against the reference rows of §1.
- **0.6** Write the first section of `docs/results/functional-form-v2.md`, marked preliminary (D_SG derivatives, existing cache). Add a short slide summary: the numbers, the figure paths, and one sentence per figure.
- **Gate G0 (Sun 15 Nov):** Krishna decides what goes into the talk.

### Phase 1: infrastructure (16–22 Nov; DFD week, so queue the long jobs on Sunday or Monday)

- **1.1 Cache v2.** Rebuild from `ClosureData` on the full record with D_G(σ), σ ∈ {0.25, 0.5, 1.0}.
  - Fields: h, q, h_x, h_xx, h_xxx, q_x, q_xx, `int_u2_dy`, ΔF, `delta_Mx_ML`, `delta_Tw_ML`.
  - Check: ∂ₓΔF against `delta_Mx_ML`.
  - Include the 8 cases from 3 Nov and the Re050 rerun if they have finished and their closures have been extracted. Refresh the manifest. Default split for the 3 Nov cases: extrapolation set (D3).
- **1.2 Analytic closure in `lavalle-cpp`.**
  - A `form_model = <file>` option in `ml.txt`, next to `sum_model`. The file lists terms: id, coefficient, parameter factor.
  - Evaluated pointwise at every RK4 stage on the ROM state, with the same filtered fields as the ML hook. Added as −S.
  - Lines tagged `// FORMCLOSURE`.
  - Leave room for route J: a ΔF part evaluated as a face flux.
  - Tests:
    1. all coefficients zero → byte-identical to Lavalle;
    2. on the DNS state at t = 1800 and 1850, S from C++ matches Python to ≤ 1e-10 (extend `scripts/ml_hook_check.py`).
- **1.3 Closed-loop harness.** An array script that runs N candidate files on the oracle case and writes `summary_form.csv`, with the reference rows (lavalle, Tw_Mx oracle, ml_sum_hhx) and the §2.4 metrics. Add a `reinit_every` option (5 or 10 time units) that resets h, q (and r) to the DNS state from `rom_forcing.h5`, to separate closure error from phase drift.
- **1.4 More closed-loop cases.**
  - List the monochromatic DNS available (`BasC/fixed-frequency/`, `BasC/liu1994/`, L800): for each, the dumps, output intervals and forcing.
  - Propose two more cases that span δ. One should be low-δ, ideally Liu & Gollub Re 19.33, β 6.4°, 1.5 Hz.
  - Give the DNS cost of a `filmflow_restart_mono.c` restart with profiles (T_PROF = 0.01, 100–200 time units) for each.
  - **Do not submit new DNS without Krishna's approval (D4).**
- **1.5 Well-posedness tool.** For a given form:
  - (a) the quasi-linear (h, q) Jacobian including the form's first-derivative terms (terms linear in h_x, q_x change the characteristic speeds just as a flux would);
  - (b) the effective diffusion matrix from its h_xx and q_xx terms (negative diffusion means ill-posed).
  
  Evaluate on DNS states and on ROM states. Report only; this is not a selection gate.
- **Gate G1 (~23 Nov):** the 1.1 and 1.2 checks pass.

### Phase 2: core results (23 Nov – 6 Dec), all on cache v2 with σ = 0.5

- **2.1 Route S, per case.** Tier 1 fixed, plus Tier 1–2 selection with forward selection and STLSQ. Coefficients against δ with bootstrap intervals.
  - Low-δ theory check: do the lowest-δ cases recover §3.1? If they are far off, look for a units, sign or derivative bug before going on. Report the result either way.
- **2.2 Route P1: per-case trends to a parametric form.**
  - Model each coefficient as c_k(δ): constant, linear, or saturating (c₀ + c₁δ/(1 + c₂δ)).
  - The ROM needs δ from the case parameters (D2). Default: ε_lin, the most-amplified wavenumber of the linearised Lavalle model, computed from (Re, Ca, β). Compare it with the measured ε_k from 0.3c.
- **2.3 Route P2: pooled parametric.**
  - Tiers 1–2 with the fixed parameter factors of §3, plus the generic set {1, Re, 1/Re, 1/We, cot β/Re}.
  - Forward selection and STLSQ, with stability selection (bootstrap over cases).
  - Score on LOCO, held-out, β-holdout and the extrapolation set.
- **2.4 P1 vs P2:** M1 and M2 on every split, the number of terms, and the coefficients against theory.
- **2.5 MLP on the same footing.**
  - Re-score `ablation_02_sum/layers1_width16/{h_hx, h_hx_q}` and the sum of the `ablation_01` separate models on identical samples and metrics.
  - Retrain the 1×16 sum MLP and the input ablation (local only, + first derivatives) on cache v2 inputs, so the derivative operator is not a confound. These are the Paper 1 *a priori* MLP results.
  - GPU limit: 2 jobs at a time (`%2`).
- **2.6 φ diagnostic (§3.4):** per case and pooled; the residual variance it explains, against δ.
- **2.7 Closed loop.**
  - The trade-off-curve candidates from P1 and P2 (about 5–10 each) on the oracle case, 100 time units, with and without `reinit_every = 10`. Select by §2.5.
  - Run the 1.5 tool on the selected form.
- **Gate G2 (~2 Dec):** if no route-S candidate beats the ML sum, start route J immediately (task 3.1 first). Krishna decides.
- Update `functional-form-v2.md` and `project-status.md`.

### Phase 3: the comprehensive study (from 7 Dec, iterating with the writing)

- **3.1 Routes J and C head-to-head with S.**
  - J: S ≈ ∂ₓ(Θ_F ξ_F) + Θ_T ξ_T. Loss = ‖S res‖² + λ_F‖ΔF res‖² + λ_T‖ΔT_w res‖², sweeping λ. ΔF is fitted at the flux level against `int_u2_dy`, so no data differentiation is needed.
  - C is the limit of large λ, i.e. separate fits.
  - Run all three in closed loop.
- **3.2 Tier 3,** aimed at the high-Ca group (Re10 Ca0.1, Re20 Ca0.06 β90, Re40 Ca0.1).
- **3.3 Robustness.**
  - σ sensitivity (0.25, 0.5, 1.0).
  - Ensemble / stability-selection coefficient distributions.
  - A PySR cross-check on the residual of the best form: test `import pysr` with `envs/pysr_env.sh` first, and cap the compute.
- **3.4 Profile projection** onto φ, ψ_a, ψ_b (ordering matrix §6, item 4).
  - Use 2D snapshots, subsampled, for about 6 cases spanning δ.
  - Look at the amplitude s along the waves against the non-hyperbolic band, and test the row-1 small-deviation assumption.
- **3.5 More closed-loop cases** (from 1.4, if approved), plus broadband statistics for 2–3 broadband cases: h PDF, crest-height distribution, celerity and spectra, for ROM with the form vs DNS vs Lavalle vs ML sum. The ROM's Chang forcing uses the same F₀, M and f_cut. The inlets are not identical, so compare statistics only.
- **3.6 Baseline sensitivity (optional):** one closed-loop run of the selected form with the consistent 9/2 diffusion (`richard-comparison.md` §6.5). Do not change the Paper 1 baseline.
- **3.7 Integration:** a closed-loop run of the selected form plus the current best ΔP_x closure, once that thread has a stable model.
- **3.8 Paper-ready outputs (iterative, no freeze):** figures and tables (the coefficient table against theory; *a priori* and closed-loop tables) generated by scripts, so they can be regenerated whenever results change; keep `functional-form-v2.md` and `project-status.md` current. Before the co-author draft (~4 Jan) and before submission, regenerate everything from the latest results.

## 5. Schedule

| Dates | Work |
|---|---|
| 13–15 Nov | Phase 0 (talk slice); G0 Sun 15 Nov |
| 16–22 Nov | Phase 1 (jobs queued before the talk); DFD talk 22 Nov |
| 23 Nov–6 Dec | Phase 2; G2 ~2 Dec |
| From 7 Dec | Phase 3, iterating with the writing (lighter over the holidays); no hard results freeze |
| ~4 Jan | Draft to co-authors |
| 4–14 Jan | Revisions, journal template |
| 15 Jan 2027 | Submit |

## 6. Decisions still needed from Krishna

- **D1:** the definition of "beats the ML sum" (§2.5).
- **D2:** how the ROM gets δ for route P1 (default ε_lin from linear theory).
- **D3:** the split for the 3 Nov cases (default: extrapolation set, never fitted).
- **D4:** which extra closed-loop cases to use, and the DNS compute for them (task 1.4).
- **D5:** the content of the talk (G0).

**Answers (13 Nov), recorded in `docs/decisions.md`:** D1 on par with `ml_sum_hhx` a priori and a posteriori on Re120 Ca0.01 β45, 10 Hz (tolerance proposed: 10%); D2 ε = O(0.1), δ = Re/10 (with the caveat noted there); D3 3 Nov cases left out until after APS; D4 Liu & Gollub Re 19.33 β 6.4° at 4.5 and 7 Hz (new monochromatic DNS needed), others on 14 Nov; D5 Lavalle theory, Lavalle in and out of range, closure-term ablation, a posteriori comparisons, and the preliminary Phase 0 form.

## 7. Rules for this work (from `CLAUDE.md`)

- **Resources:** at most 2 GPU jobs at a time and 64 CPUs on qos `bala1s`; use `bala1s-b` beyond that, and `hpg-dev` for short tests.
- **Data and git:** no data, figures, logs or caches in git. Never `git pull` in or move the data store. Write outputs under `$THINFILM_STORE/codes/ml-models/functional_forms/<run>/`.
- **DNS** is built against `~/basilisk-old/src` only. When restarting, link the dump as `restart.dump`.
- **Docs:** update `project-status.md` and `decisions.md` in the same commit as the code that changes the science. Every results write-up has a "Where it fails" section. Report numbers with the exact data window, metric and split.

## 8. Risks

- **Collinearity** (q ≈ h³) makes individual coefficients unstable. Mitigations: the (r, g) basis, stability selection, and reporting combined monomial coefficients.
- **The δ ≳ 1 cases** may be beyond any Markovian (h, q) form. The φ diagnostic (2.6) quantifies this, and it is a result in its own right for Paper 1.
- **Closed-loop drift** may come from phase error rather than the closure. The `reinit_every` runs separate the two.
- **Only one oracle case exists today.** Without the 1.4 cases, the closed-loop evidence rests on Re120 Ca0.01 β45 plus the broadband statistics.
