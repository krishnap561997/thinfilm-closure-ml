# Functional form for S = ΔM_x + ΔT_w, v2 (Tier-1 library)

Plan: `docs/plans/functional-form-plan.md`. Earlier library and structure tests: `functional-form-structure.md`.

## Phase 0 (13 Nov 2026) — PRELIMINARY

**Preliminary:** Savitzky–Golay derivatives (D_SG, window 21, order 3) on the existing `functional_mlp` cache. Phase 1 replaces both (cache v2, Gaussian D_G, matched to the ROM inputs).

- **Code:** `ml/functional_forms/library.py` (Tier-1 columns as sympy expressions, flux terms expanded by the chain rule), `fit_form.py`, `hygiene_phase0.json`; figures `scripts/figures/functional-form-v2/phase0_figures.py`; monomial table `scripts/functional_form_monomials.py`. Branch `functional-form-v2`.
- **Run:** job 51181562 (54 s). Output `$THINFILM_STORE/codes/ml-models/functional_forms/phase0_tier1/` (`phase0.json`, `figures/fig{1,2,3}_*.{png,pdf}`).
- **Data:** 22 cases (15 dev, 7 held-out), 100 < x < 0.95 L, every 4th x. Windows: entire record (default, decision of 10 Nov) and t > 1000 (only to compare like for like with `sparse_01`).
- **Hygiene (task 0.2):** the cache arrays date from 22 Oct, before the 10 Nov overwrites, so they hold the original data; the listed snapshots are dropped anyway: Re050 t > 1050 (95 snapshots, unforced restart), Re10 Ca0.01 t = 1960, Re10 Ca0.1 t = 1560 and 1570, Re40 Ca0.1 t = 1810 (1820 is not in the cache).
- **Checks:** the expanded flux columns F1–F4 match a numerical ∂ₓ of their fluxes (corr ≥ 0.9997, 0.1–2.6% rms difference from SG smoothing); raw and (r, g) bases give identical theory monomials. cot β is set exactly to 0 at β = 90°, so T5 and F4 are not identifiable there (reported as missing).
- **Metric:** NRMSE = RMSE/std(truth) per case, averaged (plan M1), unless marked "rms" (RMSE/rms, as in `sparse_01`). Lavalle alone (S = 0) gives 1.00 in every case.

### Results

**(a) Per-case Tier-1 fits (9 columns, theory parameter factors, entire record, in-sample).** NRMSE 0.07–0.41, mean 0.22. Best: Re120 Ca0.005 β90 (0.066), Re120 Ca0.01 β90 (0.104). Worst: the high-Ca group (Re20 Ca0.06 β90 0.41, Re40 Ca0.1 0.40, Re10 Ca0.1 0.36) and β = 6.4° at Re 40–80 (0.34–0.35).

**Theory coefficients as they are (row-1 q-gauge, no fit):** better than Lavalle in only 7 of 22 cases (NRMSE 0.45–0.90), all with Ca ≤ 0.01 and Re ≤ 120; 1.06–3.36 in the others.

**(b) Pooled forward selection (Tier-1 columns × parameter factors).**

| | best k | LOCO | held-out (7) | β = 6.4° holdout |
|---|---|---|---|---|
| old library `sparse_01`, t > 1000 (rms) | 19 | 0.322 | – | – |
| Tier 1 raw, t > 1000 (rms) | 3 | 0.322 | 0.291 | 0.390 |
| Tier 1 raw, entire record | 3 | 0.329 | 0.289 | 1.30 |
| Tier 1 (r, g), entire record | 3 | 0.328 | 0.294 | 1.30 |

Best 3-term form (raw, entire record), all with parameter factor 1: S ≈ 0.0713·F2 + 0.0881·T3 + 0.1201·F1, i.e. 0.0713 ∂ₓ[(q − h³)h²] + 0.0881 q q_x/h + 0.1201 ∂ₓ[(q − h³)q/h]. Theory: 0.2, 1/35 and 8/35.

- Tier 1 matches the best old-library accuracy with **3 terms instead of 19**, but is not more accurate. Beyond 3–4 terms LOCO gets worse (added parameter-factor terms overfit).
- The three selected terms are the flux-disequilibrium fluxes (F1, F2) and the inertial T3. None carries a parameter factor.
- Per-case fits (mean 0.22) are much better than the pooled form (LOCO 0.33): the structure fits each case, the **coefficients vary between cases** and the fixed parameter factors do not capture that variation.

**(c) Case placement.** ε_k = k_peak h_N (time-averaged spectrum of h over x > 100): 0.048–0.30, median about 0.086; ε_g = rms(h_x)/mean(h): 0.07–0.17. δ_k agrees with δ = Re/10 (decision D2) within a factor of about 2, except Re80 Ca0.01 β45 (ε_k = 0.30, δ_k = 24 vs 8), Re200 Ca0.01 β6.4 (34 vs 20) and Re200 Ca0.01 β45 (9.5 vs 20). Ordering-matrix rows (δ < 3.2: row 1; 3.2–32: row 2): **only the Re ≤ 20 cases are in row 1** (plus Re40 Ca0.01 β90 and Re40 Ca0.1 β45 by ε_k); everything else is row 2, where the two-equation slaved closure is formally inconsistent (ordering matrix §4).

**Monomial check against theory (§3.5; `scripts/functional_form_monomials.py`).** Comparing combined monomial coefficients (identifiable, unlike the per-column ones) for Re10 Ca0.01, Re20 Ca0.01, Re10 Ca0.1 and Re120 Ca0.01 β45:

- h⁴h_x (Lavalle's own term): theory −0.40, fits −0.30 to −0.59. Agrees.
- Inertial terms with factor Re (from F3): same sign as theory, about half its size at Re 10–20 and about 1/10 at Re 120, as expected if row-1 theory over-predicts beyond δ ~ 1.
- **Relaxation term T1 (h³ − q)/(Re h²): opposite sign to theory** at Re10 and Re20 Ca0.01 (fit −0.45 and −0.56 against +0.5), about 0 at Re10 Ca0.1. The theory sign follows from s_a = (q_eq − q)/(2q) and matches the definition of ΔT_w. Not resolved; one candidate is that Tier 1 uses h³ in place of q_eq, which also contains the capillary h³h_xxx/(3Ca) (a Tier-3 term).

### Where it fails

- **β = 6.4°:** a form trained without the β = 6.4° cases predicts them badly on the entire record (NRMSE 1.30, worse than Lavalle; 0.39 on t > 1000). Low-β cases need terms or parameter dependence that the 45°/90° cases do not constrain.
- **High-Ca cases** (Ca ≥ 0.06 at Re ≤ 40): NRMSE 0.36–0.41 even per case. Tier 3 (capillary) terms are aimed at these (plan task 3.2).
- **Theory coefficients:** do not transfer as they are beyond the lowest-δ, low-Ca cases, and the sign of T1 disagrees even there.
- **Coefficients across cases:** the per-case/pooled gap (0.22 vs 0.33) is the main missing piece: routes P1 (coefficient trends) and P2 (pooled parametric) of Phase 2.
- Not yet compared with the ML sum model on the same samples (task 2.5); no closed-loop run (stretch task 0.5 not done).

### Slide summary (talk, preliminary)

1. **`fig1_loco_vs_terms`** — A physics-derived Tier-1 library reaches the old 19-term library's leave-one-case-out error (0.32) with 3 terms.
2. **`fig2_coef_vs_delta`** — Per-case coefficients against δ = Re/10: Lavalle's h⁴h_x term is recovered, the inertial terms fall below row-1 theory as δ grows, and the coefficients spread between cases, which a single pooled form cannot capture yet.
3. **`fig3_percase_nrmse`** — Per case, the best 3-term pooled form cuts the error of Lavalle alone by 33–91% (NRMSE 0.09–0.67); worst at β = 6.4° and high Ca.

Figures: `$THINFILM_STORE/codes/ml-models/functional_forms/phase0_tier1/figures/`.
