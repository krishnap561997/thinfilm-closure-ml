# Structure of ΔM_x and ΔM_x + ΔT_w (towards a functional form)

_Figures: [`figures/functional-form-structure/`](figures/functional-form-structure/)_

Before searching for closed forms: how do ΔM_x and ΔT_w relate, and does ΔM_x follow from a flux defect ΔF(h, q)? All with t > 1000, 100 < x < 0.95 L, 22 cases from the `ml/functional_mlp` cache.

## Cancellation of ΔM_x and ΔT_w (`scripts/check_mx_tw_cancellation.py`)

The ROM source is S = −ΔM_x − ΔT_w + …, so cancellation means ΔM_x ≈ −ΔT_w.

- corr(ΔM_x, ΔT_w) = −0.93 to −0.999 (−0.975 overall); opposite signs in 86–100% of the large-correction samples.
- Not exact: ΔT_w ≈ −0.64 ΔM_x (slope −0.60 to −0.69, about −0.69 at β = 6.4° and −0.61 at β = 90°). The sum keeps about 36% of ΔM_x; rms(sum) is 0.40–0.55 of the typical rms of the two.
- Consequence: ΔT_w alone has the opposite sign of the net source and 1.8 times its size, which fits the oracle crash with ΔT_w alone (`oracle-a-posteriori.md`). Learn the pair consistently and score the sum.

## Chain-rule test (`scripts/chain_rule_test.py`)

ΔF = ∫u² − q²/h − Λ²h⁵/45 (Λ = 3), so ΔM_x = ∂ₓΔF. Per-case fits on the first half of the snapshots, scored on the second half; degree-4 polynomials in (h, q); Savitzky–Golay window 21, order 3. Mean NRMSE over 22 cases:

| Model | ΔM_x | ΔM_x + ΔT_w |
|---|---|---|
| (h, q) only | 1.05 | 1.03 |
| linear in h_x, q_x | 0.080 | 0.223 |
| + h_xx, q_xx | 0.073 | 0.190 |
| nonlinear in (h, q, h_x, q_x) | 0.057 | 0.134 |
| strict chain rule F_h h_x + F_q q_x | 0.119 | – |

- corr(∂ₓΔF, ΔM_x) ≥ 0.990 in every case; ΔF ≈ F(h, q) with R² ≈ 0.997.
- ΔM_x is nearly linear in first gradients with coefficients depending on (h, q); second derivatives add little.
- The strict chain rule is worse than the free linear form, likely because differentiating a fitted F amplifies its error (untested; fit F through its gradient instead).
- The sum is harder: first gradients are not enough (0.223), second derivatives help (0.190), and no flux-defect shortcut applies.
- Hardest: large-Ca cases (Re40 Ca0.1 β45, Re20 Ca0.06 β90, Re10 Ca0.1 β45). Re40 Ca0.04 β45 (held out) overfits with the nonlinear model (sum 0.181 → 0.301).

## Caveats

One degree, one derivative window, one chronological split; per-case fits, so no parameter dependence yet.

## Pooled sparse regression (`ml/functional_forms/fit_sparse.py`, job 48911517)

Greedy forward selection on base(h,q) × gradient × parameter factor, trained on the 15 development cases (t > 1000), each case weighted by 1/rms_c; selection redone inside every leave-one-case-out (LOCO) fold. Parameter factors: 1, Re, 1/Re, 1/We, 1/Fr², cosβ/Fr². NRMSE here = RMSE/rms(truth), per case, averaged. Output: `$THINFILM_STORE/codes/ml-models/functional_forms/sparse_01/sparse_path.json`.

**ΔM_x:**

| terms | train | LOCO | 7 held-out | β = 6.4° holdout |
|---|---|---|---|---|
| 3 | 0.135 | 0.141 | 0.113 | 0.152 |
| 5 | 0.119 | 0.129 | 0.093 | 0.127 |
| 15 | 0.100 | 0.123 | 0.083 | 0.353 |
| 20 | 0.096 | 0.207 | 0.081 | 0.361 |

5-term form (all parameter factors = 1): ΔM_x ≈ −2.067 h²q h_x + 0.273 q q_x + 0.102 h q² h_x + 0.109 q_x + 0.668 q² h_x. Held-out per case 0.05–0.15 (worst Re80 Ca0.005 β6.4).

- ΔM_x is described by a single function of (h, q, h_x, q_x) with no Re, Ca, β in the fitted terms; it generalises across β (β-holdout 0.127 ≈ LOCO 0.129).
- Adding Re/We/Fr² terms does not improve LOCO and makes the β holdout much worse (0.31–0.36): overfitting.
- Gap to the per-case linear fit (≈ 0.08): some parameter dependence is not captured by this library.

**ΔM_x + ΔT_w:** no adequate sparse form. LOCO plateaus at 0.35–0.40 (4–20 terms), held-out 0.23–0.29, worst Re80 Ca0.005 (0.43–0.52); the per-case nonlinear model reached 0.134. The first selected terms carry Re factors with tiny coefficients, which suggests the library lacks the right terms.

**Caveats:** metric (rms-normalised) and time window differ from `mlp-heldout-eval.md`, so no MLP comparison yet; q ≈ h³ makes the monomials collinear (coefficients not individually interpretable); greedy selection is noisy (LOCO not monotonic in k).

## What drives the error? (`scripts/error_vs_parameters.py`)

The 5-term ΔM_x form and the 9-term sum form evaluated on all 22 cases (t > 1000), Spearman correlation of the per-case relative error (RMSE/rms of the truth) with the case parameters. Dev cases are in-sample for the pooled fit.

| | ΔM_x all 22 | ΔM_x held-out 7 | sum all 22 | sum held-out 7 |
|---|---|---|---|---|
| We | +0.12 | −0.21 | +0.11 | −0.07 |
| Fr² | −0.71 | −0.61 | −0.75 | −0.71 |
| β | −0.39 | −0.47 | −0.49 | −0.76 |
| Re | −0.44 | −0.47 | −0.45 | −0.20 |
| rms(ΔC)/rms(M_x,Lav) | −0.60 | −0.57 | −0.62 | −0.79 |

- **We is not the driver.** The lowest-We cases are well predicted (Re40 Ca0.01 β90, We = 0.4: 0.047; Re20 Ca0.01 β45, We = 0.2: 0.092). The earlier reading of the Re80 Ca0.005 β6.4 error as a low-We effect was wrong; it is a low-β effect.
- **Low Fr², small corrections are hard in relative terms.** The β = 6.4° cases have ΔM_x of only 22–30% of the Lavalle term (rms), NRMSE 0.08–0.15 against 0.05–0.09 for most β = 45° and 90° cases. Their absolute RMSE is nearly constant (0.007–0.010); an error floor of about 0.01 (derivative smoothing or extraction noise?) would explain this. Untested.
- **A second hard group is high Ca** (Re10 Ca0.1, Re20 Ca0.06, Re40 Ca0.1 at β45/90): NRMSE 0.22–0.27 (ΔM_x) and 0.43–0.56 (sum), the largest absolute errors (0.11–0.20). Separate failure, probably missing terms for strongly nonlinear, low-σ waves.
- Caveats: Fr² = Re sinβ/3, so Fr², β and relative correction size are confounded (22 points cannot separate them); correlations only; two sparse forms, not the MLP.

## Open items

1. Test whether a ~0.01 absolute error floor comes from derivative smoothing or extraction noise (vary the Savitzky–Golay window, compare with the level-15 mesh data if available).
2. Re-score the 1×16 MLP and the 5-term form on the same t > 1000 data, metric and nonlinear region.
2. Better library for the sum: use s = q − h³ instead of q, h_xx/q_xx with Re and We factors.
3. Fit F through its gradient (strict chain rule) and test a collinearity-reduced basis for ΔM_x.
4. PySR cross-check: setup (Julia download) was not confirmed finished; test `import pysr` with `envs/pysr_env.sh`.
5. Pending reminder: verify the early-transient claim behind the t > 1000 cut (see `docs/decisions.md`, 2026-11-01).
