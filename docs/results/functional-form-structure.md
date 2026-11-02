# Structure of ΔM_x and ΔM_x + ΔT_w (towards a functional form)

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

## Open items

1. Re-score the 1×16 MLP and the 5-term form on the same t > 1000 data, metric and nonlinear region.
2. Better library for the sum: use s = q − h³ instead of q, h_xx/q_xx with Re and We factors.
3. Fit F through its gradient (strict chain rule) and test a collinearity-reduced basis for ΔM_x.
4. PySR cross-check: setup (Julia download) was not confirmed finished; test `import pysr` with `envs/pysr_env.sh`.
5. Pending reminder: verify the early-transient claim behind the t > 1000 cut (see `docs/decisions.md`, 2026-11-01).
