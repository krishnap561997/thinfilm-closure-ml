# 1×16 MLP on held-out cases (ΔM_x, ΔT_w)

_Figures: [`figures/mlp-heldout-eval/`](figures/mlp-heldout-eval/)_

Is the shallow MLP good enough on unseen cases, especially in the nonlinear wave regions, to justify looking for a closed functional form in (h, q, h_x, q_x, Re, Ca, β)?

## Setup

- **Models:** `ml/functional_mlp`, 1 hidden layer × 16, inputs h, q, h_x, q_x plus Re, Ca, angle (`first_derivatives`), seed 42, one model per target. Checkpoints: `$THINFILM_STORE/codes/ml-models/functional_mlp/runs/ablation_01/layers1_width16/first_derivatives/<target>/seed42/`.
- **Evaluation:** `evaluate.py --split heldout` (jobs 48907123–48907126, heldout and val for both targets). Held-out cases are excluded from training and normalisation: Re20 Ca0.01 β45, Re40 Ca0.04 β45, Re80 Ca0.005 β6.4, Re80 Ca0.01 β90, Re120 Ca0.01 β45, Re120 Ca0.02 β90, Re200 Ca0.01 β6.4.
- **Summary:** `scripts/mlp_wave_region_summary.py`. Nonlinear region: x > 100 and (h > 1.1 or h < 0.9), no h_x threshold. Errors are RMSE / std of the truth over the whole case. Baseline is ΔC = 0 (Lavalle alone).

## Results

Validation split reproduces the earlier numbers: pooled NRMSE 0.168 (ΔM_x), 0.225 (ΔT_w).

| | pooled NRMSE (all) | NRMSE nonlinear | baseline nonlinear | R² nonlinear | NRMSE band | NRMSE inlet |
|---|---|---|---|---|---|---|
| ΔM_x | 0.109 | 0.130 | 1.236 | 0.989 | 0.066 | 0.011 |
| ΔT_w | 0.100 | 0.121 | 1.233 | 0.990 | 0.048 | 0.011 |

Per case, nonlinear-region NRMSE (ΔM_x / ΔT_w):

| Case | ΔM_x | ΔT_w |
|---|---|---|
| Re20 Ca0.01 β45 | 0.089 | 0.110 |
| Re40 Ca0.04 β45 | 0.107 | 0.078 |
| Re80 Ca0.005 β6.4 | 0.286 | 0.226 |
| Re80 Ca0.01 β90 | 0.057 | 0.057 |
| Re120 Ca0.01 β45 | 0.082 | 0.068 |
| Re120 Ca0.02 β90 | 0.195 | 0.186 |
| Re200 Ca0.01 β6.4 | 0.178 | 0.149 |

- The nonlinear region holds 49–75% of the samples, so it is most of the domain, not a few crests.
- Median peak amplitude error of max|ΔC| per snapshot: 1.5–3.3% for ΔM_x; 1.4–3.6% for ΔT_w except Re200 β6.4 (8.2%). Peak position is within one grid cell (ΔT_w Re80 Ca0.005: two).
- Bias is below 0.05σ everywhere.
- Worst case for both targets: Re80 Ca0.005 β6.4 (We = 0.4). The budget check flags the same case, but that concerns the one-step ∂ₜ; the error is not tied to low We (see "What drives the error" in `functional-form-structure.md`: it follows Fr² and small correction size). The β = 6.4° cases are the weakest group.

## Caveats

- Held-out cases are interleaved with the development cases in (Re, Ca, β) and share the time window: this is interpolation, not extrapolation.
- Single seed.
- Not compared with the 4×64 or full-derivative models on the held-out set.

## Is the Re80 Ca0.005 β6.4 error a data-quality problem? (no)

`scripts/mlp_error_vs_mass_residual.py`: NRMSE (x > 100) at the 11 snapshots shared with the budget check (t = 1000–2000 every 100), against the per-snapshot mass residual.

- Re80 Ca0.005 β6.4: NRMSE on snapshots with mass residual < 0.05 is 0.225 (ΔM_x) / 0.185 (ΔT_w); on snapshots with residual > 0.1 it is 0.221 / 0.190. Rank correlation 0.25 / 0.47 over 11 points. The error is the same on good and poor snapshots, so it is not noisy targets: the model is genuinely less accurate at this low-β point (low We is not the driver, see `functional-form-structure.md`). Only one snapshot is < 0.05 and few cases have enough of both kinds to say more.
- By time, the error in this case and in Re200 Ca0.01 β6.4 is flat from t = 200 on (0.22 and 0.13 for ΔM_x), i.e. a systematic misfit and not a startup transient.
- Re120 Ca0.02 β90 behaves differently: 0.325 (ΔM_x) / 0.303 (ΔT_w) at t = 200–500, 0.16 at 500–1000, and 0.08–0.10 for t ≥ 1000. Its large held-out error is an early-time transient (waves still developing); the developed state is predicted well. Its 0.195 / 0.186 in the table above is dominated by t < 1000, while the budget check and the oracle use t ≥ 1000.

## Fewer inputs: (h, h_x) only (4 Nov 2026)

Two more cases of the same ablation (`ablation_01/layers1_width16/h_hx/{delta_Mx_ML,delta_Tw_ML}/seed42`, tasks 16–17, SLURM job 49560900): 1×16, 100 epochs, no early stopping, seed 42, same cache. Inputs h and h_x plus Re, Ca, angle (no q, no q_x). Held-out evaluation with `evaluate.py --split heldout` and `scripts/mlp_wave_region_summary.py` as above; validation table from `scripts/ablation_summary.py`.

**Validation** (sqrt of normalised MSE, 100 epochs; the best epoch is the last one for every row except the local baseline):

| Inputs (plus Re, Ca, β) | ΔM_x | ΔT_w |
|---|---|---|
| h, q | 0.949 | 0.953 |
| h, q, q_x | 0.230 | 0.263 |
| h, q, h_x | 0.177 | 0.222 |
| **h, h_x** | **0.171** | **0.213** |
| h, q, h_x, q_x | 0.158 | 0.212 |
| h, q, h_x, q_x, h_xx, q_xx | 0.154 | 0.211 |

**Held-out, nonlinear region** (NRMSE / σ_case; Lavalle alone ≈ 1.24):

| | (h, h_x) | (h, q, h_x, q_x) |
|---|---|---|
| ΔM_x pooled / mean over the 7 cases | 0.146 / 0.164 | **0.130 / 0.142** |
| ΔT_w pooled / mean over the 7 cases | **0.115** / 0.135 | 0.121 / **0.125** |
| R² nonlinear (ΔM_x / ΔT_w) | 0.986 / 0.991 | 0.989 / 0.990 |

Per case, nonlinear-region NRMSE, (h, h_x) vs (h, q, h_x, q_x):

| Case | ΔM_x | ΔT_w |
|---|---|---|
| Re20 Ca0.01 β45 | 0.099 vs 0.089 | 0.132 vs 0.110 |
| Re40 Ca0.04 β45 | 0.129 vs 0.107 | 0.074 vs 0.078 |
| Re80 Ca0.005 β6.4 | 0.248 vs 0.286 | 0.250 vs 0.226 |
| Re80 Ca0.01 β90 | 0.068 vs 0.057 | 0.059 vs 0.057 |
| Re120 Ca0.01 β45 | 0.101 vs 0.082 | 0.082 vs 0.068 |
| Re120 Ca0.02 β90 | 0.210 vs 0.195 | 0.161 vs 0.186 |
| Re200 Ca0.01 β6.4 | **0.296** vs 0.178 | 0.186 vs 0.149 |

- h and h_x alone remove about 90% of the Lavalle error in the nonlinear region (R² ≥ 0.986 for both targets). Peak positions stay within one grid cell (≤ 0.07) and peak-amplitude errors are 0.7–6.4%.
- Adding q to (h, h_x) does not help: (h, q, h_x) is worse than (h, h_x) on validation for both targets, so q carries no useful information once h_x is present. q_x is what gives the remaining gain (ΔM_x 0.171 → 0.158 on validation).
- ΔT_w: equal on validation (0.213 vs 0.212) and slightly better pooled on held-out, but worse in 5 of 7 cases and 8% worse on the case mean; the pooled value is dominated by the large-σ cases. Not an improvement.
- ΔM_x: 12–15% worse on held-out, and the loss is concentrated at Re200 Ca0.01 β6.4 (0.296 vs 0.178, bias −0.088σ, peak-amplitude error 6.4%). This suggests q_x (or q) matters at high Re and low β; not tested directly. Re80 Ca0.005 β6.4 improves for ΔM_x (0.286 → 0.248) but remains among the worst.
- Caveats: single seed, so differences of 0.01–0.02 may be within seed spread (not measured); the models are not converged at 100 epochs; held-out cases are interleaved with the development cases.

## Sum target ΔM_x + ΔT_w: (h, h_x, q) vs (h, h_x) (5 Nov 2026)

Two models for the combined correction that enters S_ML, for the *a posteriori* tests (`ablation_02_sum/layers1_width16/{h_hx_q,h_hx}/delta_MxTw_ML/seed42`, SLURM job 49820013). The target `delta_MxTw_ML` = `delta_Mx_ML` + `delta_Tw_ML` is built on the fly in `ml/functional_mlp/datasets.py`; the cache is unchanged. 1×16, 100 epochs, no early stopping, seed 42, same cache and splits as `ablation_01`. Inputs: (h, h_x, q) or (h, h_x), plus Re, Ca, angle. Evaluated with `evaluate.py` (val, heldout) and `scripts/mlp_wave_region_summary.py`.

NRMSE here is normalised by the std of the *sum*, which is smaller than that of ΔM_x or ΔT_w because they partly cancel. The numbers are therefore not comparable with the single-target tables above.

| | (h, h_x, q) | (h, h_x) |
|---|---|---|
| Validation √loss (final epoch) | 0.300 | **0.289** |
| Validation pooled NRMSE / R² | 0.318 / 0.899 | **0.306 / 0.906** |
| Held-out pooled NRMSE / R² | 0.260 / 0.933 | **0.243 / 0.941** |
| Held-out nonlinear region, pooled NRMSE (Lavalle alone 1.235) | 0.308 | **0.287** |
| Held-out R² nonlinear | 0.938 | **0.946** |

Held-out nonlinear-region NRMSE per case, (h, h_x, q) vs (h, h_x):

| Case | NRMSE | Bias (h, h_x) |
|---|---|---|
| Re20 Ca0.01 β45 | **0.220** vs 0.269 | 0.005 |
| Re40 Ca0.04 β45 | 0.314 vs **0.291** | 0.034 |
| Re80 Ca0.005 β6.4 | **0.557** vs 0.672 | **−0.308** |
| Re80 Ca0.01 β90 | 0.176 vs **0.159** | −0.010 |
| Re120 Ca0.01 β45 | 0.219 vs **0.206** | −0.013 |
| Re120 Ca0.02 β90 | 0.398 vs **0.356** | 0.016 |
| Re200 Ca0.01 β6.4 | 0.600 vs **0.563** | 0.073 |

- Both remove 75–80% of the Lavalle error in the nonlinear region (1.235 → 0.29–0.31), but the sum is about twice as hard as either component (0.25 vs 0.11 pooled held-out): the small residual of two large, anticorrelated terms is predicted less accurately.
- (h, h_x) is slightly better than (h, h_x, q) in aggregate (validation, held-out pooled, nonlinear region), consistent with q adding nothing once h_x is present. The margin is 5–8% and from a single seed, so it is not significant.
- Per case the picture is mixed: (h, h_x) wins in 5 of 7 cases but is clearly worse at Re20 Ca0.01 β45 (0.269 vs 0.220) and at Re80 Ca0.005 β6.4 (0.672 vs 0.557). The latter has a large negative nonlinear-region bias (−0.308σ) and an inlet error of 0.34σ, against 0.27σ for the model with q.
- Both are poor at the two β = 6.4° cases (NRMSE 0.56–0.67 and 0.56–0.60, R² 0.77–0.84). At Re200 Ca0.01 β6.4 the median peak-position offset (pkDx) is 1.7 for both models, against ≤ 0.2 elsewhere. The sum for these cases may need more than local (h, h_x).
- Caveats: single seed, not converged at 100 epochs, and no *a posteriori* result yet; a small *a priori* NRMSE on the sum does not guarantee a stable ROM.

## Where it fails

Figures: [`limit01`](figures/mlp-heldout-eval/limit01_error_vs_time.png), [`limit02`](figures/mlp-heldout-eval/limit02_learning_curves.png), [`limit03`](figures/mlp-heldout-eval/limit03_worst_case_errors.png); the R² versions of the ablation figures are `fig04` and `fig05`.

- **Worst held-out cases.** Re80 Ca0.005 β6.4: nonlinear-region NRMSE 0.286 (ΔM_x) and 0.226 (ΔT_w), R² 0.954 and 0.970 for inputs (h, q, h_x, q_x). With inputs (h, h_x) only, Re200 Ca0.01 β6.4 reaches 0.296 for ΔM_x (R² 0.947). The residuals are spikes of 1–1.4 standard deviations at wave fronts and ripples, on top of a smooth offset of about 0.1–0.2 standard deviations in Re80 Ca0.005 (`limit03`).
- **Early-time transient and systematic misfit are different problems.** Re120 Ca0.02 β90 has NRMSE 0.33 for t < 500, 0.18 for 500–1000 and 0.09 for 1500–2000 (ΔM_x). In the β = 6.4° cases the error does not depend on time (0.22–0.24 for Re80 Ca0.005, 0.14–0.16 for Re200 Ca0.01, ΔM_x; `limit01`).
- **Inputs without gradients never learn.** With only (h, q) the validation NRMSE stays at 0.95 for both targets for all 100 epochs (`limit02`).
- **Training is not converged.** For every input set with gradients the validation NRMSE still fell by 0.003–0.013 (1–8%) over the last 10 epochs; the best epoch is the last one (`limit02`).
- **The comparisons are weak.** One seed (42); differences between input sets of 0.005–0.02 in NRMSE are not shown to exceed the seed-to-seed spread. The held-out cases lie between development cases in (Re, Ca, β), so this tests interpolation only.
- **R² makes the errors look small.** For standardised targets R² ≈ 1 − NRMSE², so an NRMSE of 0.2 is an R² of 0.96 (`fig04` and `fig05` against `fig01` and `fig02`).

## Open items

1. Decide how to weight the β = 6.4° (small-correction) cases when designing the functional form; their error is systematic.
2. ~~Decide whether the form is fitted on t ≥ 1000 only or on the whole record.~~ Decided 10 Nov: the whole record (`docs/decisions.md`).
3. Measure the seed-to-seed spread (2 more seeds for the (h, h_x) and (h, q, h_x, q_x) groups) before drawing conclusions from differences below ~0.02.
4. Test whether adding q_x (or q) to (h, h_x) removes the Re200 β6.4 ΔM_x gap, i.e. whether q_x is needed at high Re and low β.
