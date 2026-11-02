# 1×16 MLP on held-out cases (ΔM_x, ΔT_w)

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
- Worst case for both targets: Re80 Ca0.005 β6.4 (We = 0.4), a case the budget check already flags (`budget-check.md`). The β = 6.4° cases are the weakest group.

## Caveats

- Held-out cases are interleaved with the development cases in (Re, Ca, β) and share the time window: this is interpolation, not extrapolation.
- Single seed.
- Not compared with the 4×64 or full-derivative models on the held-out set.

## Is the Re80 Ca0.005 β6.4 error a data-quality problem? (no)

`scripts/mlp_error_vs_mass_residual.py`: NRMSE (x > 100) at the 11 snapshots shared with the budget check (t = 1000–2000 every 100), against the per-snapshot mass residual.

- Re80 Ca0.005 β6.4: NRMSE on snapshots with mass residual < 0.05 is 0.225 (ΔM_x) / 0.185 (ΔT_w); on snapshots with residual > 0.1 it is 0.221 / 0.190. Rank correlation 0.25 / 0.47 over 11 points. The error is the same on good and poor snapshots, so it is not noisy targets: the model is genuinely less accurate at this low-We, low-β point. Only one snapshot is < 0.05 and few cases have enough of both kinds to say more.
- By time, the error in this case and in Re200 Ca0.01 β6.4 is flat from t = 200 on (0.22 and 0.13 for ΔM_x), i.e. a systematic misfit and not a startup transient.
- Re120 Ca0.02 β90 behaves differently: 0.325 (ΔM_x) / 0.303 (ΔT_w) at t = 200–500, 0.16 at 500–1000, and 0.08–0.10 for t ≥ 1000. Its large held-out error is an early-time transient (waves still developing); the developed state is predicted well. Its 0.195 / 0.186 in the table above is dominated by t < 1000, while the budget check and the oracle use t ≥ 1000.

## Open items

1. Decide how to weight the β = 6.4° (and low-We) cases when designing the functional form; their error is systematic.
2. Decide whether the form is fitted on t ≥ 1000 only (developed waves) or on the whole record.
