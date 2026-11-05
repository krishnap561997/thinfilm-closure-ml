# ΔP_x with the stencil MLP (validation data)

Can a neighbour-stencil MLP predict the pressure-gradient correction ΔP_x as well as ΔM_x and ΔT_w? On validation data it cannot: pooled R² is 0.70 for ΔP_x against 0.99 for the other two, and it is about 0 in three cases. This write-up has no figures; the numbers come from `metrics.csv` of the run below.

## Setup

- **Model:** `ml/thinfilm_mlp`, neighbour-stencil MLP, 4 hidden layers × 64, half-window J = 6 with stride 3 (half-width 0.88 h_N), trained on all four corrections (ΔD_x, ΔM_x, ΔP_x, ΔT_w) for 100 epochs, seed 42. Run: `$THINFILM_STORE/codes/ml-models/thinfilm_mlp/runs/sweep_01/corrections_j6_s3/`.
- **Data:** the validation split of the 20 development cases (`eval_val/metrics.csv`, from `evaluate.py --split val`). This is not held-out data. R² and NRMSE (RMSE / std of the truth) are per case.

## Results

Pooled: ΔP_x R² 0.704, NRMSE 0.544; case average R² 0.626, NRMSE 0.563. For comparison, pooled R² is 0.992 (ΔM_x), 0.993 (ΔT_w) and 0.978 (ΔD_x).

| Case | Re | Ca | β | We | R² ΔP_x | NRMSE ΔP_x | R² ΔM_x | R² ΔT_w |
|---|---|---|---|---|---|---|---|---|
| Re40 Ca0.002 β45 | 40 | 0.002 | 45 | 0.08 | 0.007 | 0.996 | 0.991 | 0.978 |
| Re10 Ca0.005 β45 | 10 | 0.005 | 45 | 0.05 | 0.001 | 1.000 | 0.954 | 0.954 |
| Re80 Ca0.005 β45 | 80 | 0.005 | 45 | 0.4 | 0.672 | 0.572 | 0.997 | 0.997 |
| Re120 Ca0.005 β45 | 120 | 0.005 | 45 | 0.6 | 0.706 | 0.542 | 0.997 | 0.998 |
| Re120 Ca0.005 β90 | 120 | 0.005 | 90 | 0.6 | 0.528 | 0.687 | 0.998 | 0.998 |
| Re200 Ca0.005 β6.4 | 200 | 0.005 | 6.4 | 1 | 0.481 | 0.720 | 0.978 | 0.988 |
| Re10 Ca0.01 β45 | 10 | 0.01 | 45 | 0.1 | 0.029 | 0.986 | 0.979 | 0.988 |
| Re40 Ca0.01 β90 | 40 | 0.01 | 90 | 0.4 | 0.731 | 0.518 | 0.998 | 0.998 |
| Re50 Ca0.01 β45 | 50 | 0.01 | 45 | 0.5 | 0.679 | 0.567 | 0.997 | 0.998 |
| Re80 Ca0.01 β45 | 80 | 0.01 | 45 | 0.8 | 0.468 | 0.729 | 0.988 | 0.992 |
| Re120 Ca0.01 β6.4 | 120 | 0.01 | 6.4 | 1.2 | 0.716 | 0.533 | 0.987 | 0.991 |
| Re120 Ca0.01 β90 | 120 | 0.01 | 90 | 1.2 | 0.816 | 0.429 | 0.996 | 0.997 |
| Re200 Ca0.01 β45 | 200 | 0.01 | 45 | 2 | 0.729 | 0.521 | 0.990 | 0.992 |
| Re40 Ca0.02 β6.4 | 40 | 0.02 | 6.4 | 0.8 | 0.621 | 0.616 | 0.981 | 0.989 |
| Re80 Ca0.02 β45 | 80 | 0.02 | 45 | 1.6 | 0.924 | 0.276 | 0.996 | 0.997 |
| Re200 Ca0.02 β6.4 | 200 | 0.02 | 6.4 | 4 | 0.837 | 0.404 | 0.972 | 0.985 |
| Re20 Ca0.06 β90 | 20 | 0.06 | 90 | 1.2 | 0.950 | 0.224 | 0.987 | 0.990 |
| Re10 Ca0.1 β45 | 10 | 0.1 | 45 | 1 | 0.983 | 0.132 | 0.993 | 0.996 |
| Re40 Ca0.1 β45 | 40 | 0.1 | 45 | 4 | 0.927 | 0.269 | 0.989 | 0.987 |
| Re120 Ca0.1 β45 | 120 | 0.1 | 45 | 12 | 0.712 | 0.537 | 0.963 | 0.956 |

- **R² is about 0 in three cases**, all with We ≤ 0.1: Re40 Ca0.002 β45 (0.007), Re10 Ca0.005 β45 (0.001) and Re10 Ca0.01 β45 (0.029). There the model learns nothing about ΔP_x.
- **R² is at least 0.9 in only four cases**: Re80 Ca0.02 β45 (0.924), Re20 Ca0.06 β90 (0.950), Re10 Ca0.1 β45 (0.983) and Re40 Ca0.1 β45 (0.927). The other 13 cases lie between 0.47 and 0.84 (median over all 20: 0.71).
- **It is not just Ca.** Spearman correlation of R² with log Ca is 0.73, with log We 0.69, with log Re 0.06 and with β 0.19 (20 cases). `project-status.md` quotes 0.65 and 0.31 for log Ca and log Re from an earlier calculation on a slightly different case set; I have not reconciled the two. Re120 Ca0.1 β45 (We = 12) reaches only 0.71.
- **ΔM_x and ΔT_w are predicted in every case** (R² 0.954–0.998 for both), so the problem is specific to ΔP_x.

## What it suggests

- ΔP_x contains the capillary pressure term, which depends on the third derivative of h; its failure at low Ca (strong surface tension) suggests the stencil in (h, q) does not carry enough information. The curvature input in `ml/thinfilm_mlp/augment_curvature.py` correlates better with ΔP_x but has not been evaluated on this split; a memory (non-Markovian) term is untested.
- **Part of the failure may be noisy targets rather than a model limit.** The three R² ≈ 0 cases include the one where the one-step ∂ₜ and the momentum budget do not close (Re10 Ca0.01 β45, see `budget-check.md`), and the mesh study found the extracted ΔP_x not mesh-converged (relative difference 0.87 between level 14 and 15, 0.44 after smoothing; see `mesh-sensitivity.md`).
- ΔP_x also has the most high-frequency content in time of the corrections (`oracle-a-posteriori.md`, fig04). In the oracle the run with all four terms stops at the outlet; ΔP_x has not been shown to be harmful in the interior.

## Caveats

- Validation data of the development cases only; no held-out evaluation exists for ΔP_x.
- One seed, one stencil, one model class; the target is the extracted ΔP_x including its grid-scale noise.

## Open items

1. Held-out evaluation of ΔP_x.
2. The curvature input on the same split, and the same models against a smoothed ΔP_x target.
3. A memory term if curvature does not help.
4. Decide whether Paper 1's *a posteriori* run needs ΔP_x: the oracle with ΔT_w + ΔM_x only is stable for 100 time units without it.
