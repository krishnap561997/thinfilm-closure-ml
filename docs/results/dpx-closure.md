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

## Extraction (10 Nov 2026)

Is part of the failure the target? `pipeline/extract_closures/dpx_diagnostics.py` (job 50636699) recomputed ΔP_x on 2D snapshots of 6 cases (t = 100–1900; fewer where runs stop early) and on the mono L14/L15 runs (t = 550–650), with ∫₀ʰ∂ₓp computed three ways: (a) the pipeline (∂ₓp on every row, including rows cut by the interface), (b) the Leibniz form on the liquid pressure, ∂ₓ∫₀ʰp_l dz − p_l(h)h_x, with p_l(h) extrapolated from fully liquid cells, (c) ∂ₓp integrated over fully liquid cells only. Output: `ClosureData/dpx_diagnostics/`. Numbers below are medians over snapshots.

- **The stored ΔP_x is mostly short-wave noise from the Lavalle h_xxx term.** The RMSE-selected Savitzky–Golay window is always the smallest (m = 11, about 0.5 h_N), and 1/We amplifies the noise in h_xxx. In five of six cases 49–92% of the raw ΔP_x variance is at wavelengths below 1 h_N. In Re10 Ca0.1 the raw ΔP_x varies 6.5× more than ∫∂ₓp (0.97× after a 0.5 h_N filter). The high R² there is probably partly the MLP reproducing h_xxx from the h stencil.
- **Method a carries the interface pressure jump.** Variance at wavelengths below 0.25 h_N: 0.51 (a) against 0.16 (b) for Re10 Ca0.01. Methods b and c agree within 1–5% after a 0.5 h_N filter; a differs by 20–70% raw.
- **b closes the momentum budget better where it can be checked (We ≥ 0.4).** DNS-grid residual ||∂ₜq − RHS||/||∂ₜq||, a → b: Re120 Ca0.01 0.078 → 0.043 raw, 0.019 → 0.009 at σ = 0.5; mono L14 0.073 → 0.024, 0.016 → 0.008. Exception Re10 Ca0.1 (a slightly better: 0.022 against 0.027 at σ = 0.5).
- **No method closes the budget for We ≤ 0.1** (Re10 Ca0.005, Re10 Ca0.01, Re40 Ca0.002): momentum residual 0.15–0.6 even at σ = 0.5, while the mass residual drops to about 0.001 after smoothing. The cause cannot be isolated with a one-step ∂ₜ.
- **Mesh (crest-aligned L14 vs L15):** filtered ΔP_x differs by 0.41 (σ = 0.5) and 0.26 (σ = 1), the same as κ, which depends on h only (0.37, 0.29). The two meshes have different capillary ripples (transient window, DT mismatch, +1.2% celerity), so this test cannot judge the extraction (`scripts/mesh_study/compare_dpx_mesh.py`).

**Chosen:** `delta_Px_ML_v2` = −G_σ(I_p,liq) − [h κ_x/We − cosβ/Fr² h h_x + q_xx/Re]_σ with method b and a Gaussian filter of σ = 0.5 h_N (analytic derivatives, exact curvature). σ = 1 closes the budget no better and removes about 90% of a 3 h_N capillary ripple (σ = 0.5: about 40%). Stored with `int_px_dy_liq`, `p_liq_interface`, `kappa_s`, `kappa_x_s` in the closure files (`extract_closure_terms.py`, appended to the 22 existing files and extracted for 7 new runs with data past t = 1500; jobs 50638701–50638730).

## v2 ablation: filtered curvature inputs (12 Nov 2026)

Setup:
- Target `delta_Px_ML_v2`, with a 4×64 functional MLP (pointwise inputs, no stencil) and seed 42, trained for 100 epochs.
- 18 development cases; the We ≤ 0.1 cases are left out.
- 19 input groups: A, B, C, F, G and H, each at σ = 0.25, 0.5 and 1 h_N, plus A_sg (`ml/functional_mlp/sweep_config_dpx.json`).
- Sweep: `$THINFILM_STORE/codes/ml-models/functional_mlp/runs/dpx_v2_sweep01/layers4_width64`. Training array 50649593; evaluation arrays 50933081–85 and 50933682.
- Summary: `scripts/eval_sweep_summary.py <sweep>`.

| Group | Val R² | Val NRMSE | Held-out R² | Held-out NRMSE | R² We 0.2 | R² We 0.4 | R² We 0.8–2.4 |
|---|---|---|---|---|---|---|---|
| G_s0p5 (A + h_xxx + q_xx) | 0.897 | 0.321 | 0.808 | 0.439 | 0.72 | 0.33 | 0.94–0.98 |
| H_s0p5 (A + κ_x/We) | 0.896 | 0.322 | 0.815 | 0.430 | 0.74 | 0.32 | 0.94–0.98 |
| F_s0p5 (A + h_xxx) | 0.893 | 0.327 | 0.775 | 0.474 | 0.71 | 0.32 | 0.79–0.97 |
| A_sg (old SG derivatives) | 0.846 | 0.392 | 0.714 | 0.535 | 0.69 | 0.31 | 0.61–0.93 |
| A_s1 | 0.718 | 0.531 | 0.463 | 0.733 | 0.58 | 0.11 | 0.19–0.68 |

All values are pooled except the per-We columns. The held-out cases by We are:
- We 0.2: Re20 Ca0.01 β45
- We 0.4: Re80 Ca0.005 β6.4
- We 0.8: Re80 Ca0.01 β90
- We 1.2: Re120 Ca0.01 β45
- We 1.6: Re40 Ca0.04 β45
- We 2: Re200 Ca0.01 β6.4
- We 2.4: Re120 Ca0.02 β90

What the results show:
- **The gate fails.** The best held-out pooled NRMSE is 0.43; the gate is 0.25, which is equivalent to R² ≥ 0.94. Low-We R² reaches only 0.72 and 0.33, against a gate of 0.8.
- **The curvature inputs work where We ≥ 0.8.** Held-out R² there is 0.94–0.98 for G, H and F at σ = 0.5, against 0.61–0.93 for A_sg.
- **The third derivative is the input that matters.** G, H and F (h_xxx or κ_x) beat B and C (second derivative only), and those beat A.
- **σ = 0.5 is the best filter width** for G, H and F. σ = 1 is worst for every group.
- **G and H are equivalent.** Adding q_xx to F (giving G) helps in the We 1.6 and 2.4 cases.
- **Low We fails in training and held-out cases alike.** In validation for G_s0p5, Re20 Ca0.02 β6.4 (We 0.4) has R² 0.29 and Re40 Ca0.005 β45 (We 0.2) has 0.50. All other validation cases are at 0.88 or above. In those two cases the Markovian inputs explain little of ΔP_x, whatever the filter. This matches the budget not closing at low We: it is either extraction error in the target or memory. The dense-profile budget check (jobs 50644567–70) should tell which.

### Figures

Made with `scripts/figures/dpx-closure/make_dpx_figures.py` and stored in `figures/dpx-closure/`. All are for held-out cases. The best cases are Re120 Ca0.01 β45 and Re80 Ca0.01 β90; the worst are Re80 Ca0.005 β6.4 and Re20 Ca0.01 β45. The models compared are G_s0p5, H_s0p5, F_s0p5 and A_sg.

- `fig01_extraction_strategies.png`: ΔP_x from the DNS by method a/b/c, raw and filtered, and the old target against v2 (Re120 Ca0.01 β45, t = 1500, 400 < x < 600).
  - The three I_p methods agree away from the sharp troughs.
  - The raw ΔP_x is dominated by grid-scale spikes; filtering with σ = 0.5 keeps the wave-scale signal.
- `fig02_r2_vs_We.png`: per-case R² against We.
  - Left: the input groups at σ = 0.5.
  - Middle: the filter width, for G and A.
  - Right: validation and held-out cases for G_s0p5. The low-We failure appears in training cases too.
- `fig03`–`fig06`: snapshots at t = 1500 for the best cases, in the linear region (x < 100) and the nonlinear region (400 < x < 600).
- `limit01`–`limit04`: the same for the worst cases.
  - In the nonlinear window even Re80 Ca0.005 β6.4 reaches window R² 0.92–0.98.
  - In the linear window the DNS ΔP_x has isolated spikes of 0.3–0.5 that no model predicts. Some sit at the edges of an exactly flat h plateau (x ≈ 46–57, h = 0.978), and others have no feature in h at all.
- `fig07_crest_average.png`: crest-aligned averages over about 800–1000 solitary waves (t > 1000, x > 150). Every model reproduces the mean ΔP_x around the crest and capillary ripples in all four cases. The RMS error is well below the RMS of the DNS signal.
- `fig08_truth_vs_pred.png`: 2D histograms of prediction against DNS.
  - G and H lie on the 1:1 line in the best cases, with slope 0.96–1.03. A_sg is visibly damped (slope 0.86–0.91).
  - Every case has a horizontal band at prediction ≈ 0: DNS events the model cannot see. This band dominates the worst cases.
- `fig09_spectra.png`: wavenumber spectra of ΔP_x.
  - In the best cases the error of G_s0p5 is 1–2 decades below the signal at the wave scale. In the worst cases it is comparable to the signal over 1/λ < 0.3.
  - Beyond λ = 1 h_N the predictions keep short-wave power that the filtered target does not have. It is small in absolute terms.
- `limit05_error_map.png`: x–t map for Re80 Ca0.005 β6.4. **The error is almost entirely at x < 250**, as intermittent spikes at every time.
- `limit06_lag_correlation.png`: correlation of the G_s0p5 error with lagged and shifted inputs.
  - There is no correlation with inputs at earlier times, but 10-unit sampling cannot resolve wave passage.
  - The error is anti-correlated with the DNS ΔP_x at the same point (−0.6 for We 0.4). That is, the model misses an amplitude component.
  - The error has a correlation length of about 1–2 h_N, and no correlation with upstream h or κ_x.

**Where the low-We error comes from.** Split at x = 250, R² for G_s0p5:

| Case | x < 250 | x ≥ 250 | Share of squared error from x < 250 |
|---|---|---|---|
| Re120 Ca0.01 β45 | 0.94 | 0.99 | 0.21 |
| Re80 Ca0.01 β90 | 0.80 | 0.98 | 0.39 |
| Re80 Ca0.005 β6.4 | 0.00 | 0.92 | 0.96 |
| Re20 Ca0.01 β45 | 0.14 | 0.86 | 0.59 |

So the low-We failure is mostly a short-wave component in the nearly flat upstream film, not the solitary waves. Downstream of x = 250 the Markovian model with h_xxx or κ_x inputs is at R² 0.86–0.99.

**Hypothesis, not yet tested.** h_N is about 20.5 cells, so in the nearly flat upstream film the interface sits close to a cell face. There the fully liquid top cell (h − 1.5Δ) can switch from column to column, and p_l(h) and ∂ₓ∫p_l jump with it. The flat h plateau at 20.0 cells points the same way.

To test it, compare ΔP_x against the position of the interface within its cell, (h mod Δ)/Δ, at x < 250, and use the dense-profile runs to see whether the spikes persist in time (physical) or flicker (extraction). If they are extraction, re-score with x < 250 excluded or with a fixed-depth liquid fit before going to memory models.

## Caveats

- One seed, one network, pointwise inputs.
- The first section (stencil MLP, old target) uses validation data of the development cases only.
- One seed, one stencil, one model class; the target is the extracted ΔP_x including its grid-scale noise.

## Open items

1. Low We (≤ 0.4): decide between extraction error and memory, using the dense-profile budget check, then the Step 3 memory diagnostic.
2. A memory term (Step 4), since the Markovian gate fails at low We.
4. ΔP_x is required for Paper 1's comprehensive passive-gas model (decision of 10 Nov), even though the oracle with ΔT_w + ΔM_x only is stable for 100 time units without it.
