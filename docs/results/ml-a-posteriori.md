# Learned-closure *a posteriori* runs (ΔM_x + ΔT_w, then + ΔP_x)

_Figures: [`figures/ml-a-posteriori/`](figures/ml-a-posteriori/)_

Couple the 1×16 MLPs to the ROM and run it from the DNS state, on the same case and window as the oracle runs (`oracle-a-posteriori.md`). The question is whether a closure that is good *a priori* stays good once it is driven by the ROM's own state (closed loop). Source for Paper 1 Sec. 5.3.

## Setup (Re120, Ca0.01, β45°, 10 Hz, t = 1800–1900)

- **Case and inputs:** identical to the oracle runs. DNS initial state (r = √(h/We) h_x) and DNS inlet h(0, t), q(0, t) from `rom_forcing.h5` (`oracle.txt` with `terms = none`). N = 4000, dt = 0.005, output every 0.1. Re120 Ca0.01 β45 is a held-out case for all models used here.
- **Hook:** `rom/lavalle-cpp/{include,src}/ml/` (see the solver README). Models are evaluated at every RK4 stage on the ROM state; h_x and q_x from central differences on the ROM grid (dx = 0.2); the result is added to the q-equation as −(ΔM_x + ΔT_w). Weights exported with `ml/functional_mlp/export_for_cpp.py`; the C++ forward pass matches PyTorch to 6 digits at test points.
- **Runs:**
  - `ml_sum_hhx`: one model for ΔM_x + ΔT_w, inputs (h, h_x, Re, Ca, β) (`ablation_02_sum/layers1_width16/h_hx`; *a priori* held-out nonlinear NRMSE 0.287, `mlp-heldout-eval.md`).
  - `ml_mx_tw`: separate ΔM_x and ΔT_w models, inputs (h, q, h_x, q_x, Re, Ca, β) (`ablation_01/layers1_width16/first_derivatives`). These were the best individual models on held-out data for ΔM_x; for ΔT_w the (h, h_x) model was marginally better pooled (0.115 vs 0.121) but worse on the case mean (0.135 vs 0.125).
  - Reference runs from the oracle study: `lavalle` (no correction) and `Tw_Mx` (exact DNS ΔT_w + ΔM_x).
- **Cost:** 155 s (sum model) and 301 s (two models) for 100 time units on one core, against ≈ 50 s for Lavalle alone.
- **Data:** `$THINFILM_DATA/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom/`: `models/*.txt`, `runs/{ml_sum_hhx,ml_mx_tw}/` (`ml.txt`, `run.log`, `thinfilm_training_data.h5`), `summary_ml.csv` (`scripts/oracle_summary.py`, every 0.1).
- **Figures:** `scripts/figures/ml-a-posteriori/fig01_h_profiles.py`, `anim01_h_full_domain.py` (mp4, 0 < x < 800, not in git: regenerate with the script), both with `--set 2026-11-05`; output in `figures/ml-a-posteriori/comparisons_2026-11-05/`.

## Results

h error, rms(h_ROM − h_DNS) / std(h_DNS) over 20 < x < 760, against elapsed time since t = 1800:

| elapsed | Lavalle | oracle ΔT_w+ΔM_x | ML sum (h, h_x) | ML ΔM_x + ML ΔT_w |
|---|---|---|---|---|
| 0.1 | 0.006 | 0.006 | 0.006 | 0.006 |
| 1 | 0.115 | 0.060 | 0.057 | 0.058 |
| 2 | 0.260 | 0.095 | 0.095 | 0.096 |
| 5 | 0.536 | 0.150 | 0.167 | 0.161 |
| 10 | 0.606 | 0.172 | 0.219 | 0.218 |
| 20 | 0.590 | 0.199 | 0.282 | 0.373 |
| 30 | 0.571 | 0.196 | 0.393 | 0.564 |
| 50 | 0.585 | 0.192 | 0.467 | 1.029 |
| 75 | 0.848 | 0.344 | 0.820 | 1.689 |
| 99.9 | 1.066 | 0.550 | 1.258 | 2.045 |

Max h (ROM; DNS in brackets): elapsed 5: Lavalle 1.66, oracle 2.09, ML sum 2.06, ML separate 2.10 (2.10); elapsed 50: 1.45, 2.10, 2.38, 2.26 (2.12); elapsed 99.9: 1.43, 2.04, 2.50, 2.34 (2.08).

- **Both learned closures are stable for 100 time units** (no crash; min hyperbolicity discriminant 0.065 and 0.027 at the end, h_min 0.36 and 0.23).
- **For the first ~10 time units both match the oracle.** The error is 0.057 / 0.058 after 1 time unit against 0.060 for the oracle and 0.115 for Lavalle, and the wave height is restored (max h 2.06–2.10 against 2.10 at elapsed 5, Lavalle 1.66).
- **The sum model is the better of the two in closed loop.** It stays below Lavalle until about elapsed 75 (0.47 against 0.59 at 50) and has about half the error of the separate models from elapsed 20 on. This agrees with the *a priori* ranking and with the oracle finding that ΔM_x and ΔT_w must act together: two separately trained models do not keep their partial cancellation once the state drifts.
- Neither learned closure reaches the oracle after elapsed 10 (oracle 0.19 at 50).

## Where it fails

Figures: [`fig01`](figures/ml-a-posteriori/comparisons_2026-11-05/fig01_h_profiles.png); the animation shows the whole domain.

- **The sum model over-amplifies the waves.** Max h grows past the DNS from elapsed ≈ 20: 2.25 at 30, 2.38 at 50, 2.50 at 75–99.9 against 2.04–2.12. In `fig01` the crests are also ahead of the DNS crests from elapsed 50 (taller waves travel faster), with a longer, thinner substrate behind them (h ≈ 0.55 where the DNS has 0.8–1.0 at elapsed 75–99.9, read from the figure). By elapsed 99.9 the error (1.26) is above Lavalle alone (1.07).
- **The separate models fail sooner and differently.** From elapsed 20 the error grows faster (0.37 at 20, 1.03 at 50, 2.05 at 99.9). The crests broaden into flat-topped humps (h ≈ 2.2–2.3 over ≈ 15 h_N at elapsed 99.9, `fig01`) and the substrate develops spurious bumps (h up to 1.4 at x ≈ 550, elapsed 75).
- **The closed-loop drift is the main failure, not the *a priori* error.** Over the first 10 time units, when the ROM state is still close to the DNS, both models do as well as the oracle. Once the ROM waves are taller or shifted, the models are evaluated on states that differ from the DNS and appear to reinforce the deviation rather than correct it. Not tested: whether this is a model error outside the training distribution, a phase-drift effect that also limits the oracle, or both.
- **Derivative mismatch (untested).** Training used Savitzky–Golay derivatives of the DNS profiles (`profiles_to_closures.py`); the ROM uses central differences on a grid 4× coarser (dx = 0.2 against 0.049). Capillary ripples ahead of the crests give large, noisy h_x. The `hx_smooth` option exists but was not used.
- **Scope.** One case, one frequency, 100 time units (about 2 forcing periods), single seed per model, models not converged at 100 epochs. No error bars.

## Adding the ΔP_x v2 MLP (12 Nov 2026)

Same case, window and setup as above, with a learned ΔP_x on top of the ΔM_x + ΔT_w models. SLURM array 51017895 (`rom/run_ml_px.sh`; one core per run, ≈ 6500 s each for 100 time units). Summary and figures: job 51031622 (`rom/make_figures_px.sh`).

- **ΔP_x models:** 4×64 from `functional_mlp/runs/dpx_v2_sweep01/layers4_width64` (`dpx-closure.md`), target `delta_Px_ML_v2`. Re120 Ca0.01 β45 is held out.
  - `G_s0p5`: inputs h, q and h_x, q_x, h_xxx, q_xx filtered at σ = 0.5 h_N; best validation R² (0.897), held-out R² 0.81.
  - `H_s0p5`: h, q and filtered h_x, q_x, κ_x/We; held-out R² 0.815.
- **Coupling:** added as +ΔP_x. The filtered inputs are computed on the ROM grid with the training operator (Gaussian, analytic derivatives, σ = 2.5 cells). ΔP_x v2 is defined against exact curvature (P_Lav,s uses h κ_x/We) and the ROM's capillarity is the linearised h h_xxx/We, so the runs also add h_s(κ_x,s − h_xxx,s)/We (`px_curvature_fix = 1`). Not corrected: the ROM's other P_Lav terms (hydrostatic, q_xx/Re) are unfiltered, so the ROM and the target differ at wavelengths below about 1 h_N.
- **Check:** `scripts/ml_hook_check.py` on the DNS state at t = 1800 and 1850. The C++ inputs agree with Python to 5e-12 and the ΔP_x outputs with PyTorch to 1e-7 relative. rms on that state: ΔP_x 0.055, curvature term 0.021, against 0.37 (ΔM_x) and 0.23 (ΔT_w).
- **Runs** (in `rom/runs/`): `ml_sum_pxG` (sum model (h, h_x) + `G_s0p5`), `ml_mx_tw_pxG` (separate ΔM_x and ΔT_w models + `G_s0p5`), `ml_sum_pxH` (sum model + `H_s0p5`). Data: `summary_ml_px.csv`. Figures: `comparisons_2026-11-12/` (`--set 2026-11-12`).

### Results

h error as above (rms / std of DNS h, 20 < x < 760):

| elapsed | Lavalle | oracle | ML sum | ML sum + ΔP_x (G) | ML sep. + ΔP_x (G) | ML sum + ΔP_x (H) |
|---|---|---|---|---|---|---|
| 0.1 | 0.006 | 0.006 | 0.006 | 0.006 | 0.005 | 0.006 |
| 1 | 0.115 | 0.060 | 0.057 | 0.027 | 0.026 | 0.027 |
| 2 | 0.260 | 0.095 | 0.095 | 0.046 | 0.046 | 0.044 |
| 5 | 0.536 | 0.150 | 0.167 | 0.157 | 0.204 | 0.137 |
| 10 | 0.606 | 0.172 | 0.219 | 0.317 | 0.524 | 0.308 |
| 20 | 0.590 | 0.199 | 0.282 | 0.460 | 0.828 | 0.553 |
| 30 | 0.571 | 0.196 | 0.393 | 0.733 | 0.975 | 0.839 |
| 50 | 0.585 | 0.192 | 0.467 | 1.540 | 1.124 | 1.735 |
| 75 | 0.848 | 0.344 | 0.820 | 2.038 | 2.138 | 2.122 |
| 99.9 | 1.066 | 0.550 | 1.258 | 1.967 | 2.397 | 2.127 |

Max h (ROM; DNS in brackets): elapsed 5: sum + G 2.28, sep. + G 2.27, sum + H 2.24 (2.10); elapsed 30: 2.81, 2.35, 3.00 (2.10); elapsed 99.9: 3.25, 3.18, 2.97 (2.08).

- **ΔP_x halves the short-time error.** At elapsed 1–2 all three runs have about half the error of the same closures without ΔP_x, and of the oracle (which has exact ΔT_w + ΔM_x but no ΔP_x): 0.027 against 0.057–0.060 at elapsed 1. In `fig01` the crest and the capillary ripples follow the DNS almost exactly at elapsed 1. The learned ΔP_x therefore carries real information that the other two closures lack.
- **It destabilises the waves after elapsed ≈ 5.** The crests overshoot from elapsed 5 (max h 2.24–2.28 against 2.10). From elapsed 10 all three runs are worse than without ΔP_x, and from elapsed 30 worse than Lavalle alone.
- **G and H behave the same way.** H is slightly better up to elapsed 5 and slightly worse afterwards. The input set does not change the closed-loop behaviour.
- All three runs reach t = 1900. At the end h_min is 0.19 (sum + G), 0.11 (sum + H) and 0.017 (sep. + G); the minimum hyperbolicity discriminant is 0.021, 0.011 and 1.7e-3.

### Where it fails

Figures: [`fig01`](figures/ml-a-posteriori/comparisons_2026-11-12/fig01_h_profiles.png) (window 630 < x < 750) and the animation (whole domain).

- **The overshoot starts at the steep crest fronts.** At elapsed 30 the animation shows narrow spikes (h ≈ 2.7–2.8) right at the fronts (x ≈ 430, 610), where the filtered h_xxx is largest. This points to ΔP_x feeding back on itself at the front rather than to a slow drift.
- **The waves end as wide, flat-topped humps on a thin substrate.** With the sum model (G and H), the crests become plateaus at h ≈ 2.8–3.2 about 20 h_N wide by elapsed 75–99.9, with substrate h ≈ 0.2–0.4 between them (DNS 0.6–1.0). Mass collects in fewer, taller waves; these travel faster and lose phase with the DNS.
- **The separate models with ΔP_x come close to breakdown.** By elapsed 99.9 the region x ≈ 130–470 is irregular, with h close to zero in places (h_min 0.015, discriminant 1.5e-3 at step 19701).
- **The inlet region (x < 130) is fine in every run;** the failure builds up downstream as the waves mature.
- **Causes not tested:**
  - the ΔP_x MLP is evaluated outside its training range once the crests exceed ≈ 2.2;
  - the ROM's unfiltered hydrostatic and q_xx/Re terms differ from the filtered P_Lav,s of the target at short wavelengths;
  - the `px_curvature_fix` term adds feedback of its own.
- **Scope.** As above: one case, one frequency, 100 time units, one seed per model.
- **Animation:** the title is clipped at the top and the y-limit (2.8) cuts off the ΔP_x crests (up to 3.25); to fix when regenerated.

## Open items

1. Rerun with `hx_smooth` (e.g. 1–2 cells) to test the derivative mismatch.
1. ΔP_x: rerun without `px_curvature_fix`, and with ΔP_x limited where |h_xxx| or h is outside the training range; evaluate ΔP_x on the ROM state at elapsed 5 against the DNS state to find where the feedback starts.
2. Re-initialise from the DNS every 5–10 time units to separate closure error from phase drift (also open for the oracle).
3. Closed-loop-aware training (e.g. train on ROM states, or rollout loss); constrain the sum model near crests.
4. Other cases, including broadband forcing with statistical metrics.
