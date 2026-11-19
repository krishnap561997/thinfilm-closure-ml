# Flat-film-exact MLP: plan to a journal-quality model (19 Nov 2026)

_Status: proposed, not started. Owner: Krishna. Target: Paper 1 first draft mid-December 2026._

## Goal

Make the flat-film-exact MLP closure (S = N(h, q, h_x, q_x, h_xx; p) − N(h, h³, 0, 0, 0; p), OS constraint λ_lin 10) defensible for a journal:
- every model choice is justified by an ablation;
- training is shown to be converged;
- the model is selected without looking at the test set;
- the closed-loop ROM never fails;
- results are reported with their seed spread.

The aim is evidence for the current design, not a new architecture.

## What we know (19 Nov)

**Training is short and converged on held-out data.**
- 100 epochs take 57 s.
- Held-out a priori NRMSE is 0.184 / 0.183 / 0.184 at epochs 60 / 80 / 100 (seed 0; seed 1 is similar).
- The training loss is still falling (0.0153 → 0.0139). Longer training alone would fit the training cases better, not the held-out ones.

**Capacity is not the limit.**
- Training-case NRMSE is about 0.09–0.12, against 0.18 held out.
- The closed-loop wave-back shape error (about 0.3 above the exact closure on the oracle case) is shared by V1, `ml_sum_hhx` and the flat MLP, which have very different structures.
- This points to missing input information (non-local or higher derivatives, or memory), not to network size.

**The test set has been used for selection.** The 7 broadband held-out cases are the `eval_cases` printed during training, and σ 0.5 was preferred partly from their closed-loop results. A reviewer would call this selection on the test set.

**Robustness.** 2 of 14 seed-0 broadband runs failed with loss of hyperbolicity, because the film thinned below the training range. The safeguard (training-range clamp and thin-film taper, `safeguard.py`) is being validated.

## Steps

### 1. Fix the protocol (no compute)

**Validation set for all model choices:**
- the oracle case (Re 120 Ca 0.01 β 45, 10 Hz, monochromatic: it is not the broadband test);
- the dense restart of Re 40 Ca 0.002 β 45 (`dense_aposteriori.py`; a training case);
- 2 training cases with a held-out time window, e.g. the last 20% of t.

The training script's `eval_cases` stay as a diagnostic, but are not used for choices.

**The test set:** the 7 broadband held-out cases, run once on the final model(s) only. The seed-0 broadband runs already done are reported as a preliminary result in the docs, not as the selection basis.

**Selection metric:**
- primary: the oracle h error at t = 50 and t = 100;
- secondary: comb growth error against OS (Re 20/120/200), and the dense-restart h error at t = 10.

### 2. Training protocol (GPU, minutes)

**Schedule:** about 300 epochs with cosine (or step) learning-rate decay from 1e-2. Keep the model with the best validation loss, from a priori NRMSE on the held-out time window of the training cases.

**Reporting:** loss curves (data loss, linear-constraint loss, validation NRMSE) for the paper's appendix.

**Seeds:** 3 per configuration (0, 1, 2).

**Jobs:** one SLURM array, `%2` throttle. About 1–5 min per model.

### 3. Ablation (a priori, then the oracle case)

| Factor | Values |
|---|---|
| width × depth | 32, 64, 128 × 2, 3 |
| inputs | current (h, q, h_x, q_x, h_xx); + q_xx; + q_xx, h_xxx; two filter widths (σ 0.25 and 0.5 together) |
| σ | 0.25, 0.5 |
| OS constraint | λ_lin 0, 10 (have both: L0 fails at Re 200) |

**Cost.**
- Not every combination is needed. Vary one factor at a time around the current model, plus the best two-factor combination.
- That is about 12 configurations × 3 seeds = 36 trainings, about 1 GPU-hour in total.
- The ROM oracle run is about 1 h on 4 threads per model (36 jobs on bala1s-b, or throttled on bala1s).

**Inputs with q_xx or h_xxx need ROM support:**
- `FlatMLPClosure` currently accepts h, q, h_x, q_x and h_xx, so add the new inputs, with the zero-sum `gauss_deriv_hN`;
- the subtraction term N(h, h³, 0, …) stays exact.

**Output:** a table in `docs/results/structured-closure.md` with a priori NRMSE, the oracle error at t = 50 and 100, and comb growth error (mean ± spread over seeds) for each configuration.

### 4. Robustness (in progress)

1. Validate the safeguard. Pass criteria:
   - all 7 broadband cases complete;
   - oracle, comb and completed broadband results within about ±0.01 of the unsafeguarded runs.
2. Report it as part of the model in the paper: the clamp ranges (0.01–99.99 percentiles of the development data) and the taper h 0.15–0.25 (from the DNS minima).
3. If the safeguard changes results by more than this, alternatives:
   - add thin-film / steep-gradient samples to training;
   - apply a penalty on |S| outside the data range.

### 5. Final model and test

1. **Pick the final configuration** from steps 2–3 on the validation set.
2. **Ensemble:** average S over 3–5 seeds in the ROM. This needs a small `FlatMLPClosure` change: several networks, one N0 table each. Compare the ensemble against single seeds on the oracle case.
3. **Run the broadband test once:** 7 cases × (Lavalle, `ml_sum_hhx`, final model or ensemble), plus the seed spread of single models.
4. **Figures:**
   - `broadband_fig.py metrics/waves`;
   - `comb_growth_fig.py`;
   - `dense_error_fig.py`;
   - oracle snapshots and phase.

### 6. Not planned unless the ablation asks for it

- **Non-local stencil inputs** (`ml/thinfilm_mlp`) or memory: only if the input ablation shows that more information keeps improving the closed loop.
- **Short-rollout fine-tuning** (plan Phase F): the most direct fix for closed-loop error, but the most work. Ask before starting.
- **Closed forms / symbolic regression** (Phase E): de-emphasised, 18 Nov.

## Order and rough timing

| When | What |
|---|---|
| 19–21 Nov | step 4 (safeguard validation); step 1 (protocol, no compute) |
| 21–26 Nov | steps 2–3 (training protocol, ablation, oracle runs) |
| 26 Nov–2 Dec | step 5 (ensemble, final broadband test, figures) |
| from 2 Dec | write results sections |

## Decisions needed from Krishna

1. The validation set in step 1: is it enough, or should 1–2 more broadband cases be moved from test to validation (leaving 5–6 test cases)?
2. Is an ensemble acceptable as the reported model, or should it be one network (simpler to describe)?
3. The scope of the input ablation: does adding q_xx and h_xxx (ROM change) fit before the draft?
