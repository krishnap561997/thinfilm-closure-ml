# Evaluating a functional-MLP sweep

How to score trained models with `evaluate.py` and summarise them, using the ΔP_x v2 ablation (`runs/dpx_v2_sweep01`) as the worked example. The same steps apply to any sweep made with `generate_sweep.py`; change the paths and the target name.

## 1. Where things are

| What | Path |
|---|---|
| Code (run everything from here) | `$THINFILM_ROOT/ml/functional_mlp` |
| Python env | `$THINFILM_ENV` (`source $THINFILM_ROOT/env.sh` sets it) |
| Cache (ΔP_x v2) | `$THINFILM_STORE/codes/ml-models/functional_mlp/cache_dpx_v2` |
| Sweep (ΔP_x v2) | `$THINFILM_STORE/codes/ml-models/functional_mlp/runs/dpx_v2_sweep01` |
| One trained model | `<sweep>/<network>/<input group>/<target>/seed<seed>/` |

For the ΔP_x v2 sweep a model directory is, for example, `runs/dpx_v2_sweep01/layers4_width64/F_s0p5/delta_Px_ML_v2/seed42/`. It contains:
- `best.pt`: the checkpoint at the best validation epoch;
- `last.pt`: the last epoch;
- `history.csv`: the training and validation loss per epoch;
- `run_config.json`.

The `input_groups` block of `sweep_config_dpx.json` defines the input groups. Every group also gets h, q, Re, Ca and β. The suffix `_s0p25`, `_s0p5` or `_s1` is the Gaussian filter width in h_N.

| Group | Inputs |
|---|---|
| A | h, q, h_x, q_x |
| B | A + h_xx |
| C | A + κ |
| F | A + h_xxx |
| G | A + h_xxx + q_xx |
| H | A + κ_x/We |
| A_sg | as A, but the old unfiltered Savitzky–Golay derivatives (reference) |

Set up a shell once per session:

```bash
source /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml/env.sh
cd $THINFILM_ROOT/ml/functional_mlp
export CACHE=$THINFILM_STORE/codes/ml-models/functional_mlp/cache_dpx_v2
export SWEEP=$THINFILM_STORE/codes/ml-models/functional_mlp/runs/dpx_v2_sweep01/layers4_width64
PY="env -u PYTHONHOME $THINFILM_ENV/bin/python"   # unset PYTHONHOME: a loaded python module breaks the env
```

## 2. Check that training has finished

```bash
sacct -u $USER --name=thinfilm_layers4_width64 -X --format=JobID,State,Elapsed | tail -n 25
for h in $SWEEP/*/delta_Px_ML_v2/seed42/history.csv; do echo "$(tail -n +2 $h | wc -l) $h"; done
```

Each finished run has 100 epochs in `history.csv`. If a task stopped early (for example on the time limit), resubmitting the same array task resumes it from `last.pt`.

## 3. First look: validation ranking, no GPU needed

```bash
$PY ../../scripts/ablation_summary.py $SWEEP --targets delta_Px_ML_v2
```

This prints the best and final validation NRMSE of each group, ordered best first. The values are the square root of the normalised MSE, so 1.0 means "predicts the mean". Use this ranking to choose which models to evaluate on the held-out cases. Do not choose models on the test split.

## 4. Evaluate one model

**As a batch job (GPU, about 10–30 min):**

```bash
RUN_DIR=$SWEEP/F_s0p5/delta_Px_ML_v2/seed42 SPLIT=heldout sbatch --export=ALL run_evaluate_single.sh
```

- `SPLIT` is one of `train`, `val`, `test` or `heldout`. Use `val` for comparing models and `heldout` for the final numbers.
- The job writes `$RUN_DIR/eval_$SPLIT/`: `metrics.csv` plus one `<case>.h5` per case with the predictions (`--save-predictions`).
- The job's log goes to `eval_single_<jobid>.log` and `.err` in this folder.

**Interactively on CPU** (fine for one model; the held-out split has up to about 20 million samples, so expect several minutes):

```bash
srun --account=bala1s --qos=bala1s --partition=hpg-dev --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
# in the new shell: repeat the setup lines from section 1, then
$PY evaluate.py --cache $CACHE --checkpoint $SWEEP/F_s0p5/delta_Px_ML_v2/seed42/best.pt \
    --split heldout --output $SWEEP/F_s0p5/delta_Px_ML_v2/seed42/eval_heldout --save-predictions --device cpu
```

The output directory must not exist yet; `evaluate.py` refuses to overwrite. To redo an evaluation, delete the old `eval_<split>` folder or pass a new `--output`.

## 5. Evaluate every model in the sweep

```bash
for ck in $SWEEP/*/delta_Px_ML_v2/seed42/best.pt; do
  d=$(dirname $ck)
  [ -d $d/eval_heldout ] && continue          # skip models already evaluated
  RUN_DIR=$d SPLIT=heldout sbatch --export=ALL run_evaluate_single.sh
done
```

`submit_evaluate_all.sh` in this folder is out of date: it points at the `thinfilm_mlp` cache and calls a `run_evaluate.sh` that does not exist. Use the loop above instead.

## 6. Read the results

**`metrics.csv`** has one row per (case, target) and the columns `case, target, n, RMSE, MAE, R2, NRMSE`:
- `NRMSE` = RMSE / std of the truth within that case;
- `POOLED` = all samples together;
- `CASE_AVERAGE` = the mean of the per-case rows.

To compare the pooled and per-case numbers of all groups at once:

```bash
for m in $SWEEP/*/delta_Px_ML_v2/seed42/eval_heldout/metrics.csv; do
  g=$(echo $m | awk -F/ '{print $(NF-4)}')
  awk -F, -v g=$g '$1=="POOLED"||$1=="CASE_AVERAGE"{printf "%-10s %-13s R2=%.3f NRMSE=%.3f\n", g, $1, $6, $7}' $m
done | sort -k4
```

**Wave-region summary.** This gives the error in the nonlinear region (x > 100 and h > 1.1 or h < 0.9) against the Lavalle-only baseline, plus peak errors. It needs `--save-predictions`:

```bash
$PY ../../scripts/mlp_wave_region_summary.py --eval-dir $SWEEP/F_s0p5/delta_Px_ML_v2/seed42/eval_heldout --cache $CACHE
```

**Held-out cases and their We** (none of them is used in training or normalisation):

| Case | We |
|---|---|
| Re20 Ca0.01 β45 | 0.2 |
| Re80 Ca0.005 β6.4 | 0.4 |
| Re80 Ca0.01 β90 | 0.8 |
| Re120 Ca0.01 β45 | 1.2 |
| Re40 Ca0.04 β45 | 1.6 |
| Re200 Ca0.01 β6.4 | 2 |
| Re120 Ca0.02 β90 | 2.4 |

**Gate for ΔP_x** (plan of 10 Nov): held-out pooled NRMSE ≤ 0.25, and R² ≥ 0.8 in the We ≤ 0.4 cases (Re20 Ca0.01 β45 and Re80 Ca0.005 β6.4 here).
- If the best group passes, ΔP_x goes into the ROM as a Markovian closure.
- If not, the next step is the memory diagnostic on the dense-in-time restart data.

For comparison, the old target with the 4×64 stencil MLP reached a validation R² of 0.70 (pooled), and about 0 in the low-We cases (`docs/results/dpx-closure.md`). Those numbers were for the old, noisy target, so they are a reference, not a like-for-like baseline.

## 7. Record what you find

- Put the numbers in `docs/results/dpx-closure.md` (a new section, with the sweep path and job IDs).
- If the result changes the plan, add a line to `docs/project-status.md`.
- Do not commit `eval_*` folders or `.h5` predictions. They live in the data store, and `.gitignore` covers `runs/`.
