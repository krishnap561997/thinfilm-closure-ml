# Thin-film window MLP

A configurable PyTorch MLP for central-location closure quantities from neighboring
`h, q, signed relative x`, followed by `Re, Ca, angle` (radians). All values use
the existing Nusselt nondimensionalization. No previous-time (`_m`) fields are used.
The supplied particle-force code and original notebook are not modified.

## Quick start

Use Python 3.10+ and a PyTorch environment appropriate for your GPU. Install missing
packages from `requirements.txt` (do not replace a working cluster CUDA PyTorch
installation with a CPU build). Submit commands from this extracted directory.

1. Edit `data_config.json`: replace `data_dir` with the absolute `ClosureData` path.
   Set `cache_dir` to a scratch/project location with several GB free.
2. Prepare the cache once on a compute node:

   ```bash
   python prepare_data.py --config data_config.json
   # Or: sbatch run_prepare.sh
   ```

3. Train:

   ```bash
   python train.py --cache cache/default --run-dir runs/corrections_j4_s1 \
     --half-window 4 --stride 1
   # Or: sbatch run_train.sh
   ```

4. After selecting a model using validation, evaluate its temporal test subset:

   ```bash
   python evaluate.py --cache cache/default \
     --checkpoint runs/corrections_j4_s1/best.pt --split test \
     --output runs/corrections_j4_s1/eval_test --save-predictions
   ```

5. Later, evaluate the eight untouched cases with `--split heldout` and a new
   output directory. Neither test nor heldout metrics are used during training.

The SLURM resources are editable templates, not a verified current cluster policy.
All three launchers prepend your existing `thinfilm-ml-py311/bin` directory to PATH.
For direct Python commands, first run:

```bash
export PATH="/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin:$PATH"
```
They use one process and one GPU, not distributed training. No mail recipient is
configured. Default data-loader workers are zero; increase after measuring I/O.

## Files

- `data_config.json`: explicit 20/8 case lists, filters, caps, split and cache settings.
- `prepare_data.py`: validates HDF5 fields, caches float32 arrays, saves common indices.
- `datasets.py`: memory-mapped fields; vectorized windows per batch.
- `model.py`: Tanh MLP, linear output, embedded input/output normalization.
- `train.py`: Adam, validation-driven scheduler, early stopping, checkpoints.
- `evaluate.py`: streaming per-case, pooled and case-average metrics; optional HDF5 predictions.
- `plot_results.py`: spatial profile comparison from a saved prediction file.
- `run_prepare.sh`, `run_train.sh`, `run_evaluate.sh`: SLURM launchers.
- `test_pipeline.py`: temporary synthetic end-to-end test.

## Data preparation and fixed comparisons

Development cases: all 28 named files except the eight explicitly listed in
`heldout_cases`. For each development case with nt snapshots:

- train: `[0, floor(0.65 nt))`
- validation: `[floor(0.65 nt), floor(0.80 nt))`
- test: `[floor(0.80 nt), nt)`

The default temporal gap is zero, as in the notebook. A nonzero
`temporal_gap_snapshots` drops that many snapshots from the start of validation
and test. Percentages describe snapshots; mask and caps change sample percentages.
Held-out files use all their snapshots, subject to the same spatial filters.

`sample_caps.train = 150000` is a per-case cap, not an oversampling requirement.
Set it to another positive integer, or null for all eligible centers. The default
validation, test, and heldout caps are null (all eligible centers). These may be
capped for a cheaper experiment, but then evaluation measures that saved subset.

Preparation saves `(time_index, space_index)` arrays once, using deterministic
case/split seeds. Train/validation/test selections never change between runs that
reuse the cache. Different model initialization seeds affect training only.
Training shuffles these fixed samples each epoch. Preparation refuses to overwrite
an existing cache; changed filters, case lists or budgets require a new cache path.
A failed preparation may leave an incomplete directory; choose a new path or
remove only that incomplete cache before retrying. Do not run preparation twice
concurrently on the same destination.

All cached available targets must be finite at the center. All h and q values
within the maximum radius must be finite, and h positive. This common eligibility
is intentionally stricter than an individual sparse stencil, so models use exactly
the same centers even when targets or stride change. Adding a new available target
requires a new cache. HDF5 fields must be `(nx, nt)`; no silent transposition occurs.
Parameters may be scalar datasets or attributes under `parameters`.

The cache stores required fields as time-major float32 `.npy` files, coordinates
as float64 and indices as int32. Only one source case is loaded during preparation;
training memory-maps the cache and constructs windows by batch. No giant flattened
window matrix is stored. Cache preparation reads all 28 files, but no held-out
values enter training statistics, optimization or model selection. Training does
not open held-out field files. Keep the cache immutable: checkpoint compatibility
uses its manifest ID, not a repeated full data checksum on every epoch.

## Windows

At center i, sample `i + m*stride`, m=-j,...,j where j=`half_window`.
The feature order is interleaved:

`[h_left, q_left, dx_left, ..., h_center, q_center, 0, ..., h_right, q_right, dx_right, Re, Ca, angle]`

Input dimension is `3*(2*j+1)+3`. `--half-window 0` gives the local baseline.
Stride must be a positive integer. Index sampling requires uniform x coordinates;
the actual coordinates determine signed offsets and the strict maximum radius.
No interpolation or periodic wrapping is used. `j*stride*dx` cannot exceed 2 by
default. With dx=0.0488: j=40,s=1 gives 1.952; j=41,s=1 is rejected.

All centers have a common margin of 2 inside the allowed domain, even for j=0.
The outlet cutoff is `x_min + 0.95*(x_max-x_min)`. Here x_min/x_max are the stored
coordinate bounds; if they are cell centers the cutoff is based on that extent.
Every stencil stays inside this cutoff. `run_config.json` records each case's
actual sampling spacing and half-width.

Examples (run names must be unique):

```bash
python train.py --run-dir runs/local --half-window 0
python train.py --run-dir runs/j4_s3 --half-window 4 --stride 3
python train.py --run-dir runs/j40_s1 --half-window 40 --stride 1
HALF_WINDOW=4 STRIDE=3 RUN_DIR=runs/j4_s3 sbatch run_train.sh
```

## Flat-film mask

Set `flat_mask` to false to disable. When enabled, each snapshot is retained up to
its last point satisfying `abs(h-flat_reference_height)>flat_height_tolerance`,
found BEFORE the excluded outlet region. No disturbed point means no retained
centers. Defaults: reference 1, tolerance 0.1. Only the center is masked; neighbors
may lie in undisturbed film. Flat regions between waves remain eligible. Outlet
exclusion and common radius margins remain in force when flat_mask is false.

## Target selection

Defaults: delta_Dx_ML, delta_Mx_ML, delta_Px_ML, delta_Tw_ML.

```bash
python train.py --run-dir runs/physical_j4_s1 --half-window 4 --stride 1 \
  --targets dudy_wall int_u2_dy u_interface
python train.py --run-dir runs/wall_only --targets dudy_wall
```

No target is recomputed or renamed. Prediction outputs have the units/scaling of
the stored nondimensional targets. The network has no positivity constraints.
`--target-weights` supplies one positive weight per selected target; default is all
ones. Loss is a weighted average of squared standardized errors. h and q each
share normalization over window positions. Re, Ca and angle are standardized
separately. Signed offsets are divided by the common max radius, with no centering.
Each target has its own mean/std. All statistics use selected training samples
only; near-zero std is replaced by 1. Constant global inputs remain constant.

Training uses per-sample case weights so each development case has equal total
loss weight even if one has fewer than 150,000 eligible samples. Validation model
selection uses the average of each case's mean standardized loss. Normalization
is sample-weighted over the selected training pool, not explicitly case-weighted.

## Training and reproducibility

Defaults: 4 hidden layers, 64 units, Tanh, batch 8192, Adam lr=1e-3, 100 maximum
epochs. Validation plateau halves lr after 8 plateau epochs. Early stopping uses
25 epochs without improvement. All configurable with `train.py --help`.

`best.pt` retains the lowest-validation-loss model; `last.pt` retains the last
completed epoch and optimizer, scheduler, shuffle-generator and Torch RNG state.
`history.csv` contains complete loss history. Resume with the SAME settings and a
larger maximum epoch count:

```bash
python train.py --run-dir runs/corrections_j4_s1 --epochs 200 --resume
```

Custom settings from the original command must also be supplied on resume. The
script rejects incompatible arguments and cache IDs. Workers/device/paths may be
changed; exact numerical reproducibility across hardware or library versions is
not promised. Resuming an early-stopped run retains its patience counter.
Only load checkpoints you trust (`torch.load` uses weights_only=False for the
optimizer and metadata). The model's ordinary forward method accepts raw inputs
and returns de-standardized outputs; `standardized()` is used for loss computation.

## Evaluation

Metrics are computed over all samples of the chosen saved split, in original
nondimensional target units. R2 and NRMSE are NaN for effectively constant truth.
POOLED weights cases by sample counts. CASE_AVERAGE averages the case metrics
(and retains NaN if a constituent case metric is undefined). NRMSE uses the
standard deviation of truth in that evaluation set, not training std.

Prediction HDF5 contains `truth`, `prediction`, `indices`, `x`, `t` and ordered target
names. Indices have columns `(time_index, space_index)` and preserve cache order.
No predictions are fabricated at filtered-out locations. Example:

```bash
python plot_results.py runs/corrections_j4_s1/eval_test/Re010_Ca0_010_theta45.h5 \
  --time-index 180 --output profile.png
```

The requested snapshot must belong to the saved split and contain retained centers.
Use validation for neighborhood selection. Run the temporal test and eight held-out
cases after selection; repeatedly tuning against either removes its untouched-test
status. This release performs offline learning only, without ROM integration or
claims of dynamical stability.

## Check the installation

```bash
python test_pipeline.py
```

The synthetic test checks fixed splits, outlet-before-front masking, flat neighbors,
window offsets and radius rejection, reusable sampling, mask disabling, training,
resume, and test/heldout evaluation. It never touches your DNS files.

## Automated 100-epoch sweeps

Edit `sweep_config.json` to specify your existing cache path, arrays of
`half_windows` and `strides`, named target groups and shared training settings.
Prepare the cache only once. All sweep runs reuse its exact saved indices for
train, validation and test, including the same training sample cap. The held-out
cases remain untouched by optimization and evaluation during the sweep.

```bash
python generate_sweep.py --config sweep_config.json --output runs/sweep_01
sbatch runs/sweep_01/submit_array.sh
```

The generator enumerates the Cartesian product and skips any window that exceeds
the cached maximum radius in any case. It records the skip reasons. At j=0 it
keeps only the smallest requested stride to avoid identical local-only experiments.
With the example grid and dx=0.0488 in all cases, this creates 24 model runs.
Check the printed list before submitting. Edit the generated SLURM resource lines
if your allocation needs different resources. `max_concurrent=4` allows at most
four independent one-GPU jobs at once. The generation step does not submit jobs.

Folders are named `physical_j4_s3/`, `corrections_j4_s3/`, etc., under the sweep
directory, each containing best.pt, last.pt, history.csv and run_config.json.
The sweep root also contains sweep.json (the fixed task mapping), requested_config.json,
submit_array.sh and per-task logs. A new sweep requires a new output directory.
Keep generated sweep.json immutable. Generated commands contain absolute paths;
regenerate a sweep if you move the code or cache before submission.

Every run requests exactly 100 total epochs with early stopping disabled. The
learning-rate scheduler remains active. best.pt is the best validation epoch;
last.pt is epoch 100 when the job completes. Errors, preemption, or wall-time limits
can still interrupt a job. Re-submit just an interrupted task, for example:

```bash
sbatch --array=3 runs/sweep_01/submit_array.sh
```

The task resumes automatically from its last completed epoch, preserving the
100-epoch total rather than adding 100 more epochs. A completed task performs no
additional epochs on resubmission. Do not submit duplicate copies of the same task
concurrently. If interrupted before last.pt exists, inspect the partial run folder;
remove only that incomplete folder before retrying (there is no checkpoint to resume).

Inspect a task command without training:

```bash
python run_sweep_task.py --sweep runs/sweep_01/sweep.json --task-id 0 --dry-run
```

Use history.csv validation loss or best.pt metadata to compare models. Compare
window settings within each target group; losses over different target quantities
are not a like-for-like physical error comparison. Test/heldout evaluation remains
an explicit separate step after selection, using evaluate.py as described above.
