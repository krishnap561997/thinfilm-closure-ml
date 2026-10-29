# Conditional closure statistics

Computes exact spatial Pearson curves and conditional PDFs from the uploaded
`check_correlations.ipynb`, with **corrected region labels**. No model fitting,
Spearman ranking, random sampling, or DNS solver execution is involved.

## Install and configure

Unzip this directory on HiPerGator. Use the Python environment that runs your
notebook, or install `python -m pip install -r requirements.txt`.
Edit `config.json`:

- `data_dir`: directory containing `thinfilm_closure_data_ReXXX_CaX_XXX_thetaXX.h5`.
- `output_dir`: results directory. Relative paths resolve against the config file.
- `simulation`: `Re`, `Ca`, `beta` for single-case mode. Scan/watch mode analyzes
  every recognized file and ignores this selector.
- `qois`: exact HDF5 dataset names. No implicit derivatives or aliases are used.
  In particular, `delta_Mx_ML` must exist; `delta_M_ML` will not be differentiated
  automatically if selected instead.
- `offsets`: integer offsets in grid cells, either `{"min":-100,"max":100,"step":1}`
  or an explicit list, e.g. `[-40,-20,-10,0,10,20,40]`.
- `regions`: explicit rules applied to the **central point's raw h and h_x**.
  Default Highlands: `h > 1.15` **and** `|h_x| <= 0.05`, with finite h_x.
  Default Lowlands: `h < 0.85`, with finite h_x, no slope-magnitude restriction.
  These preserve the notebook's selection rules while switching its reversed labels.
  Delete `abs_hx_le` from Highlands if you want ALL points with h > 1.15.
  `h_gt` and `h_lt` are strict inequalities; `abs_hx_le` is inclusive.
  These height thresholds are not spatial neighborhood distances.
- `correlation_use_fluctuations`: default true; subtract the time mean separately
  at each x from the QoI, h and q. Pearson additionally recenters the selected
  conditional samples, as in the notebook.
- `pdf_use_fluctuations`: default false; PDFs use raw QoI values.
- `pdf.bins`, `pdf.n_std`: histogram bins and standardized range (default ±3 std).
- `watch.interval_seconds`, `watch.stable_seconds`: scan cadence and minimum age
  of the latest source modification in watch mode.

All stored times and all complete spatial centers are included, equally weighted.
`dx=x[1]-x[0]`, and a uniform increasing grid is required. The largest absolute
requested offset K excludes K centers from each spatial boundary, even when an
asymmetric list of offsets is selected. Every requested h/q neighbor must be finite.
Each QoI then adds its own finite-value mask. No last-10% or nonflat filter is added.
Arrays in (nx,nt) and (nt,nx) layouts are accepted, using (nx,nt) if ambiguous.
No independent-observation assumption or confidence interval is imposed on the
spatial/temporal samples. Constant correlations and empty-region PDFs are NaN.

## Run one case

```bash
python -u analyze_closures.py --config config.json --mode single
```

## Run every existing case once

```bash
python -u analyze_closures.py --config config.json --mode scan
```

## Keep scanning for new/changed cases

```bash
python -u analyze_closures.py --config config.json --watch
```

This is a foreground program, not a service installed by this package. Run it
inside a compute allocation, or stop with Ctrl-C. It cannot access your cluster
from ChatGPT, and no cluster schedule has been installed. Each rescan skips
unchanged complete outputs. New cases, modified source files, and prior failures
are retried. A changed analysis config creates a new results subdirectory.

For Slurm, edit `run_batch.sh` to use your working Python executable, modules,
account/QoS/partition and suitable time/memory settings. Then use:

```bash
sbatch run_batch.sh          # one scan
sbatch run_batch.sh --watch  # repeat within the job's walltime
```

The supplied 4-hour job is only a starting allocation, not a runtime guarantee.
There is one source file in memory at a time and one QoI at a time. Long files,
many offsets, or many QoIs can still require substantial CPU time. The Slurm launcher
uses one CPU thread by default to avoid small dot products oversubscribing BLAS.

## Results and restart behavior

Each case produces `output_dir/<config-hash>/<source-stem>_statistics.h5`.
The hash includes calculation settings and the program version. Results also
record absolute source path, source size/mtime, UTC creation time, full config,
case parameters parsed from the filename, dx, nx, nt, and valid center bounds.
A source changing during calculation is rejected; its partial result is removed.
An unchanged existing complete result is skipped; `--force` reruns a single scan.
New outputs publish with atomic rename. Advisory locks prevent duplicate analysis
of the same source/config by simultaneous processes on a filesystem supporting flock.
Do not read source files while another process is actively writing them. The
watch stability delay is a practical heuristic, not proof a source is complete.

```
/x, /t
/offsets                       integer grid offsets
/distances                     offsets * dx (same units as source x)
/time_means/h, /time_means/q
/qois/<QoI>/time_mean
/qois/<QoI>/<region>/pearson/h
/qois/<QoI>/<region>/pearson/q
/qois/<QoI>/<region>/pdf/raw/bin_edges
/qois/<QoI>/<region>/pdf/raw/counts
/qois/<QoI>/<region>/pdf/raw/density
/qois/<QoI>/<region>/pdf/standardized/bin_edges
/qois/<QoI>/<region>/pdf/standardized/counts
/qois/<QoI>/<region>/pdf/standardized/density
```

Pearson group attributes store n_points and each field's strongest absolute
correlation, its integer offset, and physical separation (first in offset order
if tied). PDF group attributes store n_points, mean, std, variance, RMS. Standardized
PDF attributes store underflow, overflow, fraction included, and status.
Raw PDF bin edges are shared across regions for each QoI. Each raw histogram
integrates to 1 when nonempty. Each standardized PDF uses that region's own
mean/std; tails outside ±n_std are omitted **without** renormalization, so its
integral can be below 1. Zero-variance standardized PDFs are undefined, not spikes.

## Plotting stored statistics

The plotting scripts use the saved statistics and peak CSV files. They do not reopen the original simulation data or recalculate correlations.

### Individual correlation and PDF plots: `plot_statistics.py`

Each selected QoI/region combination produces a separate figure.

```bash
python plot_statistics.py results/HASH/CASE_statistics.h5 \
  --regions All Highlands Lowlands \
  --kind both \
  --pdf standardized \
  --format both \
  --output-dir figures_by_region/HASH
```

Available options:

| Option | Values | Default |
|---|---|---|
| `--qois` | Selected stored QoI names | All stored QoIs |
| `--regions` | Selected stored regions | All, Highlands, Lowlands |
| `--kind` | `correlations`, `pdfs`, `both` | `correlations` |
| `--pdf` | `raw`, `standardized` | `standardized` |
| `--format` | `png`, `pdf`, `both` | `png` |
| `--dpi` | Positive integer | 300 |

Files are organized by region:

```text
figures_by_region/HASH/All/pearson/
figures_by_region/HASH/All/pdfs_standardized/
figures_by_region/HASH/All/<simulation>_correlation_peaks.csv
```

The same structure is created for Highlands and Lowlands. Figure filenames follow:

```text
<simulation>_<QoI>_<region>.png
```

Correlation plots show signed Pearson correlation against physical separation:

r = n × dx, where dx = x[1] - x[0].

Negative separation represents upstream neighbors for flow toward increasing x. Distances have the same units as the source x coordinate.

The marked peak is the location of the largest absolute correlation, even if its signed value is negative. Peak distances, signed correlations, magnitudes, and offsets are printed and saved in each region's peak CSV.

#### Displaying the complete separation range

The supplied plotting file originally restricted the x-range to zero crossings around the strongest peaks. To display the full stored separation range, use:

```python
ax.set_xlim(r.min(), r.max())
ax.set_ylim(
    *visible_extents(r, list(curves.values()), r.min(), r.max())
)
```

This replaces the corresponding calls using `left` and `right`. Peak selection remains unchanged; only the displayed range and its y-limits change. The CSV's left/right lobe limits still describe the zero-crossing calculation.

### Submit plotting through Slurm

Submit from the directory containing `run_plot.sh`, `plot_statistics.py`, and `config.json`:

```bash
FIGURES_DIR=figures_by_region PLOT_KIND=both PLOT_FORMAT=both \
  sbatch run_plot.sh
```

The script reads `output_dir` from `config.json` and recursively finds all `*_statistics.h5` files, including files inside configuration-hash folders.

To process one configuration only:

```bash
RESULTS_DIR=results/HASH FIGURES_DIR=figures_by_region/HASH \
  PLOT_KIND=both sbatch run_plot.sh
```

Environment variables go before `sbatch`. QoI and region selections go after the script name:

```bash
FIGURES_DIR=figures_by_region PLOT_KIND=both \
  sbatch run_plot.sh --qois delta_Tw_ML delta_Mx_ML --regions All
```

## Scatter plots of correlation peaks: `plot_peak_scatter.py`

This script recursively loads `*_correlation_peaks.csv` files. Each region gets a separate figure with one panel per selected QoI and a separate combined CSV.

By default:

- x-axis: physical peak distance.
- y-axis: peak absolute correlation.
- Color: simulation.
- Marker: circle for h, triangle for q.

To plot signed correlations for one configuration:

```bash
python plot_peak_scatter.py \
  --root figures_by_region/HASH \
  --output-dir peak_scatter/HASH \
  --y peak_correlation \
  --format both
```

To select one region:

```bash
python plot_peak_scatter.py \
  --root figures_by_region/HASH \
  --output-dir peak_scatter/HASH \
  --regions Highlands \
  --y peak_correlation \
  --format both
```

The signed value is evaluated at the location of maximum absolute correlation.

Additional options include:

- `--qois`: select QoIs.
- `--fields h` or `--fields q`: select one neighboring field.
- `--x` and `--y`: choose `peak_distance`, `peak_offset`, `peak_correlation`, `peak_absolute_correlation`, `Re`, `Ca`, or `beta`.
- `--color-by`: choose `simulation`, `Re`, `Ca`, or `beta`.
- `--columns`: maximum QoI panels per row; default 2.
- `--label-points`: annotate points with simulation names.

For example, plot peak distance against Reynolds number:

```bash
python plot_peak_scatter.py \
  --root figures_by_region/HASH \
  --output-dir peak_scatter/HASH \
  --regions All \
  --x Re --y peak_distance --color-by Ca \
  --format both
```

### Keep scatter outputs separate by configuration

```bash
for config_dir in figures_by_region/*/; do
    [ -d "$config_dir" ] || continue
    config_id=$(basename "$config_dir")

    python plot_peak_scatter.py \
        --root "$config_dir" \
        --output-dir "peak_scatter/$config_id" \
        --y peak_correlation \
        --format both
done
```

Outputs are stored under:

```text
peak_scatter/HASH/All/
peak_scatter/HASH/Highlands/
peak_scatter/HASH/Lowlands/
```

Each region folder contains its scatter figures and `combined_peak_data.csv`. This prevents results from different analysis configurations from being combined.

If old aggregate CSVs coexist with region-specific CSVs, the loader removes duplicate peak records and gives priority to the region-specific files.


### Conditional PDF figures

Select `--kind pdfs` for PDF figures alone, or `--kind both` for correlations and
PDF figures. `--pdf raw` selects raw QoI values; `--pdf standardized` selects
`(Y - region mean) / region std` and is the default PDF type.

```bash
python plot_statistics.py results/HASH/CASE_statistics.h5 \
  --kind both --pdf standardized --format both --output-dir figures/HASH
```

With `--kind both`, separate directories avoid filename collisions:

```
figures/HASH/pearson/<simulation>_<QoI>_<region>.png
figures/HASH/pdfs_standardized/<simulation>_<QoI>_<region>.png
figures/HASH/<simulation>_correlation_peaks.csv
```

Each PDF figure contains only its selected QoI/region combination. Its x-range
is the stored histogram range and its y-range follows the density values. Empty
or undefined PDFs are skipped. The standardized PDF range was set by
`pdf.n_std` during analysis; plotting does not change its stored bins or
renormalize omitted tails. The zero-crossing x-limit rule applies only to
correlation plots.

For raw PDF figures alone:

```bash
python plot_statistics.py results/HASH/CASE_statistics.h5 \
  --kind pdfs --pdf raw --format png --output-dir figures/HASH/pdfs_raw
```

For a single kind, files go directly into `--output-dir`. Use different output
directories for raw PDFs, standardized PDFs, and correlations if invoking them
separately: their simulation/QoI/region basenames are intentionally identical.

### Automatically plot every stored statistics file

Run this Bash loop to process existing results across all configuration folders:

```bash
find results -type f -name '*_statistics.h5' -print0 |
while IFS= read -r -d '' file; do
    config_id=$(basename "$(dirname "$file")")

    if ! python plot_statistics.py "$file" \
        --format png \
        --output-dir "figures/$config_id"; then
        printf 'Plotting failed: %s\n' "$file" >&2
    fi
done
```

This creates correlations and peak CSVs. Use `--format pdf` or `--format both`
for other image formats. Add `--kind both --pdf standardized` to also create
standardized conditional PDFs. Configuration-specific output folders prevent
figures from different analyses of the same simulation overwriting one another.
The loop processes files present during that invocation; rerun it to plot newly
completed analyses. It does not install a scheduled job or continually watch for
new files. Images are regenerated on each invocation; the analysis program's
skip-unchanged behavior does not apply to plotting.

### Submit plotting as a Slurm job

Place `run_plot.sh` beside `plot_statistics.py` and `config.json`. Submit from that directory.

Generate correlation plots as PNGs for all stored statistics files:

```bash
sbatch run_plot.sh
```

Generate both correlation plots and conditional PDF plots:

```bash
PLOT_KIND=both sbatch run_plot.sh
```

Save both kinds of plots in PNG and PDF formats:

```bash
PLOT_KIND=both PLOT_FORMAT=both sbatch run_plot.sh
```

Saves plots by region:
```bash
FIGURES_DIR=figures_by_region sbatch run_plot.sh
```
Environment variables must appear **before `sbatch`**, not after the script name.

Available settings:

| Variable | Options | Default |
|---|---|---|
| `PLOT_KIND` | `correlations`, `pdfs`, `both` | `correlations` |
| `PLOT_FORMAT` | `png`, `pdf`, `both` | `png` |
| `PDF_TYPE` | `raw`, `standardized` | `standardized` |

For example, generate raw conditional PDFs only, saved as PDF files:

```bash
PLOT_KIND=pdfs PDF_TYPE=raw PLOT_FORMAT=pdf sbatch run_plot.sh
```

To restrict the plotted QoIs and regions, pass plotting arguments after the script name:

```bash
PLOT_KIND=both sbatch run_plot.sh \
  --qois delta_Tw_ML delta_Mx_ML \
  --regions All Highlands Lowlands
```

The job scans all `*_statistics.h5` files under the `output_dir` specified in `config.json`. Figures are saved under `figures/`, separated by configuration folder. With `PLOT_KIND=both`, correlation and conditional PDF figures are placed in separate subdirectories. Peak correlation locations are printed in the job log and saved as CSV files.


### Load stored results in Python or Jupyter

```python
import h5py
with h5py.File('results/HASH/CASE_statistics.h5', 'r') as f:
    distance = f['distances'][:]
    group = f['qois/delta_Tw_ML/Highlands']
    Rh = group['pearson/h'][:]
    Rq = group['pearson/q'][:]
    pdf = group['pdf/standardized/density'][:]
    bin_edges = group['pdf/standardized/bin_edges'][:]
```

## Validation

`python -m unittest -v test_statistics.py` runs small synthetic checks for notebook
Pearson agreement (including missing data), PDF normalization and tail handling,
transposed fields, empty/constant regions, skip/recompute behavior, and rejection
of sources modified during analysis. Production DNS data is not included.

## Plot Scatter plots

for config_dir in figures_by_region/*/; do
    [ -d "$config_dir" ] || continue
    config_id=$(basename "$config_dir")

    python plot_peak_scatter.py \
        --root "$config_dir" \
        --output-dir "peak_scatter/$config_id" \
        --y peak_correlation \
        --format both
done
