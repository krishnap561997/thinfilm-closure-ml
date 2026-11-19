# CLAUDE.md — thinfilm-closure-ml

Context for Claude (Claude Code on HiPerGator via VS Code, and Claude Cowork on the laptop).
Read this first, then `docs/project-status.md` for where the science currently stands.

## Project in one paragraph

ML-augmented reduced-order model (ROM) for thin liquid films falling down an inclined plane
(K.A. Krishnaprasad, S. Moghaddam, S. Balachandar — UF; funded by ONR).

- **Training data:** 2D two-phase DNS with Basilisk C (CLS-VOF, broadband inlet forcing) across (Re, Ca, β).
- **Baseline ROM:** the first-order two-equation model of Lavalle et al. (2015), with state variables h(x,t) and q(x,t).
- **ML step:** learn corrections ΔC = C_DNS − C_Lavalle for the unclosed terms M_x, T_w, P_x, D_x, then embed them back into the ROM solver.

Paper 1 is the comprehensive passive-gas model (memory closures included if needed). Paper 2: gas-sheared films. Paper 3: heated films. Later: microchannel flow boiling.

## Repository map

| Path | What it is |
|---|---|
| `dns/production/` | **Production Basilisk setup**: `filmflow0.c`, `filmflow_restart.c` (nondimensional, Nusselt scaling), `Makefile` (builds against `~/basilisk-old`), `run.sh` (SLURM), `count_sim_files.sh` |
| `dns/production/cases/<case>/` | `params.in` per DNS case (plus case-specific `.c` where one existed) |
| `dns/include/hdf5_headers/` | HDF5/XDMF output headers included by the Basilisk sources |
| `dns/include/column_profiles.h` | In-situ 1D output: per-column h, q, ∫u², ∫₀ʰ∂ₓp, ∂_z u\|₀, interface velocity (at t and t−dt), ~1 MB per output instead of a 133 MB 2D snapshot |
| `dns/oracle/` | `filmflow0_mono.c`: production `filmflow0.c` with a monochromatic inlet (`FREQ` in Hz, `AMP`), for from-scratch spin-up runs; `filmflow_restart_mono.c`: monochromatic (10 Hz) restart with column-profile output every `T_PROF`, for the oracle *a posteriori* runs |
| `dns/tests/` | Physics/BC test setups (bc-blowup-fix, clsvof, white-noise domain tests, vertical-confined, hdf5-output) |
| `dns/legacy/` | Older dimensional setup (`inclined-whitenoise-dim`) and generic sources — not used for production |
| `rom/lavalle-cpp/` | C++ Lavalle two-equation solver (Rusanov + MUSCL, RK4, Noble–Vila capillarity, Chang noise forcing, HDF5 out; `oracle/` and `ml/` closure modules). Build: `module load gcc/14.2.0 hdf5 && make` |
| `rom/matlab/` | Original MATLAB ROM (monochromatic + white-noise), `cases/ROM_Ca_Re_sweep/*/input.txt` |
| `pipeline/extract_closures/` | DNS → depth-integrated quantities and closure terms (`extract_closure_terms.py`); `generate_extraction_jobs.py` writes per-case SLURM jobs into the data store; `profiles_to_closures.py` does the same from in-situ 1D profiles; `check_momentum_budget.py` + `run_budget_check.sh` check that the budget closes (see its README) |
| `scripts/mesh_study/` | Mesh-sensitivity tools: transient-exit finder and crest-aligned comparison of two closure files (see its README) |
| `analysis/conditional_statistics/` | Conditional statistics of closure terms (peaks, regions) |
| `analysis/orr_sommerfeld/` | Orr–Sommerfeld solver (`os_solver.py`: temporal/spatial/saddle, linearised Lavalle), verification (`verify.py`), OS case table (`case_table.py`), linear closure targets and (a,b,c,d) fits (`linear_targets.py`), DNS linear-region checks (`b5_dns_linear.py`, `b7_dns_check.py`, `comb_dns/analyse_comb.py`), ROM comb-inlet linear test (`b6_rom/`). See its README |
| `scripts/structured-closure/` | Phase A diagnostics of the current MLPs (`a1`–`a4`), numpy MLP evaluator `mlp_np.py`; oracle a posteriori figures `oracle_snapshots.py` (h(x) panels) and `oracle_phase.py` (shift, shifted error, crest height over time); `comb_growth_fig.py` (ROM comb growth vs OS and comb DNS), `dense_error_fig.py` (dense-restart a posteriori error), `broadband_fig.py` (broadband held-out statistics per case; h(x) and h PDFs), `crash_diag.py` (network inputs, S and training range at the thinnest cell from `crash_states.h5`) |
| `dns/comb/` | Short comb-forced DNS from a flat film (LX 200, same mesh): `setup_comb_runs.py`, `run_comb.sh`; run/analysis instructions in `docs/plans/comb-dns-instructions.md` |
| `ml/structured_closure/` | Structured closure V1 (plan Phase C): `model.py`, `losses.py` (OS dispersion constraint L2), `os_dispersion.py`, `train.py`, `run_sweep.sh`, `summarize_runs.py`, `export_tables.py` (1D tables for the ROM), `rom_tests.py` + `run_rom_tests.sh` (uniform, comb, oracle a posteriori), `test_structured.py`; flat-film-exact MLP (`train.py --model mlp`, `export_mlp.py`); `broadband.py` + `run_broadband.sh` (broadband held-out a posteriori: ROM vs DNS wave statistics); `chain_seed_rom.sh` (after training: submit ROM tests and broadband runs); `dense_aposteriori.py` (ROM from the dense DNS restarts, DNS state and inlet, 10 time units); `safeguard.py` (training-range input clamp and thin-film taper for the flat MLP; closure names ending in `_sg`). Runs: `$THINFILM_STORE/codes/ml-models/structured_closure/` |
| `scripts/meetings/` | Figure and deck builders for meeting slides (`advisor_2026_11_17_figs.py`, `advisor_2026_11_17_deck.py`; deck needs `python-pptx` on `PYTHONPATH`, not in the env). Decks (`docs/meetings/*.pptx`) and PNGs are not in git; slide explainers (`docs/meetings/*-explainer.md`) are |
| `pipeline/cache_v2/` | Cache v2 builder (full record, D_G(σ) derivatives σ = 0.25/0.5/1, ΔF, hygiene drops) → `$THINFILM_STORE/codes/ml-models/cache_v2/` |
| `ml/functional_mlp/` | Shallow MLPs on (h, q, derivatives) — ablations for ΔM_x and ΔT_w |
| `ml/thinfilm_mlp/` | Neighbour-stencil MLP and curvature-feature experiments |
| `ml/legacy/` | Early scripts that use the old `Pytorch_ML_env` |
| `notebooks/analysis/`, `notebooks/ml/` | Exploratory and figure notebooks. Outputs are stripped, and many still contain absolute `/blue` paths |
| `datasets/` | **Manifests only** (`MANIFEST_dns.csv`, `MANIFEST_rom.csv`) — no data in git |
| `docs/` | `project-status.md` (short, mirrored to claude.ai), `equations.md`, `data-layout.md`, `hipergator.md`, `decisions.md`, `results/` (full write-ups, each with a "Where it fails" section: `budget-check.md`, `oracle-a-posteriori.md`, `mesh-sensitivity.md`, `mlp-heldout-eval.md`, `functional-form-structure.md`, `functional-form-v2.md`, `dpx-closure.md`, `ml-a-posteriori.md`, `orr-sommerfeld.md`, `structured-closure.md`; functional-form planning notes: `ordering-matrix.md`, `richard-comparison.md` (scripts in `scripts/ordering-matrix/`)); `plans/` (`structured-closure-os-plan.md` = **active plan**, `structured-closure-progress.md` = task tracker and running jobs, `functional-form-plan.md` = secondary, `comb-dns-instructions.md`, the Kalliadasis et al. book PDF); `explainers/` (explanations written for Krishna) |
| `envs/environment.yml` | Python env spec (the env itself lives on /blue) |
| `env.sh` | `source env.sh` → sets THINFILM_ROOT / THINFILM_STORE / THINFILM_DATA / THINFILM_ENV |

## Where things live (HiPerGator)

- **Code (this repo, new clone):** `$THINFILM_ROOT` (suggested `/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml`).
- **Data store (old repo directory, now treated as data only):** `$THINFILM_STORE = /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml`
  - DNS runs: `$THINFILM_STORE/datasets/BasC/DNS/param_space/<case>/` (`output/*.h5`, `infc/`, dumps)
  - Extracted closures: `$THINFILM_STORE/datasets/BasC/DNS/param_space/ClosureData/` (budget-check results in `budget_check/`)
  - Oracle DNS runs: `$THINFILM_STORE/datasets/BasC/DNS/oracle/<case>_f<freq>Hz/run_t<start>/` (`profiles/*.h5`, `output/`, dumps)
  - Monochromatic Re120 Ca0.01 β45 10 Hz reference run (L800): `$THINFILM_STORE/codes/basilisk/physics-tests/bc-blowup-fix/length-check/L800/` (dumps every 50 to t = 1800)
  - ROM sweep runs: `$THINFILM_STORE/datasets/ROM_Ca_Re_sweep/<case>/`
  - Python env: `$THINFILM_STORE/software/envs/thinfilm-ml-py311`
  - ML caches/runs: `$THINFILM_STORE/codes/ml-models/{functional_mlp,thinfilm_mlp}/{cache_dir,runs}`
- Many configs and notebooks still hard-code these `/blue/.../thinfilm-closure-ml/...` paths. They keep working because the store has not moved. **Do not move or rename the store directory.**
- On 2026-10-30 `BasC/param_space` moved to `BasC/DNS/param_space`; references were rewritten on 30 Oct and again on 4 Nov (the first pass skipped git-ignored folders such as `cache_dir/`; see `docs/decisions.md`). Old SLURM logs and `.ipynb_checkpoints` deliberately keep the old path. When you move anything in the store, search with `grep -rIl` (it ignores `.gitignore`; `rg` does not unless given `--no-ignore`) and update hard-coded paths.

## Conventions and rules

- **Never commit data or build products.**
  - This covers `*.h5`, dumps, compiled `filmflow*`, SLURM `*.out`, `runs/`, `cache_dir/`, figures and `.npz` files.
  - `.gitignore` enforces it; check `git status` before committing.
- **Strip notebook outputs before committing** with `nbstripout --install` once per clone. Put reusable code in `.py` modules, not in notebooks.
- **Run DNS cases in the data store, not inside the repo.**
  1. Copy `dns/production/{filmflow0.c,filmflow_restart.c,Makefile,run.sh}` plus the case's `params.in` into the case directory.
  2. Run `make` and then `make run`.
  3. Refresh the manifest with `python datasets/update_manifest.py`, then fill in `split` and `notes` by hand.
- **Ignore run directories:** per-run subdirectories should be named `runs/`, `run_*/` or `scratch/`. They are ignored as a whole, including their `.sh` files.
- **Use nondimensional Nusselt scaling everywhere:** h_N, u_N, Re = u_N h_N/ν, Ca = μ u_N/σ, Fr² = Re sinβ/3, We = Re·Ca, Λ = 3. See `docs/equations.md`.
- **Use these closure names consistently:** `delta_Mx_ML`, `delta_Tw_ML`, `delta_Px_ML`, `delta_Dx_ML` (ΔC = C_DNS − C_Lav). The combined source term is S_ML = −ΔM_x + ΔP_x − ΔT_w + ΔD_x.
- **SLURM settings:** account and qos `bala1s` (burst: `bala1s-b`); use `hpg-dev` for short jobs.
- **Resource limits (hard rules):**
  - At most **2 GPU jobs at a time** in total. Submit a GPU batch (training sweep, evaluations) as one array throttled with `%2`, for example `run_evaluate_array.sh` or `submit_array.sh` with `--array=0-N%2`, and set `max_concurrent: 2` in sweep configs.
  - At most **64 CPUs at a time on qos `bala1s`**, summed over all running and pending jobs. Use `bala1s-b` (no limit) for anything beyond that.
- **Modules:**
  - DNS: `gcc/14.2.0 hdf5 imagemagick`
  - **All** thin-film DNS use the Basilisk at `/home/krishnap.kalivel/basilisk-old/src`. It is the version compatible with the HDF5 output headers. The newer `$BASILISK` install belongs to the heterogeneous-nucleation project and is **not** compatible, so never build these cases against it.
- **Line endings:** `.gitattributes` keeps `.sh`, `Makefile`, `.c` and `.py` files LF, so files edited on Windows still run on HiPerGator.

## Keeping context in sync (laptop ↔ HiPerGator ↔ claude.ai)

- This repo is the **single source of truth** for project context.
- When something important is learned or decided, update `docs/project-status.md` or `docs/decisions.md` in the same commit as the code.
- Pull before starting work on either machine and push when done.
- The claude.ai project "P1 - Thin-film SciML closure" mirrors `docs/project-status.md`. Claude Cowork re-syncs it from the repo.

## Known gotchas

- `dns/legacy/inclined-whitenoise-dim` is the **dimensional** predecessor of the production sources. Do not use it for new runs.
- The forcing cut-off `FCUT_DIM = 28 Hz` is dimensional and identical for all DNS cases, so the nondimensional cut-off varies from case to case. The MATLAB ROM sweep used 20 Hz.
- `T_END` is 2000 for most DNS cases. Some, e.g. Re120_Ca0_010_theta45, use 10000.
- On HiPerGator, `/blue` is not backed up. Keep irreplaceable results (paper figures, final models) archived elsewhere as well.
- The source of the original L800 monochromatic binary is lost. `dns/oracle/filmflow_restart_mono.c` reconstructs it from the production restart plus the `BasC/fixed-frequency` inlet (amplitude 0.03). The nondimensional forcing frequency is 0.0214, so one period is about 47 time units.
- Strongly capillary low-Re cases (We ≲ 0.4, e.g. Re10 Ca0.01) have unreliable one-step ∂ₜ and do not close the momentum budget. Check the per-snapshot mass residual in `ClosureData/budget_check/` before trusting a snapshot.
- Restarts before 10 Nov 2026 ran without inlet forcing (noise phases never initialised after `restore()`; fixed in `filmflow_restart.c`). Check `inlet_signal.dat` of any restarted run; see `docs/decisions.md`.
- Basilisk ends a run as soon as no time-scheduled event remains: in short tests keep `T_OUT` and `T_DUMP` ≤ `T_END`.
- `lavalle-cpp` and the DNS share the Chang inlet; `MNOISE = M` gives an exact comb at k·f_cut/M (used for linear tests). `./thinfilm input.txt none ml.txt` runs an ML closure without a DNS oracle file. ROM outputs written before 16 Nov have a non-physical first cell (fixed in `5df0595`).
- scipy's `gaussian_filter1d` second derivative (and the ROM's `gauss_deriv`) is not zero on a constant field (truncation at 4σ): h_xx(1) = −3.5e-3 at σ 0.5 on the DNS grid, −1.8e-3 on the ROM grid. Cache-v2 h_xx/q_xx carry this bias. The structured closure uses the zero-sum `gauss_deriv_hN` in the ROM and subtracts c0·h in training (`ml/structured_closure/data.py`).
- `./thinfilm input.txt <oracle|none> sc.txt` with a `structured_table` key runs the structured closure (`StructuredClosure`, tag `// STRUCTCLOSURE`).
- A config with a `flat_mlp` key runs the flat-film-exact MLP (`FlatMLPClosure`). It tabulates N(h, h³, 0, 0, 0) once per run and runs its cell loop with OpenMP: set `OMP_NUM_THREADS` (the job scripts set it to `SLURM_CPUS_PER_TASK`). The `thinfilm` binary is built with `-fopenmp`.
- Since 19 Nov the hyperbolicity discriminant is computed without cancellation (exactly 4 dP/dh for φ = 1); build with `-DOLD_DISC` to reproduce older runs bit for bit. A run that throws writes its last 20 time units to `crash_states.h5`. Flat-MLP safeguard keys: `clip_h`, `clip_dq`, `clip_h_x`, `clip_q_x`, `clip_h_xx` (= lo hi) and `taper_h` (= h0 h1); without them the old path runs unchanged. Build test binaries to another `TARGET=` while jobs use `rom/lavalle-cpp/thinfilm`.
- The scratchpad under `/tmp` is local to the login node: put anything a SLURM job must read in the data store.
- Python that imports torch and then scipy needs `LD_LIBRARY_PATH=$THINFILM_ENV/lib` (system libstdc++ is too old).
- PDF tools (pdftotext/poppler, pypdf) are not installed on HiPerGator; plain-text extraction of PDFs needs a stdlib zlib parse.
- When restarting from a dump, link it under a different name (`RESTART_FILE = restart.dump`). A run writes its own `dump-XXXXXX`, and writing through a link with the same name would overwrite the original.
