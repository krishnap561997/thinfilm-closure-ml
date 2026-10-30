# CLAUDE.md — thinfilm-closure-ml

Context for Claude (Claude Code on HiPerGator via VS Code, and Claude Cowork on the laptop).
Read this first, then `docs/project-status.md` for where the science currently stands.

## Project in one paragraph

ML-augmented reduced-order model (ROM) for thin liquid films falling down an inclined plane
(K.A. Krishnaprasad, S. Moghaddam, S. Balachandar — UF; funded by ONR).

- **Training data:** 2D two-phase DNS with Basilisk C (CLS-VOF, broadband inlet forcing) across (Re, Ca, β).
- **Baseline ROM:** the first-order two-equation model of Lavalle et al. (2015), with state variables h(x,t) and q(x,t).
- **ML step:** learn corrections ΔC = C_DNS − C_Lavalle for the unclosed terms M_x, T_w, P_x, D_x, then embed them back into the ROM solver.

Future extensions: gas-sheared films, heated films, and microchannel flow boiling.

## Repository map

| Path | What it is |
|---|---|
| `dns/production/` | **Production Basilisk setup**: `filmflow0.c`, `filmflow_restart.c` (nondimensional, Nusselt scaling), `Makefile` (builds against `~/basilisk-old`), `run.sh` (SLURM), `count_sim_files.sh` |
| `dns/production/cases/<case>/` | `params.in` per DNS case (plus case-specific `.c` where one existed) |
| `dns/include/hdf5_headers/` | HDF5/XDMF output headers included by the Basilisk sources |
| `dns/tests/` | Physics/BC test setups (bc-blowup-fix, clsvof, white-noise domain tests, vertical-confined, hdf5-output) |
| `dns/legacy/` | Older dimensional setup (`inclined-whitenoise-dim`) and generic sources — not used for production |
| `rom/lavalle-cpp/` | C++ Lavalle two-equation solver (Rusanov + MUSCL, RK4, Noble–Vila capillarity, Chang noise forcing, HDF5 out) |
| `rom/matlab/` | Original MATLAB ROM (monochromatic + white-noise), `cases/ROM_Ca_Re_sweep/*/input.txt` |
| `pipeline/extract_closures/` | DNS → depth-integrated quantities and closure terms (`extract_closure_terms.py`); `generate_extraction_jobs.py` writes per-case SLURM jobs into the data store |
| `analysis/conditional_statistics/` | Conditional statistics of closure terms (peaks, regions) |
| `ml/functional_mlp/` | Shallow MLPs on (h, q, derivatives) — ablations for ΔM_x and ΔT_w |
| `ml/thinfilm_mlp/` | Neighbour-stencil MLP and curvature-feature experiments |
| `ml/legacy/` | Early scripts that use the old `Pytorch_ML_env` |
| `notebooks/analysis/`, `notebooks/ml/` | Exploratory and figure notebooks. Outputs are stripped, and many still contain absolute `/blue` paths |
| `datasets/` | **Manifests only** (`MANIFEST_dns.csv`, `MANIFEST_rom.csv`) — no data in git |
| `docs/` | `project-status.md`, `equations.md`, `data-layout.md`, `hipergator.md`, `decisions.md` |
| `envs/environment.yml` | Python env spec (the env itself lives on /blue) |
| `env.sh` | `source env.sh` → sets THINFILM_ROOT / THINFILM_STORE / THINFILM_DATA / THINFILM_ENV |

## Where things live (HiPerGator)

- **Code (this repo, new clone):** `$THINFILM_ROOT` (suggested `/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/code/thinfilm-closure-ml`).
- **Data store (old repo directory, now treated as data only):** `$THINFILM_STORE = /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml`
  - DNS runs: `$THINFILM_STORE/datasets/BasC/DNS/param_space/<case>/` (`output/*.h5`, `infc/`, dumps)
  - Extracted closures: `$THINFILM_STORE/datasets/BasC/DNS/param_space/ClosureData/`
  - ROM sweep runs: `$THINFILM_STORE/datasets/ROM_Ca_Re_sweep/<case>/`
  - Python env: `$THINFILM_STORE/software/envs/thinfilm-ml-py311`
  - ML caches/runs: `$THINFILM_STORE/codes/ml-models/{functional_mlp,thinfilm_mlp}/{cache_dir,runs}`
- Many configs and notebooks still hard-code these `/blue/.../thinfilm-closure-ml/...` paths. They keep working because the store has not moved. **Do not move or rename the store directory.**

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
