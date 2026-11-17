# Decision log

Record each decision as a short entry: what was decided and why. Newest entries go at the top.

## 2026-11-17 — Priority: a posteriori accuracy over interpretability

- **User's decision:** focus on a posteriori accuracy rather than on interpretable 1D coefficient functions. Closures may take (h, q or r = q/h³, h_x, q_x) as inputs where the guarantees allow it (relaxation part, or a flat-film-exact plain MLP); the flux part keeps Φ(h) unless V2 is tried. Options and order: `docs/plans/structured-closure-progress.md`, "Next options".

## 2026-11-17 — Structured closure V1: constraint form, λ_lin, zero-sum derivative, GPU

- **L2 is imposed on the dispersion relation, not on fitted (a, b, c, d).** The flux part is differentiated unfiltered by the finite-volume scheme, the relaxation part sees D_G(σ), so the filter factor exp(−k²σ²/2) applies to the relaxation part only (`ml/structured_closure/losses.py`). With σ = 0 this reproduces B.8.
- **L2 imposed at the 85 OS case-table points** minus those with held-out parameters (user's decision); it needs no DNS.
- **λ_lin = 10** (user's decision, from the λ = 1/10/100 scan: 10 halves the growth error of 1 at no held-out cost; 100 costs 15% held-out NRMSE).
- **Sweep with one seed** (user's decision), seed 1 kept as a backup: λ_lin {0, 10} × λ_F {0, 0.1} × σ {0.5, 0.25}.
- **Training on GPU:** 12× faster than 8 CPU cores; the user relaxed the 2-GPU limit for this session only.
- **Zero-sum Gaussian derivative** for the structured closure (bug found by the uniform-film ROM test): the truncated kernel gives h_xx ≠ 0 on constants, differently on the DNS and ROM grids. ROM: `gauss_deriv_hN` subtracts c0 × smoothed field; training subtracts c0·h. The older ML models are unchanged.

## 2026-11-17 — Phase C training data: excluded cases, derivative filter σ; comb DNS window

- **Excluded from structured-closure training** (user's decision): Re10 Ca0.001 β90, Re20 Ca0.001 β6.4 and Re40 Ca0.1 β90.
  - Their cache-v2 check (∂ₓΔF against the extracted ΔM_x) fails at every σ: NRMSE 0.84–0.96, against below 0.45 for every other case.
  - All three are partial runs; Re40 Ca0.1 β90 has only 25 snapshots.
  - Revisit them when the runs are finished and re-extracted.
- **Derivative filter for Phase C** (user's decision): σ = 0.5 h_N by default, with σ = 0.25 as a comparison. 0.5 is about 2.5 ROM cells, so it stays clear of grid noise; it halves derivatives at a wavelength of about 2.7 h_N. 0.25 keeps capillary ripples (better cache check at Ca 0.06–0.1) but is only about 1.25 ROM cells. Choose between them in closed loop, not on the a priori fit.
- **Comb DNS Re40 Ca0.04 β45** is analysed in x = 30–100. In the standard 60–150 window its 31 comb modes add up to a nonlinear wave height of 0.28 (`docs/results/orr-sommerfeld.md`). No rerun.

## 2026-11-16 — Closure extractions redone/added; cache v2

- **New extractions** (user's decision): Re050 Ca0.01 β45 (rerun complete, job 51592781). Also Re10 Ca0.005, Re20 Ca0.002 β90, Re20 Ca0.005 β6.4, Re40 Ca0.002 β45 and Re40 Ca0.1 β90 (no closure file yet), and the partial 3 Nov runs Re10 Ca0.001 β90 (to t = 1100) and Re20 Ca0.001 β6.4 (to t = 1690). Jobs 51592928–34. The two 3 Nov cases must be extracted again when their runs finish.
- **Full re-extraction of stale closure files** (the "closure files to redo" of 10 Nov): Re10 Ca0.01, Re10 Ca0.1, Re40 Ca0.1 β45 and Re10 Ca0.04 β90. Their snapshots at the overwritten times were regenerated on 10 Nov with the fixed restart, and the runs now reach t = 2000. The old files still contain the overwritten snapshots, and two stop at t = 1580 and 1810. Jobs 51592935–38 write `thinfilm_closure_data_<case>.h5.new` (configs in `ClosureData/configs_redo_20261116/`). **Pending:** after cache-v2 job 51592842 has finished, rename each old file to `.stale_20261116` and the `.new` file into place, then rebuild those cases in cache v2.
- **Cache v2** (`pipeline/cache_v2/build_cache_v2.py`, `$THINFILM_STORE/codes/ml-models/cache_v2/`):
  - full record and no flat mask;
  - D_G(σ) derivatives for σ = 0.25, 0.5 and 1.0, the same operator as the ROM;
  - snapshots overwritten on 10 Nov are dropped wherever they are still present;
  - the split follows `ml/functional_mlp/data_config.json`; other cases are marked `extra`.

## 2026-11-10 — Broadband restart bug: restarts ran without inlet forcing

- `dns/production/filmflow_restart.c` set the noise phases in an `event init_noise (i = 0)`. `restore()` replays the event schedule with `events(false)`, so that event never ran after a restart and the phases stayed uninitialised. Inlet RMS after a restart: about 0.002 against about 0.025 (Re10 Ca0.005: 0.026 → 0.0015). Fixed in commit 6426624 (phases set in `main()`, same seed and `rand()` sequence, so the forcing is continuous); `check_profile_restart.py` tests this on every new profile restart.
- **Affected data.** Re050 Ca0.01 β45 was restarted at t = 1050 (RMS 0.053 → 0.0054), so its record from t = 1050 on, including t = 1050–2000 in its closure file, was not forced as intended. Re040 Ca0.1 β90 was restarted at t = 100. The 11 `rs_*` restarts of 10 Nov were cancelled; they had overwritten these original snapshots: Re10 Ca0.005 760; Re10 Ca0.01 1960; Re10 Ca0.04 β90 1910, 1920; Re10 Ca0.1 1560, 1570; Re20 Ca0.005 β6.4 1210; Re40 Ca0.002 1110; Re40 Ca0.1 1810, 1820.
- Re050 (decided by the user): rerun with the fixed source. dump-001050 no longer exists, so the rerun starts from dump-001000 (original run) to T_END = 2000. Its dumps and snapshots for t > 1050 and its old closure file are in `Re050_Ca0_010_theta45/bad_forcing_restart_t1050_20261110/`; re-extract after the rerun.
- Deleted on 10 Nov (user's decision): the Re050 quarantine folder (t > 1050, 53 GB, incl. the old closure file); the 11 overwritten snapshots listed above (with their .xmf); and in Re040 Ca0.1 β90 the snapshots t = 110–280 and dumps 150–250 written by its unforced restart. Its original t = 100 dump (`dump.stale_2026-10-21`) was renamed `dump-000100` and `dump` links to it. Closure files to redo after the restarts: Re050, Re10 Ca0.01, Re10 Ca0.1, Re40 Ca0.1 (their v2 append was refused by the h check) and Re10 Ca0.04 β90 (full extraction included snapshots 1910, 1920).
- The 12 affected case directories have the fixed `filmflow_restart` (old source and binary kept as `*.pre_fix_20261110`). No data before a restart point is affected, so nothing is rerun from t = 0.

## 2026-11-10 — ΔP_x target v2; low-We cases out of the ΔP_x training for now

- Decided with the user: the ΔP_x target is `delta_Px_ML_v2` (liquid-pressure Leibniz form of ∫∂ₓp, Gaussian σ = 0.5 h_N, exact curvature in the Lavalle capillary term). The old `delta_Px_ML` stays in the files for reproducibility. Evidence: `docs/results/dpx-closure.md`, section "Extraction".
- The L14/L15 mesh test is not used as an acceptance criterion for ΔP_x: the two meshes have different ripples (κ differs as much as ΔP_x).
- Cases with We ≤ 0.1, where the momentum budget does not close with any extraction, are left out of the ΔP_x training until dense-in-time restart data shows whether the snapshots or the runs are at fault.

## 2026-11-10 — Train and score on the entire time record

- Decided by the user: ML closures (and any new functional-form fits) are trained and scored on the **entire** time record, not only t > 1000. This supersedes the 2026-11-01 entry "Fit the ΔM_x / ΔT_w functional forms on t > 1000 only", and retires the reminder in that entry.
- Results already reported on t > 1000 (`functional-form-structure.md`, the budget check, the oracle window) stay as historical numbers; re-running them on the full record is a separate task.
- New dense DNS data for ΔP_x and memory models covers the whole record by restarting from several dumps (t = 200, 800, 1500).

## 2026-11-10 — Paper 1 scope: a comprehensive passive-gas model; Papers 2 and 3

- Decided by the user: Paper 1 is a comprehensive ML-augmented reduced model for falling films with a **passive gas**. It is not limited to Markovian closures: if ΔP_x (or another term) needs memory, the non-Markovian closure belongs in Paper 1.
- Paper 2: gas-sheared films. Paper 3: heated films (heat transfer). Microchannel flow boiling comes later.
- Supersedes the scope bullet of the 2026-10-29 entry.

## 2026-11-01 — Fit ΔM_x + ΔT_w together; test the chain rule first

- Decided by the user: the functional forms are developed for the sum ΔM_x + ΔT_w (and for ΔM_x), not for ΔT_w alone, because the two cancel to about 64% (ΔT_w ≈ −0.64 ΔM_x, corr −0.975). Details and the chain-rule test: `docs/results/functional-form-structure.md`.
- Approach: sparse regression on a physical library first, PySR as a cross-check; pysindy and pysr installed in `thinfilm-ml-py311` (`envs/pysr_env.sh` sets the Julia paths on /blue). Scores are per case, normalised by each case's σ.

## 2026-11-01 — Fit the ΔM_x / ΔT_w functional forms on t > 1000 only

- **Superseded on 2026-11-10:** train and score on the entire time record (see the entry above). The reminder below is retired.
- Decided by the user: the functional forms for ΔM_x and ΔT_w are fitted (and scored) on snapshots with t > 1000, i.e. developed waves. The budget check and the oracle also use t ≥ 1000.
- Why: the held-out error of the 1×16 MLP for Re120 Ca0.02 β90 is dominated by an early-time transient (NRMSE 0.33 at t = 200–500, 0.16 at 500–1000, 0.08–0.10 for t ≥ 1000). See `docs/results/mlp-heldout-eval.md`.
- **Caveat: not verified yet.** The claim that the early-time error is a developing-wave transient comes from Claude's analysis of the prediction files by time bin. The user has not verified it in person (no look at the fields or time series for that case), and it has been checked for only four of the seven held-out cases. The cut at t > 1000 rests on it.
- **Reminder for Claude:** once the functional-form problem has been addressed (a form is fitted and evaluated), raise this decision again and ask the user to verify the transient claim and confirm that t > 1000 is still the right cut (and whether t = 1000 itself is included; the cache has snapshots every 10).

## 2026-10-30 — Oracle DNS writes 1D column profiles in situ

- For the oracle restart, Basilisk writes depth-integrated profiles (`dns/include/column_profiles.h`) instead of 2D snapshots: h, q, ∫u², ∫₀ʰ∂ₓp, ∂_z u|₀ and the raw interface velocity, at t and t−dt. That is about 1 MB per output instead of 133 MB, so the output interval can be 0.01 or finer.
- Derivatives and ΔC are still computed in Python (`pipeline/extract_closures/profiles_to_closures.py`), reusing the functions in `extract_closure_terms.py`. On the same state this matches the 2D extraction to round-off (relative difference ≤ 5×10⁻¹² for every field).
- Oracle source: `dns/oracle/filmflow_restart_mono.c` (production restart, 10 Hz inlet as in `BasC/fixed-frequency`). The source of the original L800 monochromatic binary was not found, so equivalence is checked against the existing L800 snapshots at t = 1810–1830.

## 2026-10-30 — DNS case directories moved under BasC/DNS

- `$THINFILM_STORE/datasets/BasC/param_space` is now `$THINFILM_STORE/datasets/BasC/DNS/param_space`; its contents, including `ClosureData/`, are unchanged. `fixed-frequency/` and `liu1994/` stay in `BasC/`.
- Every text reference was rewritten: repo configs, scripts, manifests, docs and notebooks; the ClosureData extraction configs; and the old code copies under `$THINFILM_STORE/codes/`. The list of edited files and a backup of the originals are in `BasC/DNS/PATH_MIGRATION_2026-10-30.txt`.
- ML caches or saved runs that store absolute data paths in binary files (`.npz`, `.pt`) were not rewritten. Rebuild a cache if it fails to find its source files.
- **Correction (4 Nov):** the 30 Oct search used `rg`, which skips git-ignored paths, so it missed `codes/ml-models/functional_mlp/cache_dir/manifest.json` and two scripts in the store (`ClosureData/run.sh`, `param_space/count_sim_files.sh`). They were fixed on 4 Nov (originals in `BasC/DNS/path_migration_2026-11-04_originals.tgz`). The manifest's cache `id` is stored, not recomputed, so it is unchanged. Remaining old-path mentions are intentional: 50 SLURM logs in `ClosureData/logs/`, Jupyter `.ipynb_checkpoints`, and the migration records.

## 2026-10-29 — Paper 1 scope, venue and closure targets

- **Venue and dates.** Target IJHMT or AI Thermal Fluids. Submit 15 Dec 2026. Results for the APS DFD talk (22 Nov) freeze on 15 Nov.
- **Scope** (*superseded on 2026-11-10*: Paper 1 is a comprehensive passive-gas model, memory closures included if needed; Paper 2 = shear, Paper 3 = heat transfer). Paper 1 covers ΔM_x, ΔT_w and ΔP_x, with ΔP_x via curvature inputs. A memory-model Paper 2 happens only if the oracle shows ΔP_x matters *a posteriori* and curvature does not fix it; otherwise the next paper is the gas-sheared or heated film.
- **Flux-form target for M_x.** ΔF = ∫u² − (q²/h + Λ²h⁵/45), the momentum part only, and ΔM_x = ∂ₓΔF. This keeps the equation conservative and gives a smoother target. The −Λ²h⁵/75 that Lavalle moves into the flux from T_w stays with T_w; ΔT_w remains a pointwise source measured against the full T_w,Lav.
- **Oracle a posteriori first.** Run the ROM with exact DNS corrections over short horizons from a DNS state. Start with the monochromatic Re = 120, β = 45°, 10 Hz case, re-run as a short restart window with output every 0.1–0.5 time units, because 10-unit h5 output is too coarse to interpolate in time.
- **Statistical metrics for broadband cases.** Broadband ROM and DNS trajectories decorrelate, so compare h PDFs, amplitude and celerity, spectra and growth. Use the monochromatic cases for snapshot-by-snapshot comparison.

## 2026-10 — Basilisk version

- All thin-film DNS are built with `~/basilisk-old/src`, which is compatible with the HDF5/XDMF output headers.
- The newer Basilisk install belongs to the heterogeneous-nucleation project and cannot write these HDF5 files.
- Eight case Makefiles reference `$BASILISK`, but those cases were also run with basilisk-old.

## 2026-10 — Repository restructure

- Code and data were split. The old /blue checkout is now the data store, and git holds only source code, small configs and manifests.
- Pre-restructure state is tagged `pre-restructure-2026-10`. Every removed binary, log, notebook output and figure is recoverable from that tag.

## Earlier (reconstructed from the ONR report, Oct 2026)

**Baseline model**

- The baseline ROM is the first-order Lavalle et al. (2015) two-equation model.
- It is Galilean invariant, has a mathematical energy structure, admits Noble–Vila stable capillarity, and gives good neutral curves (Richard et al. 2019).

**DNS setup**

- The DNS uses broadband inlet forcing (Chang et al. 1996) rather than a sweep over monochromatic frequencies, so that a single run samples many wave states.
- Settings: F₀ = 0.025, M = 1000, f_cut = 28 Hz.
- f_cut is dimensional and the same for every case. Checked on 16 Nov 2026 against OS (`docs/results/orr-sommerfeld.md`, B.4): f_cut ≥ 2 f_max (spatial) in only 7 of the 36 DNS cases. In 13 it lies below f_max itself (β 45°/90°, Re ≥ 40), and in the β 6.4° cases at Re ≤ 40 it extends into the stable band. The production runs stay as they are; the broadband DNS do not force the linearly most amplified mode at high Re.
- The parameter space is (Re, Ca, β), following Dhopeshwar et al. (2025), so that inertia and capillarity can be varied independently.
- Candidate cases were kept only if a glycerol–water mixture can realise them (≤ 90 wt%, h_N between 0.1 and 2 mm, u_N < 0.5 m/s).
- MAXLEVEL 14 was chosen over 15 for cost; level 15 changes only the wave speed slightly.

**ML scope**

- ΔD_x is dropped from the ML targets because it is negligible.
- Priority targets are ΔM_x and ΔT_w; ΔP_x is under investigation.

- 2026-11-03: Mesh study level 14 vs 15 (Re120 Ca0.01 β45, 10 Hz, t = 550–650, crest-aligned): celerity +1.2 % at level 15; ΔM_x, ΔT_w peaks within 1 %; raw ΔP_x/ΔD_x differ (extraction noise). Level 15 stops at t = 650 and has a CFL-limited dt vs the DT = 0.003 cap at level 14. Details: `docs/results/mesh-sensitivity.md`.
