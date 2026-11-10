# Decision log

Record each decision as a short entry: what was decided and why. Newest entries go at the top.

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
- Open question: f_cut is dimensional and the same for every case. Check it against 2f₀ for each case.
- The parameter space is (Re, Ca, β), following Dhopeshwar et al. (2025), so that inertia and capillarity can be varied independently.
- Candidate cases were kept only if a glycerol–water mixture can realise them (≤ 90 wt%, h_N between 0.1 and 2 mm, u_N < 0.5 m/s).
- MAXLEVEL 14 was chosen over 15 for cost; level 15 changes only the wave speed slightly.

**ML scope**

- ΔD_x is dropped from the ML targets because it is negligible.
- Priority targets are ΔM_x and ΔT_w; ΔP_x is under investigation.

- 2026-11-03: Mesh study level 14 vs 15 (Re120 Ca0.01 β45, 10 Hz, t = 550–650, crest-aligned): celerity +1.2 % at level 15; ΔM_x, ΔT_w peaks within 1 %; raw ΔP_x/ΔD_x differ (extraction noise). Level 15 stops at t = 650 and has a CFL-limited dt vs the DT = 0.003 cap at level 14. Details: `docs/results/mesh-sensitivity.md`.
