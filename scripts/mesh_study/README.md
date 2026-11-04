# scripts/mesh_study

Mesh-sensitivity comparison of two DNS runs (written for MAXLEVEL 14 vs 15, Re120 Ca0.01 β45; see `docs/results/mesh-sensitivity.md`).

- `find_transient_exit.py RUN_DIR --t1 650 [--csv out.csv]`: per snapshot max h, its x and the outlet-region max, to find when the initial disturbance leaves the domain. Do not rely on the default `--h-big` 1.5 threshold: developed waves also exceed it, read the front position instead.
- `compare_mesh.py A.h5 B.h5 --labels L14 L15 --out DIR`: takes two closure files from `pipeline/extract_closures/extract_closure_terms.py` (same snapshot times), tracks h crests (celerity, wavelength, amplitude), matches crests at equal t by nearest crest (`--match-tol`), and compares h, q, ΔM_x, ΔT_w, ΔP_x, ΔD_x and S_ML in ξ = x − x_crest (raw and Gaussian-smoothed). Writes CSVs, an npz and figures to `--out`.

Run with `$THINFILM_ENV/bin/python -I`. Outputs go in the data store (`ClosureData/mesh_study/`), not in git.
