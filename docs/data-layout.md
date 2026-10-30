# Data layout

There is no data in git. The data store is the pre-restructure checkout on /blue:

```
$THINFILM_STORE = /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml
├── datasets/BasC/DNS/param_space/<case>/        DNS runs: params.in, output/snapshot_*.h5, infc/, dump-*
├── datasets/BasC/DNS/param_space/ClosureData/   extracted closure data + generated extraction jobs (configs/*.json)
├── datasets/ROM_Ca_Re_sweep/<case>/         MATLAB ROM sweep runs
├── codes/ml-models/*/cache_dir, runs/       ML caches, training runs, logs
└── software/envs/thinfilm-ml-py311/         Python env
```

**Case naming**

- DNS cases are named `Re{Re:03d}_Ca{Ca with . → _}_theta{β with . → _}`, for example `Re120_Ca0_010_theta6_4`.
- ROM cases are named `Ca_{Ca:07.4f}_Re_{Re:07.3f}_theta_{β:06.3f}`.

**Manifests**

- `datasets/MANIFEST_dns.csv` has one row per DNS case:
  - the parameters from `params.in`;
  - automatic columns `n_h5_snapshots`, `t_last`, `status` (not_started / running / partial / complete) and `closure_extracted`;
  - hand-filled columns `split` (train / val / test / exclude) and `notes`;
  - `store_path`.
- `datasets/MANIFEST_rom.csv` has one row per ROM sweep case, with the fields from `input.txt`.

To refresh the automatic columns, run `python $THINFILM_ROOT/datasets/update_manifest.py` on HiPerGator. Add `--dry-run` to print only the summary. Hand-filled columns are preserved.
