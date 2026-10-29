# Data layout

There is no data in git. The data store is the pre-restructure checkout on /blue:

```
$THINFILM_STORE = /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml
├── datasets/BasC/param_space/<case>/        DNS runs: params.in, output/snapshot_*.h5, infc/, dump-*
├── datasets/BasC/param_space/ClosureData/   extracted closure data + generated extraction jobs (configs/*.json)
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
  - `basilisk_build` (old or current);
  - `n_h5_snapshots`, from the last file count on 2026-10-29;
  - `closure_config`, which records whether an extraction config exists;
  - `status`, which you fill in;
  - `store_path`.
- `datasets/MANIFEST_rom.csv` has one row per ROM sweep case, with the fields from `input.txt`.

To refresh the snapshot counts, run `dns/production/count_sim_files.sh` and update the manifest.
