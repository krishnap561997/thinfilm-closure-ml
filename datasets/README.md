# datasets/

This folder holds **manifests only**. The data itself lives in the data store on HiPerGator; see `docs/data-layout.md`.

- `MANIFEST_dns.csv` has one row per Basilisk case.
  - Refresh it on HiPerGator with `python datasets/update_manifest.py`.
  - The script fills in the parameters, `n_h5_snapshots`, `t_last`, `status` and `closure_extracted`, and it preserves the hand-filled columns.
  - Hand-filled columns:
    - `split`: `train`, `val`, `test` or `exclude`. This records which cases each model or figure used.
    - `notes`: anything worth remembering, e.g. "blew up at t=830, restarted", "used in ONR Fig. 4", "validation vs Liu & Gollub".
  - The `n_h5_snapshots` values committed now come from the 2026-10-29 file count; `t_last`, `status` and `closure_extracted` are blank until the first refresh.
- `MANIFEST_rom.csv` has one row per MATLAB ROM sweep case, with the fields from `input.txt`.
