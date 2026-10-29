# datasets/

This folder holds **manifests only**. The data itself lives in the data store on HiPerGator; see `docs/data-layout.md`.

- `MANIFEST_dns.csv` has one row per Basilisk case.
  - Most columns are generated from each case's `params.in` and the 2026-10-29 snapshot counts.
  - `status` and `notes` are filled in by hand.
  - `basilisk_build` records which Basilisk the case was compiled with:
    - `basilisk-old` means `~/basilisk-old/src`;
    - `basilisk-current` means `$BASILISK`.
- `MANIFEST_rom.csv` has one row per MATLAB ROM sweep case, with the fields from `input.txt`.
