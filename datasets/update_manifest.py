#!/usr/bin/env python3
"""Refresh datasets/MANIFEST_dns.csv from the data store (run on HiPerGator).

For every case in dns/production/cases/ (or any Re*_Ca*_theta* directory already in the data store)
this fills the automatic columns:

  params from params.in, n_h5_snapshots, t_last, status, closure_extracted, store_path

and keeps the hand-filled columns (split, notes) from the existing manifest.

status:  not_started  no output/snapshot_*.h5
         running      newest snapshot written within --running-hours
         partial      stopped before T_END
         complete     last snapshot time >= T_END - T_OUT

Usage:
  source $THINFILM_ROOT/env.sh
  python $THINFILM_ROOT/datasets/update_manifest.py            # writes the manifest in place
  python $THINFILM_ROOT/datasets/update_manifest.py --dry-run  # print a summary only
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("THINFILM_DATA",
             "/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets")) / "BasC/DNS/param_space"
MANIFEST = REPO / "datasets/MANIFEST_dns.csv"
SNAP = re.compile(r"snapshot_(\d+(?:\.\d+)?)\.h5$")
CASE = re.compile(r"Re\d{3}_Ca\d+_\d+_theta\d+(?:_\d+)?$")

PARAMS = ["Re", "Ca", "beta_deg", "LX", "MAXLEVEL", "T_END", "T_OUT", "T_DUMP",
          "F0_NOISE", "FCUT_DIM_Hz", "MNOISE", "NOISE_SEED"]
KEYS = ["RE", "CA", "ANGLE_DEG", "LX", "MAXLEVEL", "T_END", "T_OUT", "T_DUMP",
        "F0_NOISE", "FCUT_DIM", "MNOISE", "NOISE_SEED"]
AUTO = ["n_h5_snapshots", "t_last", "status", "closure_extracted"]
MANUAL = ["split", "notes"]
COLS = ["case"] + PARAMS + AUTO + MANUAL + ["store_path"]


def read_params(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--running-hours", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    old = {}
    if MANIFEST.exists():
        with open(MANIFEST, newline="") as f:
            old = {r["case"]: r for r in csv.DictReader(f)}

    cases = {p.name for p in (REPO / "dns/production/cases").iterdir() if p.is_dir()}
    if a.store.is_dir():
        cases |= {p.name for p in a.store.iterdir() if p.is_dir() and CASE.match(p.name)}

    rows = []
    for c in sorted(cases):
        store_case = a.store / c
        prm = read_params(store_case / "params.in") or read_params(REPO / "dns/production/cases" / c / "params.in")
        row = {"case": c, "store_path": str(store_case)}
        row.update({col: prm.get(k, "") for col, k in zip(PARAMS, KEYS)})

        snaps = []
        out = store_case / "output"
        if out.is_dir():
            for p in out.iterdir():
                m = SNAP.match(p.name)
                if m:
                    snaps.append((float(m.group(1)), p))
        t_last = max((t for t, _ in snaps), default=None)
        t_end = float(prm.get("T_END") or "nan")
        t_out = float(prm.get("T_OUT") or 0)
        if not snaps:
            status = "not_started"
        elif time.time() - max(p.stat().st_mtime for _, p in snaps) < a.running_hours * 3600:
            status = "running"
        elif t_last is not None and t_last >= t_end - t_out:
            status = "complete"
        else:
            status = "partial"
        closure = a.store / "ClosureData" / f"thinfilm_closure_data_{c}.h5"

        row.update({
            "n_h5_snapshots": len(snaps),
            "t_last": "" if t_last is None else f"{t_last:g}",
            "status": status,
            "closure_extracted": "yes" if closure.exists() else "no",
        })
        prev = old.get(c, {})
        row.update({k: prev.get(k, "") for k in MANUAL})
        if not row["notes"] and prev.get("notes"):
            row["notes"] = prev["notes"]
        rows.append(row)

    summary = {}
    for r in rows:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    print(f"{len(rows)} cases:", ", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
          f"| closure extracted: {sum(r['closure_extracted'] == 'yes' for r in rows)}")
    if a.dry_run:
        return
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
