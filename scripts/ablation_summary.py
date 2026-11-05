#!/usr/bin/env python3
"""Compare the input groups of one functional-MLP ablation from the training histories.

For each (input group, target) with a finished seed run: best and final validation NRMSE
(sqrt of the normalised validation loss; targets are standardised, so 1.0 means "predict the
mean") and the epoch of the best value. Prints a table, groups ordered by ΔM_x validation error.

    python scripts/ablation_summary.py <runs/ablation_01/layers1_width16> [--seed 42]
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group_dir", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sweep = json.loads((a.group_dir / "sweep.json").read_text())
    fields = {r["input_group"]: r["input_fields"] for r in sweep["runs"]}
    res = {}
    for g in fields:
        for t in ("delta_Mx_ML", "delta_Tw_ML"):
            p = a.group_dir / g / t / f"seed{a.seed}" / "history.csv"
            if not p.is_file(): continue
            rows = list(csv.DictReader(p.open()))
            val = np.array([float(r["val_loss"]) for r in rows])
            tr = np.array([float(r["train_loss"]) for r in rows])
            res[(g, t)] = (np.sqrt(val.min()), int(val.argmin()) + 1, np.sqrt(val[-1]), np.sqrt(tr[-1]), len(rows))

    groups = sorted({g for g, _ in res}, key=lambda g: res.get((g, "delta_Mx_ML"), (9,))[0])
    print(f"{'input group':32s} {'inputs':26s}" + "".join(f"{t[6:8]+' best':>10s}{'@ep':>5s}{'final':>8s}{'train':>8s}" for t in ("delta_Mx_ML", "delta_Tw_ML")))
    for g in groups:
        line = f"{g:32s} {','.join(fields[g]):26s}"
        for t in ("delta_Mx_ML", "delta_Tw_ML"):
            r = res.get((g, t))
            line += f"{r[0]:10.3f}{r[1]:5d}{r[2]:8.3f}{r[3]:8.3f}" if r else f"{'-':>31s}"
        print(line)
    print("\n(values are sqrt of normalised MSE; train = last epoch; every model also receives Re, Ca, beta)")


if __name__ == "__main__":
    main()
