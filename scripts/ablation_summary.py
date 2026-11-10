#!/usr/bin/env python3
"""Compare the input groups of one functional-MLP ablation from the training histories.

For each (input group, target) with a finished seed run: best and final validation NRMSE
(sqrt of the normalised validation loss; targets are standardised, so 1.0 means "predict the
mean") and the epoch of the best value. Prints a table, groups ordered by the first target's
validation error.

    python scripts/ablation_summary.py <runs/ablation_01/layers1_width16> [--seed 42]
    python scripts/ablation_summary.py <runs/dpx_v2_sweep01/layers4_width64> --targets delta_Px_ML_v2
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
    ap.add_argument("--targets", nargs="+", default=["delta_Mx_ML", "delta_Tw_ML"])
    a = ap.parse_args()

    sweep = json.loads((a.group_dir / "sweep.json").read_text())
    fields = {r["input_group"]: r["input_fields"] for r in sweep["runs"]}
    res = {}
    for g in fields:
        for t in a.targets:
            p = a.group_dir / g / t / f"seed{a.seed}" / "history.csv"
            if not p.is_file(): continue
            rows = list(csv.DictReader(p.open()))
            val = np.array([float(r["val_loss"]) for r in rows])
            tr = np.array([float(r["train_loss"]) for r in rows])
            res[(g, t)] = (np.sqrt(val.min()), int(val.argmin()) + 1, np.sqrt(val[-1]), np.sqrt(tr[-1]), len(rows))

    groups = sorted({g for g, _ in res}, key=lambda g: res.get((g, a.targets[0]), (9,))[0])
    w = max([26] + [len(','.join(v)) for v in fields.values()])
    print(f"{'input group':16s} {'inputs':{w}s}" + "".join(f"{t[6:]+' best':>14s}{'@ep':>5s}{'final':>8s}{'train':>8s}" for t in a.targets))
    for g in groups:
        line = f"{g:16s} {','.join(fields[g]):{w}s}"
        for t in a.targets:
            r = res.get((g, t))
            line += f"{r[0]:14.3f}{r[1]:5d}{r[2]:8.3f}{r[3]:8.3f}" if r else f"{'-':>35s}"
        print(line)
    print("\n(values are sqrt of normalised MSE; train = last epoch; every model also receives Re, Ca, beta)")


if __name__ == "__main__":
    main()
