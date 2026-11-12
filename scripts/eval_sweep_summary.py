#!/usr/bin/env python3
"""Summarise evaluate.py metrics over a sweep: one row per input group.

    python eval_sweep_summary.py SWEEP_DIR [--target delta_Px_ML_v2] [--seed 42] [--sort val]

Columns: pooled R2 / NRMSE on val and heldout, held-out case-average R2, and held-out R2 per
case (ordered by We). Sorted by validation pooled NRMSE (model choice is made on val).
"""
import argparse
import csv
from pathlib import Path


def we(case):
    p = case.split("_")
    return float(p[0][2:]) * float(p[1][2:] + "." + p[2])


def load(path):
    if not path.exists():
        return None
    with open(path) as fh:
        return {r["case"]: r for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--target", default="delta_Px_ML_v2")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--sort", choices=("val", "heldout"), default="val")
    a = ap.parse_args()
    rows, cases = [], None
    for g in sorted(p for p in a.sweep.iterdir() if p.is_dir()):
        d = g / a.target / f"seed{a.seed}"
        v, h = load(d / "eval_val/metrics.csv"), load(d / "eval_heldout/metrics.csv")
        if v is None or h is None:
            continue
        if cases is None:
            cases = sorted((c for c in h if c not in ("POOLED", "CASE_AVERAGE")), key=we)
        rows.append((g.name, v, h))
    key = (lambda r: float(r[1]["POOLED"]["NRMSE"])) if a.sort == "val" else (lambda r: float(r[2]["POOLED"]["NRMSE"]))
    rows.sort(key=key)
    hdr = f"{'group':<9} {'val R2':>7} {'val NR':>7} {'ho R2':>7} {'ho NR':>7} {'ho cAvR2':>8} | " + " ".join(f"We{we(c):<5g}" for c in cases)
    print(hdr)
    for name, v, h in rows:
        f = lambda r, k: float(r[k])  # noqa: E731
        print(f"{name:<9} {f(v['POOLED'],'R2'):7.3f} {f(v['POOLED'],'NRMSE'):7.3f} {f(h['POOLED'],'R2'):7.3f} "
              f"{f(h['POOLED'],'NRMSE'):7.3f} {f(h['CASE_AVERAGE'],'R2'):8.3f} | "
              + " ".join(f"{f(h[c],'R2'):7.3f}" for c in cases))
    print("held-out cases:", ", ".join(f"{c} (We {we(c):g})" for c in cases))


if __name__ == "__main__":
    main()
