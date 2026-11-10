#!/usr/bin/env python3
"""Crest-aligned L14 vs L15 comparison of the ΔP_x variants written by
pipeline/extract_closures/dpx_diagnostics.py (<label>_fields.h5).

Crests are matched at the same t by nearest crest (crests() from compare_mesh.py); every
dPx_* / Ip_* / kappa_* field is cut in xi = x - x_crest (|xi| < --half) and compared:
median relative L2 difference ||B - A|| / ||A|| and median correlation over matched waves.
Filtered variants (s<σ>) are compared as stored, i.e. each at its own filter width.

    python scripts/mesh_study/compare_dpx_mesh.py A_fields.h5 B_fields.h5 [--half 40]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mesh import crests  # noqa: E402


def load(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][...] for k in f.keys()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--half", type=float, default=40.0)
    ap.add_argument("--match-tol", type=float, default=12.0)
    ap.add_argument("--x-max-match", type=float, default=700.0)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    assert np.allclose(A["t"], B["t"]), "snapshot times differ"
    fields = sorted(k for k in A if k.startswith(("dPx_", "Ip_", "kappa")) and k in B)
    xi = np.arange(-a.half, a.half + 1e-9, 0.05)
    store = {f: ([], []) for f in fields}
    nmatch = 0
    for i in range(len(A["t"])):
        ca = [c for c in crests(A["x"], A["h"][:, i]) if c[0] < a.x_max_match and c[0] - a.half > 5]
        cb = [c for c in crests(B["x"], B["h"][:, i]) if c[0] < a.x_max_match and c[0] - a.half > 5]
        for xa, _ in ca:
            if not cb:
                break
            xb = min(cb, key=lambda c: abs(c[0] - xa))[0]
            if abs(xb - xa) >= a.match_tol:
                continue
            nmatch += 1
            for f in fields:
                store[f][0].append(np.interp(xa + xi, A["x"], A[f][:, i]))
                store[f][1].append(np.interp(xb + xi, B["x"], B[f][:, i]))
    print(f"matched waves: {nmatch} over {len(A['t'])} snapshots")
    print(f"{'field':<18}{'relL2 med':>10}{'q25':>8}{'q75':>8}{'corr med':>10}{'pk ratio':>10}")
    for f in fields:
        X, Y = np.array(store[f][0]), np.array(store[f][1])
        rl = np.linalg.norm(Y - X, axis=1) / np.linalg.norm(X, axis=1)
        cc = [np.corrcoef(X[n], Y[n])[0, 1] for n in range(len(X))]
        pk = np.abs(Y).max(1) / np.abs(X).max(1)
        print(f"{f:<18}{np.median(rl):10.3f}{np.percentile(rl,25):8.3f}{np.percentile(rl,75):8.3f}"
              f"{np.median(cc):10.3f}{np.median(pk):10.3f}")


if __name__ == "__main__":
    main()
