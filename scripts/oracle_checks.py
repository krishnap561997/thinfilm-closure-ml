#!/usr/bin/env python3
"""Two short checks on an oracle DNS run.

(a) Reproduction: h and q of the oracle run's 2D snapshots against the reference run's
    snapshots at the same times (e.g. L800 at t = 1810, 1820, 1830).
(b) Time resolution: for a closure file at output interval dt, the error of linear
    interpolation over 2 dt (f_k vs (f_{k-1} + f_{k+1})/2), relative to rms(f), for the
    corrections on the DNS grid and on the ROM grid (N cells).

    python scripts/oracle_checks.py --new <run>/output --ref <L800>/output --times 1810,1820,1830 \
        --closure <oracle>/closures/closures_part_5.h5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline" / "extract_closures"))
from extract_closure_terms import (build_cartesian_cell_grid, compute_cell_centers,  # noqa: E402
                                   depth_integral, scalar_to_matrix, vector_x_to_matrix)
from check_momentum_budget import restrict  # noqa: E402


def h_and_q(path):
    with h5py.File(path, "r") as s:
        cc = compute_cell_centers(s["Geometry/Points"][...], s["Topology"][...])
        xm, ym, idx = build_cartesian_cell_grid(cc)
        f = scalar_to_matrix(s["Cells/f"][...], idx, xm.shape)
        u = vector_x_to_matrix(s["Cells/u.x"][...], idx, xm.shape)
        t = float(np.asarray(s["Parameters/time"][()]).squeeze())
    x, h = depth_integral(xm, ym, f)
    _, q = depth_integral(xm, ym, u * f)
    return t, x, h, q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new"); ap.add_argument("--ref")
    ap.add_argument("--times", default="1810,1820,1830")
    ap.add_argument("--closure")
    ap.add_argument("--N", type=int, default=4000); ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--x-min", type=float, default=20.0); ap.add_argument("--x-max", type=float, default=760.0)
    a = ap.parse_args()

    if a.new and a.ref:
        print("(a) oracle run vs reference run (2D snapshots)")
        print("     t     t_new     t_ref   rms dh   max|dh|  dh/std(h)   rms dq  dq/std(q)")
        for T in (float(v) for v in a.times.split(",")):
            name = f"snapshot_{T:06.0f}.h5"
            tn, x, hn, qn = h_and_q(Path(a.new) / name)
            tr, _, hr, qr = h_and_q(Path(a.ref) / name)
            m = (x > a.x_min) & (x < a.x_max)
            dh, dq = (hn - hr)[m], (qn - qr)[m]
            print(f"  {T:6.0f} {tn:9.3f} {tr:9.3f} {np.sqrt(np.mean(dh**2)):8.2e} {np.max(abs(dh)):8.2e}"
                  f" {np.sqrt(np.mean(dh**2))/np.std(hr[m]):10.3f} {np.sqrt(np.mean(dq**2)):8.2e}"
                  f" {np.sqrt(np.mean(dq**2))/np.std(qr[m]):9.3f}")

    if a.closure:
        with h5py.File(a.closure, "r") as f:
            x, t = f["x"][:], f["t"][:]
            dt = float(np.median(np.diff(t)))
            m = (x > a.x_min) & (x < a.x_max)
            XR = (np.arange(a.N) + 0.5) * a.L / a.N
            mR = (XR > a.x_min) & (XR < a.x_max)
            ks = range(1, t.size - 1, 10)
            print(f"\n(b) linear interpolation over 2 dt = {2*dt:.3f} (midpoint vs neighbour average), "
                  f"{len(ks)} samples, t {t[0]:.2f}..{t[-1]:.2f}")
            print("  field          DNS grid    ROM grid (N=%d)" % a.N)
            for name in ("delta_Mx_ML", "delta_Tw_ML", "delta_Px_ML", "h", "q"):
                eD, fD, eR, fR = [], [], [], []
                for k in ks:
                    c, lo, hi = f[name][:, k], f[name][:, k - 1], f[name][:, k + 1]
                    e = c - 0.5 * (lo + hi)
                    eD.append(e[m]); fD.append(c[m] - (c[m].mean() if name in ("h", "q") else 0))
                    cR = restrict(x, c, a.N, a.L); eRk = restrict(x, e, a.N, a.L)
                    eR.append(eRk[mR]); fR.append(cR[mR] - (cR[mR].mean() if name in ("h", "q") else 0))
                r = lambda v: np.sqrt(np.mean(np.concatenate(v) ** 2))
                print(f"  {name:12s} {r(eD)/r(fD):10.4f} {r(eR)/r(fR):12.4f}")
            print("  (h, q relative to their fluctuation about the spatial mean)")


if __name__ == "__main__":
    main()
