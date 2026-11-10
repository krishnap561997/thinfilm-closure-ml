#!/usr/bin/env python3
"""Check a filmflow_restart_profiles run against its source run; exit 1 on failure.

1. Inlet forcing continuity: F(t) from <run>/inlet_signal.dat against the source run's
   inlet_signal.dat, interpolated to the same t. The restart must reproduce the original
   broadband signal (same seed and phases), so the relative RMS difference must be small
   and the RMS ratio close to 1. (Before the 2026-11-10 fix the RMS ratio was ~0.06.)
2. In-situ profiles against the 2D snapshot of the same state (a profile and a snapshot
   at the same t): h, q, int_px_dy, and the liquid-pressure fields int_pl_dy and
   p_liq_interface against extract_closure_terms.ip_liquid_leibniz on the 2D fields.

    python check_profile_restart.py RUN_DIR SOURCE_CASE_DIR
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_closure_terms import (  # noqa: E402
    build_cartesian_cell_grid, compute_cell_centers, compute_px, depth_integral,
    first_derivative_x, integrate_to_height, liquid_top_index, extrapolate_top, trapz_to,
    PX_GAP, scalar_to_matrix, scalar_value, vector_x_to_matrix,
)


def load_inlet(path):
    d = np.loadtxt(path, usecols=(0, 1))
    t, F = d[:, 0], d[:, 1]
    # keep the last occurrence of each time (a source file may contain earlier restarts)
    _, idx = np.unique(t[::-1], return_index=True)
    keep = np.sort(len(t) - 1 - idx)
    return t[keep], F[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run"); ap.add_argument("source")
    ap.add_argument("--inlet-tol", type=float, default=0.05)
    ap.add_argument("--field-tol", type=float, default=1e-8)
    a = ap.parse_args()
    run, src = Path(a.run), Path(a.source)
    ok = True

    tr, Fr = load_inlet(run / "inlet_signal.dat")
    ts, Fs = load_inlet(src / "inlet_signal.dat")
    m = (tr >= ts.min()) & (tr <= ts.max())
    Fi = np.interp(tr[m], ts, Fs)
    rel = np.sqrt(np.mean((Fr[m] - Fi) ** 2)) / np.std(Fi)
    ratio = np.std(Fr[m]) / np.std(Fi)
    good = rel < a.inlet_tol and abs(ratio - 1) < a.inlet_tol
    ok &= good
    print(f"inlet: {m.sum()} samples t={tr[m].min():.3f}..{tr[m].max():.3f}; rel RMS diff {rel:.3g}, "
          f"RMS ratio {ratio:.4f} -> {'OK' if good else 'FAIL'}")

    snaps = sorted((run / "output").glob("snapshot_*.h5"))
    if not snaps:
        print("no 2D snapshot in run/output -> FAIL"); sys.exit(1)
    with h5py.File(snaps[-1], "r") as s:
        t2 = scalar_value(s["Parameters/time"])
        cc = compute_cell_centers(s["Geometry/Points"][...], s["Topology"][...])
        xm, ym, idx = build_cartesian_cell_grid(cc)
        f = scalar_to_matrix(s["Cells/f"][...], idx, xm.shape)
        u = vector_x_to_matrix(s["Cells/u.x"][...], idx, xm.shape)
        p = scalar_to_matrix(s["Cells/p"][...], idx, xm.shape)
    prof = None
    for pp in sorted((run / "profiles").glob("profile_*.h5")):
        with h5py.File(pp, "r") as g:
            if abs(scalar_value(g["parameters/time"]) - t2) < 1e-9:
                prof = {k: g[k][:] for k in g if k != "parameters"}
    if prof is None:
        print(f"no profile at the snapshot time t={t2} -> FAIL"); sys.exit(1)
    x = xm[0, :]; dy = ym[1, 0] - ym[0, 0]
    _, h = depth_integral(xm, ym, f, 0.0)
    _, q = depth_integral(xm, ym, u * f, 0.0)
    Ip = integrate_to_height(ym, compute_px(x, p), h, 0.0)
    jb = liquid_top_index(h - PX_GAP * dy, dy, p.shape[0])
    plh = extrapolate_top(p, jb, h, dy)
    Ipl = trapz_to(p, jb, h, 1.5 * p[0] - 0.5 * p[1], plh, dy)
    ref = {"h": h, "q": q, "int_px_dy": Ip, "int_pl_dy": Ipl, "p_liq_interface": plh}
    print(f"profile vs 2D snapshot at t={t2}:")
    for k, v in ref.items():
        d = np.max(np.abs(prof[k] - v)) / max(np.max(np.abs(v)), 1e-300)
        good = d < a.field_tol
        ok &= good
        print(f"  {k:<16} max rel diff {d:.2e} {'OK' if good else 'FAIL'}")
    Ipl_liq = first_derivative_x(x, prof["int_pl_dy"]) - prof["p_liq_interface"] * first_derivative_x(x, prof["h"])
    print(f"  (I_p,liq from profile: std {np.std(Ipl_liq):.4g}; method a std {np.std(Ip):.4g})")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
