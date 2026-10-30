#!/usr/bin/env python3
"""Convert in-situ 1D column profiles (dns/include/column_profiles.h) into a closure file.

Each profiles/profile_<t>.h5 holds h, q, int_u2_dy, int_px_dy, dudy_wall and
u_interface_raw at t and t-dt (suffix _m). This applies the same steps as
extract_closure_terms.process_state (Savitzky-Golay derivatives, interface-velocity
repair, closure_set) and writes the same layout: field datasets of shape (nx, nt),
t, tm, dt and parameters/.

    python profiles_to_closures.py <run_dir>/profiles out.h5 [--t-start 1800 --t-end 1900 --every 1]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import savgol_filter
from extract_closure_terms import best_savgol_filter, closure_set, compute_ui, sg_derivatives

RAW = ("h", "q", "int_u2_dy", "int_px_dy", "dudy_wall", "u_interface_raw")


def scalar(ds): return float(np.asarray(ds[()]).squeeze())


def process_profile(x, P, suffix, Re, Ca, beta, Lambda):
    h, q, M, Ip, dudy, ui_raw = (P[k + suffix][:] for k in RAW)
    hx, hxx, hxxx = sg_derivatives(x, h, 3, min_poly=3)
    qx, qxx = sg_derivatives(x, q, 2, min_poly=2)
    # compute_ui interpolates u(x, y) to y = h; a y-independent field equal to the raw
    # interface velocity makes that step return ui_raw exactly, then the repair applies.
    ui = compute_ui(np.vstack([ui_raw, ui_raw]), np.array([0.0, 1e30]), h, x,
                    k=5, prominence=0.05, distance=10, Z_h_threshold=3.0, Z_u_region_threshold=2.0)
    mui, pui, _, _ = best_savgol_filter(x, ui, range(11, 42, 2), range(2, 4), metric="rmse", power=4)
    uix = savgol_filter(ui, mui, pui, deriv=1, delta=np.diff(x)[0])
    out = {"h": h, "q": q, "h_x": hx, "q_x": qx, "h_xx": hxx, "q_xx": qxx, "h_xxx": hxxx,
           "int_u2_dy": M, "u_interface": ui, "u_interface_x": uix, "int_px_dy": Ip, "dudy_wall": dudy}
    out.update(closure_set(x, h, q, hx, hxx, hxxx, qxx, M, ui, uix, dudy, Ip, Re, Ca, beta, Lambda))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("output_file", type=Path)
    ap.add_argument("--t-start", type=float, default=-np.inf)
    ap.add_argument("--t-end", type=float, default=np.inf)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th profile")
    ap.add_argument("--lambda", dest="Lambda", type=float, default=3.0)
    a = ap.parse_args()

    files = []
    for p in sorted(a.profile_dir.glob("profile_*.h5")):
        with h5py.File(p, "r") as f: t = scalar(f["parameters/time"])
        if a.t_start - 1e-6 <= t <= a.t_end + 1e-6: files.append((t, p))  # restart times carry ~1e-9 offsets
    files.sort()
    files = files[::a.every]
    if not files: raise SystemExit("no profiles in range")

    with h5py.File(files[0][1], "r") as f:
        x = f["x"][:]
        params = {k: scalar(v) for k, v in f["parameters"].items() if k not in ("time", "time_minus", "dt")}
    Re, Ca, beta, Lam = params["Re"], params["Ca"], params["angle"], a.Lambda
    nx, nt = x.size, len(files)
    print(f"{nt} profiles t={files[0][0]:.4f}..{files[-1][0]:.4f}, nx={nx}")

    a.output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.output_file, "w") as out:
        out.attrs["description"] = "DNS thin-film closure variables from in-situ column profiles"
        out.attrs["source"] = str(a.profile_dir)
        out.create_dataset("x", data=x)
        pg = out.create_group("parameters")
        for k, v in params.items(): pg.create_dataset(k, data=v)
        pg.create_dataset("Lambda", data=Lam)
        pg.create_dataset("We", data=Re * Ca)
        pg.create_dataset("Fr2", data=Re * np.sin(beta) / Lam)
        ds, tt = {}, {k: out.create_dataset(k, (nt,), "f8") for k in ("t", "tm", "dt")}
        for it, (t, p) in enumerate(files):
            with h5py.File(p, "r") as f:
                cur = process_profile(x, f, "", Re, Ca, beta, Lam)
                prv = process_profile(x, f, "_m", Re, Ca, beta, Lam)
                tm = scalar(f["parameters/time_minus"])
            if not ds:
                for n in cur:
                    for s in ("", "_m"):
                        ds[n + s] = out.create_dataset(n + s, (nx, nt), "f8", chunks=(nx, 1),
                                                       compression="gzip", compression_opts=4)
            for n, v in cur.items(): ds[n][:, it] = v
            for n, v in prv.items(): ds[n + "_m"][:, it] = v
            tt["t"][it], tt["tm"][it], tt["dt"][it] = t, tm, t - tm
            if (it + 1) % 100 == 0 or it + 1 == nt: print(f"  [{it+1}/{nt}] t={t:.4f}")
    print(f"Wrote {a.output_file}")


if __name__ == "__main__":
    main()
