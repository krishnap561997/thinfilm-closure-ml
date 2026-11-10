#!/usr/bin/env python3
"""Depth-integrated budget from dense in-situ profiles (filmflow_restart_profiles runs).

Tests whether the momentum budget, which fails for We <= 0.1 with the one-step ∂_t of the
2D snapshots, closes when ∂_t is replaced by the integral form over a window 2Δ:

    q(t+Δ) - q(t-Δ) = ∫_{t-Δ}^{t+Δ} RHS dt,   RHS = -∂_x M - I_p - τ_w/Re + q_xx/Re + Λh/Re
    h(t+Δ) - h(t-Δ) = -∫ q_x dt

(trapezoid over the profiles; ΔD_x neglected, as in dpx_diagnostics.py). I_p is either the
pipeline integral (method a, int_px_dy) or the liquid-pressure Leibniz form (method b,
∂_x int_pl_dy - p_liq_interface h_x). Residual = ||LHS - ∫RHS|| / ||LHS|| over
x0 < x < 0.95 L, raw and after a Gaussian filter of width σ (h_N); median over windows.
The "1 step" row uses (q - q_m)/dt against the mean RHS of the two states, i.e. what the
2D snapshots allow.

    python profile_budget.py RUN_DIR/profiles [--deltas 0.01,0.02,0.05,0.1,0.25,0.5,1] [--csv out.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_closure_terms import first_derivative_x, gauss_deriv, sg_derivatives  # noqa: E402


def scalar(ds):
    return float(np.asarray(ds[()]).squeeze())


def rhs_parts(x, g, sfx, Re, Lam):
    h, q, M, Ipa, dudy, Ipl, plh = (g[k + sfx][:] for k in
                                    ("h", "q", "int_u2_dy", "int_px_dy", "dudy_wall", "int_pl_dy", "p_liq_interface"))
    qxx = sg_derivatives(x, q, 2, min_poly=2)[1]
    common = -first_derivative_x(x, M) - dudy / Re + qxx / Re + Lam * h / Re
    Ipb = first_derivative_x(x, Ipl) - plh * first_derivative_x(x, h)
    return h, q, first_derivative_x(x, q), common - Ipa, common - Ipb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument("--deltas", default="0.01,0.02,0.05,0.1,0.25,0.5,1")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--x0", type=float, default=100.0)
    ap.add_argument("--centre-step", type=float, default=0.5, help="spacing of window centres")
    ap.add_argument("--lambda", dest="Lam", type=float, default=3.0)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    files = sorted(a.profile_dir.glob("profile_*.h5"))
    if len(files) < 3:
        raise SystemExit("need at least 3 profiles")
    with h5py.File(files[0], "r") as g:
        x = g["x"][:]; Re = scalar(g["parameters/Re"]); Ca = scalar(g["parameters/Ca"])
    dx = x[1] - x[0]; L = x[-1] + 0.5 * dx
    mask = (x > a.x0) & (x < 0.95 * L)
    nt, nx = len(files), x.size
    T = np.empty(nt); H = np.empty((nt, nx)); Q = np.empty((nt, nx)); QX = np.empty((nt, nx))
    RA = np.empty((nt, nx)); RB = np.empty((nt, nx))
    one = {k: [] for k in ("mass_raw", "mass_s", "a_raw", "b_raw", "a_s", "b_s")}
    G = lambda v: gauss_deriv(v, a.sigma, dx)  # noqa: E731
    nrm = lambda v: np.linalg.norm(v[mask])  # noqa: E731
    for i, pth in enumerate(files):
        with h5py.File(pth, "r") as g:
            T[i] = scalar(g["parameters/time"]); dt = T[i] - scalar(g["parameters/time_minus"])
            H[i], Q[i], QX[i], RA[i], RB[i] = rhs_parts(x, g, "", Re, a.Lam)
            hm, qm, qxm, ram, rbm = rhs_parts(x, g, "_m", Re, a.Lam)
        if i % 10 == 0:  # one-step check on a subset
            dth, dtq = (H[i] - hm) / dt, (Q[i] - qm) / dt
            mres = dth + 0.5 * (QX[i] + qxm)
            one["mass_raw"].append(nrm(mres) / nrm(dth)); one["mass_s"].append(nrm(G(mres)) / nrm(G(dth)))
            for m, r, rm in (("a", RA[i], ram), ("b", RB[i], rbm)):
                res = dtq - 0.5 * (r + rm)
                one[f"{m}_raw"].append(nrm(res) / nrm(dtq)); one[f"{m}_s"].append(nrm(G(res)) / nrm(G(dtq)))
    order = np.argsort(T); T, H, Q, QX, RA, RB = T[order], H[order], Q[order], QX[order], RA[order], RB[order]
    tp = np.median(np.diff(T))
    print(f"{a.profile_dir}: Re={Re:g} Ca={Ca:g} We={Re*Ca:g}; {nt} profiles t={T[0]:.3f}..{T[-1]:.3f} (Δt_prof {tp:.4g})")

    rows = [("1 step", *[np.median(one[k]) for k in ("mass_raw", "mass_s", "a_raw", "b_raw", "a_s", "b_s")], len(one["a_raw"]))]
    for D in [float(v) for v in a.deltas.split(",")]:
        n = int(round(D / tp))
        if n < 1 or 2 * n >= nt:
            continue
        cs = np.arange(n, nt - n, max(1, int(round(a.centre_step / tp))))
        acc = {k: [] for k in ("mass_raw", "mass_s", "a_raw", "b_raw", "a_s", "b_s")}
        for c in cs:
            sl = slice(c - n, c + n + 1); tt = T[sl]
            dq = Q[c + n] - Q[c - n]; dh = H[c + n] - H[c - n]
            mres = dh + np.trapezoid(QX[sl], tt, axis=0)
            acc["mass_raw"].append(nrm(mres) / nrm(dh)); acc["mass_s"].append(nrm(G(mres)) / nrm(G(dh)))
            for m, R in (("a", RA), ("b", RB)):
                res = dq - np.trapezoid(R[sl], tt, axis=0)
                acc[f"{m}_raw"].append(nrm(res) / nrm(dq)); acc[f"{m}_s"].append(nrm(G(res)) / nrm(G(dq)))
        rows.append((f"Δ={D:g}", *[np.median(acc[k]) for k in ("mass_raw", "mass_s", "a_raw", "b_raw", "a_s", "b_s")], len(cs)))

    hdr = ("window", "mass raw", f"mass s{a.sigma:g}", "mom a raw", "mom b raw", f"mom a s{a.sigma:g}", f"mom b s{a.sigma:g}", "n")
    print("".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print(f"{r[0]:>12}" + "".join(f"{v:12.4g}" for v in r[1:-1]) + f"{r[-1]:12d}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)


if __name__ == "__main__":
    main()
