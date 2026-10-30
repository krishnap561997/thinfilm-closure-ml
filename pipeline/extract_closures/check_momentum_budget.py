#!/usr/bin/env python3
"""Momentum-budget closure check for an extracted closure file (pre-oracle sanity check).

A) DNS grid: does the depth-integrated budget close with the extracted terms?
     d_t h + q_x = 0
     d_t q = -M_x + P_x - T_w + D_x + Lambda h/Re
   with d_t from the (t, t-dt) pair in each snapshot and the RHS averaged over both states.
   Evaluated twice: from the raw integrals (int_u2_dy, int_px_dy, dudy_wall) and from
   C_Lav + delta_C, which checks that the stored deltas are consistent with the raw terms.

B) ROM grid: restrict the DNS state to the lavalle-cpp grid, evaluate a Python replica of
   the ROM's discrete q-RHS (MUSCL/minmod + Rusanov, Lavalle pressure law, Noble-Vila
   capillarity with r = sqrt(h/We) h_x, 2 q_xx/Re), and test whether
     d_t q - RHS_ROM  ~=  S_ML = -dMx + dPx - dTw + dDx
   i.e. whether the oracle source closes the ROM as discretised. --smooth W applies a
   Gaussian filter of width W (h_N) to the ROM-grid fields first: the one-step DNS time
   derivative carries grid-scale interface jitter in strongly capillary cases (We ~ 0.1).
"""
from __future__ import annotations
import argparse
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from extract_closure_terms import best_savgol_filter

FIELDS = ("h", "q", "h_x", "q_x", "q_xx", "h_xxx", "int_u2_dy", "int_px_dy", "dudy_wall",
          "delta_Mx_ML", "delta_Tw_ML", "delta_Px_ML", "delta_Dx_ML")


def load(path, cols):
    with h5py.File(path, "r") as f:
        x = f["x"][:]
        P = {k: float(f["parameters"][k][()]) for k in ("Re", "Ca", "angle")}
        P["Lambda"] = float(f["parameters/Lambda"][()]) if "Lambda" in f["parameters"] else 3.0
        cur = {k: f[k][:, cols] for k in FIELDS}
        prv = {k: f[k + "_m"][:, cols] for k in FIELDS}
        t, dt = f["t"][cols], f["dt"][cols]
    return x, P, cur, prv, t, dt


def sg_dx(x, v):
    m, p, _, _ = best_savgol_filter(x, v, range(11, 42, 2), range(2, 4), metric="rmse", power=4)
    return m, p, savgol_filter(v, m, p, deriv=1, delta=np.diff(x)[0])


def dns_terms(x, s, j, P):
    """Momentum-budget terms for column j of state dict s (DNS grid)."""
    Re, Lam, b = P["Re"], P["Lambda"], P["angle"]
    We, Fr2 = Re * P["Ca"], Re * np.sin(b) / Lam
    h, q, hx, qx, qxx, hxxx = (s[k][:, j] for k in ("h", "q", "h_x", "q_x", "q_xx", "h_xxx"))
    m, p, Mx = sg_dx(x, s["int_u2_dy"][:, j])
    Mx_lav = savgol_filter(q**2 / h + Lam**2 * h**5 / 45.0, m, p, deriv=1, delta=np.diff(x)[0])
    raw = {"Mx": Mx, "Px": -s["int_px_dy"][:, j], "Tw": s["dudy_wall"][:, j] / Re,
           "Dx": qxx / Re + s["delta_Dx_ML"][:, j], "G": Lam * h / Re}
    lav = {"Mx": Mx_lav + s["delta_Mx_ML"][:, j],
           "Px": h * hxxx / We - np.cos(b) / Fr2 * h * hx + qxx / Re + s["delta_Px_ML"][:, j],
           "Tw": 3 * q / (Re * h**2) - Lam**2 * h**4 * hx / 15.0 + s["delta_Tw_ML"][:, j],
           "Dx": raw["Dx"], "G": raw["G"]}
    S = -s["delta_Mx_ML"][:, j] + s["delta_Px_ML"][:, j] - s["delta_Tw_ML"][:, j] + s["delta_Dx_ML"][:, j]
    return raw, lav, S, qx


def rhs_sum(T): return -T["Mx"] + T["Px"] - T["Tw"] + T["Dx"] + T["G"]


# ---------------------------------------------------------------- ROM replica
def restrict(x, f, N, L):
    """Conservative cell average of a DNS cell-centred field onto N uniform ROM cells."""
    d = x[1] - x[0]
    xf = np.concatenate(([x[0] - d / 2], x + d / 2))
    F = np.concatenate(([0.0], np.cumsum(f * d)))
    Xf = np.linspace(0.0, L, N + 1)
    return np.diff(np.interp(Xf, xf, F)) / (L / N)


def minmod(a, b): return np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0.0)


def rom_rhs_q(h, q, r, dx, P):
    """Interior q-RHS of lavalle-cpp evaluate_rhs (ng=1, phi_sf=1). Arrays include ghosts."""
    Re, Lam, b = P["Re"], P["Lambda"], P["angle"]
    We, F2 = Re * P["Ca"], Re * np.sin(b) / Lam
    Pp = lambda H: np.cos(b) * H**2 / (2 * F2) + 2 * Lam**2 * H**5 / 225.0
    dP = lambda H: np.cos(b) * H / F2 + (2.0 / 45.0) * Lam**2 * H**4
    sh, sq = np.zeros_like(h), np.zeros_like(q)
    sh[1:-1] = minmod(h[1:-1] - h[:-2], h[2:] - h[1:-1])
    sq[1:-1] = minmod(q[1:-1] - q[:-2], q[2:] - q[1:-1])
    hL, qL = np.maximum(h[:-1] + sh[:-1] / 2, 1e-12), q[:-1] + sq[:-1] / 2
    hR, qR = np.maximum(h[1:] - sh[1:] / 2, 1e-12), q[1:] - sq[1:] / 2

    def lam_max(H, Q):
        a, bb = -Q**2 / H**2 + dP(H), 2 * Q / H
        root = np.sqrt(np.maximum(bb**2 + 4 * a, 0.0))
        return np.maximum(abs(0.5 * (bb - root)), abs(0.5 * (bb + root)))
    c = np.maximum(lam_max(hL, qL), lam_max(hR, qR))
    FL, FR = qL**2 / hL + Pp(hL), qR**2 / hR + Pp(hR)
    flux_c = 0.5 * (FL + FR)
    flux = flux_c - 0.5 * c * (qR - qL)
    out = {"flux": -(flux[1:] - flux[:-1]) / dx,
           "rusanov": -(flux[1:] - flux_c[1:] - flux[:-1] + flux_c[:-1]) / dx,
           "src": ((Lam * h - 3 * q / h**2) / Re)[1:-1],
           "visc": (2 / Re) * (q[2:] - 2 * q[1:-1] + q[:-2]) / dx**2}
    w, phi = r / h, np.sqrt(1 / We) * (0.5 * (h[:-1] + h[1:]))**1.5
    pw = phi * np.diff(w) / dx
    out["cap"] = (pw[1:] - pw[:-1]) / dx
    out["total"] = out["flux"] + out["src"] + out["visc"] + out["cap"]
    return out


def with_ghosts(f): return np.concatenate(([f[0]], f, [f[-1]]))


def rms(a): return float(np.sqrt(np.nanmean(np.square(a))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("closure_file")
    ap.add_argument("--every", type=int, default=10, help="use every n-th snapshot")
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--N", type=int, default=4000, help="ROM cells")
    ap.add_argument("--x-min", type=float, default=20.0, help="exclude inlet region")
    ap.add_argument("--x-max-frac", type=float, default=0.95, help="exclude outlet region")
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian width (h_N) on ROM grid")
    ap.add_argument("--save", help="optional .npz with one-snapshot profiles for plotting")
    a = ap.parse_args()

    with h5py.File(a.closure_file, "r") as f: nt = f["t"].shape[0]
    cols = list(range(a.every - 1, nt, a.every))
    x, P, cur, prv, t, dt = load(a.closure_file, cols)
    print(f"{a.closure_file}\nRe={P['Re']:g} Ca={P['Ca']:g} beta={np.degrees(P['angle']):.1f} deg; "
          f"{len(cols)} snapshots t={t[0]:.0f}..{t[-1]:.0f}, solver dt {dt.min():.2e}..{dt.max():.2e}")
    mD = (x > a.x_min) & (x < a.x_max_frac * a.L)
    XR = (np.arange(a.N) + 0.5) * a.L / a.N
    mR = (XR > a.x_min) & (XR < a.x_max_frac * a.L)
    dxR = a.L / a.N
    We = P["Re"] * P["Ca"]

    acc = {k: [] for k in ("dth", "mass", "dtq", "R_raw", "R_lav", "raw_minus_lav", "S",
                           "Mx", "Px", "Tw", "Dx", "G",
                           "R_dtq", "R_E0", "R_E1", "R_S", "R_rus", "R_cap", "R_caplav")}
    for j in range(len(cols)):
        dth = (cur["h"][:, j] - prv["h"][:, j]) / dt[j]
        dtq = (cur["q"][:, j] - prv["q"][:, j]) / dt[j]
        rc, lc, Sc, qxc = dns_terms(x, cur, j, P)
        rp, lp, Sp, qxp = dns_terms(x, prv, j, P)
        Rraw = 0.5 * (rhs_sum(rc) + rhs_sum(rp))
        Rlav = 0.5 * (rhs_sum(lc) + rhs_sum(lp))
        S = 0.5 * (Sc + Sp)
        acc["dth"].append(dth[mD]); acc["mass"].append((dth + 0.5 * (qxc + qxp))[mD])
        acc["dtq"].append(dtq[mD]); acc["R_raw"].append((dtq - Rraw)[mD])
        acc["R_lav"].append((dtq - Rlav)[mD]); acc["raw_minus_lav"].append((Rraw - Rlav)[mD])
        acc["S"].append(S[mD])
        for k in ("Mx", "Px", "Tw", "Dx", "G"): acc[k].append(0.5 * (rc[k] + rp[k])[mD])

        # ROM grid
        E0, caps, rus, caplav = [], [], [], []
        for s in (cur, prv):
            hR = restrict(x, s["h"][:, j], a.N, a.L)
            qR = restrict(x, s["q"][:, j], a.N, a.L)
            hxR = restrict(x, s["h_x"][:, j], a.N, a.L)
            hxxxR = restrict(x, s["h_xxx"][:, j], a.N, a.L)
            rR = np.sqrt(hR / We) * hxR
            o = rom_rhs_q(with_ghosts(hR), with_ghosts(qR), with_ghosts(rR), dxR, P)
            E0.append(o["total"]); caps.append(o["cap"]); rus.append(o["rusanov"])
            caplav.append(hR * hxxxR / We)
        sm = (lambda v: gaussian_filter1d(v, a.smooth / dxR)) if a.smooth > 0 else (lambda v: v)
        dtqR = sm(restrict(x, dtq, a.N, a.L))
        SR = sm(restrict(x, S, a.N, a.L))
        E0 = dtqR - sm(0.5 * (E0[0] + E0[1]))
        acc["R_dtq"].append(dtqR[mR]); acc["R_E0"].append(E0[mR]); acc["R_E1"].append((E0 - SR)[mR])
        acc["R_S"].append(SR[mR]); acc["R_rus"].append((0.5 * (rus[0] + rus[1]))[mR])
        acc["R_cap"].append((0.5 * (caps[0] + caps[1]))[mR])
        acc["R_caplav"].append((0.5 * (caplav[0] + caplav[1]))[mR])

        if a.save and j == len(cols) // 2:
            np.savez(a.save, x=x[mD], dtq=dtq[mD], R_raw=(dtq - Rraw)[mD], S=S[mD], h=cur["h"][mD, j],
                     XR=XR[mR], dtqR=dtqR[mR], E0=E0[mR], E1=(E0 - SR)[mR], SR=SR[mR], t=t[j])

    A = {k: np.concatenate(v) for k, v in acc.items()}
    print("\nA) DNS grid (rms over x in mask and snapshots)")
    print(f"  mass:  rms(d_t h) = {rms(A['dth']):.3e}   rms(d_t h + q_x) = {rms(A['mass']):.3e}"
          f"   ratio {rms(A['mass'])/rms(A['dth']):.3f}")
    print("  momentum terms (RHS sign convention):  " +
          "  ".join(f"{k} {rms(A[k]):.3e}" for k in ("Mx", "Px", "Tw", "Dx", "G")))
    print(f"  rms(d_t q) = {rms(A['dtq']):.3e}")
    print(f"  residual raw   rms = {rms(A['R_raw']):.3e}   ratio to d_t q {rms(A['R_raw'])/rms(A['dtq']):.3f}")
    print(f"  residual Lav+d rms = {rms(A['R_lav']):.3e}   ratio to d_t q {rms(A['R_lav'])/rms(A['dtq']):.3f}")
    print(f"  raw - (Lav+delta)  = {rms(A['raw_minus_lav']):.3e}   (delta consistency)")
    print(f"  rms(S_ML) = {rms(A['S']):.3e}")
    print(f"\nB) ROM grid, N={a.N} (dx={dxR:g}), smoothing width {a.smooth:g} h_N")
    print(f"  rms(d_t q)                       = {rms(A['R_dtq']):.3e}")
    print(f"  E0 = d_t q - RHS_ROM       rms   = {rms(A['R_E0']):.3e}")
    print(f"  S_ML (restricted)          rms   = {rms(A['R_S']):.3e}")
    print(f"  E1 = E0 - S_ML             rms   = {rms(A['R_E1']):.3e}   E1/E0 = {rms(A['R_E1'])/rms(A['R_E0']):.3f}")
    c = np.corrcoef(A["R_E0"], A["R_S"])[0, 1]
    print(f"  corr(E0, S_ML) = {c:.3f}")
    print(f"  Rusanov dissipation part rms     = {rms(A['R_rus']):.3e}")
    print(f"  Noble-Vila cap vs h h_xxx/We     : rms diff {rms(A['R_cap']-A['R_caplav']):.3e}"
          f"  (rms cap {rms(A['R_cap']):.3e})")


if __name__ == "__main__":
    main()
