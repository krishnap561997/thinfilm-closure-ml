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
   i.e. whether the oracle source closes the ROM as discretised. --smooth 0,1 applies a
   Gaussian filter of each width (h_N) to the ROM-grid fields: the one-step DNS time
   derivative carries grid-scale interface jitter in strongly capillary cases (We ~ 0.1).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
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


def snapshot_arrays(x, P, cur, prv, j, dt, a, mD, mR):
    """Masked residual fields for snapshot j; ROM-grid fields per smoothing width."""
    N, L, dxR, We = a.N, a.L, a.L / a.N, P["Re"] * P["Ca"]
    dth = (cur["h"][:, j] - prv["h"][:, j]) / dt
    dtq = (cur["q"][:, j] - prv["q"][:, j]) / dt
    rc, lc, Sc, qxc = dns_terms(x, cur, j, P)
    rp, lp, Sp, qxp = dns_terms(x, prv, j, P)
    Rraw = 0.5 * (rhs_sum(rc) + rhs_sum(rp))
    Rlav = 0.5 * (rhs_sum(lc) + rhs_sum(lp))
    S = 0.5 * (Sc + Sp)
    D = {"dth": dth[mD], "mass": (dth + 0.5 * (qxc + qxp))[mD], "dtq": dtq[mD],
         "R_raw": (dtq - Rraw)[mD], "raw_minus_lav": (Rraw - Rlav)[mD], "S": S[mD]}
    for k in ("Mx", "Px", "Tw", "Dx", "G"): D[k] = 0.5 * (rc[k] + rp[k])[mD]

    rhs, caps, rus, caplav = [], [], [], []
    for s in (cur, prv):
        hR, qR = restrict(x, s["h"][:, j], N, L), restrict(x, s["q"][:, j], N, L)
        hxR, hxxxR = restrict(x, s["h_x"][:, j], N, L), restrict(x, s["h_xxx"][:, j], N, L)
        o = rom_rhs_q(with_ghosts(hR), with_ghosts(qR), with_ghosts(np.sqrt(hR / We) * hxR), dxR, P)
        rhs.append(o["total"]); caps.append(o["cap"]); rus.append(o["rusanov"]); caplav.append(hR * hxxxR / We)
    rhs = 0.5 * (rhs[0] + rhs[1])
    dtqR, SR = restrict(x, dtq, N, L), restrict(x, S, N, L)
    D["rom_rus"] = (0.5 * (rus[0] + rus[1]))[mR]
    D["rom_cap"] = (0.5 * (caps[0] + caps[1]))[mR]
    D["rom_caplav"] = (0.5 * (caplav[0] + caplav[1]))[mR]
    D["rom_dtq_raw"] = dtqR[mR]
    for w in a.smooth:
        sm = (lambda v: gaussian_filter1d(v, w / dxR)) if w > 0 else (lambda v: v)
        E0 = sm(dtqR) - sm(rhs)
        D[f"rom_dtq_w{w:g}"], D[f"rom_E0_w{w:g}"] = sm(dtqR)[mR], E0[mR]
        D[f"rom_S_w{w:g}"], D[f"rom_E1_w{w:g}"] = sm(SR)[mR], (E0 - sm(SR))[mR]
    return D


def metrics(D, smooth):
    """Scalar diagnostics from (possibly pooled) residual fields."""
    m = {"rms_dth": rms(D["dth"]), "mass_rel": rms(D["mass"]) / rms(D["dth"]),
         "rms_dtq": rms(D["dtq"]), "dns_res_rel": rms(D["R_raw"]) / rms(D["dtq"]),
         "S_rel": rms(D["S"]) / rms(D["dtq"]), "delta_consistency": rms(D["raw_minus_lav"])}
    for k in ("Mx", "Px", "Tw", "Dx", "G"): m[f"rms_{k}"] = rms(D[k])
    m["rom_rusanov_rel"] = rms(D["rom_rus"]) / rms(D["rom_dtq_raw"])
    m["rom_cap_mismatch_rel"] = rms(D["rom_cap"] - D["rom_caplav"]) / rms(D["rom_cap"])
    for w in smooth:
        g = lambda k: D[f"rom_{k}_w{w:g}"]
        d = rms(g("dtq"))
        m[f"rom_E0_rel_w{w:g}"] = rms(g("E0")) / d
        m[f"rom_S_rel_w{w:g}"] = rms(g("S")) / d
        m[f"rom_E1_rel_w{w:g}"] = rms(g("E1")) / d
        m[f"rom_E1_over_E0_w{w:g}"] = rms(g("E1")) / rms(g("E0"))
        m[f"rom_corr_w{w:g}"] = float(np.corrcoef(g("E0"), g("S"))[0, 1])
    return m


def select_columns(t, a):
    if a.times:
        t0, t1, step = (float(v) for v in a.times.split(":"))
        want = np.arange(t0, t1 + 0.5 * step, step)
        cols = [int(np.argmin(abs(t - w))) for w in want if np.min(abs(t - w)) < 1e-6]
        return cols, [w for w in want if np.min(abs(t - w)) >= 1e-6]
    return list(range(a.every - 1, len(t), a.every)), []


def write_csv(path, rows, append=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    new = not (append and path.exists())
    with path.open("w" if not append else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new: w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("closure_file")
    ap.add_argument("--every", type=int, default=10, help="use every n-th snapshot (ignored with --times)")
    ap.add_argument("--times", help="start:stop:step snapshot times, e.g. 1000:2000:100")
    ap.add_argument("--L", type=float, default=800.0)
    ap.add_argument("--N", type=int, default=4000, help="ROM cells")
    ap.add_argument("--x-min", type=float, default=20.0, help="exclude inlet region")
    ap.add_argument("--x-max-frac", type=float, default=0.95, help="exclude outlet region")
    ap.add_argument("--smooth", type=lambda v: [float(w) for w in v.split(",")], default=[0.0],
                    help="comma-separated Gaussian widths (h_N) on the ROM grid, e.g. 0,1")
    ap.add_argument("--csv", help="write one row per snapshot time to this file")
    ap.add_argument("--summary-csv", help="append one pooled row for this case to this file")
    a = ap.parse_args()

    case = Path(a.closure_file).stem.replace("thinfilm_closure_data_", "")
    with h5py.File(a.closure_file, "r") as f: t_all = f["t"][:]
    cols, missing = select_columns(t_all, a)
    if not cols: raise SystemExit(f"{case}: no snapshots selected")
    x, P, cur, prv, t, dt = load(a.closure_file, cols)
    print(f"{a.closure_file}\nRe={P['Re']:g} Ca={P['Ca']:g} beta={np.degrees(P['angle']):.1f} deg; "
          f"{len(cols)} snapshots t={t[0]:.0f}..{t[-1]:.0f}, solver dt {dt.min():.2e}..{dt.max():.2e}"
          + (f"; missing t=" + " ".join(f"{w:g}" for w in missing) if missing else ""))
    mD = (x > a.x_min) & (x < a.x_max_frac * a.L)
    XR = (np.arange(a.N) + 0.5) * a.L / a.N
    mR = (XR > a.x_min) & (XR < a.x_max_frac * a.L)

    base = {"case": case, "Re": P["Re"], "Ca": P["Ca"], "beta_deg": round(float(np.degrees(P["angle"])), 3)}
    rows, pooled = [], {}
    for j in range(len(cols)):
        D = snapshot_arrays(x, P, cur, prv, j, dt[j], a, mD, mR)
        rows.append({**base, "t": float(t[j]), "dt_solver": float(dt[j]), **metrics(D, a.smooth)})
        for k, v in D.items(): pooled.setdefault(k, []).append(v)
    M = metrics({k: np.concatenate(v) for k, v in pooled.items()}, a.smooth)

    print(f"\nA) DNS grid (pooled rms over x in mask and snapshots)")
    print(f"  mass residual / d_t h            = {M['mass_rel']:.3f}")
    print("  momentum terms (RHS convention): " + "  ".join(f"{k} {M['rms_'+k]:.3e}" for k in ("Mx", "Px", "Tw", "Dx", "G")))
    print(f"  rms(d_t q) = {M['rms_dtq']:.3e}; residual / d_t q = {M['dns_res_rel']:.3f}; "
          f"S_ML / d_t q = {M['S_rel']:.3f}; raw-(Lav+delta) = {M['delta_consistency']:.1e}")
    print(f"\nB) ROM grid, N={a.N}: Rusanov / d_t q = {M['rom_rusanov_rel']:.3f}; "
          f"Noble-Vila cap mismatch = {M['rom_cap_mismatch_rel']:.3f}")
    for w in a.smooth:
        print(f"  smoothing {w:g} h_N: E0/d_t q {M[f'rom_E0_rel_w{w:g}']:.3f}  S/d_t q {M[f'rom_S_rel_w{w:g}']:.3f}  "
              f"E1/d_t q {M[f'rom_E1_rel_w{w:g}']:.3f}  E1/E0 {M[f'rom_E1_over_E0_w{w:g}']:.3f}  "
              f"corr(E0,S) {M[f'rom_corr_w{w:g}']:.3f}")
    print("\n  per time:  t   mass   dns_res   " + "   ".join(f"E1/E0(w={w:g})" for w in a.smooth))
    for r in rows:
        print(f"  {r['t']:6.0f}  {r['mass_rel']:.3f}  {r['dns_res_rel']:.3f}     "
              + "      ".join(f"{r[f'rom_E1_over_E0_w{w:g}']:.3f}" for w in a.smooth))

    if a.csv: write_csv(a.csv, rows)
    if a.summary_csv:
        write_csv(a.summary_csv, [{**base, "n_times": len(cols), "t_first": float(t[0]), "t_last": float(t[-1]),
                                   "missing_times": " ".join(f"{w:g}" for w in missing), **M}], append=True)


if __name__ == "__main__":
    main()
