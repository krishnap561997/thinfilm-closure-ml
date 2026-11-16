"""B.7 and B.8: linear closure targets from the OS mode, and linear closure coefficients.

B.7.  Along the OS temporal mode (real k, eta_hat = 1), the depth-integrated closure terms of
docs/equations.md sec. 2 are linear functionals of phi:
    q'      = phi(1) + U_s                       (= c, checked)
    M_x'    = ik [ int_0^1 2 U phi' dz + U_s^2 ]
    T_w'    = phi''(0) / Re
    P_x'    = -ik int_0^1 p_hat dz,  p_hat = -(U - c) phi' + U' phi + (phi''' - k^2 phi') / (ik Re)
    D_x'    = -(k^2/Re) phi(1)
The linearised Lavalle closures (sec. 3) at the same (eta = 1, xi = c):
    M_Lav' = 2ikc;  T_Lav' = (3/Re)(c - 2) - 0.6 ik;  P_Lav' = -ik^3/We - 3ik cot b/Re - k^2 c/Re;
    D_Lav' = -k^2 c/Re
Deltas = OS - Lavalle, complex per unit eta: Re part in phase with h, Im part in quadrature.
Check: OS terms close the linear momentum balance  -ik c^2 + M' = P' - T' + D' + 3/Re, up to the
surface viscous term (1/Re) u_z|_h = -(k^2/Re) phi(1) that sec. 2 drops (it assumes u_z|_h = 0; the
passive-gas condition is u_z + w_x = 0).  The script reports |resid + (k^2/Re) phi(1)|, which must be ~0.

B.8.  Fit the linear part of S = dMx + dTw (ROM adds -S), with the uniform-film condition built in,
    S_lin = a (xi - 3 eta) + b eta_x + c xi_x [+ d eta_xx],
so that the linearised augmented ROM matches the OS *spatial* branch for omega in (0.05, 0.95) omega_c
(20 points, uniform; tags abc, abcd), and over the forced / most amplified band
omega in (0.05 omega_c, min(2 omega_max, 0.95 omega_c)) (tags band_abc, band_abcd).  The long-wave limit is imposed exactly: D_aug = D_OS = 1.2 Re - cot b fixes c
given (a, b).  Residuals: (k_r - k_r,OS)/k_r,OS and (k_i - k_i,OS)/max|k_i,OS|.
Usage: python linear_targets.py [--cases ...] [--ncpu N]
"""
import argparse, csv, math, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import os_solver as osv

STORE = os.environ.get("THINFILM_STORE", ".")
OUT = os.path.join(STORE, "analysis/orr_sommerfeld")


def cc_weights(N):
    """Clenshaw-Curtis weights on x_j = cos(pi j/N), mapped to z in [0, 1]."""
    th = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1); v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1 / (N * N - 1)
        for kk in range(1, N // 2):
            v -= 2 * np.cos(2 * kk * th[1:-1]) / (4 * kk * kk - 1)
        v -= np.cos(N * th[1:-1]) / (N * N - 1)
    else:
        w[0] = w[N] = 1 / (N * N)
        for kk in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * kk * th[1:-1]) / (4 * kk * kk - 1)
    w[1:-1] = 2 * v / N
    return w / 2          # dz = dx/2


def os_closures(k, Re, Ca, cotb, N=48, c_guess=None):
    c, z, phi = osv.eigenfunction(k, Re, Ca, cotb, N, c_guess=c_guess)
    _, D, D2, D3 = osv._grid(N)
    w = cc_weights(N)
    U, Up, Us = 3 * z - 1.5 * z ** 2, 3 - 3 * z, osv.U_S
    p1, p2, p3 = D @ phi, D2 @ phi, D3 @ phi
    q = phi[0] + Us
    M = 1j * k * (w @ (2 * U * p1) + Us ** 2)
    T = p2[-1] / Re
    ph = -(U - c) * p1 + Up * phi + (p3 - k * k * p1) / (1j * k * Re)
    P = -1j * k * (w @ ph)
    Dx = -(k * k / Re) * phi[0]
    We = Re * Ca
    lav = dict(M=2j * k * c, T=(3 / Re) * (c - 2) - 0.6j * k,
               P=-1j * k ** 3 / We - 3j * k * cotb / Re - k * k * c / Re, D=-k * k * c / Re)
    resid = -1j * k * c * c + M - (P - T + Dx + 3 / Re) + (k * k / Re) * phi[0]
    return dict(c=c, q=q, M=M, T=T, P=P, Dx=Dx, dM=M - lav["M"], dT=T - lav["T"], dP=P - lav["P"],
                dD=Dx - lav["D"], resid=resid)


# ---------------------------------------------------------------- B.8
def k_aug(omega, Re, Ca, cotb, a, b, c, d, k_ref):
    We, G = Re * Ca, -0.6 + 3 * cotb / Re
    r = np.roots([1j / We, -d, 2 * omega / Re + 1j * G + 1j * b,
                  2j * omega + 1j * c * omega - 9 / Re - 3 * a, 3 * omega / Re + a * omega - 1j * omega ** 2])
    return r[np.argmin(np.abs(r - k_ref))]


def c_from_D(a, b, Re, cotb):
    return (3.6 - 3 * cotb / Re - b - (1.2 * Re - cotb) * (3 / Re + a)) / 3


def fit_case(row):
    case = row["case"]; Re, Ca, beta = float(row["Re"]), float(row["Ca"]), float(row["beta_deg"])
    cotb = 1 / math.tan(math.radians(beta)); t0 = time.time()
    out = dict(case=case, Re=Re, Ca=Ca, beta=beta, delta_lin=float(row["delta_lin"]))
    try:
        wc, wm = float(row["os_omega_c"]), float(row["os_omega_max"])
        om_full = np.linspace(0.05, 0.95, 20) * wc
        K_full, _ = osv.spatial_curve(om_full, Re, Ca, cotb)
        om_band = np.linspace(0.05 * wc, min(2 * wm, 0.95 * wc), 20)
        K_band, _ = osv.spatial_curve(om_band, Re, Ca, cotb)
        om, K = om_full, K_full
        kimax = np.max(np.abs(K.imag))

        def res(p, with_d):
            a, b = p[0], p[1]; d = p[2] if with_d else 0.0
            c = c_from_D(a, b, Re, cotb)
            ka = np.array([k_aug(o, Re, Ca, cotb, a, b, c, d, kr) for o, kr in zip(om, K)])
            return np.r_[(ka.real - K.real) / K.real, (ka.imag - K.imag) / kimax]
        r0 = res([0, 0], False)
        out.update(lav_err_kr=np.max(np.abs(r0[:20])), lav_err_ki=np.max(np.abs(r0[20:])))
        for tag, wd, p0, band in (("abc", False, [0.0, 0.0], False), ("abcd", True, [0.0, 0.0, 0.0], False),
                                  ("band_abc", False, [0.0, 0.0], True), ("band_abcd", True, [0.0, 0.0, 0.0], True)):
            om, K = (om_band, K_band) if band else (om_full, K_full)
            kimax = np.max(np.abs(K.imag))
            if tag == "band_abc":
                r0 = res([0, 0], False)
                out.update(band_lav_err_kr=np.max(np.abs(r0[:20])), band_lav_err_ki=np.max(np.abs(r0[20:])))
            sol = least_squares(res, p0, args=(wd,), x_scale="jac")
            a, b = sol.x[0], sol.x[1]; d = sol.x[2] if wd else 0.0
            r = res(sol.x, wd)
            out.update({f"{tag}_a": a, f"{tag}_b": b, f"{tag}_c": c_from_D(a, b, Re, cotb), f"{tag}_d": d,
                        f"{tag}_err_kr": np.max(np.abs(r[:20])), f"{tag}_err_ki": np.max(np.abs(r[20:])),
                        f"{tag}_rms": np.sqrt(np.mean(r ** 2))})
        # B.7 at the OS temporal k_max and at k of the spatial maximum
        for tag, k in (("kmax", float(row["os_k_max"])), ("kwmax", float(row["os_kr_at_omega_max"]))):
            L = os_closures(k, Re, Ca, cotb)
            out.update({f"b7_{tag}_k": k, f"b7_{tag}_resid": abs(L["resid"]), f"b7_{tag}_qcheck": abs(L["q"] - L["c"])})
            for n in ("dM", "dT", "dP", "dD"):
                out[f"b7_{tag}_{n}_re"] = L[n].real; out[f"b7_{tag}_{n}_im"] = L[n].imag
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    out["wall_s"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--ncpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    A = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(OUT, "case_table.csv"))))
    if A.cases:
        rows = [r for r in rows if r["case"] in A.cases]
    if A.ncpu > 1:
        with Pool(A.ncpu) as p:
            res = p.map(fit_case, rows, chunksize=1)
    else:
        res = [fit_case(r) for r in rows]
    keys = sorted({k for r in res for k in r}, key=lambda k: (k not in ("case", "Re", "Ca", "beta", "delta_lin"), k))
    fn = os.path.join(OUT, "linear_targets.csv" if not A.cases else "linear_targets_subset.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{len(res)} cases, errors: {sum('error' in r for r in res)} -> {fn}")
    for r in res:
        if "error" in r:
            print("  ", r["case"], r["error"])
