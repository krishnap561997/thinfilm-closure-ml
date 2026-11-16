"""B.4: OS (and linearised-Lavalle) stability characteristics for every case in
datasets/MANIFEST_dns.csv.  Writes case_table.csv to $THINFILM_STORE/analysis/orr_sommerfeld/
and prints a short summary.

Per case (project Nusselt scaling; Hz via T0 = h_N/u_N from the case's params.in, as filmflow0.c):
  temporal : k_max, growth_max, c_r(k_max), cut-off k_c, omega_c = k_c c_r(k_c)
  spatial  : omega_max (max of -k_i over real omega), -k_i there, k_r there
  Hz       : f_c, f_max_spatial, and the forcing cut-off 28 Hz in nondimensional units
  long-wave: eps_lin = k_max (temporal), delta_lin = eps_lin * Re
  Lavalle  : the same quantities from the analytic linearised Lavalle model

Usage: python case_table.py [--ncpu 8] [--cases Re120_Ca0_010_theta45 ...]
"""
import argparse, csv, math, os, sys, time, traceback
from multiprocessing import Pool
import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import os_solver as osv

G = 9.81


DEFAULTS = {"RHO_L": "1072", "MU_L": "0.00673", "FCUT_DIM": "28.0"}   # filmflow0.c defaults


def read_params(case):
    p = dict(DEFAULTS)
    with open(os.path.join(ROOT, "dns/production/cases", case, "params.in")) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                p[k.strip()] = v.split("#")[0].strip()
    return p


def T0_of(Re, beta_deg, rho, mu):
    s = math.sin(math.radians(beta_deg))
    H0 = (3 * Re * mu * mu / (rho * rho * G * s)) ** (1 / 3)
    U0 = G * s * H0 * H0 * rho / mu / 3
    return H0 / U0, H0, U0


def os_c(k, Re, Ca, cb, c_guess=None):
    try:
        return osv.temporal(k, Re, Ca, cb, c_guess=c_guess)
    except RuntimeError:
        return osv.temporal(k, Re, Ca, cb, N=80, dN=24, c_guess=c_guess)


def temporal_props(growth, cr, kgrid):
    """k_max, growth_max, k_c from a growth function on a log grid + refinement."""
    g = []
    for k in kgrid:                      # stop 3 points after the first + -> - crossing
        g.append(growth(k))
        n = len(g)
        if n > 4 and max(g) > 0 and all(x <= 0 for x in g[-3:]) and g[-4] > 0 or \
           (n > 4 and max(g[:-3]) > 0 and all(x <= 0 for x in g[-3:])):
            break
    kgrid = kgrid[:len(g)]
    g = np.array(g)
    if g.max() <= 0:
        return dict(stable=True)
    i = int(np.argmax(g))
    lo, hi = kgrid[max(i - 1, 0)], kgrid[min(i + 1, len(kgrid) - 1)]
    r = minimize_scalar(lambda k: -growth(k), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-7))
    kmax = r.x
    j = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    j = j[j >= i]
    if len(j) == 0:
        raise RuntimeError("no cut-off in k grid")
    kc = brentq(growth, kgrid[j[0]], kgrid[j[0] + 1], xtol=1e-10)
    return dict(stable=False, k_max=kmax, growth_max=growth(kmax), cr_kmax=cr(kmax), k_c=kc, omega_c=kc * cr(kc))


def one_case(row):
    case = row["case"]
    Re, Ca, beta = float(row["Re"]), float(row["Ca"]), float(row["beta_deg"])
    cb = 1 / math.tan(math.radians(beta))
    out = dict(case=case, Re=Re, Ca=Ca, beta_deg=beta, status=row["status"], Re_c=5 / 6 * cb, error="")
    t = time.time()
    try:
        p = read_params(case)
        T0, H0, U0 = T0_of(Re, beta, float(p["RHO_L"]), float(p["MU_L"]))
        fcut = float(p.get("FCUT_DIM", 28.0))
        out.update(T0_s=T0, hN_mm=H0 * 1e3, uN_m_s=U0, omega_fcut=2 * math.pi * fcut * T0)
        kgrid = np.geomspace(1e-3, 4.0, 80)
        # ---- OS temporal
        tp = temporal_props(lambda k: k * os_c(k, Re, Ca, cb).imag, lambda k: os_c(k, Re, Ca, cb).real, kgrid)
        # ---- Lavalle temporal
        lp = temporal_props(lambda k: k * osv.lavalle_temporal(k, Re, Ca, cb).imag,
                            lambda k: osv.lavalle_temporal(k, Re, Ca, cb).real, kgrid)
        if tp["stable"]:
            out["error"] = "OS stable (Re < Re_c?)"
            return out
        out.update({f"os_{k}": v for k, v in tp.items() if k != "stable"})
        if not lp["stable"]:
            out.update({f"lav_{k}": v for k, v in lp.items() if k != "stable"})
        # ---- OS spatial: grid in omega, then refine around the maximum
        wc = tp["omega_c"]
        om = np.linspace(0.02, 0.97, 30) * wc
        K, C = osv.spatial_curve(om, Re, Ca, cb)
        i = int(np.argmax(-K.imag))
        i = min(max(i, 1), len(om) - 2)
        cache = {}

        def mki(w):
            k, c = osv.spatial(w, Re, Ca, cb, k0=K[i], c0=C[i])
            cache[w] = k
            return k.imag
        r = minimize_scalar(mki, bounds=(om[i - 1], om[i + 1]), method="bounded", options=dict(xatol=1e-7))
        ks = osv.spatial(r.x, Re, Ca, cb, k0=K[i], c0=C[i])[0]
        out.update(os_omega_max=r.x, os_mki_max=-ks.imag, os_kr_at_omega_max=ks.real,
                   os_cph_at_omega_max=r.x / ks.real)
        # ---- Lavalle spatial (grid + bounded refine)
        if not lp["stable"]:
            wl = lp["omega_c"]
            f = lambda w: osv.lavalle_spatial(w, Re, Ca, cb).imag
            omg = np.linspace(0.02, 0.97, 60) * wl
            j = int(np.argmin([f(w) for w in omg])); j = min(max(j, 1), len(omg) - 2)
            r2 = minimize_scalar(f, bounds=(omg[j - 1], omg[j + 1]), method="bounded", options=dict(xatol=1e-8))
            out.update(lav_omega_max=r2.x, lav_mki_max=-f(r2.x))
        # ---- derived
        tw = 2 * math.pi * T0
        out.update(f_c_Hz=wc / tw, f_max_spatial_Hz=r.x / tw,
                   fcut_over_fc=fcut / (wc / tw),
                   eps_lin=tp["k_max"], delta_lin=tp["k_max"] * Re)
        if not lp["stable"]:
            out.update(lav_f_c_Hz=lp["omega_c"] / tw, lav_eps=lp["k_max"], lav_delta=lp["k_max"] * Re)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        traceback.print_exc(file=sys.stderr)
    out["wall_s"] = round(time.time() - t, 1)
    return out


COLS = ["case", "Re", "Ca", "beta_deg", "status", "Re_c", "T0_s", "hN_mm", "uN_m_s",
        "os_k_max", "os_growth_max", "os_cr_kmax", "os_k_c", "os_omega_c",
        "os_omega_max", "os_mki_max", "os_kr_at_omega_max", "os_cph_at_omega_max",
        "f_c_Hz", "f_max_spatial_Hz", "omega_fcut", "fcut_over_fc", "eps_lin", "delta_lin",
        "lav_k_max", "lav_growth_max", "lav_cr_kmax", "lav_k_c", "lav_omega_c", "lav_omega_max", "lav_mki_max",
        "lav_f_c_Hz", "lav_eps", "lav_delta", "wall_s", "error"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncpu", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--out", default=os.path.join(os.environ.get("THINFILM_STORE", "."), "analysis/orr_sommerfeld"))
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(ROOT, "datasets/MANIFEST_dns.csv"))))
    if a.cases:
        rows = [r for r in rows if r["case"] in a.cases]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    if a.ncpu > 1:
        with Pool(a.ncpu) as pool:
            res = pool.map(one_case, rows, chunksize=1)
    else:
        res = [one_case(r) for r in rows]
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "case_table.csv" if not a.cases else "case_table_subset.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    bad = [r for r in res if r.get("error")]
    print(f"{len(res)} cases, {len(bad)} with errors/notes, {time.time() - t0:.0f} s -> {fn}")
    for r in bad[:15]:
        print("  ", r["case"], r["error"])
