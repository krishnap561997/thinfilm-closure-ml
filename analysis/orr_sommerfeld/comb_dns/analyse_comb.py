"""Analysis of the comb-forced DNS (D3; dns/comb/setup_comb_runs.py): B.5, B.7 and B.8 from DNS.

For each case under $THINFILM_STORE/datasets/BasC/DNS/comb/<case>/ (COMB_INFO.txt, run_t<T>/profiles):

0. Settling: mean h over the window and its drift (first vs second half), max over the fit range.
1. B.5  Per comb mode omega_k = k d_omega: A_k(x) = (2/n) sum_t (h - <h>_t) e^{i omega_k t} over exactly one
   comb period.  -k_i and k_r = slopes of ln|A_k| and arg A_k over XA < x < XB; R^2; x_nl = first x > XB
   where the local growth (20-unit sliding fit) departs from the window value by > 20%.
   Compared with OS (spatial branch) and analytic linearised Lavalle.
2. B.7  Closure transfer ratios per mode, from projections of the *undifferentiated* defects
      F_M = int u^2 - q^2/h - h^5/5,   G = (u_z|_0 - 3q/h^2)/Re,   H = h^5
   with d/dx -> i k (k = the fitted complex k of the mode):
      dMx/eta = i k F_M,k / A_k,   dTw/eta = (G_k + i k (3/25) H_k) / A_k        (median over XA..XB)
   (same definitions as pipeline/extract_closures/extract_closure_terms.closure_set), against the OS
   targets along the OS spatial mode (linear_targets.os_closures).
3. B.8  Fit (a, b, c, d) with D = D_OS imposed (as linear_targets.fit_case) to the DNS k(omega) of the
   reliable modes, and to OS at the same frequencies for comparison.
Reliable mode: R^2 > R2_MIN and wavelength < 2 (XB - XA) and |A_k| above 1e-7 everywhere in the window.

Outputs in --out (default $THINFILM_STORE/analysis/orr_sommerfeld/comb_dns/): comb_modes_<case>.csv,
comb_fits.csv, comb_growth_<case>.png.  Prints a short summary.
Usage: python analyse_comb.py [--cases ...] [--xa 60 --xb 150]
"""
import argparse, csv, glob, math, os, sys
import h5py, numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import os_solver as osv
from linear_targets import os_closures, k_aug, c_from_D

STORE = os.environ["THINFILM_STORE"]
BASE = os.path.join(STORE, "datasets/BasC/DNS/comb")
R2_MIN = 0.95


def info(case_dir):
    d = {}
    for line in open(os.path.join(case_dir, "COMB_INFO.txt")):
        k, *v = line.split()
        if v:
            d[k] = v[0] if k == "case" else (v[0] if k == "window" else float(v[0]))
    return d


def load(case_dir, I):
    runs = [d for d in glob.glob(os.path.join(case_dir, "run_t*")) if glob.glob(os.path.join(d, "profiles", "profile_*.h5"))]
    if not runs:
        raise FileNotFoundError(f"no profiles under {case_dir}/run_t*/profiles")
    pref = os.path.join(case_dir, f"run_t{I['T_SPIN']:g}")
    run = pref if pref in runs else sorted(runs)[-1]
    files = sorted(glob.glob(os.path.join(run, "profiles", "profile_*.h5")))
    t0 = I["T_SPIN"] + 5.0
    tt, rows = [], []
    with h5py.File(files[0]) as f:
        x = f["x"][:]
        par = {k: float(np.asarray(f["parameters"][k][()]).squeeze()) for k in ("Re", "Ca", "angle")}
    for fn in files:
        with h5py.File(fn) as f:
            t = float(np.asarray(f["parameters/time"][()]).squeeze())
            if t0 - 1e-6 <= t < t0 + I["TC"] - 1e-6:
                tt.append(t)
                rows.append([f[k][:] for k in ("h", "q", "int_u2_dy", "dudy_wall")])
    A = np.array(rows)  # (nt, 4, nx)
    return np.array(tt), x, A, par, run


def fit_line(x, y):
    p = np.polyfit(x, y, 1)
    r2 = 1 - np.var(y - np.polyval(p, x)) / max(np.var(y), 1e-300)
    return p[0], r2


def analyse_case(case, xa, xb, out, base=BASE, t0=None, tc=None):
    cdir = os.path.join(base, case)
    I = info(cdir)
    if t0 is not None:            # testing only: override the projection window
        I["T_SPIN"] = t0 - 5.0
    if tc is not None:            # testing only: shorter window, same comb frequencies (not orthogonal)
        I["TC"] = tc
    tt, x, A, par, run = load(cdir, I)
    Re, Ca = par["Re"], par["Ca"]
    beta = par["angle"] if par["angle"] < 3.2 else math.radians(par["angle"])
    cotb = 1 / math.tan(beta)
    n_exp = int(round(I["TC"] / I["T_PROF"]))
    res = dict(case=case, Re=Re, Ca=Ca, beta_deg=math.degrees(beta), n_snap=len(tt), n_expected=n_exp, run=run)
    if len(tt) != n_exp:
        res["error"] = f"window has {len(tt)} snapshots, expected {n_exp} (run unfinished?)"
        return res, []
    h, q, M, dudy = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    m = (x > xa) & (x < xb)
    hm = h.mean(0); half = len(tt) // 2
    res.update(mean_h=float(hm[m].mean()), max_mean_dev=float(np.abs(hm[m] - 1).max()),
               max_drift=float(np.abs(h[:half].mean(0) - h[half:].mean(0))[m].max()),
               max_amp=float(np.abs(h - hm)[:, m].max()))
    FM = M - q ** 2 / h - h ** 5 / 5
    G = (dudy - 3 * q / h ** 2) / Re
    H5 = h ** 5
    dw, Mm = I["domega"], int(I["M"])
    om = dw * np.arange(1, Mm + 1)
    Kos, Cos = osv.spatial_curve(om, Re, Ca, cotb)
    modes = []
    for j, w in enumerate(om):
        E = np.exp(1j * w * tt)[:, None]
        proj = lambda f: (2 / len(tt)) * (E * (f - f.mean(0))).sum(0)
        Ah, AF, AG, AH = proj(h), proj(FM), proj(G), proj(H5)
        la, ph = np.log(np.abs(Ah[m]) + 1e-300), np.unwrap(np.angle(Ah[m]))
        gi, r2 = fit_line(x[m], la); kr, _ = fit_line(x[m], ph)
        k = complex(kr, -gi)
        # departure from linear growth beyond XB
        xnl = np.nan
        s20 = int(round(20 / (x[1] - x[0])))
        lal = np.log(np.abs(Ah) + 1e-300)
        for i0 in np.where(x > xb)[0][::max(1, s20 // 4)]:
            if i0 + s20 >= len(x):
                break
            g_loc = np.polyfit(x[i0:i0 + s20], lal[i0:i0 + s20], 1)[0]
            if abs(g_loc - gi) > 0.2 * max(abs(gi), 1e-4):
                xnl = float(x[i0 + s20 // 2]); break
        dM = np.median((1j * k * AF[m] / Ah[m]).real) + 1j * np.median((1j * k * AF[m] / Ah[m]).imag)
        dT_ = (AG[m] + 1j * k * 0.12 * AH[m]) / Ah[m]
        dT = np.median(dT_.real) + 1j * np.median(dT_.imag)
        try:
            L = os_closures(Kos[j], Re, Ca, cotb, c_guess=Cos[j]); osM, osT = L["dM"], L["dT"]
        except Exception:
            osM = osT = complex(np.nan, np.nan)
        klav = osv.lavalle_spatial(w, Re, Ca, cotb)
        reliable = (r2 > R2_MIN) and (2 * np.pi / abs(kr) < 2 * (xb - xa)) and (np.abs(Ah[m]).min() > 1e-7)
        modes.append(dict(omega=w, dns_mki=gi, dns_kr=kr, dns_cph=w / kr, r2=r2, amp_xa=float(abs(Ah[m][0])),
                          amp_xb=float(abs(Ah[m][-1])), x_nonlin=xnl, reliable=int(reliable),
                          os_mki=-Kos[j].imag, os_kr=Kos[j].real, os_cph=w / Kos[j].real,
                          lav_mki=-klav.imag, lav_kr=klav.real,
                          dns_dM_re=dM.real, dns_dM_im=dM.imag, dns_dT_re=dT.real, dns_dT_im=dT.imag,
                          os_dM_re=osM.real, os_dM_im=osM.imag, os_dT_re=osT.real, os_dT_im=osT.imag))
    rel = [md for md in modes if md["reliable"]]
    res["n_reliable"] = len(rel)
    if rel:
        sc = max(md["os_mki"] for md in modes)
        res["growth_err_dns_vs_os"] = max(abs(md["dns_mki"] - md["os_mki"]) for md in rel) / sc
        res["growth_err_lav_vs_os"] = max(abs(md["lav_mki"] - md["os_mki"]) for md in rel) / sc
        res["growth_err_lav_vs_dns"] = max(abs(md["lav_mki"] - md["dns_mki"]) for md in rel) / sc
        res["speed_err_dns_vs_os"] = max(abs(md["dns_kr"] / md["os_kr"] - 1) for md in rel)
        res["omega_rel_lo"], res["omega_rel_hi"] = min(md["omega"] for md in rel), max(md["omega"] for md in rel)
        # B.8 fits on the reliable modes
        w_r = np.array([md["omega"] for md in rel])
        Kd = np.array([complex(md["dns_kr"], -md["dns_mki"]) for md in rel])
        Ko = np.array([complex(md["os_kr"], -md["os_mki"]) for md in rel])
        for tag, K in (("dns", Kd), ("os", Ko)):
            kim = np.max(np.abs(K.imag))

            def r(p, K=K, kim=kim):
                a, b, d = p; c = c_from_D(a, b, Re, cotb)
                ka = np.array([k_aug(o, Re, Ca, cotb, a, b, c, d, kk) for o, kk in zip(w_r, K)])
                return np.r_[(ka.real - K.real) / K.real, (ka.imag - K.imag) / kim]
            sol = least_squares(r, [0.0, 0.0, 0.0], x_scale="jac")
            a, b, d = sol.x; rr = r(sol.x)
            res.update({f"fit_{tag}_a": a, f"fit_{tag}_b": b, f"fit_{tag}_c": c_from_D(a, b, Re, cotb), f"fit_{tag}_d": d,
                        f"fit_{tag}_err_ki": float(np.max(np.abs(rr[len(w_r):]))),
                        f"fit_{tag}_err_kr": float(np.max(np.abs(rr[:len(w_r)])))})
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"comb_modes_{case}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(modes[0])); w.writeheader()
        for md in modes:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in md.items()})
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        o = np.array([md["omega"] for md in modes]); rl = np.array([md["reliable"] for md in modes], bool)
        for key, lab, sty in (("os_mki", "OS", "k-"), ("lav_mki", "Lavalle", "b--")):
            ax[0].plot(o, [md[key] for md in modes], sty, label=lab)
        dm = np.array([md["dns_mki"] for md in modes])
        ax[0].plot(o[rl], dm[rl], "ro", label="DNS (reliable)"); ax[0].plot(o[~rl], dm[~rl], "rx", alpha=0.4, label="DNS (other)")
        ax[1].plot(o, [md["os_cph"] for md in modes], "k-", o, [w / md["lav_kr"] for w, md in zip(o, modes)], "b--")
        cp = np.array([md["dns_cph"] for md in modes]); ax[1].plot(o[rl], cp[rl], "ro"); ax[1].plot(o[~rl], cp[~rl], "rx", alpha=0.4)
        ax[0].set_xlabel("ω"); ax[0].set_ylabel("spatial growth −k_i"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("ω"); ax[1].set_ylabel("phase speed ω/k_r"); ax[1].set_ylim(1, 3.5)
        fig.suptitle(f"{case}: comb DNS, x = {xa:g}–{xb:g}"); fig.tight_layout()
        fig.savefig(os.path.join(out, f"comb_growth_{case}.png"), dpi=140); plt.close(fig)
    except Exception as e:
        print("  (figure skipped:", e, ")")
    return res, modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--xa", type=float, default=60.0)
    ap.add_argument("--xb", type=float, default=150.0)
    ap.add_argument("--out", default=os.path.join(STORE, "analysis/orr_sommerfeld/comb_dns"))
    ap.add_argument("--base", default=BASE, help="comb data directory (testing)")
    ap.add_argument("--t0", type=float, help="testing: window start (default T_SPIN + 5)")
    ap.add_argument("--tc", type=float, help="testing: window length (default TC); comb frequencies unchanged")
    A = ap.parse_args()
    cases = A.cases or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(A.base, "Re*", "COMB_INFO.txt")))
    allres = []
    for case in cases:
        res, modes = analyse_case(case, A.xa, A.xb, A.out, A.base, A.t0, A.tc)
        allres.append(res)
        print(f"\n=== {case}")
        if "error" in res:
            print("  ERROR:", res["error"]); continue
        print(f"  settling over x {A.xa:g}-{A.xb:g}: mean h {res['mean_h']:.5f}, max|<h>-1| {res['max_mean_dev']:.1e}, "
              f"drift {res['max_drift']:.1e}, max|h-<h>| {res['max_amp']:.1e}")
        print(f"  reliable modes {res['n_reliable']}/{len(modes)}" + (f" (omega {res['omega_rel_lo']:.2f}-{res['omega_rel_hi']:.2f})" if res['n_reliable'] else ""))
        if res["n_reliable"]:
            print(f"  max growth error / max OS growth: DNS-OS {res['growth_err_dns_vs_os']:.3f}, "
                  f"Lavalle-OS {res['growth_err_lav_vs_os']:.3f}, Lavalle-DNS {res['growth_err_lav_vs_dns']:.3f};"
                  f"  max |k_r,DNS/k_r,OS - 1| {res['speed_err_dns_vs_os']:.3f}")
            for tag in ("dns", "os"):
                print(f"  fit (a,b,c,d) to {tag.upper():3s}: a {res[f'fit_{tag}_a']:+.4f} b {res[f'fit_{tag}_b']:+.3f} "
                      f"c {res[f'fit_{tag}_c']:+.3f} d {res[f'fit_{tag}_d']:+.4f}  err k_i {res[f'fit_{tag}_err_ki']:.3f} k_r {res[f'fit_{tag}_err_kr']:.3f}")
        print(f"  {'omega':>6s} {'rel':>3s} {'-ki DNS':>8s} {'OS':>8s} {'Lav':>8s} {'c DNS':>6s} {'c OS':>6s} {'x_nl':>6s} | "
              f"{'dMx/eta DNS':>17s} {'OS':>17s} | {'dTw/eta DNS':>17s} {'OS':>17s}")
        for md in modes[::max(1, len(modes) // 12)]:
            z = lambda a, b: f"{a:+.4f}{b:+.4f}i"
            print(f"  {md['omega']:6.3f} {md['reliable']:3d} {md['dns_mki']:+8.4f} {md['os_mki']:+8.4f} {md['lav_mki']:+8.4f} "
                  f"{md['dns_cph']:6.3f} {md['os_cph']:6.3f} {md['x_nonlin']:6.0f} | {z(md['dns_dM_re'], md['dns_dM_im']):>17s} "
                  f"{z(md['os_dM_re'], md['os_dM_im']):>17s} | {z(md['dns_dT_re'], md['dns_dT_im']):>17s} {z(md['os_dT_re'], md['os_dT_im']):>17s}")
    os.makedirs(A.out, exist_ok=True)
    keys = sorted({k for r in allres for k in r}, key=lambda k: (k != "case", k))
    with open(os.path.join(A.out, "comb_fits.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in allres:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\n->", A.out)


if __name__ == "__main__":
    main()
