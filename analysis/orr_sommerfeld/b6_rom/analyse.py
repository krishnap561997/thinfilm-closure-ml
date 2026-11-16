"""B.6 analysis: spatial growth -k_i and phase speed of every comb mode in the linear ROM runs
(make_runs.py), against OS, analytic linearised Lavalle and, for ML runs, the analytic dispersion of
the model's linear part at h = q = 1 (A.1 coefficients, uniform-film slope e included).

Per run: A_k(x) = (2/n) sum_t (h - <h>_t) e^{i k dw t} over the last comb period TC (exact);
-k_i, k_r = slopes of ln|A_k| and arg A_k over X_A L < x < X_B L.  Prints per case a table and the
band-averaged errors; writes b6_summary.csv.
Usage: python analyse.py [--case Re120_Ca0_010_theta45]
"""
import argparse, csv, math, os, sys
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld")); sys.path.insert(0, os.path.join(ROOT, "scripts/structured-closure"))
import os_solver as osv
from mlp_np import sum_model, sep_models
from a1_uniform_state import lin_coeffs

STORE = os.environ["THINFILM_STORE"]
BASE = os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom")
X_A, X_B = 0.1, 0.8
CASE_P = {"Re020_Ca0_010_theta45": (20.0, 0.01, 45.0), "Re120_Ca0_010_theta45": (120.0, 0.01, 45.0),
          "Re200_Ca0_010_theta45": (200.0, 0.01, 45.0)}
MODELS = ["lavalle", "ml_sum_hhx", "ml_mx_tw"]
CASE_L = {"Re020_Ca0_010_theta45": 250.0, "Re120_Ca0_010_theta45": 400.0, "Re200_Ca0_010_theta45": 450.0}


def k_aug_spatial(omega, Re, Ca, cotb, co, k_ref):
    We, G = Re * Ca, -0.6 + 3 * cotb / Re
    a, b, c, e = (co[n] for n in "abce")
    r = np.roots([1j / We, 0, 2 * omega / Re + 1j * G + 1j * b,
                  2j * omega + 1j * c * omega - 9 / Re + e - 3 * a, 3 * omega / Re + a * omega - 1j * omega ** 2])
    return r[np.argmin(np.abs(r - k_ref))]


def run_modes(d):
    cm = dict(l.split() for l in open(os.path.join(d, "comb.txt")))
    TC, M, dw = float(cm["TC"]), int(cm["M"]), float(cm["domega"])
    log = open(os.path.join(d, "run.log")).read()
    if "exit 0" not in log:
        return None, "not finished or failed"
    with h5py.File(os.path.join(d, "thinfilm_training_data.h5")) as f:
        t, x = f["time"][:], f["x"][:]
        sel = np.where(t > t[-1] - TC + 1e-9)[0]
        H = f["h"][sel[0]:sel[-1] + 1]
        tt = t[sel]
    if not np.all(np.isfinite(H)):
        return None, "non-finite h"
    L = x[-1] + 0.5 * (x[1] - x[0])
    m = (x > X_A * L) & (x < X_B * L)
    Hm = H - H.mean(0)
    res = []
    for k in range(1, M + 1):
        A = (2 / len(tt)) * (np.exp(1j * k * dw * tt)[:, None] * Hm).sum(0)
        la, ph = np.log(np.abs(A[m]) + 1e-300), np.unwrap(np.angle(A[m]))
        g = np.polyfit(x[m], la, 1); kr = np.polyfit(x[m], ph, 1)[0]
        r2 = 1 - np.var(la - np.polyval(g, x[m])) / max(np.var(la), 1e-300)
        res.append((k * dw, g[0], kr, r2, np.abs(A[m]).max()))
    info = dict(max_dev=float(np.abs(H[:, m] - 1).max()), mean_h=float(H[:, m].mean()),
                mean_h_end=float(H[:, x > X_B * L].mean()))
    return np.array(res), info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--case", nargs="*", default=list(CASE_P)); A = ap.parse_args()
    S = {"ml_sum_hhx": sum_model(), "ml_mx_tw": sep_models()}
    out = []
    for case in A.case:
        Re, Ca, beta = CASE_P[case]; cotb = 1 / math.tan(math.radians(beta)); ang = math.radians(beta)
        R = {}
        for mdl in MODELS:
            r, info = run_modes(os.path.join(BASE, case, mdl))
            R[mdl] = (r, info)
        om = next(v[0][:, 0] for v in R.values() if v[0] is not None)
        Kos, _ = osv.spatial_curve(om, Re, Ca, cotb)
        Klav = np.array([osv.lavalle_spatial(o, Re, Ca, cotb) for o in om])
        Kan = {m: np.array([k_aug_spatial(o, Re, Ca, cotb, lin_coeffs(S[m], Re, Ca, ang), kl) for o, kl in zip(om, Klav)])
               for m in S}
        print(f"\n=== {case}  (-k_i; c_ph in brackets.  'an' = analytic linearisation at h = q = 1)")
        for mdl in MODELS:
            r, info = R[mdl]
            print(f"  {mdl:11s}: " + (f"max|h-1| {info['max_dev']:.2e}, <h> window {info['mean_h']:.4f}, outlet {info['mean_h_end']:.4f}"
                                      if r is not None else info))
        hdr = f"{'omega':>6s} {'OS':>16s} {'Lav an':>16s} {'Lav ROM':>16s} {'sum an':>16s} {'sum ROM':>16s} {'sep an':>16s} {'sep ROM':>16s}"
        print(hdr)
        cell = lambda k, o: f"{-k.imag:+.4f} ({o / k.real:5.3f})" if k is not None else " " * 16
        for i, o in enumerate(om):
            rk = {m: (complex(R[m][0][i, 2], -R[m][0][i, 1]) if R[m][0] is not None else None) for m in MODELS}
            print(f"{o:6.3f} {cell(Kos[i], o)} {cell(Klav[i], o)} {cell(rk['lavalle'], o)} {cell(Kan['ml_sum_hhx'][i], o)} "
                  f"{cell(rk['ml_sum_hhx'], o)} {cell(Kan['ml_mx_tw'][i], o)} {cell(rk['ml_mx_tw'], o)}")
            out.append(dict(case=case, omega=o, os_mki=-Kos[i].imag, os_kr=Kos[i].real, lav_an_mki=-Klav[i].imag, lav_an_kr=Klav[i].real,
                            **{f"{m}_mki": (-rk[m].imag if rk[m] is not None else np.nan) for m in MODELS},
                            **{f"{m}_kr": (rk[m].real if rk[m] is not None else np.nan) for m in MODELS},
                            **{f"{m}_r2": (R[m][0][i, 3] if R[m][0] is not None else np.nan) for m in MODELS},
                            sum_an_mki=-Kan["ml_sum_hhx"][i].imag, sep_an_mki=-Kan["ml_mx_tw"][i].imag,
                            **{f"{m}_reliable": 1 for m in MODELS}))
        sc = max(abs(Kos.imag))
        Lw = (X_B - X_A) * CASE_L[case]
        print(f"  band summary over reliable modes (wavelength < {0.5 * Lw:.0f} = half the fit window, R2 of ln|A| > 0.95):"
              "  max |(-k_i) - OS| / max OS growth;  max |c_ph/c_ph,OS - 1|;  n modes")
        for m in MODELS:
            if R[m][0] is None: continue
            gi, kr, r2 = R[m][0][:, 1], R[m][0][:, 2], R[m][0][:, 3]
            ok = (2 * np.pi / np.abs(kr) < 0.5 * Lw) & (r2 > 0.95)
            if ok.any():
                print(f"    {m:11s} ROM: growth err {np.max(np.abs(gi + Kos.imag)[ok]) / sc:.3f}, "
                      f"speed err {np.max(np.abs(Kos.real / kr - 1)[ok]):.3f}, n {ok.sum()} "
                      f"(omega {om[ok].min():.2f}-{om[ok].max():.2f})")
            for i in np.where(~ok)[0]:
                out[-len(om) + i][f"{m}_reliable"] = 0
        okl = 2 * np.pi / Klav.real < 0.5 * Lw
        print(f"    {'lavalle':11s} an : growth err {np.max(np.abs(Klav.imag - Kos.imag)[okl]) / sc:.3f}, "
              f"speed err {np.max(np.abs(Kos.real / Klav.real - 1)[okl]):.3f}")
        for m in S:
            print(f"    {m:11s} an : growth err {np.max(np.abs(Kan[m].imag - Kos.imag)[okl]) / sc:.3f}, "
                  f"speed err {np.max(np.abs(Kos.real / Kan[m].real - 1)[okl]):.3f}")
    fn = os.path.join(BASE, "b6_summary.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader()
        for r in out: w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("->", fn)


if __name__ == "__main__":
    main()
