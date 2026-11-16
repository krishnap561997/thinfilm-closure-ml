"""B.3 verification of the Orr-Sommerfeld solver.  Prints a short summary and writes
verification.json to $THINFILM_STORE/analysis/orr_sommerfeld/ (or --out).

  1. Benney limit (growth and c_r as k -> 0), error must scale like k^2.
  2. Critical Reynolds number Re_c = (5/6) cot(beta).
  3. Published OS: saddle points of Brevdo et al. (1999) as tabulated in Kalliadasis et al.
     (2012), Table 7.1 (Re 133, beta 4.6 deg, Gamma 769.8).  Book Nusselt scaling uses the
     velocity scale 3 u_N (U = y - y^2/2): k_ours = k_book, omega/c/V_ours = 3 x book;
     We_book = Gamma/(3Re)^(2/3) = sigma/(rho g sin(beta) h_N^2) = 1/(3 Ca_ours);
     Re and Ct = cot(beta) are identical.  The tabulated omega is omega - kV (moving frame).
  4. Convergence in N, collocation vs shooting, and the table in plan section 0.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from os_solver import temporal, temporal_colloc, lavalle_temporal, saddle_point, spatial_curve, lavalle_spatial

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.environ.get("THINFILM_STORE", "."), "analysis/orr_sommerfeld"))
args = ap.parse_args()
t0 = time.time()
res, ok = {}, {}

# 1. Benney
rows = []
for Re in (0.5, 2.0, 5.0, 10.0):
    for k in (0.01, 0.005, 0.0025):
        c = temporal(k, Re, 0.1, 1.0)
        b = k * k * (1.2 * Re - 1.0) - k ** 4 / 0.3
        rows.append(dict(Re=Re, k=k, growth=k * c.imag, benney=b, rel_err=(k * c.imag - b) / abs(b), cr_minus_3=c.real - 3))
res["benney"] = rows
# error ratio per halving of k at each Re (expect ~4)
ratios = [rows[i]["rel_err"] / rows[i + 1]["rel_err"] for i in range(0, len(rows), 3) for i in (i, i + 1)]
ok["benney_k2"] = all(3.5 < r < 4.5 for r in ratios)
print(f"[1] Benney: rel err at k=0.0025: " + ", ".join(f"Re{r['Re']:g} {r['rel_err']:+.1e}" for r in rows if r["k"] == 0.0025)
      + f"; error ratio per k-halving {min(ratios):.2f}-{max(ratios):.2f} (expect 4)  -> {'PASS' if ok['benney_k2'] else 'FAIL'}")

# 2. Re_c
rows = []
for beta in (10.0, 45.0, 80.0):
    cb = 1 / np.tan(np.radians(beta))
    Rc = brentq(lambda R: temporal(1e-3, R, 1.0, cb).imag, 0.2 * cb + 1e-3, 3 * cb + 0.5, xtol=1e-12)
    rows.append(dict(beta=beta, Re_c=Rc, theory=5 / 6 * cb, rel_err=Rc / (5 / 6 * cb) - 1))
res["Re_c"] = rows
ok["Re_c"] = all(abs(r["rel_err"]) < 1e-4 for r in rows)
print("[2] Re_c (k=1e-3): " + ", ".join(f"b{r['beta']:g}: {r['Re_c']:.6f} vs {r['theory']:.6f}" for r in rows)
      + f"  -> {'PASS' if ok['Re_c'] else 'FAIL'}")

# 3. Brevdo et al. via Kalliadasis Table 7.1 (book units)
Re, cb = 133.0, 1 / np.tan(np.radians(4.6))
Web = 769.8 / (3 * Re) ** (2 / 3); Ca = 1 / (3 * Web)
table = [  # V_book, branch, k_r, k_i, omega_r, omega_i (signs of k_i restored: lost in the PDF text)
    (0.575, "I", 0.170, -0.179, 0.0092, 0.0031), (0.575, "II", 0.040, -0.046, 0.0053, 0.0037),
    (0.580, "I", 0.177, -0.165, 0.0083, 0.0039), (0.580, "II", 0.0431, -0.0475, 0.0051, 0.0039)]
rows = []
for Vb, br, kr, ki, wr, wi in table:
    k, w, c = saddle_point(3 * Vb, complex(kr, ki), Re, Ca, cb)
    wm = (w - k * 3 * Vb) / 3
    rows.append(dict(V_book=Vb, branch=br, book=[kr, ki, wr, wi], ours=[k.real, k.imag, wm.real, wm.imag]))
res["brevdo_table7_1"] = dict(Re=Re, beta=4.6, Gamma=769.8, We_book=Web, Ca_ours=Ca, rows=rows)
# tolerance: half a unit in the last printed digit, plus 1% for the rounding of the inputs
def close(o, b):
    dig = [10 ** -(len(f"{abs(x):.4f}".rstrip('0').split('.')[1])) for x in b]
    return all(abs(oi - bi) <= 0.5 * d + 0.01 * abs(bi) for oi, bi, d in zip(o, b, dig))
ok["brevdo"] = all(close(r["ours"], r["book"]) for r in rows)
print("[3] Brevdo et al. saddle points (book units k_r, k_i, w_r, w_i):")
for r in rows:
    print(f"    V={r['V_book']} {r['branch']:>2}: ours " + " ".join(f"{x:+.4f}" for x in r["ours"])
          + "  book " + " ".join(f"{x:+.4f}" for x in r["book"]))
print(f"    -> {'PASS' if ok['brevdo'] else 'FAIL'}")

# 4. Convergence and section-0 table
cb = 1.0
conv = {N: temporal(0.3, 120, 0.01, cb, N=N) for N in (32, 48, 64, 80)}
spread = max(abs(v - conv[48]) for v in conv.values())
dcs = []
rows = []
for k in (0.04, 0.08, 0.15, 0.3, 0.5):
    c = temporal(k, 120, 0.01, cb); c0 = temporal_colloc(k, 120, 0.01, cb); l = lavalle_temporal(k, 120, 0.01, cb)
    dcs.append(abs(c - c0))
    rows.append(dict(k=k, os_cr=c.real, os_growth=k * c.imag, lav_cr=l.real, lav_growth=k * l.imag))
res["convergence"] = dict(N_spread_k0p3=spread, colloc_vs_shoot_max=max(dcs))
res["section0_table"] = rows
plan = [(1.951, .0151, 1.900, .0152), (1.805, .0221, 1.758, .0203), (1.731, .0287, 1.702, .0227),
        (1.718, .0338, 1.715, .0207), (1.789, .0274, 1.797, .0142)]
ok["section0"] = all(abs(r["os_cr"] - p[0]) < 6e-4 and abs(r["os_growth"] - p[1]) < 6e-5 and
                     abs(r["lav_cr"] - p[2]) < 6e-4 and abs(r["lav_growth"] - p[3]) < 6e-5 for r, p in zip(rows, plan))
ok["convergence"] = spread < 1e-8 and max(dcs) < 1e-6
print(f"[4] polished c at k=0.3 over N=32..80: spread {spread:.1e}; |collocation - shooting| max {max(dcs):.1e}"
      f"  -> {'PASS' if ok['convergence'] else 'FAIL'}")
print("    k     OS c_r  OS growth  Lav c_r  Lav growth")
for r in rows:
    print(f"    {r['k']:.2f}  {r['os_cr']:.3f}  {r['os_growth']:.4f}     {r['lav_cr']:.3f}  {r['lav_growth']:.4f}")
print(f"    plan section 0 table reproduced -> {'PASS' if ok['section0'] else 'FAIL'}")

# spatial at the forced frequency of the oracle case (10 Hz, omega ~ 0.134)
om = np.array([0.05, 0.1, 0.134, 0.2, 0.3, 0.45])
K, _ = spatial_curve(om, 120, 0.01, 1.0)
res["spatial_Re120"] = [dict(omega=o, os_kr=k.real, os_mki=-k.imag, os_cph=o / k.real,
                             lav_kr=lavalle_spatial(o, 120, .01, 1.).real, lav_mki=-lavalle_spatial(o, 120, .01, 1.).imag)
                        for o, k in zip(om, K)]
print("    spatial Re120 Ca0.01 b45: omega, -k_i OS / Lavalle: " + ", ".join(
    f"{r['omega']:.3f}: {r['os_mki']:.4f}/{r['lav_mki']:.4f}" for r in res["spatial_Re120"]))

res["pass"] = ok
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "verification.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
print(f"ALL {'PASS' if all(ok.values()) else 'FAIL: ' + str([k for k, v in ok.items() if not v])}"
      f"  ({time.time() - t0:.0f} s)  -> {args.out}/verification.json")
