"""A.1: the current MLP closures at the uniform (Nusselt) state.

S = dMx + dTw (ROM adds -S).  Linearise S about h = q = 1, zero gradients:
    S ~ S0 + alpha eta + a xi + b eta_x + c xi_x,        h = 1 + eta, q = 1 + xi
    e = alpha + 3a  = dS(h, h^3, 0, 0)/dh at h = 1   (nonzero => uniform films are not steady
                                                    to first order: kinematic speed shifts)
Long-wave (plan section 1.1, extended with e):
    kinematic speed 3 - e/(3/Re + a),  D = (3.6 - 3 cot b/Re - b - 3c)/(3/Re + a)
Finite-k: linearised Lavalle + the MLP's linear part (temporal), against OS (case_table.csv):
    -ik c^2 + (2ik + 3/Re + 2k^2/Re) c + ikG - 9/Re + ik^3/We + [S0-free part: a c + ik c_ c + e - 3a + ik b] = 0
S0 is a constant forcing of the momentum equation: it changes the base state (q_base != h^3).
Usage: python a1_uniform_state.py [--out DIR]
"""
import argparse, csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_np import sum_model, sep_models

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STORE = os.environ["THINFILM_STORE"]


def lin_coeffs(S, Re, Ca, ang, d=1e-5):
    f = lambda h=1., q=1., hx=0., qx=0.: S(h, q, hx, qx, Re, Ca, ang)
    S0 = float(f())
    alpha = (f(h=1 + d) - f(h=1 - d)) / (2 * d)
    a = (f(q=1 + d) - f(q=1 - d)) / (2 * d)
    b = (f(hx=d) - f(hx=-d)) / (2 * d)
    c = (f(qx=d) - f(qx=-d)) / (2 * d)
    return dict(S0=S0, alpha=float(alpha), a=float(a), b=float(b), c=float(c), e=float(alpha + 3 * a))


def aug_growth(k, Re, Ca, cotb, co):
    We = Re * Ca; G = -0.6 + 3 * cotb / Re
    a, b, c, e = (co[n] for n in "abce") if co else (0, 0, 0, 0)
    r = np.roots([-1j * k + 1j * k * 0, 2j * k + 3 / Re + 2 * k * k / Re + a + 1j * k * c,
                  1j * k * G - 9 / Re + 1j * k ** 3 / We + e - 3 * a + 1j * k * b])
    r = r[np.argmax(r.imag)]
    return k * r.imag, r.real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(STORE, "analysis/structured_closure"))
    A = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, "ml/functional_mlp/data_config.json")))
    split = {c: "dev" for c in cfg["development_cases"]} | {c: "heldout" for c in cfg["heldout_cases"]}
    ost = {r["case"]: r for r in csv.DictReader(open(os.path.join(STORE, "analysis/orr_sommerfeld/case_table.csv")))}
    models = {"sum_hhx": sum_model(), "sep_mx_tw": sep_models()}
    rows = []
    for case, sp in split.items():
        o = ost[case]; Re, Ca, beta = float(o["Re"]), float(o["Ca"]), float(o["beta_deg"])
        cotb, ang = 1 / math.tan(math.radians(beta)), math.radians(beta)
        D_lav = 1.2 * Re - cotb
        kos = float(o["os_k_max"]); gos = float(o["os_growth_max"])
        ks = np.linspace(0.005, 1.5 * float(o["os_k_c"]), 600)
        g_lav = max(aug_growth(k, Re, Ca, cotb, None)[0] for k in ks)
        for name, S in models.items():
            co = lin_coeffs(S, Re, Ca, ang)
            den = 3 / Re + co["a"]
            g = np.array([aug_growth(k, Re, Ca, cotb, co)[0] for k in ks])
            rows.append(dict(case=case, split=sp, Re=Re, Ca=Ca, beta=beta, model=name, **co,
                             S0_over_friction=co["S0"] / (3 / Re),
                             dc_kin=-co["e"] / den, D_lav=D_lav,
                             D_aug=(3.6 - 3 * cotb / Re - co["b"] - 3 * co["c"]) / den,
                             gmax_os=gos, gmax_lav=g_lav, gmax_aug=float(g.max()),
                             kmax_os=kos, kmax_aug=float(ks[g.argmax()])))
    os.makedirs(A.out, exist_ok=True)
    fn = os.path.join(A.out, "a1_uniform_state.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"-> {fn}")
    hdr = f"{'case':24s} {'sp':4s} {'S0':>8s} {'S0/(3/Re)':>9s} {'e':>8s} {'dc_kin':>7s} {'a':>8s} {'b':>8s} {'c':>8s} {'D_lav':>7s} {'D_aug':>8s} {'gOS':>6s} {'gLav':>6s} {'gAug':>7s}"
    for name in models:
        print(f"\n== {name} ==\n{hdr}")
        for r in sorted((r for r in rows if r["model"] == name), key=lambda r: (r["split"], r["Re"], r["Ca"])):
            print(f"{r['case']:24s} {r['split'][:4]:4s} {r['S0']:+8.4f} {r['S0_over_friction']:+9.3f} {r['e']:+8.4f} {r['dc_kin']:+7.3f} "
                  f"{r['a']:+8.4f} {r['b']:+8.4f} {r['c']:+8.4f} {r['D_lav']:7.1f} {r['D_aug']:+8.1f} "
                  f"{r['gmax_os']:6.4f} {r['gmax_lav']:6.4f} {r['gmax_aug']:7.4f}")


if __name__ == "__main__":
    main()
