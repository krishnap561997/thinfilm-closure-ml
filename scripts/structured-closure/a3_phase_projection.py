"""A.3: phase projection of the DNS correction F = -(dMx + dTw) (what the ROM adds to the q-equation).

Over the analysis window (x > X_MIN, outlet 5% excluded) and all selected snapshots:
    q' = q - <q>_x (per snapshot),   q'_x = Gaussian derivative (sigma = 0.5 h_N) of q
    F ~ alpha q' + beta q'_x   (least squares; R^2 reported)
alpha > 0: F is in phase with q' (alpha = rate of growth of the <q'^2>/2 budget of the q-equation
alone, in 1/time).  F = beta q'_x acts on the q-equation alone as an advection, dc = -beta.
CAUTION (docs/results/structured-closure.md, A.3): in the coupled h-q system at long wavelength
the roles swap: a q'_x term is the "c" coefficient of plan section 1.1 and changes the growth
(D), while an in-phase q' ~ 3 eta term shifts the kinematic speed.  "dc" is a label, not a speed.  Separate projections alpha_1 = <F q'>/<q'^2>, beta_1 = <F q'_x>/<q'_x^2> are split by
region (contribution of each region's points, in % of the total), regions on smoothed h, h_x, h_xx:
    ripples : h < 1 and |h_xx| > 1.0 std(h_xx)         (checked first)
    front   : h_x < -0.5 std(h_x)     back : h_x > +0.5 std(h_x)
    crest   : h > 1.15 (rest)         substrate : the rest
Data: oracle case from rom_forcing.h5 (every 1 time unit, elapsed 0-100); broadband cases from
ClosureData (every 10 time units, t >= T_MIN).
"""
import argparse, csv, os, sys
import h5py, numpy as np
from scipy.ndimage import gaussian_filter1d as gf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_np import ORACLE

STORE = os.environ["THINFILM_STORE"]
CD = os.path.join(STORE, "datasets/BasC/DNS/param_space/ClosureData")
CASES = ["Re010_Ca0_010_theta45", "Re020_Ca0_010_theta45", "Re040_Ca0_040_theta45",
         "Re080_Ca0_010_theta45", "Re120_Ca0_010_theta45", "Re200_Ca0_010_theta45"]
X_MIN, T_MIN, SIG = 100.0, 500.0, 0.5
REG = ["front", "crest", "back", "ripples", "substrate"]


def analyse(h, q, F, dx):
    """h, q, F: (nt, nx) on the window. Returns dict of results."""
    s = SIG / dx
    hs = gf(h, s, axis=1); hx = gf(h, s, axis=1, order=1) / dx; hxx = gf(h, s, axis=1, order=2) / dx ** 2
    qp = q - q.mean(1, keepdims=True); qx = gf(q, s, axis=1, order=1) / dx
    Fs = gf(F, s, axis=1)                      # same filter as the derivatives
    cut = int(4 * s) + 1                        # drop filter edges
    hs, hx, hxx, qp, qx, Fs = (a[:, cut:-cut].ravel() for a in (hs, hx, hxx, qp, qx, Fs))
    A = np.c_[qp, qx]
    coef, *_ = np.linalg.lstsq(A, Fs, rcond=None)
    r2 = 1 - np.sum((Fs - A @ coef) ** 2) / np.sum((Fs - Fs.mean()) ** 2)
    sx, sxx = hx.std(), hxx.std()
    lab = np.full(hs.shape, "substrate", object)
    rip = (hs < 1) & (np.abs(hxx) > sxx)
    lab[(hx < -0.5 * sx)] = "front"; lab[(hx > 0.5 * sx)] = "back"
    lab[(lab == "substrate") & (hs > 1.15)] = "crest"
    lab[rip] = "ripples"
    qq, xx = np.sum(qp * qp), np.sum(qx * qx)
    out = dict(alpha=coef[0], dc=-coef[1], R2=r2, alpha1=np.sum(Fs * qp) / qq, dc1=-np.sum(Fs * qx) / xx,
               F_rms=Fs.std(), npts=len(Fs))
    for r in REG:
        m = lab == r
        out[f"frac_{r}"] = m.mean()
        out[f"amp_{r}"] = np.sum(Fs[m] * qp[m]) / qq      # contributions; sum = alpha1
        out[f"spd_{r}"] = -np.sum(Fs[m] * qx[m]) / xx      # sum = dc1
    return out


def oracle():
    with h5py.File(os.path.join(ORACLE, "rom/rom_forcing.h5")) as f:
        t, x = f["t"][:], f["x"][:]
        idx = list(range(0, len(t), 100))
        m = (x > X_MIN) & (x < 0.95 * x[-1])
        g = lambda k: np.stack([f[k][i][m] for i in idx])
        return analyse(g("dns/h"), g("dns/q"), -(g("dMx") + g("dTw")), x[1] - x[0])


def broadband(case):
    with h5py.File(os.path.join(CD, f"thinfilm_closure_data_{case}.h5")) as f:
        x, t = f["x"][:], f["t"][:]
        m = (x > X_MIN) & (x < 0.95 * x[-1]); it = np.where(t >= T_MIN)[0]
        g = lambda k: f[k][:, it][m].T
        return analyse(g("h"), g("q"), -(g("delta_Mx_ML") + g("delta_Tw_ML")), x[1] - x[0]), len(it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(STORE, "analysis/structured_closure"))
    A = ap.parse_args()
    ost = {r["case"]: r for r in csv.DictReader(open(os.path.join(STORE, "analysis/orr_sommerfeld/case_table.csv")))}
    rows = [dict(case="Re120_Ca0_010_theta45 (10 Hz oracle)", delta=float(ost["Re120_Ca0_010_theta45"]["delta_lin"]),
                 nt=100, **oracle())]
    for c in CASES:
        r, nt = broadband(c)
        rows.append(dict(case=c, delta=float(ost[c]["delta_lin"]), nt=nt, **r))
    os.makedirs(A.out, exist_ok=True)
    with open(os.path.join(A.out, "a3_phase_projection.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"F = -(dMx+dTw) ~ alpha q' + beta q'_x; x > {X_MIN:g}, broadband t >= {T_MIN:g}")
    print(f"{'case':38s} {'d_lin':>5s} {'F_rms':>7s} {'alpha':>8s} {'dc':>7s} {'R2':>5s} | {'alpha1':>8s} {'dc1':>7s}")
    for r in rows:
        print(f"{r['case']:38s} {r['delta']:5.1f} {r['F_rms']:7.4f} {r['alpha']:+8.4f} {r['dc']:+7.3f} {r['R2']:5.2f} | {r['alpha1']:+8.4f} {r['dc1']:+7.3f}")
    for key, name in (("amp", "amplitude projection alpha1"), ("spd", "speed projection dc1")):
        print(f"\nBy region, contributions to the {name} (sum = total); point fraction in brackets")
        print(f"{'case':38s} " + " ".join(f"{r:>17s}" for r in REG))
        for r in rows:
            print(f"{r['case']:38s} " + " ".join(f"{r[f'{key}_{g}']:+9.4f} ({100 * r[f'frac_{g}']:3.0f}%)" for g in REG))
    print("->", A.out)


if __name__ == "__main__":
    main()
