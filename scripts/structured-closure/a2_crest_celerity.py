"""A.2: crest celerity against crest height, DNS and ROM runs of the oracle case
(Re120 Ca0.01 b45, 10 Hz, t = 1800-1900).

Crests: main crest of each wave (maxima with h > H_MIN, prominence >= PROM, spacing >= 0.6 LAMBDA), sub-cell position by parabolic fit, in X_MIN < x < X_MAX.
Celerity: each crest matched to the nearest crest downstream after DT (displacement 0 .. 4 DT),
c = dx/DT, height = mean of the two crest heights.  Prints binned c(h_max) per run and a linear
fit c = c0 + s (h_max - 1); writes a2_crests.csv and a figure to --out.
"""
import argparse, os, sys
import h5py, numpy as np
from scipy.signal import find_peaks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_np import ORACLE

T0, DT, PROM, X_MIN, X_MAX = 1800.0, 1.0, 0.1, 100.0, 750.0
LAMBDA, H_MIN = 87.0, 1.05      # 10 Hz wavelength (OS: 2 pi/k_r = 2 pi/0.0725); main crests only
RUNS = {"lavalle": "Lavalle", "Tw_Mx": "oracle dTw+dMx", "ml_sum_hhx": "ML sum (h,h_x)", "ml_mx_tw": "ML sep. (h,q,h_x,q_x)"}


def crests(h, x):
    p, _ = find_peaks(h, height=H_MIN, prominence=PROM, distance=int(0.6 * LAMBDA / (x[1] - x[0])))
    p = p[(p > 0) & (p < len(h) - 1)]
    a, b, c = h[p - 1], h[p], h[p + 1]
    den = a - 2 * b + c
    off = np.where(den != 0, 0.5 * (a - c) / den, 0.0)
    xp = x[p] + off * (x[1] - x[0])
    hp = b - 0.25 * (a - c) * off
    m = (xp > X_MIN) & (xp < X_MAX)
    return xp[m], hp[m]


def pairs(H, x, times):
    out = []
    for i in range(len(times) - 1):
        x0, h0 = crests(H[i], x); x1, h1 = crests(H[i + 1], x)
        dt = times[i + 1] - times[i]
        for xa, ha in zip(x0, h0):
            d = x1 - xa
            ok = (d > 0) & (d < 4 * dt)
            if ok.any():
                j = np.where(ok)[0][np.argmin(d[ok])]
                out.append((times[i] - T0, xa, 0.5 * (ha + h1[j]), d[j] / dt))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ["THINFILM_STORE"], "analysis/structured_closure"))
    A = ap.parse_args()
    res = {}
    with h5py.File(os.path.join(ORACLE, "rom/rom_forcing.h5")) as f:
        t, x = f["t"][:], f["x"][:]
        idx = [int(np.argmin(abs(t - (T0 + e)))) for e in np.arange(0, 100 + 1e-9, DT)]
        res["DNS"] = pairs(np.stack([f["dns/h"][i] for i in idx]), x, t[idx])
    for d, lab in RUNS.items():
        with h5py.File(os.path.join(ORACLE, f"rom/runs/{d}/thinfilm_training_data.h5")) as f:
            t, xr = f["time"][:], f["x"][:]
            idx = [int(np.argmin(abs(t - (T0 + e)))) for e in np.arange(0, t[-1] - T0 + 1e-9, DT)]
            res[lab] = pairs(np.stack([f["h"][i] for i in idx]), xr, t[idx])
    bins = np.arange(1.0, 2.7, 0.15)
    print(f"crest celerity c(h_max), x in ({X_MIN:g}, {X_MAX:g}), DT = {DT:g}, all elapsed times 0-100")
    print(f"{'run':24s} {'n':>5s} {'c0':>6s} {'slope':>6s} " + " ".join(f"{b:.2f}" for b in bins[:-1]))
    rows = []
    for lab, P in res.items():
        if len(P) == 0:
            print(lab, "no crests"); continue
        hm, c = P[:, 2], P[:, 3]
        s, c0 = np.polyfit(hm - 1, c, 1)
        cells = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (hm >= lo) & (hm < hi)
            cells.append(f"{np.median(c[m]):7.3f}" if m.sum() >= 10 else "      -")
        print(f"{lab:24s} {len(P):5d} {c0:6.3f} {s:6.3f} " + " ".join(cells))
        rows += [(lab, *r) for r in P]
    os.makedirs(A.out, exist_ok=True)
    with open(os.path.join(A.out, "a2_crests.csv"), "w") as f:
        f.write("run,elapsed,x,h_max,celerity\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.5f},{r[4]:.5f}\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for lab, P in res.items():
        if len(P):
            ax.scatter(P[:, 2], P[:, 3], s=3, alpha=0.35, label=lab)
    ax.set_xlabel("crest height $h_{max}$"); ax.set_ylabel("crest celerity"); ax.legend(markerscale=4, fontsize=8)
    ax.set_title("Re120 Ca0.01 β45, 10 Hz, elapsed 0–100")
    fig.tight_layout(); fig.savefig(os.path.join(A.out, "a2_crest_celerity.png"), dpi=150)
    print("->", A.out)


if __name__ == "__main__":
    main()
