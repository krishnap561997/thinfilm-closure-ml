#!/usr/bin/env python3
"""h(x) snapshots of oracle ROM runs against the DNS (Re120 Ca0.01 beta45, 10 Hz).

One row per elapsed time; each run is drawn over the DNS in the window x0 < x < x1. Prints, per run
and time, the best-fit x-shift of the ROM against the DNS (cross-correlation over 20 < x < 760,
positive = ROM ahead) and the h error after removing that shift.

    python scripts/structured-closure/oracle_snapshots.py --forcing rom_forcing.h5 \
        --run exact=.../thinfilm_training_data.h5 --run mlp=... --out fig.png
"""
import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2a9d8f", "#e76f51", "#1d3557", "#e9c46a", "#8d99ae", "#6a4c93"]


def shift(hr, hd, dx, maxlag):
    a, b = hr - hr.mean(), hd - hd.mean()
    lags = np.arange(-maxlag, maxlag + 1)
    c = [np.dot(a[maxlag + l: len(a) - maxlag + l], b[maxlag: len(b) - maxlag]) for l in lags]
    l = lags[int(np.argmax(c))]
    e = np.sqrt(np.mean((a[maxlag + l: len(a) - maxlag + l] - b[maxlag: len(b) - maxlag]) ** 2)) / b.std()
    return l * dx, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--times", default="10,50,100")
    ap.add_argument("--x0", type=float, default=400.0)
    ap.add_argument("--x1", type=float, default=700.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    times = [float(s) for s in a.times.split(",")]
    with h5py.File(a.forcing) as f:
        tD, x = f["t"][:], f["x"][:]
        kD = [int(np.argmin(abs(tD - (tD[0] + t)))) for t in times]
        HD = [f["dns/h"][k] for k in kD]
    dx = x[1] - x[0]; m = (x > 20) & (x < 760); w = (x > a.x0) & (x < a.x1); maxlag = int(40 / dx)
    runs = []
    for spec in a.run:
        name, path = spec.split("=", 1)
        with h5py.File(path) as r:
            tR = r["time"][:]
            runs.append((name, [r["h"][int(np.argmin(abs(tR - tD[k])))] for k in kD]))
    n = len(runs)
    fig, axs = plt.subplots(len(times), n, figsize=(4.2 * n, 2.4 * len(times)), sharex=True, sharey=True, squeeze=False)
    print(f"{'run':>10s} " + " ".join(f"t={t:>5.0f}: shift  err|err_shifted" for t in times))
    for j, (name, H) in enumerate(runs):
        line = f"{name:>10s} "
        for i, t in enumerate(times):
            ax = axs[i, j]
            s, es = shift(H[i][m], HD[i][m], dx, maxlag)
            e = np.sqrt(np.mean((H[i][m] - HD[i][m]) ** 2)) / HD[i][m].std()
            line += f"     {s:+6.1f}  {e:.2f}|{es:.2f}           "
            ax.plot(x[w], HD[i][w], color="k", lw=1.4, label="DNS")
            ax.plot(x[w], H[i][w], color=COLORS[j % len(COLORS)], lw=1.4, label=name)
            ax.text(0.01, 0.95, f"err {e:.2f}, shift {s:+.1f}", transform=ax.transAxes, va="top", fontsize=8)
            if i == 0: ax.set_title(name)
            if j == 0: ax.set_ylabel(f"h, t = {t:.0f}")
        print(line)
    for ax in axs[-1]: ax.set_xlabel("x")
    axs[0, 0].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
