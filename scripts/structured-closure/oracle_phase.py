#!/usr/bin/env python3
"""Phase-shift diagnostic of oracle ROM runs against the DNS (Re120 Ca0.01 beta45, 10 Hz).

At every saved time (stride --every): best-fit x-shift s(t) of h_ROM against h_DNS (cross-correlation
over 20 < x < 760, sub-cell by parabolic peak fit, positive = ROM ahead), the h error before and after
removing the shift (relative to std h_DNS), and the max h ratio. Prints a table at a few times plus a
linear fit of s(t) (drift speed, relative to the DNS celerity c = lambda / T) and writes a figure and a csv.

    python scripts/structured-closure/oracle_phase.py --forcing rom_forcing.h5 --run name=path.h5 ... \
        --out fig.png [--csv out.csv]
"""
import argparse, csv
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#8d99ae", "#000000", "#1d3557", "#e9c46a", "#e76f51", "#6a4c93", "#2a9d8f"]
T_FORCE = 1 / 0.0214      # forcing period (nondimensional)


def xcorr_shift(a, b, maxlag):
    a, b = a - a.mean(), b - b.mean(); n = len(a)
    c = np.array([np.dot(a[maxlag + l: n - maxlag + l], b[maxlag: n - maxlag]) for l in range(-maxlag, maxlag + 1)])
    i = int(np.argmax(c)); d = 0.0
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        d = 0.5 * (c[i - 1] - c[i + 1]) / den if den != 0 else 0.0
    l = i - maxlag
    e = np.sqrt(np.mean((a[maxlag + l: n - maxlag + l] - b[maxlag: n - maxlag]) ** 2)) / b.std()
    return l + d, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv")
    a = ap.parse_args()
    with h5py.File(a.forcing) as f:
        tD, x = f["t"][:], f["x"][:]
        dx = x[1] - x[0]; m = (x > 20) & (x < 760)
        # DNS wavelength from the autocorrelation of the first snapshot
        h0 = f["dns/h"][0][m]; h0 = h0 - h0.mean()
        ac = np.correlate(h0, h0, "full")[len(h0) - 1:]
        lam = (np.argmax(ac[int(30 / dx):int(200 / dx)]) + int(30 / dx)) * dx
        c = lam / T_FORCE
        maxlag = int(0.5 * lam / dx)
        print(f"DNS wavelength {lam:.1f}, celerity c = lambda/T = {c:.3f}; shift range +-{0.5 * lam:.0f}")
        res, rows = {}, []
        for spec in a.run:
            name, path = spec.split("=", 1)
            with h5py.File(path) as r:
                tR = r["time"][:]
                out = []
                for i in range(0, len(tR), a.every):
                    k = int(np.argmin(abs(tD - tR[i])))
                    if abs(tD[k] - tR[i]) > 1e-6: continue
                    hd, hr = f["dns/h"][k][m], r["h"][i][m]
                    s, es = xcorr_shift(hr, hd, maxlag)
                    e = np.sqrt(np.mean((hr - hd) ** 2)) / hd.std()
                    out.append((tR[i] - tD[0], s * dx, e, es, hr.max() / hd.max()))
                    rows.append(dict(run=name, elapsed=out[-1][0], shift=s * dx, err=e, err_shifted=es, hmax_ratio=out[-1][4]))
            res[name] = np.array(out)
    show = [10, 20, 30, 50, 75, 100]
    print(f"\nshift (positive = ROM ahead) | error | error after shift | max h ROM/DNS")
    print(f"{'run':>10s} " + "".join(f"{'t=' + str(t):>24s}" for t in show) + "   drift/c (t<50, t<100)")
    for name, R in res.items():
        line = f"{name:>10s} "
        for t in show:
            j = int(np.argmin(abs(R[:, 0] - t)))
            line += f"  {R[j,1]:+5.1f} {R[j,2]:.2f} {R[j,3]:.2f} {R[j,4]:.2f}"
        sl = [np.polyfit(R[R[:, 0] < T, 0], R[R[:, 0] < T, 1], 1)[0] / c for T in (50, 101)]
        print(line + f"   {sl[0]:+.3f} {sl[1]:+.3f}")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (name, R) in enumerate(res.items()):
        col = COLORS[j % len(COLORS)]; ls = "--" if name == "exact" else "-"
        axs[0].plot(R[:, 0], R[:, 1], ls, color=col, lw=1.6, label=name)
        axs[1].plot(R[:, 0], R[:, 3], ls, color=col, lw=1.6)
        axs[2].plot(R[:, 0], R[:, 4], ls, color=col, lw=1.6)
    axs[0].axhline(0, color="0.6", lw=0.8); axs[2].axhline(1, color="0.6", lw=0.8)
    axs[0].set_ylabel("x-shift of ROM vs DNS (+ = ahead)"); axs[1].set_ylabel("h error after removing shift")
    axs[2].set_ylabel("max h, ROM / DNS")
    for ax in axs: ax.set_xlabel("time since ROM start")
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)
    if a.csv:
        with open(a.csv, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
