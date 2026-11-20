"""APS DFD 2026 e-poster, Panel 3: oracle-race animation.

Oracle case (Re 120, Ca 0.01, beta 45, 10 Hz forcing). All ROMs start from the same DNS state at t = 1800 and run
100 time units with the DNS inlet. Three stacked strips (Lavalle, plain MLP, constrained MLP), each over the DNS h(x).

    source env.sh; python scripts/meetings/aps_dfd_2026_anim.py [--out DIR] [--x0 400 --x1 780] [--fps 30]
"""
import argparse, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, os.path.dirname(__file__))
from aps_dfd_2026_figs import ROOT, SC, ORACLE, MODEL, LAB, NAVY, GRAY, ORANGE, TEAL, style

RUNS = [(os.path.join(ORACLE, "runs/lavalle/thinfilm_training_data.h5"), LAB["lav"], GRAY),
        (os.path.join(ORACLE, "runs/ml_sum_hhx/thinfilm_training_data.h5"), LAB["plain"], ORANGE),
        (os.path.join(SC, "rom", MODEL, "oracle/thinfilm_training_data.h5"), LAB["cons"], TEAL)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/meetings/2026-11-aps-dfd/poster_figs"))
    ap.add_argument("--x0", type=float, default=400); ap.add_argument("--x1", type=float, default=780)
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds to hold the last frame")
    a = ap.parse_args()

    with h5py.File(RUNS[0][0]) as f:
        tR, x = f["time"][:], f["x"][:]
    w = np.where((x >= a.x0) & (x <= a.x1))[0]; xs = x[w]
    kR = np.arange(0, len(tR), a.stride)
    H = []
    for fn, _, _ in RUNS:
        with h5py.File(fn) as f:
            assert np.allclose(f["time"][:], tR) and np.allclose(f["x"][:], x), fn
            H.append(f["h"][:, w[0]:w[-1] + 1][kR])
    with h5py.File(os.path.join(ORACLE, "rom_forcing.h5")) as f:
        tD, xD = f["t"][:], f["x"][:]
        assert np.allclose(xD, x)
        kD = np.unique([int(np.argmin(abs(tD - t))) for t in tR[kR]])
        hD_all = f["dns/h"][kD, w[0]:w[-1] + 1]
    # DNS row for each ROM frame
    iD = np.array([int(np.argmin(abs(tD[kD] - t))) for t in tR[kR]])
    print(f"{len(kR)} frames, ROM t {tR[0]:.1f}-{tR[-1]:.1f}, max |t_DNS - t_ROM| = {np.max(abs(tD[kD][iD] - tR[kR])):.3f}")

    style(18)
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.10, hspace=0.18)
    lD, lR = [], []
    for ax, (fn, lab, c) in zip(axs, RUNS):
        d, = ax.plot(xs, hD_all[iD[0]], color=NAVY, lw=2.4)
        r, = ax.plot(xs, H[len(lR)][0], color=c, lw=2.4)
        ax.set_ylim(0.3, 2.8); ax.set_yticks([0.5, 1.5, 2.5]); ax.set_ylabel("h")
        for sp in ("top", "right"): ax.spines[sp].set_visible(True)
        ax.text(0.012, 0.93, lab, transform=ax.transAxes, color=c, fontsize=19, fontweight="bold", va="top")
        lD.append(d); lR.append(r)
    axs[0].text(0.988, 0.93, "DNS", transform=axs[0].transAxes, color=NAVY, fontsize=19, fontweight="bold", va="top", ha="right")
    axs[-1].set_xlabel("x (flow →)")
    ttl = fig.text(0.075, 0.945, "", fontsize=20, va="center")
    fig.text(0.985, 0.945, "Re 120, Ca 0.01, β 45°, f = 10 Hz", fontsize=14, color="#555555",
             ha="right", va="center")

    os.makedirs(a.out, exist_ok=True); fn = os.path.join(a.out, "p3_oracle_race.mp4")
    writer = FFMpegWriter(fps=a.fps, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    nhold = int(a.hold * a.fps)
    with writer.saving(fig, fn, dpi=150):
        for j in range(len(kR) + nhold):
            k = min(j, len(kR) - 1)
            for i in range(3):
                lD[i].set_ydata(hD_all[iD[k]]); lR[i].set_ydata(H[i][k])
            ttl.set_text(f"t = {tR[kR][k] - tD[0]:5.1f}")
            writer.grab_frame()
    print("wrote", fn)
    # still of the last frame for the poster PDF / backup
    fig.savefig(fn.replace(".mp4", "_last.png"), dpi=150); print("wrote", fn.replace(".mp4", "_last.png"))


if __name__ == "__main__":
    main()
