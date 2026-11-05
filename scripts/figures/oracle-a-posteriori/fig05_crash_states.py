"""Last saved state of the two runs that lose hyperbolicity, against the DNS at the same time."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
T0 = 1800.01
cases = [("Tw", S.DTW + " only", S.ORANGE), ("all", "all four terms", S.YELLOW)]
fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.6))
with h5py.File(S.ORACLE / "rom/rom_forcing.h5") as F:
    tD, x = F["t"][:], F["x"][:]
    for ax, (key, lab, c) in zip(axes, cases):
        with h5py.File(S.ORACLE / f"rom/runs/{key}/thinfilm_training_data.h5") as f:
            t = f["time"][:]; h = f["h"][-1]; tl = t[-1]
        hd = F["dns/h"][int(np.argmin(abs(tD - tl)))]
        j = int(np.argmin(h))
        ax.plot(x, hd, color=S.INK, lw=1.8, label="DNS")
        ax.plot(x, h, color=c, lw=1.2, label=lab)
        ax.plot(x[j], h[j], marker="X", color=c, mec=S.INK, ms=11, zorder=5)
        ax.axvspan(760, 800, color=S.MUTED, alpha=0.15, lw=0)
        ax.text(758, 2.45, "outlet region (excluded\nfrom the error statistics)", ha="right", va="top", fontsize=8.5, color=S.INK2)
        xt = x[j] + 25 if x[j] < 700 else x[j] - 230
        ax.annotate(f"min h = {h[j]:.2f} at x = {x[j]:.0f}", xy=(x[j], h[j]), xytext=(xt, 0.2), fontsize=9, color=S.INK2,
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.8))
        ax.set_title(f"{lab}: last saved state, elapsed time {tl - T0:.1f}", loc="left")
        ax.set_ylabel(S.HN); ax.set_xlim(0, 800); ax.set_ylim(-0.05, 2.6); ax.legend(loc="upper left", ncol=2)
axes[1].set_xlabel(S.XN)
fig.tight_layout()
S.save(fig, "oracle-a-posteriori", "fig05_crash_states")
