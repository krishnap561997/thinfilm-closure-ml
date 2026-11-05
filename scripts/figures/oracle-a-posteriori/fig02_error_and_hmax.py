"""Oracle ROM runs: h error against the DNS and maximum film thickness, versus elapsed time."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import style as S

S.apply()
d = pd.read_csv(S.ORACLE / "rom/summary.csv")
series = [("lavalle", "Lavalle only", S.BLUE, "o", "-"),
          ("Tw", S.DTW + " only", S.ORANGE, "s", "--"),
          ("Tw_Mx", S.DTW + " + " + S.DMX, S.AQUA, "^", "-"),
          ("all", "all four terms", S.YELLOW, "D", "--")]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
for ax, col, ylab in ((axes[0], "h_rel", r"rms error in $h$ / std($h_{DNS}$)"), (axes[1], "hmax_rom", r"maximum of $h/h_N$ over $x$")):
    for key, lab, c, mk, ls in series:
        r = d[d.run == key]
        ax.plot(r.elapsed, r[col], color=c, ls=ls, lw=1.6, marker=mk, ms=5, markevery=max(1, len(r) // 9), label=lab)
        if key in ("Tw", "all"):
            ax.plot(r.elapsed.iloc[-1], r[col].iloc[-1], marker="X", color=c, ms=11, mec=S.INK, mew=1.0, zorder=5)
    ax.set_xlabel("elapsed time since restart (t − 1800)"); ax.set_ylabel(ylab)
r = d[d.run == "lavalle"]
axes[1].plot(r.elapsed, r.hmax_dns, color=S.INK, lw=2.2, label="DNS")
axes[0].set_ylim(0, 1.8); axes[1].set_ylim(1.2, 2.9)
axes[0].set_title("(a) error against the DNS", loc="left"); axes[1].set_title("(b) wave height", loc="left")
axes[0].annotate("X = run crashed\n(hyperbolicity lost)", xy=(7.9, 1.65), xytext=(14, 1.45), fontsize=9, color=S.INK2,
                 arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.8))
axes[1].legend(loc="upper center", ncol=3, fontsize=8.5, columnspacing=1.2, handlelength=1.8)
fig.tight_layout()
S.save(fig, "oracle-a-posteriori", "fig02_error_and_hmax")
