"""Crest-aligned DeltaM_x and DeltaT_w, level 14 against level 15 (median and 25-75% band over 65 waves)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S
from _aligned import aligned

S.apply(); z = aligned(); xi = z["xi"]
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
for ax, f, col, lab in ((axes[0], "delta_Mx_ML", S.BLUE, S.DMX), (axes[1], "delta_Tw_ML", S.ORANGE, S.DTW)):
    for lv, ls, lw, al in (("L14", "-", 4.0, 0.45), ("L15", "--", 1.3, 1.0)):
        a = z[f"raw_{f}_{lv}"]; med = np.median(a, 0)
        ax.plot(xi, med, color=col, ls=ls, lw=lw, alpha=al)
        if lv == "L14": ax.fill_between(xi, np.percentile(a, 25, 0), np.percentile(a, 75, 0), color=col, alpha=0.18, lw=0)
    ax.set_xlabel(r"$\xi = x - x_{crest}$"); ax.set_title(lab, loc="left"); ax.set_xlim(-40, 40)
axes[0].set_ylabel("correction (nondimensional)")
axes[0].legend(handles=[Line2D([], [], color=S.INK2, ls="-", lw=4, alpha=0.45, label="level 14 (band: 25–75% of 65 waves)"),
                        Line2D([], [], color=S.INK2, ls="--", label="level 15")], loc="upper left", fontsize=8.5)
fig.suptitle("Re120 Ca0.01 β45°, t = 550–650: crest-aligned closures (as extracted) on the two meshes, medians over 65 waves", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); S.save(fig, "mesh-sensitivity", "fig01_crest_aligned_closures")
