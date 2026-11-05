"""DeltaP_x on the two meshes for one wave: raw (grid-scale spikes) and after Gaussian smoothing."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S
from _aligned import aligned

S.apply(); z = aligned(); xi = z["xi"]; k = 30
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharex=True)
for ax, mode, ttl in ((axes[0], "raw", "(a) as extracted"), (axes[1], "smooth", "(b) after Gaussian smoothing (width 1 $h_N$)")):
    for lv, ls, lw in (("L14", "-", 1.6), ("L15", "--", 1.3)):
        ax.plot(xi, z[f"{mode}_delta_Px_ML_{lv}"][k], color=S.AQUA, ls=ls, lw=lw)
    ax.set_title(ttl, loc="left", fontsize=10); ax.set_xlabel(r"$\xi = x - x_{crest}$"); ax.set_xlim(-40, 40)
axes[0].set_ylabel(S.DPX + " (nondimensional)")
axes[0].legend(handles=[Line2D([], [], color=S.AQUA, ls="-", label="level 14"), Line2D([], [], color=S.AQUA, ls="--", label="level 15")], loc="upper left")
fig.suptitle(f"One matched wave (number {k} of 65): spikes sit at the interface and differ between meshes", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); S.save(fig, "mesh-sensitivity", "fig02_dPx_grid_scale_noise")
