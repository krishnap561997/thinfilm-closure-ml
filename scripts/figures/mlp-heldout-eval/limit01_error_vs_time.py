"""Held-out error against time: early transient (Re120 Ca0.02 b90) against flat misfit (beta = 6.4 deg cases)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
import style as S, _heldout as H

S.apply()
HL = {"Re120_Ca0_020_theta90": ("-", "o"), "Re080_Ca0_005_theta6_4": ("--", "s"), "Re200_Ca0_010_theta6_4": (":", "^")}
edges = [0, 500, 1000, 1500, 2001]; names = ["t < 500", "500–1000", "1000–1500", "1500–2000"]
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
for ax, (t, (lab, col)) in zip(axes, H.TARGETS.items()):
    for c in H.cases():
        d = H.load("first_derivatives", t, c); right = d["x"] > 100
        v = [np.sqrt(np.mean(((d["P"] - d["Y"])[right & (d["t"] >= a) & (d["t"] < b)]) ** 2)) / d["sigma"] if (right & (d["t"] >= a) & (d["t"] < b)).any() else np.nan
             for a, b in zip(edges[:-1], edges[1:])]
        if c in HL:
            ls, mk = HL[c]; ax.plot(range(4), v, color=col, ls=ls, marker=mk, lw=1.8, ms=6, label=H.nice(c), zorder=3)
            print(f"{lab:14s} {H.nice(c):22s} " + "  ".join(f"{x:.3f}" for x in v))
        else:
            ax.plot(range(4), v, color=S.MUTED, lw=1.0, alpha=0.7, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_xlabel("time bin (the snapshots are every 10 time units)")
    ax.set_title(lab, loc="left"); ax.legend(loc="upper right", fontsize=8.5)
axes[0].set_ylabel("NRMSE for x > 100 (RMSE / std of the case)")
fig.text(0.01, 0.0, "grey: the other four held-out cases. Model: inputs $h, q, h_x, q_x$.", fontsize=8.5, color=S.INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); S.save(fig, "mlp-heldout-eval", "limit01_error_vs_time")
