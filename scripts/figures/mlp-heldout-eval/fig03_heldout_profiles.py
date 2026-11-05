"""Predicted against DNS DeltaM_x and DeltaT_w profiles for one easy and one hard held-out case."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
G = S.FMLP / "runs/ablation_01/layers1_width16"
cases = [("Re120_Ca0_010_theta45", "Re120 Ca0.01 β45° (easy)"), ("Re200_Ca0_010_theta6_4", "Re200 Ca0.01 β6.4° (hard for " + S.DMX + ")")]
targets = [("delta_Mx_ML", S.DMX, S.BLUE), ("delta_Tw_ML", S.DTW, S.ORANGE)]
def load(grp, t, case, ti0=None):
    with h5py.File(G / grp / t / "seed42/eval_heldout" / f"{case}.h5") as g:
        P, Y, idx, x = g["prediction"][:, 0], g["truth"][:, 0], g["indices"][:], g["x"][:]
    ti = np.unique(idx[:, 0]); ti0 = ti[int(0.8 * len(ti))] if ti0 is None else ti0
    m = idx[:, 0] == ti0; o = np.argsort(idx[m, 1])
    return ti0, x[idx[m, 1][o]], Y[m][o], P[m][o]
fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.8), sharex=True)
for j, (case, title) in enumerate(cases):
    ti0 = None
    for i, (t, lab, col) in enumerate(targets):
        ti0, x, y, p_hh = load("h_hx", t, case, ti0)
        _, _, _, p_fd = load("first_derivatives", t, case, ti0)
        ax = axes[i, j]; m = (x > 300) & (x < 420)
        ax.plot(x[m], y[m], color=S.INK, lw=2.2, label="DNS")
        ax.plot(x[m], p_fd[m], color=col, lw=1.4, ls="--", label=r"MLP, inputs $h,\ q,\ h_x,\ q_x$")
        ax.plot(x[m], p_hh[m], color=col, lw=1.6, label=r"MLP, inputs $h,\ h_x$")
        ax.set_ylabel(lab)
        if i == 0: ax.set_title(f"{title}", loc="left")
        if i == 1: ax.set_xlabel(S.XN)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=S.INK, lw=2.2, label="DNS"), Line2D([], [], color=S.INK2, lw=1.4, ls="--", label=r"MLP, inputs $h,\ q,\ h_x,\ q_x$"), Line2D([], [], color=S.INK2, lw=1.6, label=r"MLP, inputs $h,\ h_x$")], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Held-out cases, one snapshot each (window 300 < x < 420); the MLPs never saw these cases", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 1))
S.save(fig, "mlp-heldout-eval", "fig03_heldout_profiles")
