"""Where the held-out errors sit: residual profiles of the worst cases and a pooled parity plot (DeltaM_x)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S, _heldout as H

S.apply()
t = "delta_Mx_ML"; lab, col = H.TARGETS[t]
worst = ["Re080_Ca0_005_theta6_4", "Re200_Ca0_010_theta6_4"]
fig = plt.figure(figsize=(10.8, 6.0)); gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
for i, c in enumerate(worst):
    ax = fig.add_subplot(gs[i, 0])
    d = {g: H.load(g, t, c) for g in ("h_hx", "first_derivatives")}
    ti = np.unique(d["h_hx"]["ti"]); ti0 = ti[int(0.8 * len(ti))]
    for g, ls, a in (("first_derivatives", "--", 0.6), ("h_hx", "-", 1.0)):
        m = (d[g]["ti"] == ti0) & (d[g]["x"] > 300) & (d[g]["x"] < 420); o = np.argsort(d[g]["x"][m])
        r = ((d[g]["P"] - d[g]["Y"])[m][o]) / d[g]["sigma"]
        ax.plot(d[g]["x"][m][o], r, color=col, ls=ls, lw=1.5, alpha=a, label=r"inputs $h,\ h_x$" if g == "h_hx" else r"inputs $h,\ q,\ h_x,\ q_x$")
        print(f"{H.nice(c)} {g}: rms residual/sigma in window {np.sqrt(np.mean(r**2)):.3f}, max {np.abs(r).max():.2f}")
    ax.axhline(0, color=S.INK, lw=0.8)
    ax.set_title(f"{H.nice(c)}: residual (prediction − DNS) / std, one snapshot", loc="left", fontsize=10)
    ax.set_ylabel("residual / std of the case")
    if i == 0: ax.legend(loc="upper right", fontsize=8.5)
    if i == 1: ax.set_xlabel(S.XN)
ax = fig.add_subplot(gs[:, 1]); rng = np.random.default_rng(0)
for c in H.cases():
    d = H.load("h_hx", t, c); nl = d["x"] > 100
    sel = rng.choice(np.flatnonzero(nl), size=min(2500, int(nl.sum())), replace=False)
    hi = c in worst
    ax.scatter(d["Y"][sel] / d["sigma"], d["P"][sel] / d["sigma"], s=4 if not hi else 6, color=col if hi else S.MUTED, alpha=0.35 if hi else 0.18, lw=0, rasterized=True)
lim = np.array([-6, 6]); ax.plot(lim, lim, color=S.INK, lw=1.0)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("DNS " + lab + " / std"); ax.set_ylabel("predicted " + lab + " / std")
ax.set_title("all 7 held-out cases, x > 100, 2500 random points each\n(inputs $h, h_x$)", loc="left", fontsize=10)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=col, label="the two β = 6.4° cases"), Line2D([], [], marker="o", ls="", color=S.MUTED, label="other five")], loc="upper left", fontsize=8.5)
fig.tight_layout(); S.save(fig, "mlp-heldout-eval", "limit03_worst_case_errors", png_only=True, dpi=170)
