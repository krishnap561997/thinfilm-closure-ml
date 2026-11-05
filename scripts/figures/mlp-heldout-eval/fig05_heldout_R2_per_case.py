"""Held-out R^2 in the nonlinear region per case (the NRMSE figure fig02, restated as R^2)."""
import sys, subprocess, tempfile; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S, _heldout as H

S.apply()
script = S.ROOT / "scripts/mlp_wave_region_summary.py"
res = {}
for grp in ("h_hx", "first_derivatives"):
    for t in H.TARGETS:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            subprocess.run([sys.executable, str(script), "--eval-dir", str(H.G / grp / t / "seed42/eval_heldout"),
                            "--cache", str(S.FMLP / "cache_dir"), "--csv", str(out)], check=True, capture_output=True)
            res[(grp, t)] = pd.read_csv(out).set_index("case")["R2_NL"]
cs = H.cases(); y = np.arange(len(cs))
print("case                      R2_NL dMx (h,hx / h,q,hx,qx)    dTw")
for c in cs: print(f"{c:26s} {res[('h_hx','delta_Mx_ML')][c]:.3f} / {res[('first_derivatives','delta_Mx_ML')][c]:.3f}      {res[('h_hx','delta_Tw_ML')][c]:.3f} / {res[('first_derivatives','delta_Tw_ML')][c]:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
for ax, (t, (lab, col)) in zip(axes, H.TARGETS.items()):
    ax.scatter(res[("h_hx", t)][cs].values, y + 0.1, s=60, color=col, zorder=3)
    ax.scatter(res[("first_derivatives", t)][cs].values, y - 0.1, s=60, color=col, alpha=0.4, zorder=3)
    ax.set_title(lab, loc="left"); ax.set_xlim(0.93, 1.0); ax.set_xlabel("$R^2$ in the nonlinear region (held-out)")
    ax.grid(axis="y", visible=True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=col, label=r"inputs $h,\ h_x$"),
                       Line2D([], [], marker="o", ls="", color=col, alpha=0.4, label=r"inputs $h,\ q,\ h_x,\ q_x$")], loc="upper left", fontsize=8.5)
axes[0].set_yticks(y); axes[0].set_yticklabels([H.nice(c) for c in cs]); axes[0].invert_yaxis()
fig.tight_layout(); S.save(fig, "mlp-heldout-eval", "fig05_heldout_R2_per_case")
