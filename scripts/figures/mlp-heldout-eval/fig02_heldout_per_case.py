"""Held-out nonlinear-region NRMSE per case: (h, h_x) against (h, q, h_x, q_x)."""
import sys, subprocess, tempfile; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S

S.apply()
G = S.FMLP / "runs/ablation_01/layers1_width16"
script = S.ROOT / "scripts/mlp_wave_region_summary.py"
res = {}
for grp in ("h_hx", "first_derivatives"):
    for t in ("delta_Mx_ML", "delta_Tw_ML"):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "m.csv"
            subprocess.run([sys.executable, str(script), "--eval-dir", str(G / grp / t / "seed42/eval_heldout"),
                            "--cache", str(S.FMLP / "cache_dir"), "--csv", str(out)], check=True, capture_output=True)
            d = pd.read_csv(out); res[(grp, t)] = d.set_index("case")["NL"]
cases = list(res[("h_hx", "delta_Mx_ML")].index)
import re
def nice(c):
    re_, ca, th = re.match(r"Re(\d+)_Ca(\d_\d+)_theta(.+)", c).groups()
    return f"Re{int(re_)} Ca{float(ca.replace('_', '.')):g} β{th.replace('_', '.')}°"
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
x = np.arange(len(cases)); w = 0.38
for ax, t, col, lab in ((axes[0], "delta_Mx_ML", S.BLUE, S.DMX), (axes[1], "delta_Tw_ML", S.ORANGE, S.DTW)):
    ax.bar(x - w / 2, res[("h_hx", t)].values, w - 0.04, color=col, label=r"$h,\ h_x$")
    ax.bar(x + w / 2, res[("first_derivatives", t)].values, w - 0.04, color=col, alpha=0.38, label=r"$h,\ q,\ h_x,\ q_x$")
    ax.set_xticks(x); ax.set_xticklabels([nice(c) for c in cases], rotation=35, ha="right", fontsize=8.5)
    ax.set_title(lab, loc="left"); ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=col, label=r"inputs $h,\ h_x$"), Patch(color=col, alpha=0.38, label=r"inputs $h,\ q,\ h_x,\ q_x$")], loc="upper left", fontsize=8.5)
axes[0].set_ylabel("NRMSE in the nonlinear region (held-out cases)"); axes[0].set_ylim(0, 0.34)
axes[1].text(0.98, 0.97, "Lavalle alone (correction = 0)\nscores about 1.2 (off scale)", transform=axes[1].transAxes, ha="right", va="top", fontsize=8.5, color=S.INK2)
fig.tight_layout()
S.save(fig, "mlp-heldout-eval", "fig02_heldout_per_case")
