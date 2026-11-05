"""What is not mesh-converged: level 15 against level 14, per closure, and the budget on the two meshes."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
import style as S

S.apply()
M = S.CLOSURE / "mesh_study"
a = pd.read_csv(M / "compare/aligned_summary.csv"); b = pd.read_csv(M / "budget_summary.csv").set_index("case")
fields = [("delta_Mx_ML", S.DMX, S.BLUE), ("delta_Tw_ML", S.DTW, S.ORANGE), ("delta_Px_ML", S.DPX, S.AQUA), ("delta_Dx_ML", S.DDX, S.YELLOW), ("S_ML", r"$S_{ML}$", S.VIOLET)]
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]; x = np.arange(len(fields)); w = 0.38
for i, (f, lab, col) in enumerate(fields):
    raw = float(a[(a.field == f) & (a["mode"] == "raw")].relL2_med.iloc[0]); sm = float(a[(a.field == f) & (a["mode"] == "smooth")].relL2_med.iloc[0])
    ax.bar(i - w / 2, raw, w - 0.04, color=col); ax.bar(i + w / 2, sm, w - 0.04, color=col, alpha=0.4)
    print(f"{f:14s} raw relL2 {raw:.3f}  smoothed {sm:.3f}")
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in fields]); ax.grid(axis="x", visible=False)
ax.set_ylabel("median relative L2 difference, level 15 against 14\n(crest-aligned, 65 waves)")
ax.legend(handles=[Patch(color=S.INK2, label="as extracted"), Patch(color=S.INK2, alpha=0.4, label="after Gaussian smoothing (1 $h_N$)")], loc="upper left", fontsize=8.5)
ax.set_title("(a) closures", loc="left", fontsize=10)
ax = axes[1]; keys = [("mass_rel", "mass residual"), ("dns_res_rel", "DNS momentum\nresidual"), ("rom_E1_over_E0_w1", "ROM misfit\nleft (E1/E0)")]
for i, (k, lab) in enumerate(keys):
    v14, v15 = float(b.loc["closure_L800", k]), float(b.loc["closure_L800_Level15", k])
    ax.bar(i - w / 2, v14, w - 0.04, color=S.INK2); ax.bar(i + w / 2, v15, w - 0.04, color=S.INK2, alpha=0.4)
    print(f"{k:20s} L14 {v14:.3f}  L15 {v15:.3f}")
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in keys], fontsize=9); ax.grid(axis="x", visible=False)
ax.legend(handles=[Patch(color=S.INK2, label="level 14"), Patch(color=S.INK2, alpha=0.4, label="level 15")], loc="upper left", fontsize=8.5)
ax.set_title("(b) budget check, t = 550–650", loc="left", fontsize=10)
fig.tight_layout(); S.save(fig, "mesh-sensitivity", "limit01_not_converged")
