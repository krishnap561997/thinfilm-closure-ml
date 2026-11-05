"""Validation R^2 for each input set of the ablation (the NRMSE figure fig01, restated as R^2)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, numpy as np, matplotlib.pyplot as plt
import style as S, _heldout as H

S.apply()
fields = H.groups()
r2 = lambda g, t: 1 - float(H.val_nrmse_curve(g, t).min() ** 2)        # 1 - normalised validation MSE
exact = {}
for t in H.TARGETS:
    for row in csv.DictReader((H.G / "first_derivatives" / t / "seed42/eval_val/metrics.csv").open()):
        if row["case"] == "POOLED": exact[t] = float(row["R2"])
groups = [g for g in fields if g != "local_baseline"]
groups.sort(key=lambda g: r2(g, "delta_Mx_ML"))
y = np.arange(len(groups))
print("group                 R2 dMx   R2 dTw")
for g in groups: print(f"{g:28s} {r2(g, 'delta_Mx_ML'):.4f}  {r2(g, 'delta_Tw_ML'):.4f}")
print("local_baseline", f"{r2('local_baseline', 'delta_Mx_ML'):.3f}  {r2('local_baseline', 'delta_Tw_ML'):.3f}")
print("exact pooled val R2 (first_derivatives):", exact, " approx:", {t: round(r2('first_derivatives', t), 4) for t in H.TARGETS})
fig, ax = plt.subplots(figsize=(8.0, 4.6))
ax.scatter([r2(g, "delta_Mx_ML") for g in groups], y + 0.12, s=60, color=S.BLUE, label=S.DMX, zorder=3)
ax.scatter([r2(g, "delta_Tw_ML") for g in groups], y - 0.12, s=55, marker="s", color=S.ORANGE, label=S.DTW, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([H.label(fields[g]) for g in groups])
for g in ("h_hx", "first_derivatives"):
    i = groups.index(g)
    ax.text(r2(g, "delta_Mx_ML") + 0.0008, i + 0.12, f"{r2(g, 'delta_Mx_ML'):.3f}", va="center", fontsize=8.5, color=S.INK2)
    ax.text(r2(g, "delta_Tw_ML") + 0.0008, i - 0.12, f"{r2(g, 'delta_Tw_ML'):.3f}", va="center", fontsize=8.5, color=S.INK2)
ax.set_xlim(0.925, 0.985); ax.grid(axis="y", visible=True); ax.set_xlabel("validation $R^2$ after 100 epochs")
ax.set_title("Inputs of the 1×16 MLP (plus Re, Ca, β in every case)", loc="left")
ax.text(0.925 + 0.0005, -0.85, f"h, q only: $R^2$ = {r2('local_baseline','delta_Mx_ML'):.2f} / {r2('local_baseline','delta_Tw_ML'):.2f} (off scale)", fontsize=8.5, color=S.INK2)
ax.text(0.984, -0.85, "$R^2 = 1 -$ normalised validation MSE;\nexact pooled $R^2$ of the $h,q,h_x,q_x$ model: "
        f"{exact['delta_Mx_ML']:.3f} / {exact['delta_Tw_ML']:.3f}", fontsize=8, ha="right", color=S.INK2)
ax.set_ylim(-1.3, len(groups) - 0.4); ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=9)
fig.tight_layout(); S.save(fig, "mlp-heldout-eval", "fig04_validation_R2_by_input_set")
