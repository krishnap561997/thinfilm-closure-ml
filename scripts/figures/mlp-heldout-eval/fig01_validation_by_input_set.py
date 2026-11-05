"""Validation error of the 1x16 MLP for each input set of the ablation (both targets)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv, json, numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
G = S.FMLP / "runs/ablation_01/layers1_width16"
fields = {r["input_group"]: r["input_fields"] for r in json.loads((G / "sweep.json").read_text())["runs"]}
def val(g, t):
    rows = list(csv.DictReader((G / g / t / "seed42/history.csv").open()))
    return float(np.sqrt(min(float(r["val_loss"]) for r in rows)))
tex = {"h": "h", "q": "q", "h_x": "h_x", "h_xx": "h_{xx}", "q_x": "q_x", "q_xx": "q_{xx}"}
lab = lambda g: "$" + ",\\ ".join(tex[f] for f in fields[g]) + "$"
groups = [g for g in fields if g != "local_baseline"]
groups.sort(key=lambda g: -val(g, "delta_Mx_ML"))
y = np.arange(len(groups)); h = 0.38
fig, ax = plt.subplots(figsize=(8.0, 4.6))
mx = [val(g, "delta_Mx_ML") for g in groups]; tw = [val(g, "delta_Tw_ML") for g in groups]
ax.barh(y + h / 2, mx, height=h - 0.04, color=S.BLUE, label=S.DMX)
ax.barh(y - h / 2, tw, height=h - 0.04, color=S.ORANGE, label=S.DTW)
ax.set_yticks(y); ax.set_yticklabels([lab(g) for g in groups])
for g, i in ((g, i) for i, g in enumerate(groups) if g in ("h_hx", "first_derivatives")):
    ax.text(val(g, "delta_Mx_ML") + 0.004, i + h / 2, f"{val(g, 'delta_Mx_ML'):.3f}", va="center", fontsize=8.5, color=S.INK2)
    ax.text(val(g, "delta_Tw_ML") + 0.004, i - h / 2, f"{val(g, 'delta_Tw_ML'):.3f}", va="center", fontsize=8.5, color=S.INK2)
ax.set_xlim(0, 0.30); ax.grid(axis="y", visible=False)
ax.set_xlabel("validation NRMSE after 100 epochs (1.0 = predicting the mean)")
ax.set_title("Inputs of the 1×16 MLP (plus Re, Ca, β in every case)", loc="left")
ax.text(0.298, -0.9, f"h, q only: {val('local_baseline','delta_Mx_ML'):.2f} / {val('local_baseline','delta_Tw_ML'):.2f} (off scale)", ha="right", fontsize=8.5, color=S.INK2)
ax.set_ylim(-1.2, len(groups) - 0.4); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
fig.tight_layout()
S.save(fig, "mlp-heldout-eval", "fig01_validation_by_input_set")
