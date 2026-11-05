"""Validation error against epoch for every input set: is the training converged, and what never learns?"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
import style as S, _heldout as H

S.apply()
fields = H.groups()
HL = {"local_baseline": ("--", None), "h_hx": ("-", None), "first_derivatives": ("-.", None), "full_derivatives": (":", None)}
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
print("group                          NRMSE at epoch 90   epoch 100   drop over last 10 epochs")
for ax, (t, (lab, col)) in zip(axes, H.TARGETS.items()):
    for g in fields:
        v = H.val_nrmse_curve(g, t); ep = np.arange(1, len(v) + 1)
        if g in HL:
            ls = HL[g][0]; ax.plot(ep, v, color=col if g != "local_baseline" else S.INK, ls=ls, lw=1.8, label=H.label(fields[g]), zorder=3)
        else:
            ax.plot(ep, v, color=S.MUTED, lw=0.9, alpha=0.7, zorder=2)
        if t == "delta_Mx_ML": print(f"{g:30s} {v[89]:.4f}             {v[99]:.4f}     {v[89] - v[99]:.4f}")
    ax.set_xlabel("epoch"); ax.set_title(lab, loc="left"); ax.set_yscale("log"); ax.set_ylim(0.12, 1.2)
    ax.set_yticks([0.15, 0.2, 0.3, 0.5, 1.0]); ax.set_yticklabels(["0.15", "0.2", "0.3", "0.5", "1.0"]); ax.minorticks_off()
    ax.axvline(100, color=S.MUTED, lw=0.8)
axes[0].set_ylabel("validation NRMSE (log scale)")
axes[0].text(50, 1.02, "no gradients: never learns", fontsize=8.5, color=S.INK2, ha="center", va="bottom")
fig.text(0.01, 0.0, "grey: the other input sets. Training stopped at 100 epochs without early stopping; the best epoch is the last one for every set except the one without gradients.", fontsize=8.5, color=S.INK2)
from matplotlib.lines import Line2D
lh = [Line2D([], [], color=S.INK2, ls=HL[g][0], lw=1.8, label=H.label(fields[g])) for g in HL] + [Line2D([], [], color=S.MUTED, lw=1.0, label="other input sets")]
fig.legend(handles=lh, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.04))
fig.tight_layout(rect=(0, 0.13, 1, 1)); S.save(fig, "mlp-heldout-eval", "limit02_learning_curves")
