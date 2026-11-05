"""Per-snapshot mass residual (no closure involved) against the momentum-budget remainder."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import style as S

S.apply()
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(S.CLOSURE / "budget_check/per_case/*.csv"))])
d["We"] = d.Re * d.Ca
low = d.case.isin({"Re010_Ca0_010_theta45", "Re020_Ca0_010_theta45", "Re080_Ca0_005_theta6_4"})
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.scatter(d.mass_rel[~low], d.rom_E1_over_E0_w1[~low], s=22, color=S.BLUE, alpha=0.65, lw=0, label=f"other 19 cases ({(~low).sum()} snapshots)")
ax.scatter(d.mass_rel[low], d.rom_E1_over_E0_w1[low], s=28, color=S.ORANGE, lw=0, label=f"the 3 cases that do not close ({low.sum()} snapshots)")
rho = spearmanr(d.mass_rel, d.rom_E1_over_E0_w1)[0]
ax.set_xlabel("mass residual  |∂ₜh + qₓ| / |∂ₜh|   (no closure involved)"); ax.set_ylabel("share of the ROM misfit left after adding the exact corrections")
ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
ax.text(0.02, 0.98, f"rank correlation {rho:.2f}, all {len(d)} snapshots", transform=ax.transAxes, va="top", fontsize=9, color=S.INK2)
ax.set_title("Where the mass equation fails, the momentum budget fails too", loc="left")
fig.tight_layout()
S.save(fig, "budget-check", "fig02_mass_residual_flag")
