"""Momentum budget on the ROM grid: Lavalle misfit and what remains after adding the exact corrections."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import style as S

S.apply()
d = pd.read_csv(S.CLOSURE / "budget_check/summary.csv")
d["We"] = d.Re * d.Ca
FAIL = {"Re010_Ca0_010_theta45", "Re020_Ca0_010_theta45", "Re080_Ca0_005_theta6_4"}   # E1 > 4.5% of d_t q
d = d.sort_values(["We", "Re"], ascending=[False, False]).reset_index(drop=True)
lab = [f"Re{r.Re:g} Ca{r.Ca:g} β{r.beta_deg:g}°  (We {r.We:.2g})" for r in d.itertuples()]
y = np.arange(len(d)); h = 0.38
fig, ax = plt.subplots(figsize=(8.6, 7.2))
ax.barh(y + h / 2, d.rom_E0_rel_w1, height=h - 0.04, color=S.BLUE, label="Lavalle ROM misfit  |∂ₜq − RHS| / |∂ₜq|")
ax.barh(y - h / 2, d.rom_E1_rel_w1, height=h - 0.04, color=S.ORANGE, label="left after adding the exact corrections")
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8.5); ax.set_xlabel("rms over 20 < x < 760, t = 1000–2000, 1 $h_N$ smoothing, relative to rms(∂ₜq)")
ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
for yy in y[d.case.isin(FAIL).values]:
    ax.axhspan(yy - 0.5, yy + 0.5, color=S.MUTED, alpha=0.14, lw=0)
ax.text(0.655, y[d.case.isin(FAIL).values].min() - 0.2, "shaded: the three cases that do not close\n(all low We, 0.1–0.4)", ha="right", va="top", fontsize=8.5, color=S.INK2)
ax.set_title("Does the exact correction close the ROM's momentum equation?", loc="left")
fig.tight_layout()
S.save(fig, "budget-check", "fig01_unexplained_by_case")
