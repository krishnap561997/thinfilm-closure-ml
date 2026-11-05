"""Position of matched crests on the two meshes: the offset grows downstream (wave speed +1.2%)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import style as S

S.apply()
d = pd.read_csv(S.CLOSURE / "mesh_study/compare/matched_waves.csv")
off = d.x_L15 - d.x_L14
fig, ax = plt.subplots(figsize=(6.6, 4.0))
ax.scatter(d.x_L14, off, s=22, color=S.BLUE, alpha=0.7, lw=0)
c = np.polyfit(d.x_L14, off, 1); xx = np.array([d.x_L14.min(), d.x_L14.max()])
ax.plot(xx, np.polyval(c, xx), color=S.INK, lw=1.2, ls="--", label=f"linear fit, slope {c[0]:.3f} (offset per unit distance)")
ax.set_xlabel("crest position on level 14, $x/h_N$"); ax.set_ylabel(r"crest offset, level 15 − level 14 ($h_N$)")
ax.set_title(f"{len(d)} matched crests, t = 550–650", loc="left"); ax.legend(loc="upper left")
fig.tight_layout(); S.save(fig, "mesh-sensitivity", "fig03_crest_offset")
