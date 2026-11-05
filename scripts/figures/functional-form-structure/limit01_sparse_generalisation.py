"""Sparse closed forms: error against number of terms on training, leave-one-case-out, held-out and beta = 6.4 deg cases."""
import sys, re; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
log = (S.ROOT / "ml/functional_forms/fit_sparse_48911517.log").read_text().splitlines()
tables, cur = {}, None
for ln in log:
    m = re.match(r"== (\w+):", ln)
    if m: cur = m.group(1); tables[cur] = []; continue
    m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s", ln)
    if m and cur: tables[cur].append([float(v) for v in m.groups()])
T = {k: np.array(v) for k, v in tables.items()}
for k, v in T.items():
    print(f"{k}: {len(v)} rows; LOCO min {v[:, 2].min():.3f} at k={int(v[v[:, 2].argmin(), 0])}; "
          f"beta6.4 holdout {v[3, 4]:.3f} (k=4) ... {v[:, 4].max():.3f} (max at k={int(v[v[:, 4].argmax(), 0])}); LOCO k=20 {v[-1, 2]:.3f}")
panels = [("Mx", S.DMX + " (pooled sparse form)", S.BLUE), ("S", S.DMX + " + " + S.DTW + " (pooled sparse form)", S.SUMC)]
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=False)
for ax, (key, title, col) in zip(axes, panels):
    v = T[key]; k = v[:, 0]
    ax.plot(k, v[:, 1], color=S.MUTED, ls="--", lw=1.4, label="training (15 development cases)")
    ax.plot(k, v[:, 2], color=col, lw=1.8, marker="o", ms=4, label="leave-one-case-out")
    ax.plot(k, v[:, 3], color=col, lw=1.4, ls=":", label="7 held-out cases")
    ax.plot(k, v[:, 4], color=S.INK, lw=1.8, marker="s", ms=4, label="β = 6.4° cases left out")
    ax.set_xlabel("number of terms in the form"); ax.set_title(title, loc="left", fontsize=10); ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper right", fontsize=8.5)
axes[0].set_ylabel("mean NRMSE per case (rms-normalised)")
axes[0].set_ylim(0, 0.62); axes[1].set_ylim(0, 0.95)
fig.tight_layout(); S.save(fig, "functional-form-structure", "limit01_sparse_generalisation")
