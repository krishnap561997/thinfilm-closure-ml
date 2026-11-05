"""Per-case error of the pooled sparse forms against Fr^2 (t > 1000)."""
import sys, subprocess; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from matplotlib.ticker import ScalarFormatter, NullFormatter
import style as S

S.apply()
script = S.ROOT / "scripts/error_vs_parameters.py"
path = S.STORE / "codes/ml-models/functional_forms/sparse_01/sparse_path.json"
def run(target, k):
    out = subprocess.run([sys.executable, str(script), "--cache", str(S.FMLP / "cache_dir"), "--path", str(path),
                          "--target", target, "--k", str(k)], check=True, capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        tk = ln.split()
        if len(tk) >= 8 and tk[0].startswith("Re") and "_Ca" in tk[0]:
            held = tk[1] == "H"; v = [float(z) for z in (tk[2:] if held else tk[1:])]
            rows.append(dict(case=tk[0], held=held, We=v[0], Fr2=v[1], beta=v[2], nrmse=v[6]))
    return rows
panels = [("Mx", 5, S.BLUE, S.DMX + " (5-term form)"), ("S", 9, S.SUMC, S.DMX + " + " + S.DTW + " (9-term form)")]
mk = {6.4: "^", 45.0: "o", 90.0: "s"}
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=False)
for ax, (tg, k, col, title) in zip(axes, panels):
    rows = run(tg, k)
    for r in rows:
        ax.scatter(r["Fr2"], r["nrmse"], marker=mk[round(r["beta"], 1)], s=46, color=col if r["held"] else "none",
                   edgecolor=col, lw=1.4, zorder=3)
    rho = spearmanr([r["Fr2"] for r in rows], [r["nrmse"] for r in rows])[0]
    ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20, 40]); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel(r"$Fr^2 = Re\,\sin\beta\,/\,3$"); ax.set_title(title + f",  Spearman ρ = {rho:.2f}", loc="left", fontsize=10)
axes[0].set_ylabel("NRMSE per case (RMSE / rms of the truth)")
h = [Line2D([], [], marker=mk[b], ls="", color=S.INK2, mfc="none", label=f"β = {b:g}°") for b in (6.4, 45.0, 90.0)]
h += [Line2D([], [], marker="o", ls="", color=S.INK2, label="held-out case (filled)"),
      Line2D([], [], marker="o", ls="", color=S.INK2, mfc="none", label="development case (open, in-sample)")]
axes[1].legend(handles=h, loc="lower left", fontsize=8.2)
fig.tight_layout()
S.save(fig, "functional-form-structure", "fig02_error_vs_Fr2")
