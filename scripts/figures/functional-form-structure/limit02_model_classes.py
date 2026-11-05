"""Per-case error of the model classes of the chain-rule test (linear, + second derivatives, nonlinear, strict chain rule)."""
import sys, subprocess, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

S.apply()
cache = S.FMLP / "cache_dir"
out = subprocess.run([sys.executable, str(S.ROOT / "scripts/chain_rule_test.py"), "--cache", str(cache)], check=True, capture_output=True, text=True).stdout
roles = {e["name"]: e["role"] for e in json.loads((cache / "manifest.json").read_text())["cases"]}
rows = {}
for ln in out.splitlines():
    tk = ln.replace("|", " ").split()
    if len(tk) == 12 and tk[0].startswith("Re"):
        v = [float(z) for z in tk[1:]]
        rows[tk[0]] = dict(Mx=dict(M0=v[2], M1=v[3], M2=v[4], M3=v[5], chain=v[6]), S=dict(M0=v[7], M1=v[8], M2=v[9], M3=v[10]))
    elif tk and tk[0] == "MEAN": print("MEAN row:", ln.strip())
import re
def nice(c):
    re_, ca, th = re.match(r"Re(\d+)_Ca(\d_\d+)_theta(.+)", c).groups()
    s = f"Re{int(re_)} Ca{float(ca.replace('_', '.')):g} β{th.replace('_', '.')}°"
    return s + ("  (held out)" if roles.get(c, "").startswith("held") else "")
cs = sorted(rows, key=lambda c: rows[c]["S"]["M3"])
y = np.arange(len(cs))
mk = {"M1": ("o", "linear in $h_x, q_x$"), "M2": ("s", "+ second derivatives"), "M3": ("^", "nonlinear in $h, q, h_x, q_x$"), "chain": ("D", "strict chain rule")}
print("overfit: cases where the nonlinear model is worse than the linear one:")
for key in ("Mx", "S"):
    for c in cs:
        if rows[c][key]["M3"] > rows[c][key]["M1"]: print(f"  {key} {c}: linear {rows[c][key]['M1']:.3f} nonlinear {rows[c][key]['M3']:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.2), sharey=True)
for ax, (key, title, col) in zip(axes, (("Mx", S.DMX, S.BLUE), ("S", S.DMX + " + " + S.DTW, S.SUMC))):
    for c, yy in zip(cs, y):
        vals = [rows[c][key][m] for m in mk if m in rows[c][key]]
        ax.plot([min(vals), max(vals)], [yy, yy], color=S.GRID, lw=2.0, zorder=1)
    for m, (marker, _) in mk.items():
        if m == "chain" and key == "S": continue
        ax.scatter([rows[c][key][m] for c in cs], y, marker=marker, s=34, color=col, alpha=0.9 if m != "M2" else 0.5, zorder=3, lw=0)
    ax.set_title(title, loc="left"); ax.set_xlabel("NRMSE on the later half of the snapshots (t > 1000)"); ax.grid(axis="y", visible=True)
axes[0].set_yticks(y); axes[0].set_yticklabels([nice(c) for c in cs], fontsize=8.2); axes[0].invert_yaxis()
fig.legend(handles=[Line2D([], [], marker=mk[m][0], ls="", color=S.INK2, label=mk[m][1]) for m in mk], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.025))
fig.text(0.01, 0.0, "Fits per case, first half of the snapshots; cases sorted by the nonlinear model's error on the sum. Without gradients the error is about 1.0 (not shown).", fontsize=8.5, color=S.INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1)); S.save(fig, "functional-form-structure", "limit02_model_classes")
