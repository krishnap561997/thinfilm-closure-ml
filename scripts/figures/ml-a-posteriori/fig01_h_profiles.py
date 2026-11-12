"""h(x) at six elapsed times: DNS, Lavalle alone, oracle ΔT_w + ΔM_x and the learned closures.

    python fig01_h_profiles.py [--set 2026-11-05 | 2026-11-12]

Same layout as oracle-a-posteriori/fig01: one x window, chosen around the largest late-time error of
the set's reference run, used for every panel. Writes to ml-a-posteriori/comparisons_<set>/.
"""
import numpy as np, matplotlib.pyplot as plt
import _runs as R
S = R.S

date, cfg, outdir = R.get_set()
S.apply()
runs, times = cfg["runs"], cfg["times"]
_, x, dns, rom = R.load(runs, times)
mask = (x > 20) & (x < 760)
rel = lambda h, i: np.sqrt(np.mean((h[mask] - dns[i][mask]) ** 2)) / np.std(dns[i][mask])
err = np.abs(rom[cfg["ref"]][-1] - dns[-1]); k = int(120 / (x[1] - x[0]))
sm = np.convolve(err * mask, np.ones(k) / k, mode="same")
xc = x[int(np.argmax(sm))]; lo, hi = max(20, round(xc - 60, -1)), min(760, round(xc + 60, -1))
m = (x > lo) & (x < hi)
print(f"window {lo}-{hi} (x = {xc:.0f})")
many = len(runs) > 4
fig, axes = plt.subplots(3, 2, figsize=(10.4, 9.4 if many else 8.6), sharex=True, sharey=True)
for i, (ax, e) in enumerate(zip(axes.flat, times)):
    ax.plot(x[m], dns[i][m], color=S.INK, lw=2.2, label="DNS")
    for name, (_, c, ls, _s) in runs.items():
        ax.plot(x[m], rom[name][i][m], color=c, ls=ls, lw=1.4, label=name)
    es = [f"{short} {rel(rom[n][i], i):.2f}" for n, (*_, short) in runs.items()]
    sep = "\n" if many else " | "
    head = " | ".join(es[:3]) + sep + " | ".join(es[3:]) if many else " | ".join(es)
    ax.set_title(f"elapsed {e:g}:  error {head}", loc="left", fontsize=8.5 if many else 9)
    ax.set_xlim(lo, hi)
for ax in axes[-1]: ax.set_xlabel(S.XN)
for ax in axes[:, 0]: ax.set_ylabel(S.HN)
if many:
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.005))
else:
    axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle(f"Film thickness with learned closures, Re = 120, Ca = 0.01, β = 45°, 10 Hz (window {lo:g} < x < {hi:g}; "
             "error = rms / std of DNS h, 20 < x < 760)", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.06, 1, 1) if many else None)
S.save(fig, outdir, "fig01_h_profiles", dpi=165)
