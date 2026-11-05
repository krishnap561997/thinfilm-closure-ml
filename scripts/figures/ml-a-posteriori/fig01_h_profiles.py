"""h(x) at six elapsed times: DNS, Lavalle alone, oracle ΔT_w + ΔM_x and the two learned closures.

Same layout as oracle-a-posteriori/fig01: one x window, chosen around the largest late-time error of
the ML sum run, used for every panel.
"""
import numpy as np, matplotlib.pyplot as plt
import _runs as R
S = R.S

S.apply()
times = [1.0, 5.0, 20.0, 50.0, 75.0, 99.9]
_, x, dns, rom = R.load(times)
mask = (x > 20) & (x < 760)
rel = lambda h, i: np.sqrt(np.mean((h[mask] - dns[i][mask]) ** 2)) / np.std(dns[i][mask])
ref = r"ML sum $(h,\ h_x)$"
err = np.abs(rom[ref][-1] - dns[-1]); k = int(120 / (x[1] - x[0]))
sm = np.convolve(err * mask, np.ones(k) / k, mode="same")
xc = x[int(np.argmax(sm))]; lo, hi = max(20, round(xc - 60, -1)), min(760, round(xc + 60, -1))
m = (x > lo) & (x < hi)
print(f"window {lo}-{hi} (x = {xc:.0f})")
fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.6), sharex=True, sharey=True)
for i, (ax, e) in enumerate(zip(axes.flat, times)):
    ax.plot(x[m], dns[i][m], color=S.INK, lw=2.2, label="DNS")
    for name, (_, c, ls) in R.RUNS.items():
        ax.plot(x[m], rom[name][i][m], color=c, ls=ls, lw=1.4, label=name)
    es = [rel(rom[n][i], i) for n in R.RUNS]
    ax.set_title(f"elapsed {e:g}:  error Lav {es[0]:.2f} | oracle {es[1]:.2f} | sum {es[2]:.2f} | sep. {es[3]:.2f}",
                 loc="left", fontsize=9)
    ax.set_xlim(lo, hi)
for ax in axes[-1]: ax.set_xlabel(S.XN)
for ax in axes[:, 0]: ax.set_ylabel(S.HN)
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle(f"Film thickness with learned closures, Re = 120, Ca = 0.01, β = 45°, 10 Hz (window {lo:g} < x < {hi:g}; "
             "error = rms / std of DNS h, 20 < x < 760)", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout()
S.save(fig, "ml-a-posteriori", "fig01_h_profiles", dpi=165)
