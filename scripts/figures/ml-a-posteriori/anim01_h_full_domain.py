"""Animation of h(x), 0 < x < 800, t = 1800-1900 (every 0.1): DNS, Lavalle alone and the two learned closures.

Writes docs/results/figures/ml-a-posteriori/anim01_h_full_domain.mp4 (needs ffmpeg on PATH).
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import _runs as R
S = R.S

S.apply()
R.RUNS.pop("Oracle " + S.DTW + " + " + S.DMX)  # requested: DNS, Lavalle and the two ML runs
times, x, dns, rom = R.load()
mask = (x > 20) & (x < 760)
fig, ax = plt.subplots(figsize=(12, 3.6))
ld, = ax.plot(x, dns[0], color=S.INK, lw=1.8, label="DNS")
lines = {n: ax.plot(x, rom[n][0], color=c, ls=ls, lw=1.2, label=n)[0] for n, (_, c, ls) in R.RUNS.items()}
ax.set_xlim(0, 800); ax.set_ylim(0, 2.8)
ax.set_xlabel(S.XN); ax.set_ylabel(S.HN)
ax.legend(loc="upper left", ncol=4, fontsize=8.5)
title = ax.set_title("", loc="left", fontsize=10)
fig.tight_layout()

def frame(i):
    ld.set_ydata(dns[i])
    errs = []
    for n, l in lines.items():
        l.set_ydata(rom[n][i])
        errs.append(np.sqrt(np.mean((rom[n][i][mask] - dns[i][mask]) ** 2)) / np.std(dns[i][mask]))
    title.set_text(f"Re 120, Ca 0.01, β 45°, 10 Hz   t = {R.T0 + times[i]:7.1f}   "
                   f"h error: Lavalle {errs[0]:.2f} | ML sum {errs[1]:.2f} | ML sep. {errs[2]:.2f}")
    return [ld, title, *lines.values()]

out = S.FIGDIR / "ml-a-posteriori"; out.mkdir(parents=True, exist_ok=True)
FuncAnimation(fig, frame, frames=len(times), blit=True).save(out / "anim01_h_full_domain.mp4",
                                                          writer=FFMpegWriter(fps=25, bitrate=2400), dpi=120)
print(f"wrote {out / 'anim01_h_full_domain.mp4'} ({(out / 'anim01_h_full_domain.mp4').stat().st_size / 1e6:.1f} MB, {len(times)} frames)")
