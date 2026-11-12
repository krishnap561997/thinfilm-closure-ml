"""Animation of h(x), 0 < x < 800, from t = 1800 (every 0.1): DNS, Lavalle alone and the learned closures.

    python anim01_h_full_domain.py [--set 2026-11-05 | 2026-11-12]

The oracle run is left out (as requested for the first set). Runs to t = 1900, or to the set's
t_max. Writes ml-a-posteriori/comparisons_<set>/anim01_h_full_domain.mp4 (needs ffmpeg on PATH).
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import _runs as R
S = R.S

date, cfg, outdir = R.get_set()
S.apply()
runs = {k: v for k, v in cfg["runs"].items() if k != R.ORC}
times, x, dns, rom = R.load(runs, t_max=cfg["t_max"])
mask = (x > 20) & (x < 760)
fig, ax = plt.subplots(figsize=(12, 3.9 if len(runs) > 3 else 3.6))
ld, = ax.plot(x, dns[0], color=S.INK, lw=1.8, label="DNS")
lines = {n: ax.plot(x, rom[n][0], color=c, ls=ls, lw=1.2, label=n)[0] for n, (_, c, ls, _s) in runs.items()}
ax.set_xlim(0, 800); ax.set_ylim(0, 2.8)
ax.set_xlabel(S.XN); ax.set_ylabel(S.HN)
ax.legend(loc="upper left", ncol=4 if len(runs) <= 3 else 3, fontsize=8.5)
title = ax.set_title("", loc="left", fontsize=10)
fig.tight_layout()


def frame(i):
    ld.set_ydata(dns[i])
    errs = []
    for n, l in lines.items():
        l.set_ydata(rom[n][i])
        e = np.sqrt(np.mean((rom[n][i][mask] - dns[i][mask]) ** 2)) / np.std(dns[i][mask])
        errs.append(f"{runs[n][3]} {e:.2f}")
    title.set_text(f"Re 120, Ca 0.01, β 45°, 10 Hz   t = {R.T0 + times[i]:7.1f}   h error: " + " | ".join(errs))
    return [ld, title, *lines.values()]


out = S.FIGDIR / outdir; out.mkdir(parents=True, exist_ok=True)
f = out / "anim01_h_full_domain.mp4"
FuncAnimation(fig, frame, frames=len(times), blit=True).save(f, writer=FFMpegWriter(fps=25, bitrate=2400), dpi=120)
print(f"wrote {f} ({f.stat().st_size / 1e6:.1f} MB, {len(times)} frames)")
