"""Time content of the corrections the ROM is given: series at one x, and temporal spectra."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
terms = [("dMx", S.DMX, S.BLUE, "-"), ("dTw", S.DTW, S.ORANGE, "--"), ("dPx", S.DPX, S.AQUA, "-")]
x0 = 400.0
with h5py.File(S.ORACLE / "rom/rom_forcing.h5") as f:
    t, x = f["t"][:], f["x"][:]
    i0 = int(np.argmin(abs(x - x0))); n = t.size                   # all 100 time units
    ser = {k: f[k][:300, i0] for k, *_ in terms}
    cols = np.arange(int(100 / 0.2), int(700 / 0.2), 20)          # every 20th cell, 100 < x < 700
    spec = {}
    edges = np.logspace(np.log10(0.012), np.log10(50), 70)
    for k, *_ in terms:
        a = f[k][:n][:, cols]; a = a - a.mean(0)
        w = np.hanning(n)[:, None]
        P = (np.abs(np.fft.rfft(a * w, axis=0)) ** 2).mean(1)
        spec[k] = P
    dt = float(t[1] - t[0]); freq = np.fft.rfftfreq(n, dt)
    binned = {}
    for k in spec:
        which = np.digitize(freq[1:], edges); P = spec[k][1:]
        fc = np.array([freq[1:][which == b].mean() for b in range(1, len(edges)) if (which == b).any()])
        pc = np.array([P[which == b].mean() for b in range(1, len(edges)) if (which == b).any()])
        binned[k] = (fc, pc / pc.max())

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
ax = axes[0]
for k, lab, c, ls in terms:
    ax.plot(t[:300] - t[0], ser[k][:300], color=c, ls=ls, label=lab)
ax.set_xlabel("elapsed time"); ax.set_ylabel("correction (nondimensional)")
ax.set_title(f"(a) corrections at x = {x0:g}, first 3 time units", loc="left"); ax.legend(loc="upper right", ncol=3)
ax = axes[1]
for k, lab, c, ls in terms:
    ax.loglog(binned[k][0], binned[k][1], color=c, ls=ls, label=lab)
ax.axvline(1 / 47.0, color=S.MUTED, lw=0.8); ax.text(1 / 47.0 * 1.12, 3e-9, "forcing\nfrequency", color=S.INK2, fontsize=8.5)
ax.set_xlabel("frequency (1 / nondimensional time)"); ax.set_ylabel("power (normalised to each peak)")
ax.set_title("(b) temporal spectrum, 100 < x < 700", loc="left"); ax.set_ylim(1e-10, 3)
fig.tight_layout()
S.save(fig, "oracle-a-posteriori", "fig04_correction_time_content")
