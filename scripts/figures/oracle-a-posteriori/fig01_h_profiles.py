"""h(x) at six elapsed times, up to the end of the run: DNS, Lavalle alone, Lavalle + exact DeltaT_w + DeltaM_x.

The x window is chosen once, around the largest error of the corrected run at the last time, and
is used for every panel so that the growth of the error can be compared between panels.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
T0 = 1800.01
runs = {"Lavalle only": ("lavalle", S.BLUE), S.DTW + " + " + S.DMX: ("Tw_Mx", S.AQUA)}
times = [1.0, 5.0, 20.0, 50.0, 75.0, 99.9]
with h5py.File(S.ORACLE / "rom/rom_forcing.h5") as f:
    tD, x, hD = f["t"][:], f["x"][:], f["dns/h"]
    dns = {e: hD[int(np.argmin(abs(tD - (T0 + e))))] for e in times}
rom = {}
for name, (d, _) in runs.items():
    with h5py.File(S.ORACLE / f"rom/runs/{d}/thinfilm_training_data.h5") as f:
        t = f["time"][:]
        rom[name] = {e: f["h"][int(np.argmin(abs(t - (T0 + e))))] for e in times}

mask = (x > 20) & (x < 760)
rel = lambda h, e: np.sqrt(np.mean((h[mask] - dns[e][mask]) ** 2)) / np.std(dns[e][mask])
corr = S.DTW + " + " + S.DMX
# window around the largest smoothed |error| of the corrected run at the last time
err = np.abs(rom[corr][times[-1]] - dns[times[-1]]); k = int(120 / (x[1] - x[0]))
sm = np.convolve(err * mask, np.ones(k) / k, mode="same")
xc = x[int(np.argmax(sm))]; lo, hi = max(20, round(xc - 60, -1)), min(760, round(xc + 60, -1))
m = (x > lo) & (x < hi)
print(f"window {lo}-{hi} (centre of the largest late-time error, x = {xc:.0f})")
print("elapsed   Lavalle   corrected   (rms error / std of DNS h, 20 < x < 760)")
fig, axes = plt.subplots(3, 2, figsize=(10.4, 8.4), sharex=True, sharey=True)
for ax, e in zip(axes.flat, times):
    ax.plot(x[m], dns[e][m], color=S.INK, lw=2.2, label="DNS")
    for name, (_, c) in runs.items():
        ax.plot(x[m], rom[name][e][m], color=c, lw=1.5, label=name)
    el, ec = rel(rom["Lavalle only"][e], e), rel(rom[corr][e], e)
    print(f"{e:7.1f}   {el:7.3f}   {ec:9.3f}")
    ax.set_title(f"elapsed {e:g}:  error {ec:.2f} (Lavalle alone {el:.2f})", loc="left", fontsize=10)
    ax.set_xlim(lo, hi)
for ax in axes[-1]: ax.set_xlabel(S.XN)
for ax in axes[:, 0]: ax.set_ylabel(S.HN)
axes[0, 0].legend(loc="upper left", fontsize=8.5)
fig.suptitle(f"Film thickness, Re = 120, Ca = 0.01, β = 45°, 10 Hz (window {lo:g} < x < {hi:g}, chosen where the late-time error is largest)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
S.save(fig, "oracle-a-posteriori", "fig01_h_profiles", dpi=165)
