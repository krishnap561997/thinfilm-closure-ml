"""x-t diagram of h: DNS, Lavalle alone, Lavalle + exact DeltaT_w + DeltaM_x."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style as S

S.apply()
T0 = 1800.01
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", S.SEQ_BLUE[1], S.SEQ_BLUE[3], S.SEQ_BLUE[5]])
with h5py.File(S.ORACLE / "rom/rom_forcing.h5") as f:
    tD, x = f["t"][:], f["x"][:]
    idx = np.arange(10, tD.size, 10)             # every 0.1 time units, matches ROM output
    hD = np.array([f["dns/h"][i] for i in idx]); tt = tD[idx] - T0
panels = [("DNS", hD, tt)]
for key, lab in (("lavalle", "Lavalle only"), ("Tw_Mx", S.DTW + " + " + S.DMX)):
    with h5py.File(S.ORACLE / f"rom/runs/{key}/thinfilm_training_data.h5") as f:
        panels.append((lab, f["h"][:], f["time"][:] - T0))
fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
for ax, (lab, h, t) in zip(axes, panels):
    im = ax.imshow(h, origin="lower", aspect="auto", cmap=cmap, vmin=0.5, vmax=2.1,
                   extent=[x[0], x[-1], t[0], t[-1]], interpolation="nearest")
    ax.set_title(lab, loc="left"); ax.set_xlabel(S.XN); ax.grid(False)
axes[0].set_ylabel("elapsed time since restart")
cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015); cb.set_label(S.HN); cb.outline.set_visible(False)
fig.suptitle("Film thickness in the (x, t) plane, Re = 120, Ca = 0.01, β = 45°", x=0.01, ha="left", fontsize=11)
S.save(fig, "oracle-a-posteriori", "fig03_xt_diagram", png_only=True, dpi=110)
