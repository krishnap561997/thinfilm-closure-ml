"""The restart run (new source, profiles) against the original L800 run, from the 2D snapshots."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import numpy as np, matplotlib.pyplot as plt
import style as S
from oracle_checks import h_and_q

S.apply()
new = S.ORACLE / "run_t1800/output"
ref = S.STORE / "codes/basilisk/physics-tests/bc-blowup-fix/length-check/L800/output"
times = [1810, 1820, 1830]
rows = []
for T in times:
    tn, x, hn, _ = h_and_q(new / f"snapshot_{T:06d}.h5"); _, _, hr, _ = h_and_q(ref / f"snapshot_{T:06d}.h5")
    m = (x > 20) & (x < 760); rows.append((T, np.sqrt(np.mean((hn - hr)[m] ** 2)) / np.std(hr[m]), x, hn, hr))
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
T, e, x, hn, hr = rows[-1]; m = (x > 300) & (x < 420)
axes[0].plot(x[m], hr[m], color=S.INK, lw=2.4, label="original L800 run")
axes[0].plot(x[m], hn[m], color=S.BLUE, lw=1.2, ls="--", label="restart with profile output")
axes[0].set_xlabel(S.XN); axes[0].set_ylabel(S.HN); axes[0].set_title(f"(a) film thickness at t = {T}", loc="left"); axes[0].legend(loc="upper right")
axes[1].plot([r[0] - 1800 for r in rows], [100 * r[1] for r in rows], marker="o", color=S.BLUE)
axes[1].set_xlabel("elapsed time since restart (t − 1800)"); axes[1].set_ylabel("rms difference in h, % of std(h)")
axes[1].set_ylim(0, 1.5); axes[1].set_xlim(0, 40); axes[1].set_title("(b) difference", loc="left")
fig.tight_layout()
S.save(fig, "oracle-a-posteriori", "fig06_restart_reproduces_L800")
