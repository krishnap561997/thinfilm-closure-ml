"""DeltaM_x and DeltaT_w nearly cancel: scatter over all cases and one profile."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, numpy as np, matplotlib.pyplot as plt
import style as S

S.apply()
C = S.FMLP / "cache_dir"; man = json.loads((C / "manifest.json").read_text())
rng = np.random.default_rng(0); px, py = [], []
def load(case, name, ix, it): return np.load(C / case / f"{name}.npy", mmap_mode="r")[it][:, ix].astype(float)
for e in man["cases"]:
    t, x = np.load(C / e["name"] / "t.npy"), np.load(C / e["name"] / "x.npy")
    it = np.flatnonzero(t > 1000); ix = np.flatnonzero((x > 100) & (x < 0.95 * x.max()))
    m, w = load(e["name"], "delta_Mx_ML", ix, it).ravel(), load(e["name"], "delta_Tw_ML", ix, it).ravel()
    ok = np.isfinite(m) & np.isfinite(w); m, w = m[ok], w[ok]
    sel = rng.choice(m.size, size=min(2500, m.size), replace=False)
    px.append(m[sel] / m.std()); py.append(w[sel] / m.std())
px, py = np.concatenate(px), np.concatenate(py)
slope = (px @ py) / (px @ px); r = np.corrcoef(px, py)[0, 1]
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
ax = axes[0]
ax.scatter(px, py, s=3, color=S.MUTED, alpha=0.18, lw=0, rasterized=True)
lim = np.array([-4.2, 4.2])
ax.plot(lim, -lim, color=S.INK, lw=1.2, ls="--", label="exact cancellation (slope −1)")
ax.plot(lim, slope * lim, color=S.SUMC, lw=1.8, label=f"least squares: slope {slope:.2f}, corr {r:.3f}")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel(S.DMX + r" / std($\Delta M_x$) of its case"); ax.set_ylabel(S.DTW + r" / std($\Delta M_x$) of its case")
ax.set_title("(a) 22 cases, t > 1000, 100 < x < 0.95 L", loc="left", fontsize=10); ax.legend(loc="upper right", fontsize=8.5)
ax = axes[1]
case = "Re120_Ca0_010_theta45"; t, x = np.load(C / case / "t.npy"), np.load(C / case / "x.npy")
it = np.array([int(np.argmin(abs(t - 1500)))]); ix = np.flatnonzero((x > 300) & (x < 420))
m, w = load(case, "delta_Mx_ML", ix, it)[0], load(case, "delta_Tw_ML", ix, it)[0]
ax.plot(x[ix], m, color=S.BLUE, label=S.DMX); ax.plot(x[ix], w, color=S.ORANGE, ls="--", label=S.DTW)
ax.plot(x[ix], m + w, color=S.SUMC, lw=2.0, label=S.DMX + " + " + S.DTW + "  (the net source)")
ax.set_xlabel(S.XN); ax.set_ylabel("correction (nondimensional)")
ax.set_title(f"(b) Re120 Ca0.01 β45°, t = {t[it[0]]:.0f}", loc="left", fontsize=10); ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
S.save(fig, "functional-form-structure", "fig01_cancellation", png_only=True, dpi=170)
