#!/usr/bin/env python3
"""Figures for docs/results/dpx-closure.md (ΔP_x v2 ablation, 12 Nov 2026).

    python make_dpx_figures.py [--out docs/results/figures/dpx-closure] [--only NAME ...]

Reads the closure files (ClosureData), the dpx diagnostics fields, and the evaluate.py
predictions (eval_heldout/eval_val) of the 4×64 sweep from $THINFILM_STORE. Figures:

  fig01_extraction_strategies   ΔP_x from DNS by method a/b/c, raw and filtered (Re120 Ca0.01 β45, t = 1500)
  fig02_r2_vs_We                per-case R² against We for all input groups
  fig03-06 / limit01-04         snapshots at t = 1500, x < 100 and 400 < x < 600, DNS vs 4 models
  fig07_crest_average           crest-aligned mean h, ΔP_x and RMS error over solitary waves, t > 1000
  fig08_truth_vs_pred           2D histograms of prediction against DNS, t > 1000, x > 100
  fig09_spectra                 wavenumber spectra of DNS ΔP_x, prediction and error
  limit05_error_map             x–t map of DNS, prediction, error for Re80 Ca0.005 β6.4
  limit06_lag_correlation       correlation of the error with lagged / shifted inputs
"""
import argparse
import csv
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

STORE = Path(os.environ.get("THINFILM_STORE", "/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml"))
CD = STORE / "datasets/BasC/DNS/param_space/ClosureData"
SWEEP = STORE / "codes/ml-models/functional_mlp/runs/dpx_v2_sweep01/layers4_width64"
TARGET = "delta_Px_ML_v2"

INK, MUTED, GRID = "#0b0b0b", "#6b6a63", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODELS = {  # fixed colour per model
    "G_s0p5": ("G (σ=0.5): A + h_xxx + q_xx", SERIES[0]),
    "H_s0p5": ("H (σ=0.5): A + κ_x/We", SERIES[1]),
    "F_s0p5": ("F (σ=0.5): A + h_xxx", SERIES[2]),
    "A_sg": ("A_sg: h, q, h_x, q_x (old SG, unfiltered)", SERIES[3]),
}
BEST = ["Re120_Ca0_010_theta45", "Re080_Ca0_010_theta90"]
WORST = ["Re080_Ca0_005_theta6_4", "Re020_Ca0_010_theta45"]
CASES = BEST + WORST
T_SNAP = 1500.0
REGIONS = {"linear": (0.0, 100.0), "nonlinear": (400.0, 600.0)}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.dpi": 150,
})


def parse(case):
    p = case.split("_")
    Re, Ca, th = float(p[0][2:]), float(p[1][2:] + "." + p[2]), p[4] if len(p) > 4 else p[3]
    beta = float(case.split("theta")[1].replace("_", "."))
    return Re, Ca, Re * Ca, beta


def label(case):
    Re, Ca, We, b = parse(case)
    return f"Re {Re:g}, Ca {Ca:g}, β {b:g}° (We {We:g})"


_closure = {}


def closure(case):
    if case not in _closure:
        f = h5py.File(CD / f"thinfilm_closure_data_{case}.h5", "r")
        _closure[case] = f
    return _closure[case]


_pred = {}


def pred(group, case, split="heldout"):
    """Full (nt, nx) arrays of truth and prediction, NaN where not evaluated."""
    key = (group, case, split)
    if key not in _pred:
        with h5py.File(SWEEP / group / TARGET / "seed42" / f"eval_{split}" / f"{case}.h5", "r") as f:
            t, x, ix = f["t"][:], f["x"][:], f["indices"][:]
            P = np.full((t.size, x.size), np.nan, np.float32); T = P.copy()
            P[ix[:, 0], ix[:, 1]] = f["prediction"][:, 0]; T[ix[:, 0], ix[:, 1]] = f["truth"][:, 0]
        _pred[key] = (t, x, T, P)
    return _pred[key]


def r2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    return 1 - np.sum((b - a) ** 2) / np.sum((a - a.mean()) ** 2)


def save(fig, out, name, dpi=150):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    from PIL import Image  # 256-colour palette PNG: same look, a fraction of the size
    Image.open(out / f"{name}.png").convert("RGB").quantize(256).save(out / f"{name}.png", optimize=True)
    print(f"{name}.png  {os.path.getsize(out / f'{name}.png') / 1024:.0f} KB")


# ---------------------------------------------------------------- figures
def fig_extraction(out):
    case = "Re120_Ca0_010_theta45"
    g = h5py.File(CD / "dpx_diagnostics" / f"{case}_fields.h5", "r")
    it = int(np.argmin(abs(g["t"][:] - T_SNAP)))
    x = g["x"][:]; m = (x > 400) & (x < 600)
    c = closure(case); ic = int(np.argmin(abs(c["t"][:] - T_SNAP)))
    print("check: diagnostics dPx_b_s0.5 vs stored v2, max diff",
          np.nanmax(abs(g["dPx_b_s0.5"][:, it] - c[TARGET][:, ic])[m]))
    fig = plt.figure(figsize=(11, 8.5)); gs = GridSpec(3, 2, height_ratios=[0.7, 1, 1], figure=fig)
    ax = fig.add_subplot(gs[0, :]); ax.plot(x[m], g["h"][m, it], color=INK); ax.set_ylabel("h")
    ax.set_title("film height")
    axs = [fig.add_subplot(gs[i, j]) for i in (1, 2) for j in (0, 1)]
    names = {"a": "a: ∂ₓp on all rows (incl. interface)", "b": "b: liquid-pressure Leibniz form",
             "c": "c: ∂ₓp over fully liquid cells"}
    for k, col in zip("abc", SERIES):
        axs[0].plot(x[m], g[f"Ip_{k}"][m, it], color=col, label=names[k], lw=1.0)
        axs[1].plot(x[m], g[f"dPx_{k}_raw"][m, it], color=col, label=names[k], lw=1.0)
    axs[0].set_title("I_p = ∫₀ʰ ∂ₓp dz, raw (three methods)"); axs[0].set_ylabel("I_p")
    axs[1].set_title("ΔP_x raw (exact-curvature Lavalle term, unfiltered)"); axs[1].set_ylabel("ΔP_x")
    lim = 1.3 * np.nanmax(abs(g["dPx_b_s0.25"][m, it]))
    axs[1].set_ylim(-lim, lim)
    for s, col in zip(["raw", "s0.25", "s0.5", "s1"], [GRID, SERIES[0], SERIES[1], SERIES[2]]):
        axs[2].plot(x[m], g[f"dPx_b_{s}"][m, it], color=col if s != "raw" else MUTED,
                    lw=0.8 if s == "raw" else 1.4, label="raw" if s == "raw" else f"σ = {s[1:]} h_N")
    axs[2].set_ylim(-lim, lim); axs[2].set_title("method b: effect of the Gaussian filter"); axs[2].set_ylabel("ΔP_x")
    axs[3].plot(x[m], c["delta_Px_ML"][m, ic], color=MUTED, lw=0.8, label="old: method a + SG h_xxx (delta_Px_ML)")
    axs[3].plot(x[m], c[TARGET][m, ic], color=SERIES[0], label="v2: method b, σ = 0.5 (delta_Px_ML_v2)")
    axs[3].set_ylim(-lim, lim); axs[3].set_title("stored targets: old vs v2"); axs[3].set_ylabel("ΔP_x")
    for a in axs:
        a.legend(fontsize=7, loc="upper left"); a.set_xlabel("x / h_N")
    ax.set_xlabel("x / h_N")
    fig.suptitle(f"ΔP_x extraction strategies — {label(case)}, t = {T_SNAP:g}, 400 < x < 600 "
                 f"(panels b–d clipped at ±1.3 max|σ=0.25|)", fontsize=10)
    fig.tight_layout(); save(fig, out, "fig01_extraction_strategies")


def read_metrics(group, split):
    p = SWEEP / group / TARGET / "seed42" / f"eval_{split}" / "metrics.csv"
    with open(p) as fh:
        return {r["case"]: float(r["R2"]) for r in csv.DictReader(fh) if r["case"] not in ("POOLED", "CASE_AVERAGE")}


def fig_r2_we(out):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    groups = [("A_s0p5", "A"), ("B_s0p5", "B"), ("C_s0p5", "C"), ("F_s0p5", "F"), ("G_s0p5", "G"), ("H_s0p5", "H")]
    for (gname, lab), col in zip(groups, SERIES):
        d = read_metrics(gname, "heldout"); cs = sorted(d, key=lambda c: parse(c)[2])
        axs[0].plot([parse(c)[2] for c in cs], [d[c] for c in cs], "-o", ms=4, color=col, label=lab)
    d = read_metrics("A_sg", "heldout"); cs = sorted(d, key=lambda c: parse(c)[2])
    axs[0].plot([parse(c)[2] for c in cs], [d[c] for c in cs], "--o", ms=4, color=MUTED, label="A_sg (old)")
    axs[0].set_title("held-out, σ = 0.5 input groups")
    for gname, col in (("G", SERIES[0]), ("A", MUTED)):
        for s, ls in (("s0p25", ":"), ("s0p5", "-"), ("s1", "--")):
            d = read_metrics(f"{gname}_{s}", "heldout"); cs = sorted(d, key=lambda c: parse(c)[2])
            axs[1].plot([parse(c)[2] for c in cs], [d[c] for c in cs], ls, marker="o", ms=3.5, color=col,
                        label=f"{gname}, σ = {s[1:].replace('p', '.')}")
    axs[1].set_title("held-out, filter width (G vs A)")
    bcol = {6.4: SERIES[3], 45.0: SERIES[0], 90.0: SERIES[1]}
    for split, mk, fc in (("val", "o", "none"), ("heldout", "s", None)):
        d = read_metrics("G_s0p5", split)
        for c, v in d.items():
            We, b = parse(c)[2], parse(c)[3]
            axs[2].scatter(We, v, marker=mk, s=36, edgecolors=bcol[b], facecolors=fc or bcol[b], lw=1.4)
    for b, col in bcol.items():
        axs[2].scatter([], [], marker="s", color=col, label=f"β = {b:g}°")
    axs[2].scatter([], [], marker="o", facecolors="none", edgecolors=MUTED, label="validation (training cases)")
    axs[2].scatter([], [], marker="s", color=MUTED, label="held-out")
    axs[2].set_title("G_s0p5 per case: validation vs held-out")
    for a in axs:
        a.set_xscale("log"); a.set_xlabel("We = Re·Ca"); a.axhline(0.8, color=MUTED, lw=0.8, ls=":")
        a.set_xticks([0.2, 0.4, 0.8, 1.6, 3.2]); a.set_xticklabels(["0.2", "0.4", "0.8", "1.6", "3.2"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.legend(fontsize=7, loc="lower right"); a.set_ylim(0, 1.02)
    axs[0].set_ylabel("R² of ΔP_x (per case)")
    axs[0].text(2.6, 0.82, "gate R² = 0.8", fontsize=7, color=MUTED)
    fig.tight_layout(); save(fig, out, "fig02_r2_vs_We")


def fig_snapshots(out):
    n_best, n_worst = 3, 1
    for case in CASES:
        c = closure(case); x = c["x"][:]
        t, xp, _, _ = pred("G_s0p5", case); it = int(np.argmin(abs(t - T_SNAP)))
        ic = int(np.argmin(abs(c["t"][:] - T_SNAP)))
        for reg, (x0, x1) in REGIONS.items():
            m = (x > x0) & (x < x1)
            fig = plt.figure(figsize=(11, 8.5)); gs = GridSpec(3, 2, height_ratios=[0.7, 1, 1], figure=fig)
            ax = fig.add_subplot(gs[0, :]); ax.plot(x[m], c["h"][m, ic], color=INK); ax.set_ylabel("h")
            ax.set_xlabel("x / h_N"); ax.set_title("film height")
            axs = [fig.add_subplot(gs[i, j]) for i in (1, 2) for j in (0, 1)]
            ylo, yhi = np.inf, -np.inf
            for a, (gname, (lab, col)) in zip(axs, MODELS.items()):
                _, _, T, P = pred(gname, case)
                a.plot(x[m], T[it, m], color=INK, label="DNS")
                a.plot(x[m], P[it, m], color=col, ls="--", label="MLP")
                a.set_title(f"{lab}   R² (window) = {r2(T[it, m], P[it, m]):.2f}", fontsize=8.5)
                a.set_xlabel("x / h_N"); a.set_ylabel("ΔP_x"); a.legend(fontsize=7, loc="upper left")
                ylo, yhi = min(ylo, np.nanmin(T[it, m]), np.nanmin(P[it, m])), max(yhi, np.nanmax(T[it, m]), np.nanmax(P[it, m]))
            pad = 0.05 * (yhi - ylo)
            for a in axs:
                a.set_ylim(ylo - pad, yhi + pad); a.set_xlim(x[m][0], x[m][-1])
            ax.set_xlim(x[m][0], x[m][-1])
            tag = "best" if case in BEST else "worst"
            fig.suptitle(f"{label(case)} — held-out, {tag} — t = {t[it]:g}, "
                         f"{'linear region' if reg == 'linear' else 'nonlinear region (solitary waves + ripples)'}, "
                         f"{x0:g} < x < {x1:g}", fontsize=10)
            fig.tight_layout()
            if case in BEST:
                name = f"fig{n_best:02d}_snapshot_{case}_{reg}"; n_best += 1
            else:
                name = f"limit{n_worst:02d}_snapshot_{case}_{reg}"; n_worst += 1
            save(fig, out, name)


def crest_windows(h, x, t, half=30.0, tmin=1000.0, x0=150.0):
    dx = x[1] - x[0]; w = int(round(half / dx)); L = x[-1]
    out = []
    for it in np.where(t > tmin)[0]:
        hh = h[:, it]
        pk, _ = find_peaks(hh, prominence=0.3, distance=int(30 / dx))
        out += [(it, p) for p in pk if x[p] > x0 and x[p] < 0.95 * L - half and p - w >= 0]
    return out, w


def fig_crest(out):
    fig, axs = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [0.7, 1, 0.8]})
    for j, case in enumerate(CASES):
        c = closure(case); x = c["x"][:]; h = c["h"][:]
        t, _, T, _ = pred("G_s0p5", case)
        wins, w = crest_windows(h, x, t)
        s = (np.arange(-w, w + 1)) * (x[1] - x[0])
        H = np.array([h[p - w:p + w + 1, it] for it, p in wins])
        Tw = np.array([T[it, p - w:p + w + 1] for it, p in wins])
        axs[0, j].plot(s, H.mean(0), color=INK); axs[0, j].fill_between(s, *np.percentile(H, [10, 90], 0), color=GRID)
        axs[1, j].plot(s, np.nanmean(Tw, 0), color=INK, label="DNS", lw=2)
        for gname, (lab, col) in MODELS.items():
            P = pred(gname, case)[3]
            Pw = np.array([P[it, p - w:p + w + 1] for it, p in wins])
            axs[1, j].plot(s, np.nanmean(Pw, 0), color=col, ls="--", label=gname)
            axs[2, j].plot(s, np.sqrt(np.nanmean((Pw - Tw) ** 2, 0)), color=col, label=gname)
        axs[2, j].plot(s, np.sqrt(np.nanmean((Tw - np.nanmean(Tw)) ** 2, 0)), color=MUTED, ls=":", label="RMS of DNS")
        axs[0, j].set_title(f"{label(case)}\n{len(wins)} waves, t > 1000", fontsize=8.5)
        axs[2, j].set_xlabel("x − x_crest  (h_N; ripples ahead at > 0)")
    axs[0, 0].set_ylabel("h (mean, 10–90%)"); axs[1, 0].set_ylabel("mean ΔP_x"); axs[2, 0].set_ylabel("RMS error")
    axs[1, 0].legend(fontsize=7, loc="lower left"); axs[2, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Crest-aligned averages over solitary waves (prominence > 0.3, x > 150): best two (left), worst two (right)", fontsize=10)
    fig.tight_layout(); save(fig, out, "fig07_crest_average", dpi=120)


def fig_hist(out):
    fig, axs = plt.subplots(4, 4, figsize=(13, 12.5))
    for i, case in enumerate(CASES):
        t, x, T, _ = pred("G_s0p5", case)
        sel = (t > 1000)[:, None] & (x > 100)[None, :]
        tv = T[sel]; lo, hi = np.nanpercentile(tv, [0.1, 99.9])
        for j, (gname, (lab, col)) in enumerate(MODELS.items()):
            P = pred(gname, case)[3]; pv = P[sel]; ok = np.isfinite(tv) & np.isfinite(pv)
            a = axs[i, j]
            a.hist2d(tv[ok], pv[ok], bins=120, range=[[lo, hi], [lo, hi]], norm=LogNorm(), cmap="Blues", rasterized=True)
            a.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
            k = np.sum(tv[ok] * pv[ok]) / np.sum(tv[ok] ** 2)
            a.text(0.03, 0.97, f"R² = {r2(tv[ok], pv[ok]):.2f}\nslope = {k:.2f}", transform=a.transAxes, va="top", fontsize=7.5)
            a.grid(False)
            if i == 0:
                a.set_title(gname, fontsize=9)
            if j == 0:
                a.set_ylabel(f"{label(case)}\npredicted", fontsize=8)
            if i == 3:
                a.set_xlabel("DNS ΔP_x")
    fig.suptitle("Predicted against DNS ΔP_x (held-out, t > 1000, x > 100; log density; slope = least squares through 0)", fontsize=10)
    fig.tight_layout(); save(fig, out, "fig08_truth_vs_pred", dpi=100)


def spectrum(A, dx):
    n = A.shape[1]; win = np.hanning(n)
    F = np.fft.rfft((A - A.mean(1, keepdims=True)) * win, axis=1)
    return np.fft.rfftfreq(n, dx), np.mean(abs(F) ** 2, 0) * dx / np.sum(win ** 2)


def fig_spectra(out):
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for a, case in zip(axs, CASES):
        t, x, T, _ = pred("G_s0p5", case); dx = x[1] - x[0]
        tt = t > 1000
        m = (x > 100) & np.all(np.isfinite(T[tt]), 0)  # evaluated columns (x < ~758)
        Tm = T[tt][:, m]
        good = np.ones(Tm.shape[0], bool)
        k, St = spectrum(Tm[good], dx)
        a.loglog(k[1:], St[1:], color=INK, lw=2, label="DNS")
        for gname, ls in (("G_s0p5", "-"), ("A_sg", "-")):
            lab, col = MODELS[gname]
            Pm = pred(gname, case)[3][tt][:, m][good]
            a.loglog(k[1:], spectrum(Pm, dx)[1][1:], color=col, lw=1.2, label=f"{gname} prediction")
            a.loglog(k[1:], spectrum(Pm - Tm[good], dx)[1][1:], color=col, ls=":", lw=1.5, label=f"{gname} error")
        a.axvline(1.0, color=MUTED, lw=0.8, ls="--"); a.text(1.05, 0.03, "λ = 1 h_N", fontsize=7, color=MUTED, transform=a.get_xaxis_transform())
        a.set_xlim(1e-3, 5); a.set_title(label(case), fontsize=8.5); a.set_xlabel("1/λ  (1/h_N)")
    axs[0].set_ylabel("power spectral density"); axs[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Wavenumber spectra of ΔP_x (held-out, t > 1000, 100 < x < 758)", fontsize=10)
    fig.tight_layout(); save(fig, out, "fig09_spectra")


def fig_error_map(out):
    case = "Re080_Ca0_005_theta6_4"
    t, x, T, P = pred("G_s0p5", case)
    sl = slice(None, None, 8); vmax = np.nanpercentile(abs(T), 99)
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for a, A, ttl in zip(axs, (T, P, P - T), ("DNS ΔP_x", "G_s0p5 prediction", "error (prediction − DNS)")):
        n = A.shape[1] // 32 * 32  # pool 32 columns, keep the value of largest magnitude
        B = A[:, :n].reshape(A.shape[0], -1, 32); k = np.nanargmax(np.nan_to_num(abs(B), nan=-1), 2)
        Ap = np.take_along_axis(B, k[..., None], 2)[..., 0]; Ap[np.all(np.isnan(B), 2)] = np.nan
        im = a.imshow(Ap, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=[x[0], x[n - 1], t[0], t[-1]], interpolation="nearest", rasterized=True)
        a.set_title(ttl); a.set_ylabel("t"); a.grid(False)
        fig.colorbar(im, ax=a, pad=0.01)
    axs[-1].set_xlabel("x / h_N")
    fig.suptitle(f"{label(case)} (held-out, worst case): ΔP_x over x and t (every 10 time units; max-|·| over 32 columns)", fontsize=10)
    fig.tight_layout(); save(fig, out, "limit05_error_map", dpi=110)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))


def fig_lag(out):
    cases = ["Re120_Ca0_010_theta45", "Re080_Ca0_005_theta6_4", "Re020_Ca0_010_theta45"]
    fig, axs = plt.subplots(len(cases), 2, figsize=(11, 3.2 * len(cases)), sharey=True)
    for i, case in enumerate(cases):
        c = closure(case); t, x, T, P = pred("G_s0p5", case); E = P - T
        feats = {"h": c["h"][:].T, "κ_x (σ=0.5)": c["kappa_x_s"][:].T, "DNS ΔP_x": T, "error": E}
        cols = {"h": SERIES[0], "κ_x (σ=0.5)": SERIES[1], "DNS ΔP_x": SERIES[2], "error": INK}
        mx = (x > 100) & (x < 0.95 * x[-1]); it = np.where(t > 1000)[0]
        lags = np.arange(0, 11)
        for name, F in feats.items():
            r = [corr(E[it][:, mx], F[it - L][:, mx]) for L in lags]
            axs[i, 0].plot(10 * lags, r, "-o", ms=3.5, color=cols[name], label=name)
        dx = x[1] - x[0]; shifts = np.arange(-20, 20.01, 0.5); jj = np.where(mx)[0]
        for name, F in feats.items():
            r = [corr(E[it][:, jj], F[it][:, jj + int(round(s / dx))]) for s in shifts]
            axs[i, 1].plot(shifts, r, color=cols[name], label=name)
        axs[i, 0].set_title(f"{label(case)}: fixed x, input at t − τ", fontsize=8.5)
        axs[i, 1].set_title("same t, input at x + s (s < 0: upstream)", fontsize=8.5)
        axs[i, 0].set_ylabel("corr(error, input)")
        for a in axs[i]:
            a.axhline(0, color=MUTED, lw=0.6)
    axs[-1, 0].set_xlabel("τ (time units)"); axs[-1, 1].set_xlabel("s (h_N)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Correlation of the G_s0p5 error with lagged and shifted inputs (held-out, t > 1000, x > 100)", fontsize=10)
    fig.tight_layout(); save(fig, out, "limit06_lag_correlation")


FIGS = {"extraction": fig_extraction, "r2_we": fig_r2_we, "snapshots": fig_snapshots, "crest": fig_crest,
        "hist": fig_hist, "spectra": fig_spectra, "error_map": fig_error_map, "lag": fig_lag}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[3] / "docs/results/figures/dpx-closure")
    ap.add_argument("--only", nargs="*", choices=list(FIGS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.only or FIGS:
        FIGS[name](a.out)


if __name__ == "__main__":
    main()
