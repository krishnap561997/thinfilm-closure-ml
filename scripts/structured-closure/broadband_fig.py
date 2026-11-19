#!/usr/bin/env python3
"""Broadband held-out a posteriori figures (ml/structured_closure/broadband.py runs), t >= 1000.

metrics : per case, distance of the ROM h distribution from the DNS (W1/std) and main-crest height ratio; runs that
          stopped early (loss of hyperbolicity) are marked with x at the top of the panel
waves   : for chosen cases, h(x) at t = 2000 (DNS and ROMs, offset; different noise realisations) and the h PDF

    python scripts/structured-closure/broadband_fig.py metrics --csv broadband_seed0.csv --out fig.png
    python scripts/structured-closure/broadband_fig.py waves --out fig.png
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "ml/structured_closure"))
from broadband import OUT, load_dns, load_rom, X0, X1  # noqa: E402

NAVY, CORAL, GRAY = "#0B3C5D", "#E4572E", "#7A8B94"
CL = [("lavalle", "Lavalle", GRAY, "s"), ("ml_sum_hhx", "MLP (previous)", "#F2A541", "^"),
      ("v1_s0.25_L10_F0_seed0", "Structured V1", "#6A4C93", "D"),
      ("mlp_s0.25_L10_seed0", "MLP + OS, σ 0.25", "#4FB0C6", "v"), ("mlp_s0.5_L10_seed0", "MLP + OS, σ 0.5", "#1D7874", "o")]
CASES = ["Re020_Ca0_010_theta45", "Re040_Ca0_040_theta45", "Re080_Ca0_005_theta6_4", "Re080_Ca0_010_theta90",
         "Re120_Ca0_010_theta45", "Re120_Ca0_020_theta90", "Re200_Ca0_010_theta6_4"]


def label(case):
    re, ca, th = case.split("_Ca")[0][2:], case.split("_Ca")[1].split("_theta")[0], case.split("theta")[1]
    return f"Re {int(re)}\nCa {ca.replace('_', '.')}\nβ {th.replace('_', '.')}°"


def style():
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 220, "savefig.bbox": "tight"})


def metrics(a):
    R = {(r["case"], r["closure"]): r for r in csv.DictReader(open(a.csv))}
    fig, axs = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    xc = np.arange(len(CASES)); off = np.linspace(-0.28, 0.28, len(CL))
    for ax, key, ylab, top in [(axs[0], "pdf_dist", "h-distribution distance\nW$_1$ / std(h$_{DNS}$)", None),
                               (axs[1], "crest_ratio", "main-crest height\nROM / DNS", None)]:
        vals = []
        for (name, lab, c, mk), o in zip(CL, off):
            y = [float(R[(cs, name)][key]) if (cs, name) in R else np.nan for cs in CASES]; vals += y
            ax.plot(xc + o, y, mk, color=c, ms=9, mec="white", mew=0.7, label=lab, zorder=3)
            crash = [i for i, cs in enumerate(CASES) if (cs, name) not in R]
            ax._crash = getattr(ax, "_crash", []) + [(i + o, c) for i in crash]
        v = np.array(vals, float); v = v[np.isfinite(v)]
        if key == "pdf_dist":
            ax.set_yscale("log"); lo, hi = v.min() / 1.6, v.max() * 2.2
        else:
            ax.axhline(1, color=NAVY, lw=1.5, zorder=1); lo, hi = min(v.min(), 0.7) - 0.05, v.max() + 0.15
        ax.set_ylim(lo, hi)
        for xx, c in ax._crash:
            yy = hi / 1.35 if key == "pdf_dist" else hi - 0.06
            ax.plot(xx, yy, "x", color=c, ms=11, mew=2.5, zorder=4)
        ax.set_ylabel(ylab)
        for x in xc[:-1] + 0.5: ax.axvline(x, color="#DDDDDD", lw=0.8, zorder=0)
    axs[0].text(0.995, 0.97, "× = run blew up (loss of hyperbolicity) before t = 1000", transform=axs[0].transAxes,
                ha="right", va="top", fontsize=11, color="#444444")
    axs[1].set_xticks(xc); axs[1].set_xticklabels([label(c) for c in CASES], fontsize=11)
    axs[1].text(-0.45, 1.02, "DNS", color=NAVY, fontsize=11, va="bottom")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.04), fontsize=12)
    fig.suptitle("Held-out broadband cases, statistics over t = 1000–2000 (ROM from flat film, own inlet noise)",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, a.out)


def waves(a):
    sel = [n for n in CL if n[0] in a.closures]
    fig, axs = plt.subplots(2, len(a.cases), figsize=(5.2 * len(a.cases), 8.2), gridspec_kw=dict(height_ratios=[1.5, 1]))
    axs = np.atleast_2d(axs).reshape(2, -1)
    for j, case in enumerate(a.cases):
        r = load_rom(os.path.join(OUT, sel[0][0], case), a.t0); x = r[1]; m = (x > X0) & (x < X1)
        td, HD = load_dns(case, a.t0, x)
        curves = [("DNS", NAVY, HD)] + [(lab, c, np.asarray(load_rom(os.path.join(OUT, n, case), a.t0)[2]))
                                        for n, lab, c, _ in sel if load_rom(os.path.join(OUT, n, case), a.t0) is not None]
        ax = axs[0, j]
        for k, (lab, c, H) in enumerate(curves):
            ax.plot(x[m], H[-1][m] + a.offset * (len(curves) - 1 - k), color=c, lw=1.3)
            ax.text(X1 + 8, 1 + a.offset * (len(curves) - 1 - k), lab, color=c, fontsize=11, va="center")
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("x")
        ax.set_title(label(case).replace("\n", ", "), loc="left", fontweight="bold", fontsize=13)
        ax = axs[1, j]; bins = np.linspace(0, max(H[:, m].max() for _, _, H in curves) * 1.02, 70)
        for lab, c, H in curves:
            p, e = np.histogram(H[:, m].ravel(), bins=bins, density=True)
            ax.semilogy(0.5 * (e[1:] + e[:-1]), np.where(p > 0, p, np.nan), color=c, lw=2.2 if lab == "DNS" else 1.6, label=lab)
        ax.set_xlabel("h"); ax.set_ylim(1e-3, None)
    axs[0, 0].set_ylabel(f"h(x) at t = {a.t_end:g} (offset)"); axs[1, 0].set_ylabel("PDF of h, t ≥ 1000")
    fig.tight_layout(); save(fig, a.out)


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); fig.savefig(out); print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["metrics", "waves"])
    ap.add_argument("--csv"); ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=["Re020_Ca0_010_theta45", "Re120_Ca0_020_theta90", "Re200_Ca0_010_theta6_4"])
    ap.add_argument("--closures", nargs="+", default=["lavalle", "mlp_s0.5_L10_seed0"])
    ap.add_argument("--offset", type=float, default=2.2); ap.add_argument("--t0", type=float, default=1000.0); ap.add_argument("--t_end", type=float, default=2000.0)
    a = ap.parse_args(); style()
    metrics(a) if a.cmd == "metrics" else waves(a)


if __name__ == "__main__":
    main()
