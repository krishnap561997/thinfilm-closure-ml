#!/usr/bin/env python3
"""Figures for the APS DFD 2026 Interact e-poster (docs/meetings/2026-11-aps-dfd/story_v2.md).

Poster canvas 36 x 20.25 in, three columns about 11.5 in wide: figures are made at their size on the poster, with
18-20 pt text. One colour per model everywhere: DNS / Orr-Sommerfeld navy, Lavalle grey, plain MLP orange,
physics-constrained MLP teal (exact DNS closure: navy dashed).

    LD_LIBRARY_PATH=$THINFILM_ENV/lib python scripts/meetings/aps_dfd_2026_figs.py [equations oracle method comb waves] --out DIR
"""
import argparse, csv, os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STORE = os.environ["THINFILM_STORE"]
SC = os.path.join(STORE, "codes/ml-models/structured_closure")
ORACLE = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom")
sys.path.insert(0, os.path.join(ROOT, "ml/structured_closure"))
sys.path.insert(0, os.path.join(ROOT, "scripts/structured-closure"))

NAVY, GRAY, ORANGE, TEAL = "#0B3C5D", "#8A9AA3", "#E8871E", "#1D7874"
BLUE_UF, ORANGE_UF = "#0021A5", "#FA4616"
MODEL = "mlp_s0.5_L10_seed0"
LAB = dict(dns="DNS", lav="Lavalle", plain="plain MLP", cons="constrained MLP", exact="exact closure (from DNS)")


def style(fs=18):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                         "font.size": fs, "axes.titlesize": fs + 1, "axes.labelsize": fs + 1, "legend.fontsize": fs - 1,
                         "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1, "axes.linewidth": 1.3,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.08})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True); fn = os.path.join(out, name); fig.savefig(fn); plt.close(fig); print("wrote", fn)


# ---------------------------------------------------------------- Panel 2: equations with the unclosed terms
def equations(out):
    # mathtext pieces laid out left to right; each unclosed term gets its colour, a bracket and its name underneath
    style(); plt.rcParams["mathtext.fontset"] = "cm"
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11, 3.6)); r = fig.canvas.get_renderer(); W, H = fig.get_size_inches() * fig.dpi
    fs = 26
    fig.text(0.02, 0.88, r"$\partial_t h + \partial_x q = 0$", fontsize=fs, va="baseline")
    parts = [(r"$\partial_t q\ +\ $", "k", None), (r"$\partial_x\!\int_0^h u^2\,dz$", "#6A3D9A", r"$M_x$"),
             (r"$\ =\ $", "k", None), (r"$-\!\int_0^h \partial_x p\,dz$", "#1F78B4", r"$P_x$"),
             (r"$\ -\ $", "k", None), (r"$\frac{1}{Re}\,\partial_z u|_0$", "#D62728", r"$T_w$"),
             (r"$\ +\ $", "k", None), (r"$\frac{1}{Re}\int_0^h \partial_{xx}u\,dz$", "#2CA02C", r"$D_x$"),
             (r"$\ +\ \frac{3}{Re}\,h$", "k", None)]
    x, y = 0.02, 0.60
    for s, c, lab in parts:
        t = fig.text(x, y, s, fontsize=fs, color=c, va="baseline")
        bb = t.get_window_extent(r); x0, x1, yb = bb.x0 / W, bb.x1 / W, bb.y0 / H
        if lab:
            yl = yb - 0.035
            fig.add_artist(Line2D([x0, x0, x1, x1], [yl + 0.03, yl, yl, yl + 0.03], transform=fig.transFigure, color=c, lw=2.2))
            fig.text((x0 + x1) / 2, yl - 0.02, lab, fontsize=fs - 2, color=c, ha="center", va="top")
        x = x1 / 1.0 + 0.004
    fig.text(0.02, 0.20, r"Lavalle et al. (2015): parabolic $u(z)$ $\Rightarrow$ closed terms $C_{\mathrm{Lav}}(h,q)$", fontsize=20, va="baseline")
    fig.text(0.02, 0.03, r"ML ROM: $C = C_{\mathrm{Lav}} + \Delta C$; the network learns $S = \Delta M_x + \Delta T_w$", fontsize=20,
             va="baseline", color=TEAL)
    save(fig, out, "p2_equations.png")


# ---------------------------------------------------------------- Panel 3: oracle error against time, and h(x) late
def oracle(out):
    style()
    rows = list(csv.DictReader(open(os.path.join(SC, "rom", MODEL, "oracle_summary.csv"))))
    runs = [("lavalle", LAB["lav"], GRAY, "-", 3.0), ("oracle_TwMx", "exact closure", NAVY, "--", 2.6),
            ("ml_sum_hhx", LAB["plain"], ORANGE, "-", 3.0), ("structured", LAB["cons"], TEAL, "-", 3.6)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for key, lab, c, ls, lw in runs:
        r = [x for x in rows if x["run"] == key]
        t = np.array([float(x["elapsed"]) for x in r]); e = np.array([float(x["h_rel"]) for x in r])
        k = np.argsort(t); ax.plot(t[k], e[k], ls, color=c, lw=lw, label=lab)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.3)
    ax.set_xlabel("time from the DNS state"); ax.set_ylabel("h error / std(h$_{DNS}$)")
    ax.legend(loc="upper left", fontsize=15, handlelength=1.6, borderaxespad=0.1)
    with h5py.File(os.path.join(ORACLE, "rom_forcing.h5")) as f:
        tD, x = f["t"][:], f["x"][:]; kD = int(np.argmin(abs(tD - (tD[0] + 100)))); hD = f["dns/h"][kD]
    def at(path):
        with h5py.File(path) as r:
            tR = r["time"][:]; return r["h"][int(np.argmin(abs(tR - tD[kD])))]
    hP = at(os.path.join(ORACLE, "runs/ml_sum_hhx/thinfilm_training_data.h5"))
    hC = at(os.path.join(SC, "rom", MODEL, "oracle/thinfilm_training_data.h5"))
    w = (x > 600) & (x < 760)
    ax2.plot(x[w], hD[w], color=NAVY, lw=2.8, label="DNS"); ax2.plot(x[w], hP[w], color=ORANGE, lw=2.0); ax2.plot(x[w], hC[w], color=TEAL, lw=2.0)
    ax2.set_xlabel("x"); ax2.set_ylabel("h at t = 100"); ax2.set_ylim(0.3, 2.75)
    ax2.legend(loc="upper left", fontsize=15, borderaxespad=0.1)
    ax2.annotate("crests too tall", xy=(664, 2.5), xytext=(682, 2.62), fontsize=14, color=ORANGE, va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
    fig.text(0.99, -0.02, "oracle case: Re 120, Ca 0.01, β 45°, 10 Hz forcing", ha="right", fontsize=13, color="#555555")
    fig.tight_layout(w_pad=1.5); save(fig, out, "p3_oracle_error.png")


# ---------------------------------------------------------------- Panel 4: method schematic
def method(out):
    style()
    fig, ax = plt.subplots(figsize=(11, 4.3)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.3); ax.axis("off")
    def box(x, y, w, h, title, body, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.14", ec=ec, fc="white", lw=2.4))
        ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=17, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 0.66, body, ha="center", va="top", fontsize=14.5, linespacing=1.35)
    def arrow(x0, y0, x1, y1, c="#333333"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=24, lw=2.6, color=c))
    y, h = 2.25, 1.95
    box(0.03, y, 2.2, h, "2D DNS", "Basilisk\n15 training\ncases", NAVY)
    box(2.58, y, 2.55, h, "Closure data", "depth-integrate\nS = ΔM$_x$ + ΔT$_w$", NAVY)
    box(5.48, y, 3.0, h, "Neural network", "inputs h, q, h$_x$, q$_x$, h$_{xx}$\nand Re, Ca, β\n2 layers × 64", TEAL)
    box(8.83, y, 2.14, h, "ROM", "Lavalle\n+ learned S", TEAL)
    for x0, x1 in [(2.25, 2.56), (5.15, 5.46), (8.5, 8.81)]:
        arrow(x0, y + h / 2, x1, y + h / 2)
    box(0.9, 0.03, 4.6, 1.75, "1. Flat film exact", "S = N(h, q, …) − N(h, h³, 0, 0, 0)\nno correction for a uniform film", ORANGE_UF)
    box(5.85, 0.03, 4.25, 1.75, "2. Linear stability", "Orr–Sommerfeld growth rates\nas a term in the loss", ORANGE_UF)
    arrow(4.6, 1.8, 6.3, 2.23, ORANGE_UF); arrow(7.4, 1.8, 7.1, 2.23, ORANGE_UF)
    save(fig, out, "p4_method.png")


# ---------------------------------------------------------------- Panel 4: linear growth against OS, Re 20/120/200
def comb(out):
    style()
    from comb_growth_fig import modes
    cases = [("Re020_Ca0_010_theta45", "Re 20"), ("Re120_Ca0_010_theta45", "Re 120"), ("Re200_Ca0_010_theta45", "Re 200")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.3))
    for ax, (case, title) in zip(axs, cases):
        R = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(STORE, f"analysis/orr_sommerfeld/comb_dns/comb_modes_{case}.csv")))]
        w = np.array([r["omega"] for r in R]); rel = np.array([r["reliable"] for r in R]) > 0
        ax.plot(w, [r["os_mki"] for r in R], color=NAVY, lw=3, label="Orr–Sommerfeld")
        d = np.array([r["dns_mki"] for r in R])
        ax.plot(w[rel], d[rel], "o", mfc="none", mec=NAVY, mew=1.8, ms=10, label="DNS")
        M = modes(os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom", case, "lavalle"))
        ax.plot(M[:, 0], M[:, 1], "s", color=GRAY, ms=8, mec="white", mew=0.6, label=LAB["lav"])
        M = modes(os.path.join(SC, "rom", MODEL, f"comb_{case}"))
        ax.plot(M[:, 0], M[:, 1], "D", color=TEAL, ms=8, mec="white", mew=0.6, label=LAB["cons"])
        ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel("frequency ω")
        ax.set_xlim(0, (w[rel].max() if rel.any() else w.max()) * 1.1); ax.set_ylim(0, None)
        ax.locator_params(axis="both", nbins=4)
    axs[0].set_ylabel("growth rate −k$_i$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=16, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6); save(fig, out, "p4_linear_growth.png")


# ---------------------------------------------------------------- Panel 5: held-out cases, h(x) at t = 2000 and h PDFs
def waves(out):
    style()
    from broadband import OUT, load_dns, load_rom, X0, X1
    cases = [("Re020_Ca0_010_theta45", "Re 20, Ca 0.01, β 45°"), ("Re120_Ca0_020_theta90", "Re 120, Ca 0.02, β 90°"),
             ("Re200_Ca0_010_theta6_4", "Re 200, Ca 0.01, β 6.4°")]
    cl = [("lavalle", LAB["lav"], GRAY), ("ml_sum_hhx", LAB["plain"], ORANGE), (MODEL, LAB["cons"], TEAL)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.6), gridspec_kw=dict(height_ratios=[1.55, 1]))
    off = 1.8; x1 = 420.0
    for j, (case, title) in enumerate(cases):
        r = load_rom(os.path.join(OUT, cl[0][0], case), 1000.0); x = r[1]; m = (x > X0) & (x < X1); mw = (x > 120) & (x < x1)
        _, HD = load_dns(case, 1000.0, x)
        curves = [(LAB["dns"], NAVY, HD)] + [(lab, c, np.asarray(load_rom(os.path.join(OUT, n, case), 1000.0)[2])) for n, lab, c in cl]
        ax = axs[0, j]; n = len(curves)
        for k, (lab, c, H) in enumerate(curves):
            ax.plot(x[mw], H[-1][mw] + off * (n - 1 - k), color=c, lw=1.7)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("x"); ax.set_xticks([200, 300, 400])
        ax.set_ylim(-0.2, off * (n - 1) + 3.4)
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
        ax = axs[1, j]; bins = np.linspace(0, 3.2, 65)
        for lab, c, H in curves:
            p, e = np.histogram(H[:, m].ravel(), bins=bins, density=True)
            ax.semilogy(0.5 * (e[1:] + e[:-1]), np.where(p > 0, p, np.nan), color=c, lw=3 if lab == "DNS" else 2.2, label=lab)
        ax.set_xlabel("h"); ax.set_ylim(2e-3, 8); ax.set_xlim(0, 3.2)
        if j: ax.set_yticklabels([])
    axs[0, 0].set_ylabel("h(x), t = 2000\n(offset)"); axs[1, 0].set_ylabel("PDF of h")
    axs[0, 2].text(0.99, 1.0, "fails: ML waves 2× too long", transform=axs[0, 2].transAxes, ha="right", va="top",
                   fontsize=14, color="#B00020", style="italic")
    h, l = axs[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=16, columnspacing=1.2)
    fig.tight_layout(h_pad=0.8, w_pad=0.5); save(fig, out, "p5_heldout_waves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["equations", "oracle", "method", "comb", "waves"])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/meetings/2026-11-aps-dfd/poster_figs"))
    a = ap.parse_args()
    for f in a.figs:
        globals()[f](a.out)


if __name__ == "__main__":
    main()
