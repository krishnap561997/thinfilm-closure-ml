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
    # mathtext pieces laid out left to right; each unclosed term gets its colour, a bracket and its name underneath.
    # Both equations centred.
    style(); plt.rcParams["mathtext.fontset"] = "cm"
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11, 2.4)); r = fig.canvas.get_renderer(); W, H = fig.get_size_inches() * fig.dpi
    fs = 26
    fig.text(0.5, 0.82, r"$\partial_t h + \partial_x q = 0$", fontsize=fs, va="baseline", ha="center")
    parts = [(r"$\partial_t q\ +\ $", "k", None), (r"$\partial_x\!\int_0^h u^2\,dz$", "#6A3D9A", r"$M_x$"),
             (r"$\ =\ $", "k", None), (r"$-\!\int_0^h \partial_x p\,dz$", "#1F78B4", r"$P_x$"),
             (r"$\ -\ $", "k", None), (r"$\frac{1}{Re}\,\partial_z u|_0$", "#D62728", r"$T_w$"),
             (r"$\ +\ $", "k", None), (r"$\frac{1}{Re}\int_0^h \partial_{xx}u\,dz$", "#2CA02C", r"$D_x$"),
             (r"$\ +\ \frac{3}{Re}\,h$", "k", None)]
    gap = 0.004; wid = []
    for s_, _, _ in parts:
        t = fig.text(0, 0, s_, fontsize=fs); wid.append(t.get_window_extent(r).width / W); t.remove()
    x, y = 0.5 - (sum(wid) + gap * (len(parts) - 1)) / 2, 0.40
    for s_, c, lab in parts:
        t = fig.text(x, y, s_, fontsize=fs, color=c, va="baseline")
        bb = t.get_window_extent(r); x0, x1, yb = bb.x0 / W, bb.x1 / W, bb.y0 / H
        if lab:
            yl = yb - 0.035
            fig.add_artist(Line2D([x0, x0, x1, x1], [yl + 0.03, yl, yl, yl + 0.03], transform=fig.transFigure, color=c, lw=2.2))
            fig.text((x0 + x1) / 2, yl - 0.02, lab, fontsize=fs - 2, color=c, ha="center", va="top")
        x = x1 + gap
    save(fig, out, "p2_equations.png")


# ---------------------------------------------------------------- Panel 3: oracle error against time, and h(x) late
def oracle(out):
    style()
    rows = list(csv.DictReader(open(os.path.join(SC, "rom", MODEL, "oracle_summary.csv"))))
    runs = [("lavalle", LAB["lav"], GRAY, "-", 3.0), ("oracle_TwMx", "exact closure", NAVY, "--", 2.6),
            ("ml_sum_hhx", LAB["plain"], ORANGE, "-", 3.0), ("structured", LAB["cons"], TEAL, "-", 3.6)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for a_ in (ax, ax2):
        for sp in ("top", "right"): a_.spines[sp].set_visible(True)
    for key, lab, c, ls, lw in runs:
        r = [x for x in rows if x["run"] == key]
        t = np.array([float(x["elapsed"]) for x in r]); e = np.array([float(x["h_rel"]) for x in r])
        k = np.argsort(t); ax.plot(t[k], e[k], ls, color=c, lw=lw, label=lab)
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.3)
    ax.set_xlabel("t"); ax.set_ylabel(r"$|h - h_\mathrm{DNS}|\,/\,\sigma_{h,\mathrm{DNS}}$")
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
    ax2.set_xlabel("x"); ax2.set_ylabel("h"); ax2.set_title("t = 100"); ax2.set_ylim(0.3, 2.75)
    ax2.legend(loc="upper left", fontsize=15, borderaxespad=0.1)
    ax2.annotate("crests too tall", xy=(664, 2.5), xytext=(682, 2.62), fontsize=14, color=ORANGE, va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
    fig.text(0.99, -0.02, "Re 120, Ca 0.01, β 45°, f = 10 Hz", ha="right", fontsize=13, color="#555555")
    fig.tight_layout(w_pad=0.2); save(fig, out, "p3_oracle_error.png")


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
    box(0.03, y, 2.42, h, "2D DNS", "Basilisk C\n15 training cases", NAVY)
    box(2.80, y, 2.33, h, "Closure data", "Depth-integrate\nS = ΔM$_x$ + ΔT$_w$", NAVY)
    box(5.48, y, 3.0, h, "Neural network", "Input: h, q, h$_x$, q$_x$, h$_{xx}$\nOutput: S", TEAL)
    box(8.83, y, 2.14, h, "ROM", "Lavalle\n+ learned S", TEAL)
    for x0, x1 in [(2.47, 2.78), (5.15, 5.46), (8.5, 8.81)]:
        arrow(x0, y + h / 2, x1, y + h / 2)
    box(0.9, 0.03, 4.6, 1.75, "1. Flat film exact", "S = N(h, q, …) − N(h, h³, 0, 0, 0)\nNo correction for a uniform film", ORANGE_UF)
    box(5.85, 0.03, 4.25, 1.75, "2. Linear stability", "Orr–Sommerfeld growth rates\nas a term in the loss", ORANGE_UF)
    arrow(4.6, 1.8, 6.3, 2.23, ORANGE_UF); arrow(7.4, 1.8, 7.1, 2.23, ORANGE_UF)
    save(fig, out, "p4_method.png")


# ---------------------------------------------------------------- Panel 4: linear growth against OS, Re 20/120/200
def comb(out):
    style()
    from comb_growth_fig import modes
    cases = [("Re020_Ca0_010_theta45", "Re 20"), ("Re120_Ca0_010_theta45", "Re 120"), ("Re200_Ca0_010_theta45", "Re 200")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, (case, title) in zip(axs, cases):
        for sp in ("top", "right"): ax.spines[sp].set_visible(True)
        R = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(STORE, f"analysis/orr_sommerfeld/comb_dns/comb_modes_{case}.csv")))]
        w = np.array([r["omega"] for r in R]); rel = np.array([r["reliable"] for r in R]) > 0
        ax.plot(w, [r["os_mki"] for r in R], color=NAVY, lw=3, label="Orr–Sommerfeld")
        d = np.array([r["dns_mki"] for r in R])
        ax.plot(w[rel], d[rel], "o", mfc="none", mec=NAVY, mew=1.6, ms=8, label="DNS")
        M = modes(os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom", case, "lavalle"))
        ax.plot(M[:, 0], M[:, 1], "s", color=GRAY, ms=8, mec="white", mew=0.6, label=LAB["lav"])
        M = modes(os.path.join(SC, "rom", MODEL, f"comb_{case}"))
        ax.plot(M[:, 0], M[:, 1], "D", color=TEAL, ms=8, mec="white", mew=0.6, label=LAB["cons"])
        ax.text(0.97, 0.93, title, transform=ax.transAxes, ha="right", va="top", fontweight="bold"); ax.set_xlabel("ω", labelpad=1)
        ax.set_xlim(0, (w[rel].max() if rel.any() else w.max()) * 1.1); ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        ax.locator_params(axis="both", nbins=3)
    axs[0].set_ylabel("−k$_i$", labelpad=2)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09), fontsize=15, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=0.4); save(fig, out, "p4_linear_growth.png")


# ---------------------------------------------------------------- Panel 5: held-out cases, h(x) at t = 2000 and h PDFs
def waves(out):
    # two figures: h(x) snapshot of one held-out case (Re 120 Ca 0.02 beta 90), and h PDFs of three held-out cases
    style()
    from broadband import OUT, load_dns, load_rom, X0, X1
    cases = [("Re020_Ca0_010_theta45", "Re 20, Ca 0.01, β 45°"), ("Re120_Ca0_020_theta90", "Re 120, Ca 0.02, β 90°"),
             ("Re200_Ca0_010_theta6_4", "Re 200, Ca 0.01, β 6.4°")]
    snap = 1
    cl = [("lavalle", LAB["lav"], GRAY), ("ml_sum_hhx", LAB["plain"], ORANGE), (MODEL, LAB["cons"], TEAL)]
    fs, a0 = plt.subplots(figsize=(11, 3.8))
    fp, axp = plt.subplots(1, 3, figsize=(11, 3.5))
    off = 1.8; x1 = 420.0
    for j, (case, title) in enumerate(cases):
        r = load_rom(os.path.join(OUT, cl[0][0], case), 1000.0); x = r[1]; m = (x > X0) & (x < X1); mw = (x > 120) & (x < x1)
        _, HD = load_dns(case, 1000.0, x)
        curves = [(LAB["dns"], NAVY, HD)] + [(lab, c, np.asarray(load_rom(os.path.join(OUT, n, case), 1000.0)[2])) for n, lab, c in cl]
        if j == snap:
            n = len(curves)
            for k, (lab, c, H) in enumerate(curves):
                a0.plot(x[mw], H[-1][mw] + off * (n - 1 - k), color=c, lw=1.8, label=lab)
            a0.set_yticks([]); a0.set_xlabel("x"); a0.set_xlim(120, x1)
            for sp in ("top", "right"): a0.spines[sp].set_visible(True)
            a0.set_ylim(-0.2, off * (n - 1) + 3.0); a0.set_ylabel("h")
            a0.set_title(title + " (broadband forcing)", loc="center", fontsize=16, fontweight="bold")
        ax = axp[j]; bins = np.linspace(0, 3.2, 65)
        for sp in ("top", "right"): ax.spines[sp].set_visible(True)
        for lab, c, H in curves:
            p, e = np.histogram(H[:, m].ravel(), bins=bins, density=True)
            ax.plot(0.5 * (e[1:] + e[:-1]), p, color=c, lw=3 if lab == "DNS" else 2.2, label=lab)
        ax.set_xlabel("h", labelpad=1); ax.set_xlim(0, 3.2); ax.set_ylim(bottom=0); ax.set_title(title, fontsize=14, fontweight="bold")
    axp[0].set_ylabel("PDF", labelpad=3)
    for f in (fs, fp):
        h, l = (a0 if f is fs else axp[0]).get_legend_handles_labels()
        f.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08 if f is fs else -0.05), fontsize=16, columnspacing=1.2)
    fs.tight_layout(); save(fs, out, "p5_heldout_snapshot.png")
    fp.suptitle("Broadband forcing", fontsize=16, fontweight="bold")
    fp.tight_layout(w_pad=0.5); fp.subplots_adjust(top=0.80); fp._suptitle.set_y(0.985); save(fp, out, "p5_heldout_pdf.png")


# ---------------------------------------------------------------- Panel 3: a priori fit of the plain MLP, held-out case
FMLP = os.path.join(STORE, "codes/ml-models/functional_mlp")
APRIORI_RUN = os.path.join(FMLP, "runs/ablation_02_sum/layers1_width16/h_hx/delta_MxTw_ML/seed42/eval_heldout")


def apriori(out, case="Re120_Ca0_010_theta45", title="Re 120, Ca 0.01, β 45°", it=-1, x0=500., x1=650.):
    """DNS ΔM_x + ΔT_w against the plain MLP (ml_sum_hhx: inputs h, h_x, Re, Ca, β) at one held-out snapshot."""
    import h5py
    style()
    with h5py.File(os.path.join(APRIORI_RUN, case + ".h5")) as f:
        idx, pred, truth, t = f["indices"][:], f["prediction"][:, 0], f["truth"][:, 0], f["t"][:]
    c = os.path.join(FMLP, "cache_dir", case)
    x, h = np.load(os.path.join(c, "x.npy")), np.load(os.path.join(c, "h.npy"), mmap_mode="r")
    it = it % len(t)
    m = idx[:, 0] == it; j = idx[m, 1]; o = np.argsort(j); j, P, T = j[o], pred[m][o], truth[m][o]
    w = (x[j] >= x0) & (x[j] <= x1); j, P, T = j[w], P[w], T[w]
    nr = np.sqrt(np.mean((P - T) ** 2)) / np.std(T)
    print(f"{case} t = {t[it]:.0f}: {w.sum()} points, snapshot NRMSE {nr:.3f}")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.0))
    for a in (a0, a1):
        for sp in ("top", "right"): a.spines[sp].set_visible(True)
    a0.plot(x[j], h[it, j], color=NAVY, lw=2.2); a0.set_ylabel("h"); a0.set_yticks([1, 2])
    a0.set_xlabel("x"); a0.set_xlim(x0, x1)
    fig.suptitle(f"{title}, t = {t[it]:.0f} (held-out case)", fontsize=16, fontweight="bold")
    a1.axhline(0, color=GRAY, lw=2.2, ls="--", label=LAB["lav"])
    a1.plot(x[j], T, color=NAVY, lw=2.6, label="DNS")
    a1.plot(x[j], P, color=ORANGE, lw=1.8, label=LAB["plain"])
    a1.set_ylabel(r"$\Delta M_x + \Delta T_w$"); a1.set_xlabel("x"); a1.set_xlim(x0, x1)
    a1.set_ylim(-0.9, 2.4)
    for a in (a0, a1):  # region where the correction peaks
        for xv in (550, 585): a.axvline(xv, color="#D62728", ls=":", lw=2.2, zorder=0)
    a1.legend(loc="upper right", fontsize=15, handlelength=1.6, labelspacing=0.3, borderaxespad=0.3)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.88, wspace=0.2)
    fig._suptitle.set_y(0.965)
    save(fig, out, "p3_apriori_fit.png")


# ---------------------------------------------------------------- Column 1: ONR report Figure 2 redrawn (Lavalle vs DNS, monochromatic)
# Same data and times as notebooks/analysis/TechReport_figures.ipynb (cells 12-13).
HPG = os.path.dirname(STORE)
F2_LIU_DNS = os.path.join(HPG, "Setups/basilisk/back2basics/clsvof/LX_2.0_test/infc10.0000.dat")
F2_LIU_ROM = os.path.join(HPG, "thinfilm-solver/1.5Hz/Freq_1.5Hz_time_10.0000s.mat")
F2_L800_DNS = os.path.join(STORE, "codes/basilisk/physics-tests/bc-blowup-fix/length-check/L800/output/snapshot_000460.h5")
F2_L800_ROM = os.path.join(STORE, "codes/rom-solver/inclined-frequency/cases/Re120_Ca0_010_beta45_f10Hz_L800_N4000/thinfilm_training_data.h5")


RED_ONR = "#D62728"  # ONR report colour for Lavalle; this figure only


def lavalle_mono(out):
    import h5py
    from scipy.io import loadmat
    from matplotlib.collections import LineCollection
    style()
    # (A) Re 19.33, beta 6.4, 1.5 Hz, t = 10 s: DNS interface segments, MATLAB ROM
    r = loadmat(F2_LIU_ROM); hN = float(r["hN"].item())
    xA, hA = np.asarray(r["x_dim"]).ravel() / hN, np.asarray(r["hi"]).ravel()
    seg = (np.loadtxt(F2_LIU_DNS) / hN).reshape(-1, 2, 2)
    seg = seg[(seg[:, :, 0].max(1) >= 0) & (seg[:, :, 0].min(1) <= 800)]
    # (B) Re 120, beta 45, 10 Hz, t = 460: DNS column heights from the 2D snapshot, C++ ROM
    with h5py.File(F2_L800_ROM) as f:
        xB, tB = f["x"][:].ravel(), f["time"][:].ravel(); k = int(np.argmin(abs(tB - 460.0))); assert abs(tB[k] - 460) < 1e-8
        hB = f["h"][k, :] if f["h"].shape[0] == len(tB) else f["h"][:, k]
    with h5py.File(F2_L800_DNS) as f:
        P, T, fr = f["Geometry/Points"][:, :2], f["Topology"][:], f["Cells/f"][:].ravel()
    cx = P[T, 0].mean(1); dz = P[T, 1].max(1) - P[T, 1].min(1)
    xD, ic = np.unique(np.round(cx, 12), return_inverse=True); hD = np.bincount(ic, weights=fr * dz)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    a0.add_collection(LineCollection(seg, colors=NAVY, linewidths=2.2))
    m = (xA >= 0) & (xA <= 800); a0.plot(xA[m], hA[m], color=RED_ONR, lw=2.2, ls="--")
    a0.autoscale_view()
    a1.plot(xD, hD, color=NAVY, lw=1.8, label=LAB["dns"]); a1.plot(xB, hB, color=RED_ONR, lw=2.0, ls="--", label=LAB["lav"])
    for a, ttl in ((a0, "Re 19.3, Ca 0.01, β 6.4°, f = 1.5 Hz"), (a1, "Re 120, Ca 0.01, β 45°, f = 10 Hz")):
        for sp in ("top", "right"): a.spines[sp].set_visible(True)
        a.set_xlim(0, 800); a.set_xlabel("x"); a.set_title(ttl, fontsize=16)
    a0.set_ylabel("h"); a1.set_ylim(0.1, 2.3)
    a1.legend(loc="upper left", fontsize=15, ncol=2, borderaxespad=0.2, handlelength=1.6)
    fig.tight_layout(w_pad=0.6); save(fig, out, "p1_lavalle_mono.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=["equations", "oracle", "method", "comb", "waves", "apriori"])
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/meetings/2026-11-aps-dfd/poster_figs"))
    a = ap.parse_args()
    for f in a.figs:
        globals()[f](a.out)


if __name__ == "__main__":
    main()
