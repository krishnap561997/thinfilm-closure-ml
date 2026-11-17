"""Figures for the 17 Nov 2026 advisor deck (docs/meetings/, deck and PNGs not in git).

F1 oracle a posteriori drift of the existing ML closure     F2 comb DNS vs OS vs Lavalle (Re 80/120/200)
F3 Lavalle linear error vs delta_lin, and the (a,b,c,d) fit  F4 structured-closure schematic; ROM comb growth
F5 oracle a posteriori incl. structured; net closure power on the oracle case
Usage: python advisor_2026_11_17_figs.py [--extra NAME=PATH_TO_oracle_summary.csv ...]  (adds runs to F5a)
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STORE = os.environ["THINFILM_STORE"]
OUT = os.path.join(ROOT, "docs/meetings/figures_2026-11-17")
SC = os.path.join(STORE, "codes/ml-models/structured_closure")
NAVY, TEAL, CORAL, GOLD, GRAY, LIGHT = "#0B3C5D", "#1D7874", "#E4572E", "#F2A541", "#7A8B94", "#C9D6DF"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#5C6B73",
                     "axes.labelcolor": "#2B3A42", "xtick.color": "#2B3A42", "ytick.color": "#2B3A42", "axes.titleweight": "bold",
                     "axes.titlesize": 15, "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})


def series(fn, run):
    t, e, hm, hd = [], [], [], []
    for r in csv.DictReader(open(fn)):
        if r["run"] == run:
            t.append(float(r["elapsed"])); e.append(float(r["h_rel"])); hm.append(float(r["hmax_rom"])); hd.append(float(r["hmax_dns"]))
    return np.array(t), np.array(e), np.array(hm), np.array(hd)


def f1(summ):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run, lab, c, ls, lw in (("lavalle", "Lavalle (no closure)", GRAY, "-", 2), ("oracle_TwMx", "exact closure (DNS ΔT$_w$+ΔM$_x$)", TEAL, "--", 2.4),
                                ("ml_sum_hhx", "current ML closure", CORAL, "-", 2.8)):
        t, e, *_ = series(summ, run); ax.plot(t, e, ls, color=c, lw=lw, label=lab)
    ax.axvspan(10, 100, color=CORAL, alpha=0.07); ax.text(78, 0.08, "ML closure drifts", color=CORAL, ha="center", fontsize=14, weight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.35)
    ax.set_xlabel("time since ROM start (h$_N$/u$_N$)"); ax.set_ylabel("h error / std(h$_{DNS}$)")
    ax.set_title("Oracle case: Re 120, Ca 0.01, β 45°, 10 Hz", loc="left")
    ax.legend(loc="upper left", fontsize=12.5)
    fig.savefig(os.path.join(OUT, "f1_ml_drift.png")); plt.close(fig)


def f2():
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=False)
    for ax, (c, lab) in zip(axs, (("Re080_Ca0_010_theta45", "Re 80"), ("Re120_Ca0_010_theta45", "Re 120"), ("Re200_Ca0_010_theta45", "Re 200"))):
        R = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(STORE, f"analysis/orr_sommerfeld/comb_dns/comb_modes_{c}.csv")))]
        w = np.array([r["omega"] for r in R]); rel = np.array([r["reliable"] for r in R]) > 0
        ax.plot(w, [r["os_mki"] for r in R], color=NAVY, lw=2.6, label="Orr–Sommerfeld (exact)")
        ax.plot(w, [r["lav_mki"] for r in R], "--", color=GRAY, lw=2.2, label="Lavalle model")
        d = np.array([r["dns_mki"] for r in R])
        ax.plot(w[rel], d[rel], "o", color=CORAL, ms=7, mec="white", mew=0.8, label="DNS (comb forcing)")
        ax.set_title(lab, loc="left"); ax.set_xlabel("frequency ω"); ax.set_xlim(0, w[rel].max() * 1.08); ax.set_ylim(0, None)
    axs[0].set_ylabel("spatial growth rate −k$_i$"); axs[0].legend(fontsize=12, loc="lower center")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "f2_comb_os.png")); plt.close(fig)


def f3():
    R = list(csv.DictReader(open(os.path.join(STORE, "analysis/orr_sommerfeld/linear_targets.csv"))))
    R = [r for r in R if r.get("band_lav_err_ki") not in (None, "")]
    d = np.array([float(r["delta_lin"]) for r in R]); lav = np.array([float(r["band_lav_err_ki"]) for r in R])
    fit = np.array([float(r["band_abcd_err_ki"]) for r in R])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhspan(0, 0.10, color=TEAL, alpha=0.08); ax.text(1.15, 0.085, "within 10%", color=TEAL, fontsize=12.5, va="top")
    ax.scatter(d, lav, s=42, color=GRAY, alpha=0.85, label="Lavalle model", edgecolor="white", lw=0.6)
    ax.scatter(d, fit, s=42, color=TEAL, label="Lavalle + 4 linear coefficients (a,b,c,d)", edgecolor="white", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("δ$_{lin}$ = k$_{max}$ Re  (inertia of the most unstable wave)")
    ax.set_ylabel("max growth error / max OS growth"); ax.set_ylim(0, max(0.6, lav.max() * 1.05))
    ax.set_title(f"All {len(R)} DNS-table cases, forced band", loc="left"); ax.legend(loc="upper left", fontsize=12.5)
    fig.savefig(os.path.join(OUT, "f3_linear_error.png")); plt.close(fig)


def f4a():
    fig, ax = plt.subplots(figsize=(7.0, 3.9)); ax.set_axis_off()
    x = np.linspace(0, 10, 600)
    h = 1 + 0.85 * np.exp(-((x - 5.2) / 0.9) ** 2) * (1 + 0.25 * np.tanh(-(x - 5.2) * 3)) - 0.18 * np.exp(-((x - 3.6) / 0.35) ** 2) * np.cos(9 * (x - 3.6)) * 0.6
    ax.fill_between(x, 0, h, color=LIGHT, alpha=0.9); ax.plot(x, h, color=NAVY, lw=2.4); ax.plot([0, 10], [0, 0], color="#2B3A42", lw=3)
    ax.text(0.15, -0.28, "wall", fontsize=12, color="#2B3A42")
    ax.annotate("", xy=(2.0, 1.45), xytext=(0.4, 1.45), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2)); ax.text(0.4, 1.57, "flow", color=NAVY, fontsize=12)
    ax.text(5.4, 2.55, "Flux part  Φ(h), ΔP(h)", color=TEAL, fontsize=14, weight="bold", ha="center")
    ax.text(5.4, 2.25, "inside ∂ₓ(·): conservative, sets wave speeds", color=TEAL, fontsize=12, ha="center")
    ax.text(5.4, 0.35, "Relaxation part  κ(Q − q)/h²", color=CORAL, fontsize=14, weight="bold", ha="center")
    ax.text(5.4, 0.08, "pulls q toward Q = h³(1 + g₁hₓ + g₂qₓ + g₃hₓₓ)", color=CORAL, fontsize=12, ha="center")
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 2.8)
    fig.savefig(os.path.join(OUT, "f4a_schematic.png"), transparent=True); plt.close(fig)


def f4b():
    sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld/b6_rom")); sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld"))
    from analyse import run_modes
    case = "Re120_Ca0_010_theta45"
    R = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(STORE, f"analysis/orr_sommerfeld/comb_dns/comb_modes_{case}.csv")))]
    wos = np.array([r["omega"] for r in R]); gos = np.array([r["os_mki"] for r in R])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(wos, gos, color=NAVY, lw=2.8, label="Orr–Sommerfeld (exact)")
    for d, lab, c, mk in ((os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom", case, "lavalle"), "ROM: Lavalle", GRAY, "s"),
                          (os.path.join(SC, "rom/v1_s0.5_L0_F0_seed0", f"comb_{case}"), "ROM: structured, no OS constraint", GOLD, "^"),
                          (os.path.join(SC, "rom/v1_s0.25_L10_F0_seed0", f"comb_{case}"), "ROM: structured + OS constraint", TEAL, "o")):
        M, info = run_modes(d)
        if M is None:
            continue
        L = float([l for l in open(os.path.join(d, "input.txt")) if l.startswith("L ")][0].split("=")[1])
        rel = (M[:, 3] > 0.95) & (2 * np.pi / np.abs(M[:, 2]) < 0.35 * L)
        ax.plot(M[rel, 0], M[rel, 1], mk, color=c, ms=7.5, mec="white", mew=0.8, label=lab)
    ax.set_xlim(0, 1.55); ax.set_ylim(0, None); ax.set_xlabel("frequency ω"); ax.set_ylabel("spatial growth rate −k$_i$")
    ax.set_title("Linear waves in the ROM, Re 120", loc="left"); ax.legend(fontsize=13, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(os.path.join(OUT, "f4b_rom_comb.png")); plt.close(fig)


def f5a(summ_struct, extra):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = [(summ_struct, "lavalle", "Lavalle", GRAY, "-", 2), (summ_struct, "oracle_TwMx", "exact closure", TEAL, "--", 2.4),
            (summ_struct, "ml_sum_hhx", "current ML closure", CORAL, "-", 2.4), (summ_struct, "structured", "structured V1 (σ 0.25, OS constraint)", NAVY, "-", 2.8)]
    cols = [GOLD, "#8E5572", "#3E8E7E"]
    for k, (name, fn) in enumerate(extra):
        rows.append((fn, "structured", name, cols[k % 3], "-", 2.4))
    for fn, run, lab, c, ls, lw in rows:
        t, e, *_ = series(fn, run); ax.plot(t, e, ls, color=c, lw=lw, label=lab)
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.0); ax.set_xlabel("time since ROM start (h$_N$/u$_N$)"); ax.set_ylabel("h error / std(h$_{DNS}$)")
    ax.set_title("Oracle case a posteriori", loc="left"); ax.legend(fontsize=11.5, loc="upper left")
    fig.savefig(os.path.join(OUT, "f5a_aposteriori.png")); plt.close(fig)


def f5b():
    # net closure power <(-S) q'> on the oracle-case DNS states (diag_oracle.py, 17 Nov): ratio to DNS x DNS value
    P_dns = -3.43e-4
    vals = [("DNS\n(true)", P_dns, TEAL), ("current\nMLP", -6.50 * P_dns, CORAL), ("V1\nσ 0.5", -5.26 * P_dns, NAVY),
            ("V1\nσ 0.25", -2.97 * P_dns, NAVY)]
    fig, ax = plt.subplots(figsize=(4.7, 4.6))
    xs = np.arange(len(vals)); v = np.array([x[1] for x in vals]) * 1e3
    ax.bar(xs, v, color=[x[2] for x in vals], width=0.62)
    ax.axhline(0, color="#2B3A42", lw=1.2)
    for x, y in zip(xs, v):
        ax.text(x, y + (0.08 if y >= 0 else -0.08), f"{y:+.2f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=13, weight="bold")
    ax.set_xticks(xs, [x[0] for x in vals], fontsize=13); ax.set_ylabel("net power into waves (×10⁻³)")
    ax.set_title("Closure energy input, oracle case", loc="left"); ax.set_ylim(-0.75, 3.1)
    ax.text(0, 0.18, "damps", ha="center", color=TEAL, fontsize=12.5, weight="bold"); ax.text(2.0, 2.92, "learned closures feed the waves", ha="center", color=CORAL, fontsize=13, weight="bold")
    fig.savefig(os.path.join(OUT, "f5b_power.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", nargs="*", default=[]); A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summ = os.path.join(SC, "rom/v1_s0.25_L10_F0_seed0/oracle_summary.csv")
    f1(summ); f2(); f3(); f4a(); f4b(); f5a(summ, [tuple(e.split("=", 1)) for e in A.extra]); f5b()
    print("figures ->", OUT)
