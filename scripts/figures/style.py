"""Shared matplotlib style for the figures in docs/results/figures/.

Palette: the validated categorical order (blue, orange, aqua, yellow, magenta, green, violet,
red) on a light surface; text and axes in neutral ink, never in a series colour. Series are
also distinguished by marker/linestyle and direct labels, because aqua and yellow are below 3:1
contrast on the surface. Data are read from $THINFILM_DATA / $THINFILM_STORE (source env.sh).
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "docs" / "results" / "figures"
STORE = Path(os.environ.get("THINFILM_STORE", "/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml"))
DATA = Path(os.environ.get("THINFILM_DATA", STORE / "datasets"))
ORACLE = DATA / "BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz"
CLOSURE = DATA / "BasC/DNS/param_space/ClosureData"
FMLP = STORE / "codes/ml-models/functional_mlp"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1a1a19", "#4a4a47", "#7b7b77", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.labelsize": 10, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 100, "savefig.dpi": 200, "pdf.fonttype": 42,
    })


def save(fig, result, name, png_only=False, dpi=None):
    out = FIGDIR / result
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
    if not png_only:
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    size = (out / f"{name}.png").stat().st_size / 1024
    print(f"wrote {out / name}.png ({size:.0f} KB)")


# Math-text labels used across figures
HN = r"$h/h_N$"
XN = r"$x/h_N$"
DMX, DTW, DPX, DDX = r"$\Delta M_x$", r"$\Delta T_w$", r"$\Delta P_x$", r"$\Delta D_x$"
# the sum DeltaM_x + DeltaT_w (the ROM's net source) gets its own entity colour
SUMC = VIOLET
