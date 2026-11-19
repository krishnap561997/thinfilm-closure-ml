#!/usr/bin/env python3
"""Linear waves in the ROM (comb inlet) for chosen closures against Orr-Sommerfeld and the comb DNS, Re 20/120/200.

Spatial growth rate -k_i against frequency: OS (line), comb DNS (reliable modes), ROM runs from
rom_tests.py (comb_<case>/ directories; modes from analysis/orr_sommerfeld/b6_rom/analyse.run_modes).

    python scripts/structured-closure/comb_growth_fig.py --run "MLP + OS constraint=mlp_s0.5_L10_seed0" \
        --run "MLP, no constraint=mlp_s0.25_L0_seed0" --out fig.png
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
STORE = os.environ["THINFILM_STORE"]
SC = os.path.join(STORE, "codes/ml-models/structured_closure")
sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld/b6_rom")); sys.path.insert(0, os.path.join(ROOT, "analysis/orr_sommerfeld"))
from analyse import run_modes  # noqa: E402

NAVY, TEAL, CORAL, GOLD, GRAY = "#0B3C5D", "#1D7874", "#E4572E", "#F2A541", "#7A8B94"
STYLE = [(TEAL, "o"), (GOLD, "^"), ("#6A4C93", "D"), ("#8D99AE", "v")]
CASES = [("Re020_Ca0_010_theta45", "Re 20"), ("Re120_Ca0_010_theta45", "Re 120"), ("Re200_Ca0_010_theta45", "Re 200")]


def modes(d):
    M, _ = run_modes(d)
    if M is None:
        return None
    L = float([l for l in open(os.path.join(d, "input.txt")) if l.startswith("L ")][0].split("=")[1])
    rel = (M[:, 3] > 0.95) & (2 * np.pi / np.abs(M[:, 2]) < 0.35 * L)
    return M[rel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="LABEL=structured_closure run name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 220, "savefig.bbox": "tight"})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, (case, title) in zip(axs, CASES):
        R = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(STORE, f"analysis/orr_sommerfeld/comb_dns/comb_modes_{case}.csv")))]
        w = np.array([r["omega"] for r in R]); rel = np.array([r["reliable"] for r in R]) > 0
        ax.plot(w, [r["os_mki"] for r in R], color=NAVY, lw=2.6, label="Orr–Sommerfeld (exact)", zorder=1)
        d = np.array([r["dns_mki"] for r in R])
        ax.plot(w[rel], d[rel], "o", mfc="none", mec=CORAL, mew=1.6, ms=9, label="DNS (comb forcing)", zorder=2)
        M = modes(os.path.join(STORE, "analysis/orr_sommerfeld/b6_rom", case, "lavalle"))
        if M is not None:
            ax.plot(M[:, 0], M[:, 1], "s", color=GRAY, ms=6.5, mec="white", mew=0.6, label="ROM: Lavalle", zorder=3)
        for (spec, (c, mk)) in zip(a.run, STYLE):
            lab, run = spec.split("=", 1)
            M = modes(os.path.join(SC, "rom", run, f"comb_{case}"))
            if M is not None:
                ax.plot(M[:, 0], M[:, 1], mk, color=c, ms=7, mec="white", mew=0.6, label=f"ROM: {lab}", zorder=4)
        ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel("frequency ω")
        ax.set_xlim(0, (w[rel].max() if rel.any() else w.max()) * 1.1); ax.set_ylim(0, None)
    axs[0].set_ylabel("spatial growth rate −k$_i$")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.08), fontsize=12)
    fig.tight_layout(); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
