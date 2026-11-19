#!/usr/bin/env python3
"""h error against time for the dense broadband a posteriori runs (csv from dense_aposteriori.py analyse --csv).

    python scripts/structured-closure/dense_error_fig.py --csv dense_summary.csv --out fig.png [--label run=LABEL ...]
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"lavalle": ("#7A8B94", "-", "Lavalle"), "ml_sum_hhx": ("#E4572E", "-", "previous MLP"),
         "v1_s0.25_L10_F0_seed0": ("#F2A541", "-", "structured V1"),
         "mlp_s0.25_L10_seed0": ("#6A4C93", "--", "MLP + OS, σ 0.25"), "mlp_s0.5_L10_seed0": ("#1D7874", "-", "MLP + OS, σ 0.5")}
TITLE = {"Re020_Ca0_010_theta45": "Re 20, Ca 0.01, β 45° (held out)", "Re120_Ca0_010_theta45": "Re 120, Ca 0.01, β 45° (held out)",
         "Re040_Ca0_002_theta45": "Re 40, Ca 0.002, β 45° (training case)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    cases = list(dict.fromkeys(r["case"] for r in rows))
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 220, "savefig.bbox": "tight"})
    fig, axs = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.0), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), cases):
        for run, (col, ls, lab) in STYLE.items():
            rr = [r for r in rows if r["case"] == c and r["closure"] == run]
            if rr:
                ax.plot([float(r["elapsed"]) for r in rr], [float(r["err"]) for r in rr], ls, color=col, lw=2.2, label=lab)
        ax.set_title(TITLE.get(c, c), loc="left", fontsize=12, fontweight="bold"); ax.set_xlabel("time since DNS state")
        ax.set_xlim(0, 10)
    np.atleast_1d(axs)[0].set_ylim(0, 1.05 * max(float(r["err"]) for r in rows))
    np.atleast_1d(axs)[0].set_ylabel("h error / std(h$_{DNS}$)")
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
