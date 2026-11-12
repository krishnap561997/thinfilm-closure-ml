"""Shared loader: DNS and ROM runs for the ML a-posteriori test (Re120 Ca0.01 β45, 10 Hz, t = 1800-1900).

Each comparison set is one figure folder, docs/results/figures/ml-a-posteriori/comparisons_<date>/:
  2026-11-05  learned ΔM_x + ΔT_w (sum model and separate models) against Lavalle and the oracle
  2026-11-12  the same plus the ΔP_x v2 MLPs (G_s0p5, H_s0p5)
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np
import style as S

T0 = 1800.0
LAV, ORC = "Lavalle only", "Oracle " + S.DTW + " + " + S.DMX
SUM, SEP = r"ML sum $(h,\ h_x)$", "ML " + S.DMX + " + ML " + S.DTW
# label: (run folder, colour, linestyle, short name for panel titles)
_BASE = {
    LAV: ("lavalle", S.BLUE, "-", "Lav"),
    ORC: ("Tw_Mx", S.MUTED, ":", "oracle"),
    SUM: ("ml_sum_hhx", S.ORANGE, "-", "sum"),
}
SETS = {
    "2026-11-05": dict(runs={**_BASE, SEP: ("ml_mx_tw", S.MAGENTA, "--", "sep.")},
                       t_max=None, ref=SUM, times=[1.0, 5.0, 20.0, 50.0, 75.0, 99.9]),
    "2026-11-12": dict(runs={**_BASE,
                             SUM + " + ML " + S.DPX + " (G)": ("ml_sum_pxG", S.GREEN, "-", "sum+PG"),
                             SEP + " + ML " + S.DPX + " (G)": ("ml_mx_tw_pxG", S.YELLOW, "--", "sep.+PG"),
                             SUM + " + ML " + S.DPX + " (H)": ("ml_sum_pxH", S.AQUA, "-.", "sum+PH")},
                       t_max=None, ref=SUM + " + ML " + S.DPX + " (G)",
                       times=[1.0, 5.0, 20.0, 50.0, 75.0, 99.9]),
}


def get_set():
    """Parse --set DATE and return (date, set dict, output folder name)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default=max(SETS))
    d = ap.parse_args().set
    return d, SETS[d], f"ml-a-posteriori/comparisons_{d}"


def load(runs, times=None, t_max=None):
    """Return elapsed times, x, rom-grid DNS h and each run's h at those times.

    times=None: every saved time of the Lavalle run, up to t_max (absolute time) if given.
    """
    with h5py.File(S.ORACLE / "rom/rom_forcing.h5") as f:
        tD, x, hD = f["t"][:], f["x"][:], f["dns/h"]
        if times is None:
            with h5py.File(S.ORACLE / "rom/runs/lavalle/thinfilm_training_data.h5") as r:
                times = r["time"][:] - T0
            if t_max is not None:
                times = times[times <= t_max - T0 + 1e-9]
        kD = [int(np.argmin(abs(tD - (T0 + e)))) for e in times]
        dns = np.stack([hD[k] for k in kD])
    rom = {}
    for name, (d, *_) in runs.items():
        with h5py.File(S.ORACLE / f"rom/runs/{d}/thinfilm_training_data.h5") as f:
            t = f["time"][:]
            idx = [int(np.argmin(abs(t - (T0 + e)))) for e in times]
            if max(abs(t[idx] - (T0 + np.asarray(times)))) > 0.06:
                raise ValueError(f"{d}: saved times do not cover the requested ones (last {t.max():g})")
            rom[name] = np.stack([f["h"][i] for i in idx])
    return np.asarray(times), x, dns, rom
