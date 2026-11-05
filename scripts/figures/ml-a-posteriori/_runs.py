"""Shared loader: DNS and ROM runs for the ML a-posteriori test (Re120 Ca0.01 β45, 10 Hz, t = 1800-1900)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np
import style as S

T0 = 1800.0
# label: (run folder, colour, linestyle)
RUNS = {
    "Lavalle only": ("lavalle", S.BLUE, "-"),
    "Oracle " + S.DTW + " + " + S.DMX: ("Tw_Mx", S.MUTED, ":"),
    r"ML sum $(h,\ h_x)$": ("ml_sum_hhx", S.ORANGE, "-"),
    "ML " + S.DMX + " + ML " + S.DTW: ("ml_mx_tw", S.MAGENTA, "--"),
}


def load(times=None):
    """Return x, rom-grid DNS h and each run's h at the requested elapsed times (all saved times if None)."""
    with h5py.File(S.ORACLE / "rom/rom_forcing.h5") as f:
        tD, x, hD = f["t"][:], f["x"][:], f["dns/h"]
        if times is None:
            with h5py.File(S.ORACLE / "rom/runs/lavalle/thinfilm_training_data.h5") as r:
                times = r["time"][:] - T0
        kD = [int(np.argmin(abs(tD - (T0 + e)))) for e in times]
        dns = np.stack([hD[k] for k in kD])
    rom = {}
    for name, (d, _, _) in RUNS.items():
        with h5py.File(S.ORACLE / f"rom/runs/{d}/thinfilm_training_data.h5") as f:
            t = f["time"][:]
            rom[name] = np.stack([f["h"][int(np.argmin(abs(t - (T0 + e))))] for e in times])
    return np.asarray(times), x, dns, rom
