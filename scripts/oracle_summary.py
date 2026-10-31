#!/usr/bin/env python3
"""Short summary of oracle ROM runs against the DNS (on the ROM grid).

For each run (lavalle-cpp thinfilm_training_data.h5) and each saved time that matches a
forcing time: rms and max |h_ROM - h_DNS|, the rms error relative to the DNS h fluctuation,
the same for q, and max h (ROM vs DNS), all over x_min < x < x_max. Prints a table at a
few times and writes the full time series to --csv.

    python scripts/oracle_summary.py --forcing rom_forcing.h5 \
        --run lavalle=runs/lavalle/thinfilm_training_data.h5 --run all=... [--csv out.csv]
"""
from __future__ import annotations
import argparse
import csv
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--run", action="append", required=True, help="name=path/to/thinfilm_training_data.h5")
    ap.add_argument("--x-min", type=float, default=20.0)
    ap.add_argument("--x-max", type=float, default=760.0)
    ap.add_argument("--show", default="0.1,1,2,5,10,20,50,99.9", help="elapsed times to print")
    ap.add_argument("--csv")
    a = ap.parse_args()

    with h5py.File(a.forcing, "r") as f:
        tD, x = f["t"][:], f["x"][:]
        m = (x > a.x_min) & (x < a.x_max)
        hD, qD = f["dns/h"], f["dns/q"]
        runs, rows = {}, []
        for spec in a.run:
            name, path = spec.split("=", 1)
            with h5py.File(path, "r") as r:
                tR, hR, qR = r["time"][:], r["h"][:], r["q"][:]
            res = []
            for i, t in enumerate(tR):
                k = int(np.argmin(abs(tD - t)))
                if abs(tD[k] - t) > 1e-6: continue
                hd, qd = hD[k][m], qD[k][m]
                eh, eq = hR[i][m] - hd, qR[i][m] - qd
                row = {"run": name, "t": t, "elapsed": t - tD[0],
                       "h_rms": np.sqrt(np.mean(eh**2)), "h_max": np.max(abs(eh)),
                       "h_rel": np.sqrt(np.mean(eh**2)) / np.std(hd),
                       "q_rms": np.sqrt(np.mean(eq**2)), "q_rel": np.sqrt(np.mean(eq**2)) / np.std(qd),
                       "hmax_rom": float(np.max(hR[i][m])), "hmax_dns": float(np.max(hd)),
                       "finite": bool(np.all(np.isfinite(hR[i])))}
                res.append(row)
            runs[name] = res
            rows += res
            last = res[-1] if res else None
            print(f"{name:10s}: {len(res)} matched times, last t={last['t']:.2f}" if last else f"{name}: no matched times")

    show = [float(s) for s in a.show.split(",")]
    print(f"\nrelative rms error in h  (|h_ROM - h_DNS| / std(h_DNS), {a.x_min:g} < x < {a.x_max:g})")
    print("elapsed " + "".join(f"{n:>11s}" for n in runs))
    for s in show:
        line = f"{s:7.1f} "
        for res in runs.values():
            el = np.array([r["elapsed"] for r in res])
            line += f"{res[int(np.argmin(abs(el - s)))]['h_rel']:11.3f}" if len(el) and np.min(abs(el - s)) < 0.06 else f"{'-':>11s}"
        print(line)
    print("\nmax h (ROM / DNS) at the same times")
    for s in show:
        line = f"{s:7.1f} "
        for res in runs.values():
            el = np.array([r["elapsed"] for r in res])
            if len(el) and np.min(abs(el - s)) < 0.06:
                r = res[int(np.argmin(abs(el - s)))]
                line += f"  {r['hmax_rom']:4.2f}/{r['hmax_dns']:4.2f}"
            else:
                line += f"{'-':>11s}"
        print(line)
    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
