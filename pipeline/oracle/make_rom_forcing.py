#!/usr/bin/env python3
"""Build the lavalle-cpp oracle forcing file from closure files (DNS grid -> ROM grid).

Conservative cell averages onto N uniform cells of [0, L] of the corrections and the
DNS state, at every closure-file time:
  dMx, dTw, dPx, dDx         (nt, N)   delta_Mx_ML, delta_Tw_ML, delta_Px_ML, delta_Dx_ML
  dns/h, dns/q               (nt, N)   DNS state on the ROM grid (for comparison)
  init/h, init/q, init/h_x   (N,)      state at the first time (ROM initial condition)
  inlet/h, inlet/q, inlet/h_x (nt,)    first ROM cell (inlet boundary values)
  t (nt,), x (N,), attributes Re, Ca, angle, L.
Read by rom/lavalle-cpp/src/oracle/oracle_forcing.cpp.

    python make_rom_forcing.py <closures_part_*.h5 ...> -o rom_forcing.h5 [--N 4000 --L 800]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract_closures"))
from check_momentum_budget import restrict  # noqa: E402

TERMS = {"dMx": "delta_Mx_ML", "dTw": "delta_Tw_ML", "dPx": "delta_Px_ML", "dDx": "delta_Dx_ML"}
STATE = {"dns/h": "h", "dns/q": "q"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("closure_files", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--L", type=float, default=800.0)
    a = ap.parse_args()

    parts = []
    for p in a.closure_files:
        with h5py.File(p, "r") as f: parts.append((float(f["t"][0]), p, f["t"].shape[0]))
    parts.sort()
    nt = sum(n for _, _, n in parts)
    with h5py.File(parts[0][1], "r") as f:
        x = f["x"][:]
        par = {k: float(np.asarray(f["parameters"][k][()]).squeeze()) for k in ("Re", "Ca", "angle")}
    N, L = a.N, a.L
    XR = (np.arange(N) + 0.5) * L / N
    print(f"{len(parts)} closure files, {nt} times, DNS nx={x.size} -> ROM N={N}")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.output, "w") as out:
        for k, v in par.items(): out.attrs[k] = v
        out.attrs["L"] = L
        out.attrs["source"] = " ".join(str(p) for _, p, _ in parts)
        out.create_dataset("x", data=XR)
        tds = out.create_dataset("t", (nt,), "f8")
        big = {k: out.create_dataset(k, (nt, N), "f8", chunks=(1, N)) for k in list(TERMS) + list(STATE)}
        inlet = {k: out.create_dataset(f"inlet/{k}", (nt,), "f8") for k in ("h", "q", "h_x")}
        it = 0
        for _, p, n in parts:
            with h5py.File(p, "r") as f:
                tds[it:it + n] = f["t"][:]
                for j in range(n):
                    for name, src in {**TERMS, **STATE}.items():
                        big[name][it + j] = restrict(x, f[src][:, j], N, L)
                    hR = big["dns/h"][it + j]
                    hxR = restrict(x, f["h_x"][:, j], N, L)
                    inlet["h"][it + j], inlet["q"][it + j], inlet["h_x"][it + j] = hR[0], big["dns/q"][it + j][0], hxR[0]
                    if it + j == 0:
                        out.create_dataset("init/h", data=hR)
                        out.create_dataset("init/q", data=big["dns/q"][0])
                        out.create_dataset("init/h_x", data=hxR)
            it += n
            print(f"  {p.name}: t {tds[it - n]:.3f}..{tds[it - 1]:.3f}")
        t = tds[:]
        if np.any(np.diff(t) <= 0): raise SystemExit("closure times are not strictly increasing")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()
