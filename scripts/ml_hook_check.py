#!/usr/bin/env python3
"""Check the C++ ML hook (rom/lavalle-cpp/src/ml) against the Python features and PyTorch models.

    python ml_hook_check.py --forcing rom_forcing.h5 --ml ml.txt --px-ckpt best.pt [--snap 0]

Takes the DNS state (init, or dns/h,q at index --snap) on the ROM grid from rom_forcing.h5, pads
one ghost cell each side (the ROM has ng = 1), runs rom/lavalle-cpp/ml_check, and compares:
  * filtered inputs against ml/functional_mlp/prepare_data.filtered_derivatives;
  * the 'px' output against the PyTorch checkpoint evaluated on the Python features;
  * the 'px_fix' term h_s (kappa_x,s - h_xxx,s)/We against numpy.
Prints max abs and max relative differences and the rms of each source term.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml/functional_mlp"))
from model import PhysicalMLP  # noqa: E402
from prepare_data import filtered_derivatives  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--ml", required=True)
    ap.add_argument("--px-ckpt", required=True)
    ap.add_argument("--snap", type=int, default=-1, help="-1 = init state")
    ap.add_argument("--sigma", type=float, default=0.5)
    a = ap.parse_args()
    with h5py.File(a.forcing) as f:
        x = f["x"][:]
        h, q = (f["init/h"][:], f["init/q"][:]) if a.snap < 0 else (f["dns/h"][a.snap], f["dns/q"][a.snap])
        Re, Ca, ang = (float(f.attrs[k]) for k in ("Re", "Ca", "angle"))
    dx = x[1] - x[0]
    h, q = np.pad(h, 1, mode="edge"), np.pad(q, 1, mode="edge")
    with tempfile.TemporaryDirectory() as td:
        st, out = Path(td) / "state.txt", Path(td) / "out.txt"
        with open(st, "w") as fh:
            fh.write(f"{float(dx)!r} {Re!r} {Ca!r} {ang!r} 1 {h.size}\n")
            np.savetxt(fh, np.c_[h, q], fmt="%.17g")
        r = subprocess.run([str(ROOT / "rom/lavalle-cpp/ml_check"), a.ml, str(st), str(out)],
                           capture_output=True, text=True, check=True)
        print(r.stdout.strip())
        cols = open(out).readline().split()
        C = dict(zip(cols, np.loadtxt(out, skiprows=1).T))
    s = a.sigma
    tag = f"_s{s:g}".replace(".", "p")
    P = filtered_derivatives(h[:, None], q[:, None], s, dx, (Re, Ca, ang))
    P = {k + tag: v[:, 0] for k, v in P.items()}
    inner = slice(1, -1)
    print(f"dx = {dx:g}, sigma = {s} h_N = {s / dx:g} cells, n = {h.size}")
    for k in ("h_x" + tag, "q_x" + tag, "h_xxx" + tag, "q_xx" + tag, "kappa_x_We" + tag):
        d = np.abs(C[k] - P[k])[inner]
        print(f"  {k:<16} max|C++ - py| = {d.max():.2e}   (max|py| = {np.abs(P[k]).max():.3g})")
    ck = torch.load(a.px_ckpt, map_location="cpu", weights_only=False)
    m = PhysicalMLP(ck["architecture"], ck["normalization"]).double()
    m.load_state_dict({k: v.double() for k, v in ck["model"].items()}); m.eval()
    feats = {"h": h, "q": q, **P}
    X = np.c_[[feats[n] for n in ck["args"]["input_fields"]]].T
    X = np.c_[X, np.full((h.size, 3), (Re, Ca, ang))]
    with torch.no_grad():
        y = m(torch.from_numpy(X)).numpy()[:, 0]
    d = np.abs(C["px"] - y)[inner]
    print(f"  px  (inputs {ck['args']['input_fields']}): max|C++ - torch| = {d.max():.2e}, "
          f"rel to rms {d.max() / np.sqrt(np.mean(y[inner] ** 2)):.2e}")
    if "px_fix" in C:
        hs, hx, hxx, hxxx = (P[k + tag] if k != "h" else None for k in ("h", "h_x", "h_xx", "h_xxx"))
        from scipy.ndimage import gaussian_filter1d
        hs = gaussian_filter1d(h, s / dx, mode="nearest")
        g = 1 + hx ** 2
        fix = hs * (hxxx / g ** 1.5 - 3 * hx * hxx ** 2 / g ** 2.5 - hxxx) / (Re * Ca)
        print(f"  px_fix: max|C++ - py| = {np.abs(C['px_fix'] - fix)[inner].max():.2e}")
    print("rms of source terms on this state:",
          ", ".join(f"{k} {np.sqrt(np.mean(C[k][inner] ** 2)):.4g}" for k in ("sum", "mx", "tw", "px", "px_fix") if k in C))


if __name__ == "__main__":
    main()
