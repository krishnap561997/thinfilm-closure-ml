#!/usr/bin/env python3
"""Why did a flat-film-exact MLP ROM run fail?  Reads crash_states.h5 (last 20 time units before the error, written
by lavalle-cpp on an exception; or the last snapshots of thinfilm_training_data.h5 with --file) and prints, for the
cell of minimum h at each time: the network inputs (h, q, h_x, q_x, h_xx; zero-sum D_G(sigma) as in the ROM), the
closure S = N(state) - N(h, h^3, 0, 0, 0), Lavalle's drag 3q/h^2 for scale, and whether each input is inside the
training range (0.1-99.9 percentiles over the development cases, cache v2, same sigma).

    LD_LIBRARY_PATH=$THINFILM_ENV/lib python scripts/structured-closure/crash_diag.py <run_dir> [...] [--last 12]
"""
import argparse, math, os, sys
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d as gf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "ml/structured_closure")); sys.path.insert(0, os.path.dirname(__file__))
import data as D  # noqa: E402
from mlp_np import MLP  # noqa: E402

INP = ["h", "q", "h_x", "q_x", "h_xx"]


def gd0(f, sig, dx, m):
    """Zero-sum Gaussian derivative (ROM gauss_deriv_hN), along axis 1 (as diag_oracle.gd0)."""
    s = sig / dx
    o = gf(f, s, axis=1, order=m, mode="nearest") / dx ** m
    if m:
        c0 = gf(np.ones((1, int(10 * s) + 20)), s, axis=1, order=m, mode="nearest")[0, int(5 * s) + 10] / dx ** m
        o = o - c0 * gf(f, s, axis=1, mode="nearest")
    return o


def kv(fn):
    out = {}
    for l in open(fn):
        l = l.split("#")[0]
        if "=" in l:
            k, v = l.split("=", 1); out[k.strip()] = v.strip()
    return out


def train_range(sigma):
    dev, _ = D.split(); cols = {k: [] for k in INP}
    for c in dev:
        d = D.load_case(c, sigma=sigma)
        for k in INP: cols[k].append(d[k].ravel()[::7])
    return {k: np.nanpercentile(np.concatenate(v), [0.1, 99.9]) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--last", type=int, default=12)
    ap.add_argument("--file", default="crash_states.h5")
    a = ap.parse_args()
    rng = {}
    for run in a.runs:
        cl, inp = kv(os.path.join(run, "closure.txt")), kv(os.path.join(run, "input.txt"))
        sig = float(cl.get("sigma", 0.5)); net = MLP(cl["flat_mlp"])
        if sig not in rng: rng[sig] = train_range(sig)
        R = rng[sig]
        par = dict(Re=float(inp["Re_input"]), Ca=float(inp["Ca_input"]), angle=math.radians(float(inp["theta_deg"])))
        fn = os.path.join(run, a.file)
        if not os.path.exists(fn): print(f"== {run}: no {a.file}"); continue
        with h5py.File(fn) as f:
            t, x, H, Q = f["time"][:], f["x"][:], f["h"][:], f["q"][:]
        dx = x[1] - x[0]
        F = dict(h=H, q=Q, h_x=gd0(H, sig, dx, 1), q_x=gd0(Q, sig, dx, 1), h_xx=gd0(H, sig, dx, 2))
        S = net(**F, **par) - net(h=H, q=H ** 3, h_x=0 * H, q_x=0 * H, h_xx=0 * H, **par)
        print(f"\n== {run}\n   sigma {sig}; {len(t)} states t {t[0]:.2f}-{t[-1]:.3f}; training range (0.1-99.9%): "
              + ", ".join(f"{k} [{R[k][0]:.3g}, {R[k][1]:.3g}]" for k in INP))
        print(f"   {'t':>9s} {'x':>7s} " + " ".join(f"{k:>8s}" for k in INP) + f" {'S':>9s} {'3q/h2':>9s} {'S_max|':>9s}  out of range")
        for j in list(range(len(t)))[-a.last:]:
            i = int(np.argmin(H[j])); out = [k for k in INP if not (R[k][0] <= F[k][j, i] <= R[k][1])]
            print(f"   {t[j]:9.3f} {x[i]:7.2f} " + " ".join(f"{F[k][j, i]:8.3f}" for k in INP)
                  + f" {S[j, i]:9.3f} {3 * Q[j, i] / H[j, i] ** 2:9.3f} {np.abs(S[j]).max():9.3f}  {' '.join(out) or '-'}")
        j = -1; frac = {k: np.mean((F[k][j] < R[k][0]) | (F[k][j] > R[k][1])) for k in INP}
        print("   last state, share of cells outside the training range: " + ", ".join(f"{k} {v:.3f}" for k, v in frac.items()))


if __name__ == "__main__":
    main()
