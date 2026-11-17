"""A priori diagnosis of closures on the oracle case (Re120 Ca0.01 b45, 10 Hz, DNS on the ROM grid).

Closures are evaluated on the DNS state (rom_forcing.h5, every 1 time unit, 20 < x < 0.95 L) with the
inputs the ROM gives them:
    structured runs: zero-sum D_G(sigma) gradients (as gauss_deriv_hN), S, its flux and relaxation parts
    ml_sum_hhx     : central-difference h_x (as in its ROM run, hx_smooth = 0)
Per region (A.3 labels on D_G(0.5)-smoothed h, h_x, h_xx: ripples, front, back, crest, substrate):
    err   share of the total squared error of S (%), and rms error / rms(S_DNS) in the region
    P     power of the closure into the q fluctuations, P = <(-S) q'>, q' = q - <q>_x per snapshot,
          relative to the DNS value in the region (1 = same energy input as the true closure).
Usage: python diag_oracle.py RUN [RUN ...]
"""
import json, math, os, sys
import numpy as np
import h5py, torch
from scipy.ndimage import gaussian_filter1d as gf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts/structured-closure"))
import data as D  # noqa: E402
from model import StructuredV1  # noqa: E402
from mlp_np import sum_model  # noqa: E402

STORE = os.environ["THINFILM_STORE"]
FORC = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom/rom_forcing.h5")
Re, Ca, beta = 120.0, 0.01, math.radians(45.0); cotb = 1 / math.tan(beta)
REG = ["crest", "front", "back", "ripples", "substrate"]


def gd0(f, sig, dx, m):
    """Zero-sum Gaussian derivative (ROM gauss_deriv_hN), along axis 1."""
    s = sig / dx
    o = gf(f, s, axis=1, order=m, mode="nearest") / dx ** m
    if m:
        c0 = gf(np.ones((1, int(10 * s) + 20)), s, axis=1, order=m, mode="nearest")[0, int(5 * s) + 10] / dx ** m
        o = o - c0 * gf(f, s, axis=1, mode="nearest")
    return o


def main(runs):
    with h5py.File(FORC) as f:
        t, x = f["t"][:], f["x"][:]
        idx = list(range(0, len(t), 100))
        g = lambda k: np.stack([f[k][i] for i in idx]).astype(np.float64)
        h, q, Sd = g("dns/h"), g("dns/q"), g("dMx") + g("dTw")
    dx = float(x[1] - x[0]); m = (x > 20) & (x < 0.95 * x[-1])
    # regions (A.3)
    hs, hx5, hxx5 = gd0(h, 0.5, dx, 0), gd0(h, 0.5, dx, 1), gd0(h, 0.5, dx, 2)
    lab = np.full(h.shape, "substrate", object)
    sx, sxx = hx5[:, m].std(), hxx5[:, m].std()
    lab[hx5 < -0.5 * sx] = "front"; lab[hx5 > 0.5 * sx] = "back"
    lab[(lab == "substrate") & (hs > 1.15)] = "crest"
    lab[(hs < 1) & (np.abs(hxx5) > sxx)] = "ripples"
    qp = q - q[:, m].mean(1, keepdims=True)
    preds = {}
    hxc = np.gradient(h, dx, axis=1)
    preds["ml_sum_hhx"] = dict(S=sum_model()(h, q, hxc, None, Re, Ca, beta))
    for run in runs:
        cfg = json.load(open(os.path.join(D.BASE, "runs", run, "config.json")))
        mod = StructuredV1(cfg["width"], cfg["depth"], use_g3=not cfg["no_g3"]).double()
        mod.load_state_dict({k: v.double() for k, v in torch.load(os.path.join(D.BASE, "runs", run, "model.pt"), map_location="cpu").items()})
        sg = cfg["sigma"]
        hx, qx, hxx = gd0(h, sg, dx, 1), gd0(q, sg, dx, 1), gd0(h, sg, dx, 2)
        T = lambda a: torch.tensor(a.ravel())
        n = h.size; p = torch.tensor(D.pfeat(Re, Ca, beta), dtype=torch.float64)[None, :].repeat(n, 1)
        S, fl = mod(T(h), T(q), T(hx), T(qx), T(hxx), p, torch.full((n,), Re, dtype=torch.float64), torch.full((n,), cotb, dtype=torch.float64))
        S = S.detach().numpy().reshape(h.shape); fl = fl.detach().numpy().reshape(h.shape)
        preds[run] = dict(S=S, flux=fl, relax=fl - S)
    sel = lambda a, r=None: a[:, m][lab[:, m] == r] if r else a[:, m]
    rmsD = np.sqrt(np.mean(sel(Sd) ** 2))
    print(f"oracle case, {len(idx)} snapshots; region fractions: " + ", ".join(f"{r} {np.mean(lab[:, m] == r):.2f}" for r in REG))
    PD = {r: np.mean(-sel(Sd, r) * sel(qp, r)) for r in REG}; PD["all"] = np.mean(-sel(Sd) * sel(qp))
    print("DNS power <(-S) q'> per region: " + ", ".join(f"{r} {PD[r]:+.2e}" for r in REG + ["all"]))
    for name, pr in preds.items():
        S = pr["S"]; e = S - Sd; tot = np.sum(sel(e) ** 2)
        print(f"\n{name}: NRMSE {np.sqrt(np.mean(sel(e) ** 2)) / rmsD:.3f}")
        print(f"  {'region':10s} {'err%':>5s} {'rel rms':>7s} {'P/P_DNS':>8s}" + ("   P_flux/P_DNS P_relax/P_DNS" if "flux" in pr else ""))
        for r in REG + ["all"]:
            rr = None if r == "all" else r
            P = np.mean(-sel(S, rr) * sel(qp, rr))
            line = (f"  {r:10s} {100 * np.sum(sel(e, rr) ** 2) / tot:5.1f} {np.sqrt(np.mean(sel(e, rr) ** 2)) / np.sqrt(np.mean(sel(Sd, rr) ** 2)):7.3f} {P / PD[r]:8.2f}")
            if "flux" in pr:
                line += f"   {np.mean(-sel(pr['flux'], rr) * sel(qp, rr)) / PD[r]:12.2f} {np.mean(sel(pr['relax'], rr) * sel(qp, rr)) / PD[r]:13.2f}"
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
