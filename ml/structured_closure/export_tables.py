"""Export a trained structured closure V1 as 1D tables in h for lavalle-cpp (plan task C.5).

For one run and one case (Re, Ca, beta), on h in [0.01, 5] (5001 points):
    h  Phi  Phi_h  dP  dP_h  kappa  g1  g2  g3
with dP_h = P_Lav'(h)(exp(N_s) - 1) and dP(h) = int_1^h dP_h dh' (cumulative trapezoid; the
constant does not matter in the flux divergence).  Also writes the ROM config sc_<case>.txt.

Output: <run>/tables/<case>.txt and <run>/tables/sc_<case>.txt
Usage: python export_tables.py RUN_NAME CASE [CASE ...]
"""
import json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D  # noqa: E402
from data import pfeat  # noqa: E402
from model import StructuredV1  # noqa: E402


def case_params(case):
    import re
    m = re.match(r"Re(\d+)_Ca(\d+)_(\d+)_theta([\d_]+)", case)
    return float(m.group(1)), float(f"{m.group(2)}.{m.group(3)}"), float(m.group(4).replace("_", "."))


def export(run, case, n=5001, hmin=0.01, hmax=5.0):
    rd = os.path.join(D.BASE, "runs", run)
    cfg = json.load(open(os.path.join(rd, "config.json")))
    m = StructuredV1(cfg["width"], cfg["depth"], use_g3=not cfg["no_g3"]).double()
    m.load_state_dict({k: v.double() for k, v in torch.load(os.path.join(rd, "model.pt"), map_location="cpu").items()})
    Re, Ca, bdeg = case_params(case); b = math.radians(bdeg); cotb = 1 / math.tan(b)
    h = torch.linspace(hmin, hmax, n, dtype=torch.float64)
    p = torch.tensor(pfeat(Re, Ca, b), dtype=torch.float64)[None, :].repeat(n, 1)
    c = m.coeffs(h, p, Re, cotb)
    g = {k: v.detach().numpy() for k, v in c.items()}
    hn = h.numpy(); dPh = g["dPh"]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dPh[1:] + dPh[:-1]) * np.diff(hn))])
    dP = cum - np.interp(1.0, hn, cum)
    out = os.path.join(rd, "tables"); os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"{case}.txt")
    cols = np.column_stack([hn, g["Phi"], g["Phi_h"], dP, dPh, g["kap"], g["g1"], g["g2"], g["g3"]])
    hdr = (f"sigma {cfg['sigma']}\nRe {Re}\nCa {Ca}\nbeta_deg {bdeg}\nrun {run}\n"
           "columns h Phi Phi_h dP dP_h kappa g1 g2 g3")
    np.savetxt(fn, cols, fmt="%.10e", header=hdr, comments="# ")
    with open(os.path.join(out, f"sc_{case}.txt"), "w") as f:
        f.write(f"structured_table = {fn}\nsigma = {cfg['sigma']}\nrelax = 1\n")
    i1 = np.argmin(np.abs(hn - 1))
    print(f"{run} {case}: Phi(1) {g['Phi'][i1]:.4f} kappa(1)/(3/Re) {g['kap'][i1] * Re / 3:.3f} "
          f"g1 g2 g3 (1) {g['g1'][i1]:+.3f} {g['g2'][i1]:+.3f} {g['g3'][i1]:+.3f} -> {fn}")
    return fn


if __name__ == "__main__":
    for c in sys.argv[2:]:
        export(sys.argv[1], c)
