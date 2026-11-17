"""Checks of the structured closure code (no training data needed; seconds on CPU).

1. Uniform film: S(h, h^3, 0, 0, 0) = 0 for random weights and any h (S0 = e = 0 by construction).
2. model.linear against a finite-difference linearisation of model.forward about h = q = 1.
3. Hyperbolicity: min discriminant >= 0 over a box of states for random weights (V1).
4. Dispersion residual with sigma = 0 reproduces B.8: Lavalle and band_abcd growth errors of
   linear_targets.csv for a few cases (Newton roots from the OS k).
"""
import csv, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import data as D  # noqa: E402
from model import StructuredV1  # noqa: E402
import train as TR  # noqa: E402

torch.manual_seed(1)
Re, Ca, beta = 120.0, 0.01, math.radians(45); cotb = 1 / math.tan(beta)
p1 = torch.tensor(D.pfeat(Re, Ca, beta))
m = StructuredV1().double()
for prm in m.parameters():                     # random, non-trivial weights
    torch.nn.init.normal_(prm, std=0.3)
fails = 0

# 1
h = torch.linspace(0.3, 3.0, 50, dtype=torch.float64); z = torch.zeros_like(h)
P = p1.double()[None, :].repeat(50, 1)
S, _ = m(h, h ** 3, z, z, z, P, torch.full_like(h, Re), torch.full_like(h, cotb))
print(f"1. uniform film: max |S| = {S.abs().max():.2e}"); fails += S.abs().max() > 1e-12

# 2
L = m.linear(p1.double(), Re, cotb)
eps = 1e-6; o = torch.ones(1, dtype=torch.float64); zz = torch.zeros(1, dtype=torch.float64)
def Sx(h_, q_, hx, qx, hxx):
    return m(h_ * o, q_ * o, hx * o, qx * o, hxx * o, p1.double()[None, :], torch.tensor([Re]), torch.tensor([cotb]))[0][0]
fd = dict(a=(Sx(1, 1 + eps, 0, 0, 0) - Sx(1, 1 - eps, 0, 0, 0)) / (2 * eps),
          b=(Sx(1, 1, eps, 0, 0) - Sx(1, 1, -eps, 0, 0)) / (2 * eps),
          c=(Sx(1, 1, 0, eps, 0) - Sx(1, 1, 0, -eps, 0)) / (2 * eps),
          d=(Sx(1, 1, 0, 0, eps) - Sx(1, 1, 0, 0, -eps)) / (2 * eps),
          e=(Sx(1 + eps, (1 + eps) ** 3, 0, 0, 0) - Sx(1 - eps, (1 - eps) ** 3, 0, 0, 0)) / (2 * eps))
err = max(abs(float(fd[k]) - float(L[k])) for k in "abcd")
print(f"2. linear coeffs: a b c d = {[round(float(L[k]), 5) for k in 'abcd']}, max |FD - linear| = {err:.1e}, e = {float(fd['e']):.1e}")
fails += err > 1e-6 or abs(float(fd["e"])) > 1e-8

# 3
hh, qq = torch.meshgrid(torch.linspace(0.2, 4, 60, dtype=torch.float64), torch.linspace(0.0, 30, 60, dtype=torch.float64), indexing="ij")
hh, qq = hh.ravel(), qq.ravel()
disc = m.discriminant(hh, qq, p1.double()[None, :].repeat(len(hh), 1), torch.full_like(hh, Re), torch.full_like(hh, cotb))
print(f"3. min discriminant over h 0.2-4, q 0-30: {disc.min():.3e}"); fails += disc.min() < 0

# 4
store = os.environ["THINFILM_STORE"]
lt = {r["case"]: r for r in csv.DictReader(open(os.path.join(store, "analysis/orr_sommerfeld/linear_targets.csv")))}
js = json.load(open(os.path.join(D.BASE, "os_dispersion.json")))
for case in ("Re020_Ca0_010_theta45", "Re120_Ca0_010_theta45", "Re200_Ca0_010_theta6_4"):
    r = js[case]; Re_, Ca_, b_ = r["Re"], r["Ca"], math.radians(r["beta_deg"]); cb = 1 / math.tan(b_)
    tg = [dict(omega=torch.tensor(r["omega"], dtype=torch.float64),
               k=torch.tensor(np.array(r["k_re"]) + 1j * np.array(r["k_im"])), Re=Re_, We=Re_ * Ca_, cotb=cb, D_os=r["D_os"])]
    out = []
    for coef in ((0, 0, 0, 0), tuple(float(lt[case][f"band_abcd_{k}"]) for k in "abcd")):
        a_, b2, c_, d_ = coef; kap = a_ + 3 / Re_

        class Fake:
            def linear(self, p, Re, cotb):
                t = lambda v: torch.tensor(v, dtype=torch.float64)
                return dict(Fh=t(b2), Fq=t(c_), kap=t(kap), g1=t(0.), g2=t(0.), g3=t(-d_ / kap), a=t(a_), b=t(b2), c=t(c_), d=t(d_))
        C = dict(p=torch.zeros(1, 3), Re=torch.tensor([Re_]), cotb=torch.tensor([cb]))
        ki, kr, _, Dg = TR.roots(Fake(), C, tg, 0.0, 0)
        out.append(ki)
    ref = (float(lt[case]["band_lav_err_ki"]), float(lt[case]["band_abcd_err_ki"]))
    print(f"4. {case}: growth err Lavalle {out[0]:.4f} (B.8 {ref[0]:.4f}), band_abcd {out[1]:.4f} (B.8 {ref[1]:.4f})")
    fails += abs(out[0] - ref[0]) > 2e-3 or abs(out[1] - ref[1]) > 2e-3
print("ALL PASS" if not fails else f"{int(fails)} FAILED")
