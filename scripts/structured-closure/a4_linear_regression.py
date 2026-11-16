"""A.4: linear-region regression of the DNS correction S = dMx + dTw (oracle case, 10 Hz).

Window X_LO < x < X_HI (linear region; B.5 will fix it, until then x < 60), every 5th snapshot of
rom_forcing.h5 (dt = 0.05, elapsed 0-100).  Gaussian-filtered (sigma = 0.5) eta = h - 1, xi = q - 1
and their x-derivatives.  Fits
  (full) S = S0 + e eta + a (xi - 3 eta) + b eta_x + c xi_x
  (2+1)  S = S0 + P eta + Q eta_x         (all that a single frequency can identify: xi ~ c eta)
and reports coefficients, R^2, the condition number of the full design, and the spatial growth
-k_i and phase speed at the forcing frequency implied by each fit (linearised Lavalle + fit, see
spatial_aug), against OS and Lavalle.
"""
import argparse, os, sys
import h5py, numpy as np
from scipy.ndimage import gaussian_filter1d as gf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../../analysis/orr_sommerfeld"))
from mlp_np import ORACLE
import os_solver as osv

Re, Ca, COTB, OMEGA = 120.0, 0.01, 1.0, 2 * np.pi * 10 * 2.1389034e-3   # 10 Hz, T_N as in DNS/ROM


def spatial_aug(omega, co, Re=Re, Ca=Ca, cotb=COTB):
    """Spatial root of linearised Lavalle + S_lin (ROM adds -S), branch nearest the Lavalle root.
    k^4 (i/We) + k^2 (2w/Re + iG + ib) + k (2iw + i c w - 9/Re + e - 3a) + (3w/Re + a w - i w^2) = 0"""
    We, G = Re * Ca, -0.6 + 3 * cotb / Re
    a, b, c, e = (co.get(n, 0.0) for n in "abce")
    r = np.roots([1j / We, 0, 2 * omega / Re + 1j * G + 1j * b, 2j * omega + 1j * c * omega - 9 / Re + e - 3 * a,
                  3 * omega / Re + a * omega - 1j * omega ** 2])
    k0 = osv.lavalle_spatial(omega, Re, Ca, cotb)
    return r[np.argmin(np.abs(r - k0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlo", type=float, default=10.0)
    ap.add_argument("--xhi", type=float, default=60.0)
    A = ap.parse_args()
    with h5py.File(os.path.join(ORACLE, "rom/rom_forcing.h5")) as f:
        x = f["x"][:]; dx = x[1] - x[0]; sl = slice(0, len(f["t"]), 5)
        pad = (x < A.xhi + 20)                      # load a margin for the filter
        H, Q = f["dns/h"][sl][:, pad], f["dns/q"][sl][:, pad]
        S = (f["dMx"][sl] + f["dTw"][sl])[:, pad]
    s = 0.5 / dx
    eta, xi = gf(H - 1, s, axis=1), gf(Q - 1, s, axis=1)
    ex, xx = gf(H, s, axis=1, order=1) / dx, gf(Q, s, axis=1, order=1) / dx
    Sf = gf(S, s, axis=1)
    m = (x[pad] > A.xlo) & (x[pad] < A.xhi)
    eta, xi, ex, xx, Sf = (v[:, m].ravel() for v in (eta, xi, ex, xx, Sf))
    one = np.ones_like(eta)
    print(f"oracle Re120 Ca0.01 b45, 10 Hz (omega {OMEGA:.4f}); x in ({A.xlo:g}, {A.xhi:g}); {len(Sf)} points")
    print(f"rms eta {eta.std():.4f}; S: mean {Sf.mean():+.2e}, rms of fluctuation {Sf.std():.2e}")
    fits = {"full": (["S0", "e", "a", "b", "c"], np.c_[one, eta, xi - 3 * eta, ex, xx]),
            "2+1": (["S0", "e", "b"], np.c_[one, eta, ex])}
    kos = osv.spatial_curve(np.array([0.05, OMEGA]), Re, Ca, COTB)[0][-1]
    klav = osv.lavalle_spatial(OMEGA, Re, Ca, COTB)
    print(f"OS     : -k_i {-kos.imag:.5f}  c_ph {OMEGA / kos.real:.4f}")
    print(f"Lavalle: -k_i {-klav.imag:.5f}  c_ph {OMEGA / klav.real:.4f}")
    for name, (cols, X) in fits.items():
        Xs = X / X.std(0, where=None, keepdims=True).clip(1e-30) if False else X
        coef, *_ = np.linalg.lstsq(X, Sf, rcond=None)
        r2 = 1 - np.sum((Sf - X @ coef) ** 2) / np.sum((Sf - Sf.mean()) ** 2)
        Z = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
        cond = np.linalg.cond(Z)
        co = dict(zip(cols, coef))
        k = spatial_aug(OMEGA, co)
        print(f"[{name}] " + "  ".join(f"{n} {v:+.3e}" for n, v in co.items())
              + f"  R2 {r2:.3f}  cond {cond:.1f}  -> -k_i {-k.imag:.5f}  c_ph {OMEGA / k.real:.4f}")
        if name == "full":
            print(f"        long-wave: kinematic shift {-co['e'] / (3 / Re + co['a']):+.4f}, "
                  f"D {(3.6 - 3 * COTB / Re - co['b'] - 3 * co['c']) / (3 / Re + co['a']):.1f} (Lavalle {1.2 * Re - COTB:.1f})")


if __name__ == "__main__":
    main()
