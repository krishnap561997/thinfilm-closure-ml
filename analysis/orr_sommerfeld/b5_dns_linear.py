"""B.5: spatial growth and phase speed of the forced frequency in the DNS, against OS.

Oracle run (Re120 Ca0.01 b45, 10 Hz monochromatic, t = 1800-1900, h every 0.01 on the ROM grid,
rom_forcing.h5).  Over an integer number of forcing periods, A_n(x) = (2/T) int h e^{i n w t} dt for
n = 1..3 (w = 2 pi 10 Hz T_N).  h = Re[A e^{-iwt}], A ~ exp(i k x):  -k_i = d ln|A|/dx,
k_r = d arg A/dx.  Local rates are least-squares slopes over a sliding window of WIN.  Also
fits over fixed x-windows.  x_lin: first x (after the inlet adjustment) where the local growth of
n = 1 falls below 80% of its median over 20 < x < 60.  Writes b5_dns_linear.csv (A_n(x)).
"""
import argparse, math, os, sys
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import os_solver as osv

STORE = os.environ["THINFILM_STORE"]
ORACLE = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz")
TN, FHZ, Re, Ca, COTB = 2.1389034175e-3, 10.0, 120.0, 0.01, 1.0
WIN = 10.0


def slope(x, y, win):
    n = max(3, int(round(win / (x[1] - x[0]))))
    out = np.full_like(y, np.nan)
    for i in range(n // 2, len(x) - n // 2):
        s = slice(i - n // 2, i + n // 2 + 1)
        out[i] = np.polyfit(x[s], y[s], 1)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=2)
    ap.add_argument("--xmax", type=float, default=300.0)
    A_ = ap.parse_args()
    w = 2 * math.pi * FHZ * TN
    with h5py.File(os.path.join(ORACLE, "rom/rom_forcing.h5")) as f:
        t, x = f["t"][:], f["x"][:]
        m = x < A_.xmax
        nT = int(round(A_.periods * 2 * math.pi / w / (t[1] - t[0])))
        H = f["dns/h"][:nT][:, m]
        tt = t[:nT]
    x = x[m]; dt = tt[1] - tt[0]; T = nT * dt
    print(f"oracle 10 Hz: omega {w:.5f}, window {T:.2f} = {A_.periods} periods ({nT} samples), x < {A_.xmax:g}")
    print(f"mean h over window at x = 20, 50, 100: {H[:, np.searchsorted(x, 20)].mean():.4f} "
          f"{H[:, np.searchsorted(x, 50)].mean():.4f} {H[:, np.searchsorted(x, 100)].mean():.4f}")
    # reference: OS and Lavalle at n w
    ws = np.array([w, 2 * w, 3 * w])
    Kos, _ = osv.spatial_curve(np.r_[0.05, ws], Re, Ca, COTB); Kos = Kos[1:]
    Klav = np.array([osv.lavalle_spatial(o, Re, Ca, COTB) for o in ws])
    rows = {}
    for n in (1, 2, 3):
        E = np.exp(1j * n * w * (tt - tt[0]))
        A = (2 / nT) * (E[:, None] * (H - H.mean(0))).sum(0)
        la, ph = np.log(np.abs(A)), np.unwrap(np.angle(A))
        g, kr = slope(x, la, WIN), slope(x, ph, WIN)
        rows[n] = (A, g, kr)
        print(f"\nn = {n} (omega {n * w:.4f}):  OS -k_i {-Kos[n - 1].imag:+.5f} c_ph {n * w / Kos[n - 1].real:.4f} | "
              f"Lavalle -k_i {-Klav[n - 1].imag:+.5f} c_ph {n * w / Klav[n - 1].real:.4f}")
        print(f"  {'x window':>12s} {'|A| start':>9s} {'-k_i fit':>9s} {'c_ph fit':>8s}")
        for a, b in [(5, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 160), (160, 200)]:
            s = (x >= a) & (x < b)
            gi = np.polyfit(x[s], la[s], 1)[0]; ki = np.polyfit(x[s], ph[s], 1)[0]
            print(f"  {a:5.0f}-{b:<5.0f} {np.abs(A[s][0]):9.2e} {gi:+9.5f} {n * w / ki:8.4f}")
    g1 = rows[1][1]
    ref = np.nanmedian(g1[(x > 20) & (x < 60)])
    after = np.where((x > 20) & (g1 < 0.8 * ref))[0]
    xlin = x[after[0]] if len(after) else np.nan
    print(f"\nn = 1 local growth median 20<x<60: {ref:.5f};  x_lin (local growth < 80% of it): {xlin:.1f};"
          f"  |A_1| there {np.abs(rows[1][0][after[0]]) if len(after) else np.nan:.3f}")
    print(f"harmonic check: d ln|A_2|/dx / d ln|A_1|/dx over 40-80: "
          f"{np.nanmedian(rows[2][1][(x > 40) & (x < 80)]) / np.nanmedian(g1[(x > 40) & (x < 80)]):.2f} (2 = slaved, nonlinear)")
    out = os.path.join(STORE, "analysis/orr_sommerfeld/b5_dns_linear.csv")
    with open(out, "w") as f:
        f.write("x," + ",".join(f"A{n}_re,A{n}_im,g{n},kr{n}" for n in (1, 2, 3)) + "\n")
        for i in range(len(x)):
            f.write(f"{x[i]:.3f}," + ",".join(f"{rows[n][0][i].real:.6e},{rows[n][0][i].imag:.6e},{rows[n][1][i]:.6e},{rows[n][2][i]:.6e}" for n in (1, 2, 3)) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
