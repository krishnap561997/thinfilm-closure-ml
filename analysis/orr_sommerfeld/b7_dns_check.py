"""B.7 check against the DNS: complex transfer ratios dMx/eta, dTw/eta, dPx/eta, (dMx+dTw)/eta at the
forcing frequency in the linear region of the oracle run, against the OS targets along the OS
*spatial* mode at the same omega (complex k; linear_targets.os_closures).

DNS: rom_forcing.h5 (h, dMx, dTw, dPx every 0.01), projection on e^{i w t} over two forcing periods
at each x; ratio of complex amplitudes, median over X windows.  The phase convention matches the OS
mode: h = Re[A e^{-iwt}], eta_hat = 1.
"""
import math, os, sys
import h5py, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import os_solver as osv
from linear_targets import os_closures

STORE = os.environ["THINFILM_STORE"]
F = os.path.join(STORE, "datasets/BasC/DNS/oracle/Re120_Ca0_010_theta45_f10Hz/rom/rom_forcing.h5")
Re, Ca, COTB, W = 120.0, 0.01, 1.0, 2 * math.pi * 10 * 2.1389034175e-3

K, C = osv.spatial_curve(np.array([0.05, W]), Re, Ca, COTB)
L = os_closures(K[-1], Re, Ca, COTB, c_guess=C[-1])
print(f"OS spatial mode at omega {W:.4f}: k = {K[-1]:.5f}, c = {L['c']:.4f}, balance residual {abs(L['resid']):.1e}")
with h5py.File(F) as f:
    t, x = f["t"][:], f["x"][:]
    nT = int(round(2 * 2 * math.pi / W / (t[1] - t[0])))
    m = x < 200
    E = np.exp(1j * W * (t[:nT] - t[0]))[:, None]
    amp = lambda k: (2 / nT) * (E * (f[k][:nT][:, m] - f[k][:nT][:, m].mean(0))).sum(0)
    Ah, AM, AT, AP = amp("dns/h"), amp("dMx"), amp("dTw"), amp("dPx")
x = x[m]
print(f"{'':16s} {'dMx/eta':>18s} {'dTw/eta':>18s} {'(dMx+dTw)/eta':>18s} {'dPx/eta':>18s}")
fmt = lambda z: f"{z.real:+.4f}{z.imag:+.4f}i"
print(f"{'OS target':16s} {fmt(L['dM']):>18s} {fmt(L['dT']):>18s} {fmt(L['dM'] + L['dT']):>18s} {fmt(L['dP']):>18s}")
for a, b in [(20, 60), (60, 100), (100, 130), (130, 160)]:
    s = (x >= a) & (x < b)
    r = lambda A: np.median((A[s] / Ah[s]).real) + 1j * np.median((A[s] / Ah[s]).imag)
    print(f"{'DNS x ' + str(a) + '-' + str(b):16s} {fmt(r(AM)):>18s} {fmt(r(AT)):>18s} {fmt(r(AM + AT)):>18s} {fmt(r(AP)):>18s}")
