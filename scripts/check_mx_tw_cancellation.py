#!/usr/bin/env python
"""How often do DeltaM_x and DeltaT_w cancel in the ROM source S = -dMx - dTw (+dPx + dDx)?

Per case (x > x0, t > t0, from the functional_mlp cache), prints:
  corr      Pearson corr(dMx, dTw)  (cancellation <=> corr near -1)
  slope     least-squares dTw = slope*dMx (-1 = exact cancellation)
  rms_sum   rms(dMx+dTw) / sqrt(0.5(rms(dMx)^2+rms(dTw)^2))   (0 = perfect cancellation)
  frac<.2   fraction of samples with |dMx+dTw| < 0.2 max(|dMx|,|dTw|) among samples where
            max(|dMx|,|dTw|) > 0.1 * its case max (the large-correction samples)
  frac_opp  fraction of those large samples with opposite signs
Usage: check_mx_tw_cancellation.py --cache <cache_dir> [--x0 100] [--t0 1000]
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--cache', required=True)
    a.add_argument('--x0', type=float, default=100.)
    a.add_argument('--t0', type=float, default=1000.)
    a.add_argument('--x1-frac', type=float, default=0.95, help='drop the outlet beyond this fraction of the domain')
    a = a.parse_args()
    root = Path(a.cache)
    man = json.loads((root / 'manifest.json').read_text())
    print(f'x>{a.x0}, x<{a.x1_frac}L, t>{a.t0}')
    print(f"{'case':26s} {'role':5s} {'corr':>6s} {'slope':>6s} {'rms_sum':>7s} {'frac<.2':>7s} {'frac_opp':>8s} {'rmsMx':>7s} {'rmsTw':>7s}")
    allm, allt = [], []
    for e in man['cases']:
        d = root / e['name']
        t, x = np.load(d / 't.npy'), np.load(d / 'x.npy')
        it = np.flatnonzero(t > a.t0)
        ix = np.flatnonzero((x > a.x0) & (x < a.x1_frac * x.max()))
        m = np.load(d / 'delta_Mx_ML.npy', mmap_mode='r')[it][:, ix].astype(float).ravel()
        w = np.load(d / 'delta_Tw_ML.npy', mmap_mode='r')[it][:, ix].astype(float).ravel()
        ok = np.isfinite(m) & np.isfinite(w)
        m, w = m[ok], w[ok]
        allm.append(m / m.std()); allt.append(w / m.std())
        mx = np.maximum(np.abs(m), np.abs(w))
        big = mx > 0.1 * mx.max()
        s = m + w
        print(f"{e['name']:26s} {e['role'][:5]:5s} {np.corrcoef(m, w)[0, 1]:6.3f} {(m @ w) / (m @ m):6.3f} "
              f"{np.sqrt((s ** 2).mean() / (0.5 * ((m ** 2).mean() + (w ** 2).mean()))):7.3f} "
              f"{(np.abs(s[big]) < 0.2 * mx[big]).mean():7.2f} {(m[big] * w[big] < 0).mean():8.2f} "
              f"{np.sqrt((m ** 2).mean()):7.4f} {np.sqrt((w ** 2).mean()):7.4f}")
    m, w = np.concatenate(allm), np.concatenate(allt)
    print(f"{'ALL (each case / std Mx)':32s} {np.corrcoef(m, w)[0, 1]:6.3f} {(m @ w) / (m @ m):6.3f} "
          f"{np.sqrt(((m + w) ** 2).mean() / (0.5 * ((m ** 2).mean() + (w ** 2).mean()))):7.3f}")


if __name__ == '__main__':
    main()
