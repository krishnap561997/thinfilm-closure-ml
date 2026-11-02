#!/usr/bin/env python
"""Region-resolved summary of MLP closure predictions (ml/functional_mlp evaluate.py output).

Regions (h from the cache, x from the prediction files):
  NL     nonlinear : x > x0 and (h > h_hi or h < h_lo)
  BAND   quiescent : x > x0 and h_lo <= h <= h_hi
  INLET            : x <= x0
All errors are normalised by the std of the truth over the whole case (same sigma for
every region), so region numbers add up consistently. 'base' is the Lavalle-only
baseline (correction = 0). R2_NL is 1 - SSE/SST with SST taken inside NL.
Peak columns: per snapshot, relative error of max|DeltaC| over x > x0 (median over
snapshots) and the x-offset between the truth and predicted peak (median |dx|).

Usage: mlp_wave_region_summary.py --eval-dir <.../eval_heldout> --cache <cache_dir>
"""
import argparse
from pathlib import Path
import h5py
import numpy as np


def region_stats(y, p, mask, sigma):
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, nrmse=np.nan, base=np.nan, r2=np.nan, bias=np.nan)
    e = (p - y)[mask]
    yy = y[mask]
    sst = ((yy - yy.mean()) ** 2).sum()
    return dict(n=n, nrmse=np.sqrt((e ** 2).mean()) / sigma, base=np.sqrt((yy ** 2).mean()) / sigma,
                r2=1 - (e ** 2).sum() / sst if sst > 0 else np.nan, bias=e.mean() / sigma)


def peaks(y, p, x, ti, mask):
    amp, dx = [], []
    ti, y, p, x = ti[mask], y[mask], p[mask], x[mask]
    order = np.argsort(ti, kind='stable')
    ti, y, p, x = ti[order], y[order], p[order], x[order]
    bounds = np.flatnonzero(np.diff(ti)) + 1
    for yy, pp, xx in zip(np.split(y, bounds), np.split(p, bounds), np.split(x, bounds)):
        iy, ip = np.argmax(np.abs(yy)), np.argmax(np.abs(pp))
        amp.append((abs(pp[ip]) - abs(yy[iy])) / abs(yy[iy]))
        dx.append(abs(xx[ip] - xx[iy]))
    return np.median(np.abs(amp)), np.median(dx)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--eval-dir', required=True, help='evaluate.py output dir with <case>.h5 files')
    a.add_argument('--cache', required=True)
    a.add_argument('--x0', type=float, default=100.)
    a.add_argument('--h-hi', type=float, default=1.1)
    a.add_argument('--h-lo', type=float, default=0.9)
    a.add_argument('--csv', default=None)
    a = a.parse_args()
    files = sorted(Path(a.eval_dir).glob('*.h5'))
    print(f'{a.eval_dir}\nNL: x>{a.x0} & (h>{a.h_hi} | h<{a.h_lo}); errors / sigma_case; base = DeltaC=0')
    pooled = {}
    rows = []
    for f in files:
        case = f.stem
        hfield = np.load(Path(a.cache) / case / 'h.npy', mmap_mode='r')
        with h5py.File(f) as g:
            targets = [t.decode() if isinstance(t, bytes) else t for t in g.attrs['targets']]
            P, Y = g['prediction'][:].astype(float), g['truth'][:].astype(float)
            idx, x = g['indices'][:], g['x'][:]
        ti, si = idx[:, 0], idx[:, 1]
        h = np.asarray(hfield)[ti, si].astype(float)
        xs = x[si]
        right = xs > a.x0
        nl = right & ((h > a.h_hi) | (h < a.h_lo))
        band = right & ~nl
        inlet = ~right
        for j, t in enumerate(targets):
            y, p = Y[:, j], P[:, j]
            sigma = y.std()
            r = {k: region_stats(y, p, m, sigma) for k, m in dict(all=np.ones_like(right), NL=nl, BAND=band, INLET=inlet).items()}
            pa, pdx = peaks(y, p, xs, ti, right)
            rows.append(dict(case=case, target=t, frac_NL=nl.mean(), all=r['all']['nrmse'], NL=r['NL']['nrmse'],
                             NL_base=r['NL']['base'], R2_NL=r['NL']['r2'], bias_NL=r['NL']['bias'],
                             BAND=r['BAND']['nrmse'], INLET=r['INLET']['nrmse'], peak_amp_err=pa, peak_dx=pdx))
            acc = pooled.setdefault(t, dict(sse={k: 0. for k in r}, n={k: 0 for k in r}, ssy={k: 0. for k in r},
                                            sy2=0., ny=0, sy=0., sNL_y=0., sNL_y2=0.))
            sc = pooled[t]
            for k, m in dict(all=np.ones_like(right), NL=nl, BAND=band, INLET=inlet).items():
                sc['sse'][k] += ((p - y)[m] ** 2).sum()
                sc['ssy'][k] += (y[m] ** 2).sum()
                sc['n'][k] += int(m.sum())
            sc['sy'] += y.sum(); sc['sy2'] += (y ** 2).sum(); sc['ny'] += len(y)
            sc.setdefault('NLy', []).append((y[nl].sum(), (y[nl] ** 2).sum(), ((p - y)[nl] ** 2).sum(), int(nl.sum())))
    for t in sorted({r['target'] for r in rows}):
        print(f'\n== {t}')
        print(f"{'case':26s} {'fracNL':>6s} {'all':>6s} {'NL':>6s} {'NLbase':>6s} {'R2_NL':>6s} {'biasNL':>7s} {'BAND':>6s} {'INLET':>6s} {'pkAmp':>6s} {'pkDx':>6s}")
        for r in [r for r in rows if r['target'] == t]:
            print(f"{r['case']:26s} {r['frac_NL']:6.2f} {r['all']:6.3f} {r['NL']:6.3f} {r['NL_base']:6.3f} {r['R2_NL']:6.3f} {r['bias_NL']:7.3f} "
                  f"{r['BAND']:6.3f} {r['INLET']:6.3f} {r['peak_amp_err']:6.3f} {r['peak_dx']:6.2f}")
        sc = pooled[t]
        sigma = np.sqrt(sc['sy2'] / sc['ny'] - (sc['sy'] / sc['ny']) ** 2)
        pr = {k: np.sqrt(sc['sse'][k] / sc['n'][k]) / sigma for k in sc['sse']}
        sy, sy2, sse, n = (sum(v[i] for v in sc['NLy']) for i in range(4))
        r2 = 1 - sse / (sy2 - sy * sy / n)
        print(f"{'POOLED (pooled sigma)':26s} {n / sc['ny']:6.2f} {pr['all']:6.3f} {pr['NL']:6.3f} "
              f"{np.sqrt(sc['ssy']['NL'] / sc['n']['NL']) / sigma:6.3f} {r2:6.3f} {'':7s} {pr['BAND']:6.3f} {pr['INLET']:6.3f}")
    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
