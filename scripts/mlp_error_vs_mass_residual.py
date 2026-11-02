#!/usr/bin/env python
"""Does the MLP held-out error track the per-snapshot mass residual (data-quality flag)?

For snapshots shared by the cache and budget_check (t = 1000..2000 every 100), prints per
case the NRMSE (x > x0, error / std of truth of the case) of each target at each snapshot,
split into good (mass_rel < lo) and poor (mass_rel > hi) snapshots, and the rank correlation.
Usage: mlp_error_vs_mass_residual.py --eval-root <.../first_derivatives> --budget <.../budget_check/per_case>
(eval-root contains <target>/seed42/eval_heldout)
"""
import argparse, csv
from pathlib import Path
import h5py
import numpy as np


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--eval-root', required=True)
    a.add_argument('--budget', required=True)
    a.add_argument('--targets', nargs='+', default=['delta_Mx_ML', 'delta_Tw_ML'])
    a.add_argument('--x0', type=float, default=100.)
    a.add_argument('--lo', type=float, default=0.05)
    a.add_argument('--hi', type=float, default=0.1)
    a = a.parse_args()
    for tgt in a.targets:
        d = Path(a.eval_root) / tgt / 'seed42' / 'eval_heldout'
        print(f'\n== {tgt}  (NRMSE x>{a.x0}; good: mass<{a.lo}, poor: mass>{a.hi})')
        print(f"{'case':26s} {'n':>3s} {'all':>6s} {'good':>6s} {'poor':>6s} {'spearman':>8s}")
        for f in sorted(d.glob('*.h5')):
            with h5py.File(f) as g:
                j = [t.decode() if isinstance(t, bytes) else t for t in g.attrs['targets']].index(tgt)
                p, y = g['prediction'][:, j].astype(float), g['truth'][:, j].astype(float)
                idx, x, t = g['indices'][:], g['x'][:], g['t'][:]
            mass = {float(r['t']): float(r['mass_rel']) for r in csv.DictReader(open(Path(a.budget) / f'{f.stem}.csv'))}
            keep = x[idx[:, 1]] > a.x0
            sigma = y[keep].std()
            ts, ms, es = [], [], []
            for tm, m in mass.items():
                sel = keep & (t[idx[:, 0]] == tm)
                if sel.any():
                    ts.append(tm); ms.append(m); es.append(np.sqrt(((p - y)[sel] ** 2).mean()) / sigma)
            ms, es = np.array(ms), np.array(es)
            good, poor = es[ms < a.lo], es[ms > a.hi]
            rho = np.corrcoef(rank(ms), rank(es))[0, 1] if len(ms) > 2 else np.nan
            fm = lambda v: f'{v.mean():6.3f}' if len(v) else '   n/a'
            print(f'{f.stem:26s} {len(ms):3d} {es.mean():6.3f} {fm(good)} {fm(poor)} {rho:8.2f}')


if __name__ == '__main__':
    main()
