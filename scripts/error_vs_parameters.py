#!/usr/bin/env python
"""Which case parameters track the closure-model error?  (t > t0, x0 < x < 0.95 L, all cases)

Evaluates a sparse form from ml/functional_forms/fit_sparse.py (sparse_path.json, k terms) on every
cached case and tabulates, per case: We, Fr2, beta, Re, rms(truth), rms(truth)/rms(M_x,Lav),
absolute RMSE and NRMSE = RMSE/rms(truth). Then Spearman correlations of NRMSE and RMSE with
each parameter, over dev+heldout and over heldout only. NB: dev cases are in-sample for the
pooled fit (a 5-term form has little capacity to overfit, but read dev numbers accordingly).
Usage: error_vs_parameters.py --cache <cache_dir> --path <sparse_path.json> --target Mx --k 5
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml' / 'functional_forms'))
from fit_sparse import case_params, pfactors, PNAMES  # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--cache', required=True)
    a.add_argument('--path', required=True)
    a.add_argument('--target', choices=['Mx', 'S'], default='Mx')
    a.add_argument('--k', type=int, default=5)
    a.add_argument('--t0', type=float, default=1000.)
    a.add_argument('--x0', type=float, default=100.)
    a.add_argument('--win', type=int, default=21)
    a.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'ml' / 'functional_mlp' / 'data_config.json'))
    a = a.parse_args()
    row = json.loads(Path(a.path).read_text())[a.target][a.k - 1]
    terms = []
    for s, c in zip(row['terms'], row['coef']):
        m = re.match(r'h\^(\d+) q\^(\d+) (\w+) \[(.+)\]', s)
        terms.append((int(m.group(1)), int(m.group(2)), m.group(3), PNAMES.index(m.group(4)), c))
    held = set(json.loads(Path(a.config).read_text())['heldout_cases'])
    root = Path(a.cache); rows = []
    for e in json.loads((root / 'manifest.json').read_text())['cases']:
        n = e['name']; d = root / n
        t, x = np.load(d / 't.npy'), np.load(d / 'x.npy'); dx = x[1] - x[0]
        it = np.flatnonzero(t > a.t0); ix = np.flatnonzero((x > a.x0) & (x < 0.95 * x.max()))
        ld = lambda f: np.load(d / f'{f}.npy', mmap_mode='r')[it].astype(float)
        h, q, Mx, Tw = ld('h'), ld('q'), ld('delta_Mx_ML'), ld('delta_Tw_ML')
        D = lambda f, k: savgol_filter(f, a.win, 3, deriv=k, delta=dx, axis=1)
        g = dict(hx=D(h, 1), qx=D(q, 1), hxx=D(h, 2), qxx=D(q, 2))
        Re, Ca, beta = case_params(n); pf = pfactors(Re, Ca, beta)
        y = (Mx if a.target == 'Mx' else Mx + Tw)
        pred = sum(c * h ** i * q ** j * g[gn] * pf[p] for i, j, gn, p, c in terms)
        Mlav = D(q ** 2 / h + h ** 5 / 5, 1)
        sl = (slice(None), ix)
        yy, pp, ml = y[sl], pred[sl], Mlav[sl]
        ok = np.isfinite(yy) & np.isfinite(pp)
        rm = np.sqrt((yy[ok] ** 2).mean()); rmse = np.sqrt(((pp - yy)[ok] ** 2).mean())
        rows.append(dict(case=n, held=n in held, Re=Re, Ca=Ca, beta=beta, We=Re * Ca, Fr2=Re * np.sin(np.radians(beta)) / 3,
                         rms=rm, rel=rm / np.sqrt((ml[np.isfinite(ml)] ** 2).mean()), rmse=rmse, nrmse=rmse / rm))
    rows.sort(key=lambda r: r['We'])
    print(f"target={a.target} k={a.k}; rows sorted by We\n{'case':26s} {'':1s} {'We':>6s} {'Fr2':>6s} {'beta':>5s} {'rms':>7s} {'rms/Mlav':>8s} {'RMSE':>7s} {'NRMSE':>6s}")
    for r in rows:
        print(f"{r['case']:26s} {'H' if r['held'] else ' '} {r['We']:6.2f} {r['Fr2']:6.2f} {r['beta']:5.1f} {r['rms']:7.4f} {r['rel']:8.3f} {r['rmse']:7.4f} {r['nrmse']:6.3f}")
    for label, sel in [('all 22', rows), ('held-out 7', [r for r in rows if r['held']])]:
        print(f'\nSpearman rho ({label}):')
        for out in ['nrmse', 'rmse']:
            print(f'  {out:6s}' + ' '.join(f"{p}:{spearmanr([r[p] for r in sel], [r[out] for r in sel])[0]:6.2f}" for p in ['We', 'Fr2', 'beta', 'Re', 'Ca', 'rms', 'rel']))


if __name__ == '__main__':
    main()
