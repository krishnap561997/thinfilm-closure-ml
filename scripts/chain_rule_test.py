#!/usr/bin/env python
"""Chain-rule / linear-in-gradients structure test for DeltaM_x and S = DeltaM_x + DeltaT_w.

Hypothesis: DeltaM_x = d_x DF, DF = int_u2 - q^2/h - Lambda^2 h^5/45 (Lambda = 3), with
DF = F(h, q) only. Then DeltaM_x = F_h h_x + F_q q_x (linear in first gradients, no
second derivatives).  Per case, fit on the first half of the snapshots (t > t0) and
score on the second half (x0 < x < 0.95 L). Polynomials in (h, q) of total degree DEG.
  sanity     corr(d_x DF, DeltaM_x) and std ratio: checks the DF definition
  R2_F       R^2 of DF ~ poly(h, q)  (test half)
Models (NRMSE = RMSE/std of truth, test half), for targets Mx and S = Mx + Tw:
  M0   poly(h,q)                                   no gradients
  M1   poly(h,q)*h_x + poly(h,q)*q_x               linear in 1st gradients
  M2   M1 + poly*h_xx + poly*q_xx                  adds 2nd derivatives
  M3   poly in (h,q,h_x,q_x), total degree DEG     nonlinear in gradients
  chain  F_h h_x + F_q q_x with F fitted to DF     (Mx only; strict chain rule)
Derivatives: Savitzky-Golay, window WIN, order 3, along x.
"""
import argparse, json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

LAM2_45 = 9. / 45.


def monos(a, b, deg):
    return [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)], a, b


def pf(h, q, deg):
    return np.stack([h ** i * q ** j for i in range(deg + 1) for j in range(deg + 1 - i)], 1)


def pf4(cols, deg):
    n = len(cols)
    exps = [e for e in product(range(deg + 1), repeat=n) if sum(e) <= deg]
    return np.stack([np.prod([c ** k for c, k in zip(cols, e)], axis=0) for e in exps], 1)


def fit(X, y):
    sc = np.abs(X).max(0) + 1e-12
    w, *_ = np.linalg.lstsq(X / sc, y, rcond=1e-10)
    return w / sc


def nrmse(p, y):
    return np.sqrt(((p - y) ** 2).mean()) / y.std()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--cache', required=True)
    a.add_argument('--x0', type=float, default=100.)
    a.add_argument('--t0', type=float, default=1000.)
    a.add_argument('--deg', type=int, default=4)
    a.add_argument('--win', type=int, default=21)
    a.add_argument('--stride', type=int, default=4)
    a.add_argument('--cases', nargs='*')
    a = a.parse_args()
    root = Path(a.cache)
    man = json.loads((root / 'manifest.json').read_text())
    hdr = f"{'case':26s} {'corrDF':>6s} {'R2_F':>6s} | " + ' '.join(f'{m:>6s}' for m in ['Mx:M0', 'M1', 'M2', 'M3', 'chain']) + ' | ' + ' '.join(f'{m:>6s}' for m in ['S:M0', 'M1', 'M2', 'M3'])
    print(f'deg={a.deg} win={a.win} t>{a.t0} x>{a.x0}; NRMSE on later half of snapshots\n{hdr}')
    res = []
    for e in man['cases']:
        if a.cases and e['name'] not in a.cases:
            continue
        d = root / e['name']
        t, x = np.load(d / 't.npy'), np.load(d / 'x.npy')
        dx = x[1] - x[0]
        it = np.flatnonzero(t > a.t0)
        ix = np.flatnonzero((x > a.x0) & (x < 0.95 * x.max()))
        ld = lambda f: np.load(d / f'{f}.npy', mmap_mode='r')[it].astype(float)
        h, q, u2, Mx, Tw = ld('h'), ld('q'), ld('int_u2_dy'), ld('delta_Mx_ML'), ld('delta_Tw_ML')
        D = lambda f, k: savgol_filter(f, a.win, 3, deriv=k, delta=dx, axis=1)
        DF = u2 - q ** 2 / h - LAM2_45 * h ** 5
        sub = lambda f: f[:, ix][:, ::a.stride]
        c1 = np.corrcoef(sub(D(DF, 1)).ravel(), sub(Mx).ravel())[0, 1]
        flds = dict(h=sub(h), q=sub(q), hx=sub(D(h, 1)), qx=sub(D(q, 1)), hxx=sub(D(h, 2)), qxx=sub(D(q, 2)),
                    DF=sub(DF), Mx=sub(Mx), S=sub(Mx + Tw))
        nt = flds['h'].shape[0]; tr = np.arange(nt) < nt // 2
        g = {k: v[tr].ravel() for k, v in flds.items()}; s = {k: v[~tr].ravel() for k, v in flds.items()}
        ok = lambda D_: np.isfinite(D_['h']) & (D_['h'] > 0)
        g = {k: v[ok(g)] for k, v in g.items()}; s = {k: v[ok(s)] for k, v in s.items()}

        def design(D_, kind):
            P = pf(D_['h'], D_['q'], a.deg)
            if kind == 'M0': return P
            if kind == 'M1': return np.hstack([P * D_['hx'][:, None], P * D_['qx'][:, None]])
            if kind == 'M2': return np.hstack([P * D_[k][:, None] for k in ['hx', 'qx', 'hxx', 'qxx']])
            if kind == 'M3': return pf4([D_['h'], D_['q'], D_['hx'], D_['qx']], a.deg)
        out = {}
        for tgt in ['Mx', 'S']:
            for kind in ['M0', 'M1', 'M2', 'M3']:
                w = fit(design(g, kind), g[tgt])
                out[(tgt, kind)] = nrmse(design(s, kind) @ w, s[tgt])
        wF = fit(pf(g['h'], g['q'], a.deg), g['DF'])
        r2F = 1 - ((pf(s['h'], s['q'], a.deg) @ wF - s['DF']) ** 2).sum() / ((s['DF'] - s['DF'].mean()) ** 2).sum()
        exps = [(i, j) for i in range(a.deg + 1) for j in range(a.deg + 1 - i)]
        Fh = sum(c * i * s['h'] ** max(i - 1, 0) * s['q'] ** j for c, (i, j) in zip(wF, exps))
        Fq = sum(c * j * s['h'] ** i * s['q'] ** max(j - 1, 0) for c, (i, j) in zip(wF, exps))
        out[('Mx', 'chain')] = nrmse(Fh * s['hx'] + Fq * s['qx'], s['Mx'])
        res.append((e['name'], c1, r2F, out))
        print(f"{e['name']:26s} {c1:6.3f} {r2F:6.3f} | " + ' '.join(f"{out[('Mx', k)]:6.3f}" for k in ['M0', 'M1', 'M2', 'M3', 'chain'])
              + ' | ' + ' '.join(f"{out[('S', k)]:6.3f}" for k in ['M0', 'M1', 'M2', 'M3']), flush=True)
    m = lambda f: np.mean([f(r) for r in res])
    print(f"{'MEAN':26s} {m(lambda r: r[1]):6.3f} {m(lambda r: r[2]):6.3f} | " + ' '.join(f"{m(lambda r: r[3][('Mx', k)]):6.3f}" for k in ['M0', 'M1', 'M2', 'M3', 'chain'])
          + ' | ' + ' '.join(f"{m(lambda r: r[3][('S', k)]):6.3f}" for k in ['M0', 'M1', 'M2', 'M3']))


if __name__ == '__main__':
    main()
