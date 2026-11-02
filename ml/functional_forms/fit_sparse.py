#!/usr/bin/env python
"""Pooled sparse regression for DeltaM_x and S = DeltaM_x + DeltaT_w (route A).

Library: base(h,q) * gradient * parameter factor, with
  base      h^i q^j, i+j <= DEG
  gradient  Mx: h_x, q_x ; S: h_x, q_x, h_xx, q_xx      (every term vanishes for uniform flow)
  param     1, Re, 1/Re, 1/We, 1/Fr^2, cos(beta)/Fr^2   (We = Re Ca, Fr^2 = Re sin(beta)/3)
Data: functional_mlp cache, t > T0, X0 < x < 0.95 L, every STRIDE-th x. Each case is weighted by
1/rms_c, so errors are NRMSE_c = rms(error)/rms(truth) per case (rms, not std; both ~equal here).
Per-case Gram matrices are accumulated once, so selection, leave-one-case-out (LOCO) and
beta-holdout folds are exact and cheap. Forward selection (greedy, exact term counts) is
re-run inside every fold. Development cases are used for training/LOCO; held-out cases only
for the final test.
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

NAMES_G = {'Mx': ['hx', 'qx'], 'S': ['hx', 'qx', 'hxx', 'qxx']}
PNAMES = ['1', 'Re', '1/Re', '1/We', '1/Fr2', 'cosb/Fr2']


def case_params(name):
    m = re.match(r'Re(\d+)_Ca(\d+)_(\d+)_theta([\d_]+)', name)
    Re = float(m.group(1)); Ca = float(f'{m.group(2)}.{m.group(3)}'); beta = float(m.group(4).replace('_', '.'))
    return Re, Ca, beta


def pfactors(Re, Ca, beta):
    b = np.radians(beta); fr2 = Re * np.sin(b) / 3; we = Re * Ca
    return np.array([1., Re, 1 / Re, 1 / we, 1 / fr2, np.cos(b) / fr2])


def build_gram(root, e, a):
    d = root / e['name']
    t, x = np.load(d / 't.npy'), np.load(d / 'x.npy'); dx = x[1] - x[0]
    it = np.flatnonzero(t > a.t0)
    ix = np.flatnonzero((x > a.x0) & (x < 0.95 * x.max()))[::a.stride]
    bases = [(i, j) for i in range(a.deg + 1) for j in range(a.deg + 1 - i)]
    pf = pfactors(*case_params(e['name']))
    out = {}
    for tgt in ['Mx', 'S']:
        K = len(NAMES_G[tgt]); nf = len(bases) * K * len(pf)
        out[tgt] = dict(G=np.zeros((nf, nf)), b=np.zeros(nf), yy=0., n=0)
    ld = lambda f, rows: np.load(d / f'{f}.npy', mmap_mode='r')[rows].astype(float)
    for c0 in range(0, len(it), 10):
        rows = it[c0:c0 + 10]
        lo, hi = max(ix.min() - a.win, 0), min(ix.max() + a.win + 1, len(x))
        sl = lambda f: ld(f, rows)[:, lo:hi]
        h, q, Mx, Tw = sl('h'), sl('q'), sl('delta_Mx_ML'), sl('delta_Tw_ML')
        D = lambda f, k: savgol_filter(f, a.win, 3, deriv=k, delta=dx, axis=1)
        g = dict(hx=D(h, 1), qx=D(q, 1), hxx=D(h, 2), qxx=D(q, 2))
        cut = lambda f: f[:, ix - lo].ravel()
        h_, q_ = cut(h), cut(q); ok = np.isfinite(h_) & (h_ > 0)
        for tgt, y in [('Mx', cut(Mx)), ('S', cut(Mx + Tw))]:
            cols = []
            for (i, j) in bases:
                B = h_ ** i * q_ ** j
                for gn in NAMES_G[tgt]:
                    for p in pf:
                        cols.append(B * cut(g[gn]) * p)
            X = np.stack(cols, 1)[ok]; yv = y[ok]
            o = out[tgt]; o['G'] += X.T @ X; o['b'] += X.T @ yv; o['yy'] += yv @ yv; o['n'] += len(yv)
    return out, bases


def solve(G, b, act, ridge=1e-9):
    A = G[np.ix_(act, act)]
    return np.linalg.solve(A + ridge * np.trace(A) / len(act) * np.eye(len(act)), b[act])


def forward_select(G, b, kmax, ridge=1e-9):
    """Greedy forward selection on (standardised) Gram matrices. Returns list of active sets."""
    act, path = [], []
    for _ in range(kmax):
        best, bj = -np.inf, None
        for j in range(len(b)):
            if j in act or G[j, j] <= 0:
                continue
            a2 = act + [j]
            try:
                w = solve(G, b, a2, ridge)
            except np.linalg.LinAlgError:
                continue
            gain = w @ b[a2]
            if gain > best:
                best, bj = gain, j
        if bj is None:
            break
        act = act + [bj]; path.append(list(act))
    return path


def nrmse(w, act, c):
    sse = w @ c['G'][np.ix_(act, act)] @ w - 2 * w @ c['b'][act] + c['yy']
    return np.sqrt(max(sse, 0.) / c['yy'])


def pooled(cases, names):
    nf = len(cases[names[0]]['b'])
    G, b = np.zeros((nf, nf)), np.zeros(nf)
    for n in names:
        c = cases[n]; G += c['G'] / c['yy'] * c['n']; b += c['b'] / c['yy'] * c['n']   # weight 1/rms_c^2
    return G, b


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--cache', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'functional_mlp' / 'data_config.json'))
    a.add_argument('--t0', type=float, default=1000.)
    a.add_argument('--x0', type=float, default=100.)
    a.add_argument('--deg', type=int, default=3)
    a.add_argument('--win', type=int, default=21)
    a.add_argument('--stride', type=int, default=4)
    a.add_argument('--kmax', type=int, default=20)
    a = a.parse_args()
    root = Path(a.cache); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    dev, held = cfg['development_cases'], cfg['heldout_cases']
    man = {e['name']: e for e in json.loads((root / 'manifest.json').read_text())['cases']}
    grams = {'Mx': {}, 'S': {}}
    t_ = time.time()
    for n in dev + held:
        g, bases = build_gram(root, man[n], a)
        for tgt in grams:
            grams[tgt][n] = g[tgt]
        print(f'gram {n} {time.time() - t_:.0f}s', flush=True)
    feat = {tgt: [f'h^{i} q^{j} {gn} [{pn}]' for (i, j) in bases for gn in NAMES_G[tgt] for pn in PNAMES] for tgt in grams}
    beta = {n: case_params(n)[2] for n in dev + held}
    res = {}
    for tgt in grams:
        cs = grams[tgt]
        G, b = pooled(cs, dev)
        s = np.sqrt(np.diag(G) + 1e-300)           # column standardisation (via training diag)
        std = lambda cc: dict(G=cc['G'] / np.outer(s, s), b=cc['b'] / s, yy=cc['yy'], n=cc['n'])
        css = {n: std(c) for n, c in cs.items()}
        Gs, bs = pooled(css, dev)
        path = forward_select(Gs, bs, a.kmax)
        rows = []
        for k, act in enumerate(path, 1):
            w = solve(Gs, bs, act)
            tr = np.mean([nrmse(w, act, css[n]) for n in dev])
            ho = [nrmse(w, act, css[n]) for n in held]
            # LOCO with selection inside the fold
            lo = []
            for n in dev:
                Gf, bf = pooled(css, [m for m in dev if m != n]); pf = forward_select(Gf, bf, k)
                if len(pf) < k: continue
                lo.append(nrmse(solve(Gf, bf, pf[-1]), pf[-1], css[n]))
            # beta-holdout: train without beta=6.4 cases (dev), test on all beta=6.4 (dev+held)
            tb = [m for m in dev if beta[m] != 6.4]; Gb, bb = pooled(css, tb); pb = forward_select(Gb, bb, k)
            bh = np.mean([nrmse(solve(Gb, bb, pb[-1]), pb[-1], css[m]) for m in dev + held if beta[m] == 6.4]) if len(pb) == k else np.nan
            rows.append(dict(k=k, train=tr, loco=np.mean(lo), heldout=np.mean(ho), beta64_holdout=bh,
                             terms=[feat[tgt][j] for j in act], coef=[float(c / s[j]) for c, j in zip(w, act)],
                             heldout_cases={n: float(v) for n, v in zip(held, ho)}))
        res[tgt] = rows
        print(f'\n== {tgt}: mean NRMSE (rms-normalised, per case)\n  k   train   LOCO  heldout  beta6.4-holdout')
        for r in rows:
            print(f"{r['k']:3d} {r['train']:7.3f} {r['loco']:6.3f} {r['heldout']:7.3f} {r['beta64_holdout']:8.3f}   +{r['terms'][-1]}")
    (out / 'sparse_path.json').write_text(json.dumps(res, indent=1))
    print('wrote', out / 'sparse_path.json')


if __name__ == '__main__':
    main()
