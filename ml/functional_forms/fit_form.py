#!/usr/bin/env python
"""Phase 0 fits of the Tier-1 library for S = DeltaM_x + DeltaT_w (plan tasks 0.3a-c).

Data: functional_mlp cache, X0 < x < 0.95 L, every STRIDE-th x, Savitzky-Golay derivatives
(window WIN, order 3: the D_SG operator of the plan, talk slice only), snapshots dropped as
listed in hygiene_phase0.json. Windows: 'full' (entire record) and 't1000' (t > 1000, to compare
like for like with sparse_01).
  (a) per case: Tier-1 least squares with theory parameter factors (raw basis), coefficients
      with 95% block-bootstrap intervals (blocks = snapshots); NRMSE of the fit, of the theory
      coefficients as they are, and of Lavalle alone (S = 0)
  (b) pooled forward selection on Tier-1 columns x parameter factors (raw and rg bases), cases
      weighted by 1/rms_c as in fit_sparse.py; train / LOCO (selection inside each fold) /
      held-out / beta-6.4 holdout
  (c) case placement: eps_k = k_peak h_N (time-averaged h spectrum), eps_g = rms(h_x)/mean(h),
      delta = eps Re for both and for eps = 0.1 (decision D2), ordering-matrix row
NRMSE_std = RMSE/std(truth) (plan metric M1); NRMSE_rms = RMSE/rms(truth) (as in sparse_01).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

sys.dont_write_bytecode = True
import library as L
from fit_sparse import case_params, forward_select, solve

WINDOWS = {'full': 0., 't1000': 1000.}


def row_of(delta):
    return 1 if delta < np.sqrt(10) else (2 if delta < 10 * np.sqrt(10) else 3)


def new_acc(nf):
    return dict(G=np.zeros((nf, nf)), b=np.zeros(nf), yy=0., sy=0., n=0)


def add(acc, X, y):
    acc['G'] += X.T @ X; acc['b'] += X.T @ y; acc['yy'] += y @ y; acc['sy'] += y.sum(); acc['n'] += len(y)


def design_names(basis):
    names = []
    for k, c in L.BASES[basis].items():
        for f in [c['factor']] + [f for f in L.GENERIC_FACTORS if f != c['factor']]:
            names.append((k, f))
    return names


def build_case(root, name, a, hyg):
    d = root / name
    t, x = np.load(d / 't.npy'), np.load(d / 'x.npy'); dx = x[1] - x[0]
    Re, Ca, beta = case_params(name)
    keep = np.ones(len(t), bool)
    if name in hyg['drop_t_gt']:
        keep &= t <= hyg['drop_t_gt'][name]
    for tt in hyg['drop_t'].get(name, []):
        keep &= ~np.isclose(t, tt)
    it = np.flatnonzero(keep)
    reg = np.flatnonzero((x > a.x0) & (x < 0.95 * x.max()))
    ix = reg[::a.stride]
    fixed_ids = list(L.RAW)
    ffac = np.array([L.factor_value(L.RAW[k]['factor'], Re, Ca, beta) for k in fixed_ids])
    dn = {b: design_names(b) for b in L.BASES}
    dfac = {b: np.array([L.factor_value(f, Re, Ca, beta) for _, f in dn[b]]) for b in L.BASES}
    snap = []                                       # per-snapshot Gram of the fixed raw design
    pooled = {(b, w): new_acc(len(dn[b])) for b in L.BASES for w in WINDOWS}
    power, hx_ss, h_s, nsp = 0., 0., 0., 0
    ld = lambda f, rows: np.load(d / f'{f}.npy', mmap_mode='r')[rows].astype(float)
    win = np.hanning(len(reg))
    for c0 in range(0, len(it), 10):
        rows = it[c0:c0 + 10]
        h, q = ld('h', rows), ld('q', rows)
        S = ld('delta_Mx_ML', rows) + ld('delta_Tw_ML', rows)
        D = lambda f, k: savgol_filter(f, a.win, 3, deriv=k, delta=dx, axis=1)
        fl = dict(h=h, q=q, hx=D(h, 1), qx=D(q, 1), hxx=D(h, 2), qxx=D(q, 2))
        hr = h[:, reg]
        power = power + (np.abs(np.fft.rfft((hr - hr.mean(1, keepdims=True)) * win, axis=1)) ** 2).sum(0)
        hx_ss += (fl['hx'][:, reg] ** 2).sum(); h_s += hr.sum(); nsp += hr.size
        cols = {b: L.evaluate(b, {k: v[:, ix] for k, v in fl.items()}) for b in L.BASES}
        y = S[:, ix]
        for r in range(len(rows)):
            ok = np.isfinite(y[r]) & (h[r, ix] > 0)
            Xf = np.stack([cols['raw'][k][r] for k in fixed_ids], 1)[ok] * ffac
            acc = new_acc(len(fixed_ids)); add(acc, Xf, y[r][ok]); acc['t'] = float(t[rows[r]]); snap.append(acc)
            for b in L.BASES:
                Xd = np.stack([cols[b][k][r] for k, _ in dn[b]], 1)[ok] * dfac[b]
                for w, t0 in WINDOWS.items():
                    if t[rows[r]] > t0:
                        add(pooled[(b, w)], Xd, y[r][ok])
    kk = 2 * np.pi * np.fft.rfftfreq(len(reg), dx)
    kp = kk[3 + np.argmax(power[3:])]
    eps_g = np.sqrt(hx_ss / nsp) / (h_s / nsp)
    place = dict(Re=Re, Ca=Ca, beta=beta, We=Re * Ca, eps_k=float(kp), eps_g=float(eps_g),
                 delta_k=float(kp * Re), delta_g=float(eps_g * Re), delta_D2=Re / 10.,
                 row_k=row_of(kp * Re), row_g=row_of(eps_g * Re), row_D2=row_of(Re / 10.),
                 n_snap=len(it), dropped=int((~keep).sum()))
    return snap, pooled, place, fixed_ids


def lsq(G, b, ridge=1e-10):
    """Standardised least squares; columns that vanish (e.g. cot(beta) terms at 90 deg) get NaN."""
    act = np.flatnonzero(np.diag(G) > 0)
    s = np.sqrt(np.diag(G)[act])
    A = G[np.ix_(act, act)] / np.outer(s, s)
    w = np.full(len(b), np.nan)
    w[act] = np.linalg.solve(A + ridge * np.eye(len(act)), b[act] / s) / s
    return w


def err(w, acc):
    w = np.nan_to_num(w)
    sse = max(w @ acc['G'] @ w - 2 * w @ acc['b'] + acc['yy'], 0.)
    sst = acc['yy'] - acc['sy'] ** 2 / acc['n']
    return np.sqrt(sse / sst), np.sqrt(sse / acc['yy'])          # (std-, rms-normalised)


def sum_acc(accs):
    out = new_acc(len(accs[0]['b']))
    for c in accs:
        for k in ['G', 'b', 'yy', 'sy', 'n']:
            out[k] = out[k] + c[k]
    return out


def per_case(snap, ids, t0, nboot, rng, theory):
    sel = [s for s in snap if s['t'] > t0]
    acc = sum_acc(sel)
    w = lsq(acc['G'], acc['b'])
    boots = []
    for _ in range(nboot):
        pick = rng.integers(0, len(sel), len(sel))
        a2 = sum_acc([sel[i] for i in pick])
        boots.append(lsq(a2['G'], a2['b']))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(coef=dict(zip(ids, w.tolist())), lo=dict(zip(ids, lo.tolist())), hi=dict(zip(ids, hi.tolist())),
                nrmse_fit=err(w, acc)[0], nrmse_theory=err(theory, acc)[0], nrmse_lavalle=err(np.zeros_like(w), acc)[0],
                n=acc['n'], n_snap=len(sel))


def pooled_path(cases, dev, held, beta, kmax, names):
    def stdz(c, s):
        return dict(G=c['G'] / np.outer(s, s), b=c['b'] / s, yy=c['yy'], sy=c['sy'], n=c['n'])

    def pool(cs, nn):
        nf = len(cs[nn[0]]['b']); G, b = np.zeros((nf, nf)), np.zeros(nf)
        for n in nn:
            c = cs[n]; G += c['G'] / c['yy'] * c['n']; b += c['b'] / c['yy'] * c['n']
        return G, b
    G0, _ = pool(cases, dev)
    s = np.sqrt(np.diag(G0)) + 1e-300
    cs = {n: stdz(c, s) for n, c in cases.items()}
    Gs, bs = pool(cs, dev)
    path = forward_select(Gs, bs, kmax)
    rows = []
    for k, act in enumerate(path, 1):
        w = solve(Gs, bs, act)
        e = lambda n, ww=w, aa=act: err_sub(ww, aa, cs[n])
        lo = []
        for n in dev:
            Gf, bf = pool(cs, [m for m in dev if m != n]); pf = forward_select(Gf, bf, k)
            if len(pf) == k:
                lo.append(err_sub(solve(Gf, bf, pf[-1]), pf[-1], cs[n]))
        tb = [m for m in dev if beta[m] != 6.4]; Gb, bb = pool(cs, tb); pb = forward_select(Gb, bb, k)
        bh = [err_sub(solve(Gb, bb, pb[-1]), pb[-1], cs[m]) for m in dev + held if beta[m] == 6.4] if len(pb) == k else []
        mean = lambda v, j: float(np.mean([x[j] for x in v])) if v else float('nan')
        tr, ho = [e(n) for n in dev], [e(n) for n in held]
        rows.append(dict(k=k, terms=[f'{names[j][0]}[{names[j][1]}]' for j in act],
                         coef=[float(c / s[j]) for c, j in zip(w, act)],
                         **{f'{lab}_{m}': mean(v, j) for lab, v in [('train', tr), ('loco', lo), ('heldout', ho), ('beta64', bh)]
                            for j, m in enumerate(['std', 'rms'])},
                         heldout_cases={n: float(v[0]) for n, v in zip(held, ho)},
                         dev_cases={n: float(v[0]) for n, v in zip(dev, tr)}))
    return rows


def err_sub(w, act, c):
    G = c['G'][np.ix_(act, act)]
    sse = max(w @ G @ w - 2 * w @ c['b'][act] + c['yy'], 0.)
    return np.sqrt(sse / (c['yy'] - c['sy'] ** 2 / c['n'])), np.sqrt(sse / c['yy'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    here = Path(__file__).resolve().parent
    p.add_argument('--config', default=str(here.parent / 'functional_mlp' / 'data_config.json'))
    p.add_argument('--hygiene', default=str(here / 'hygiene_phase0.json'))
    p.add_argument('--x0', type=float, default=100.)
    p.add_argument('--win', type=int, default=21)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--kmax', type=int, default=12)
    p.add_argument('--nboot', type=int, default=200)
    a = p.parse_args()
    root, out = Path(a.cache), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text()); hyg = json.loads(Path(a.hygiene).read_text())
    dev, held = cfg['development_cases'], cfg['heldout_cases']
    rng = np.random.default_rng(0)
    res = dict(args=vars(a), placement={}, per_case={}, pooled={})
    pooled_all = {}
    t_ = time.time()
    for n in dev + held:
        snap, pooled, place, ids = build_case(root, n, a, hyg)
        Re, Ca, beta = case_params(n)
        theory = np.array([float(L.RAW[k]['theory']) for k in ids])
        res['placement'][n] = place
        res['per_case'][n] = {w: per_case(snap, ids, t0, a.nboot, rng, theory) for w, t0 in WINDOWS.items()}
        pooled_all[n] = pooled
        print(f'{n} done {time.time() - t_:.0f}s', flush=True)
    beta = {n: case_params(n)[2] for n in dev + held}
    for b in L.BASES:
        for w in WINDOWS:
            res['pooled'][f'{b}_{w}'] = pooled_path({n: pooled_all[n][(b, w)] for n in dev + held}, dev, held, beta, a.kmax, design_names(b))
    res['theory_coef'] = {k: float(c['theory']) for k, c in L.RAW.items()}
    res['factor'] = {k: c['factor'] for k, c in L.RAW.items()}
    (out / 'phase0.json').write_text(json.dumps(res, indent=1))

    print(f"\n(c) placement{'':16s} We   eps_k  eps_g  delta_k delta_g Re/10 rows(k,g,D2) dropped")
    for n in dev + held:
        q = res['placement'][n]
        print(f"{n:26s} {q['We']:5.2f} {q['eps_k']:6.3f} {q['eps_g']:6.3f} {q['delta_k']:7.2f} {q['delta_g']:7.2f} {q['delta_D2']:5.1f}  {q['row_k']},{q['row_g']},{q['row_D2']}   {q['dropped']}")
    for w in WINDOWS:
        print(f'\n(a) per-case Tier-1, window {w}: NRMSE_std of fit / theory coefficients / Lavalle (S=0)')
        for n in dev + held:
            r = res['per_case'][n][w]
            print(f"{n:26s} {'H' if n in held else ' '} fit {r['nrmse_fit']:6.3f}  theory {r['nrmse_theory']:8.3f}  lavalle {r['nrmse_lavalle']:6.3f}")
    for key, rows in res['pooled'].items():
        print(f'\n(b) pooled {key}: k  train  LOCO  heldout  beta6.4  (std; LOCO_rms)   +term')
        for r in rows:
            print(f"   {r['k']:2d} {r['train_std']:6.3f} {r['loco_std']:6.3f} {r['heldout_std']:6.3f} {r['beta64_std']:6.3f}  ({r['loco_rms']:5.3f})  +{r['terms'][-1]}")
    print('wrote', out / 'phase0.json')


if __name__ == '__main__':
    main()
