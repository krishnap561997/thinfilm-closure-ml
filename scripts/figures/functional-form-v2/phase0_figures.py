#!/usr/bin/env python
"""Phase 0 talk figures for the Tier-1 form of S = DeltaM_x + DeltaT_w (plan task 0.4).

  fig1_loco_vs_terms   LOCO NRMSE against number of terms: old library (sparse_01) vs Tier 1
  fig2_coef_vs_delta   per-case Tier-1 coefficients against delta = Re/10 (decision D2), theory lines
  fig3_percase_nrmse   per-case NRMSE of the best pooled Tier-1 form against Lavalle alone (S = 0)
Usage: phase0_figures.py --phase0 <phase0.json> --sparse <sparse_path.json> --out <dir>
Prints the numbers each figure shows, so the slide text can quote them.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reference categorical slots 1-3 (dataviz palette, light mode), fixed order; markers as second cue
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
MUTED, INK, SURF = '#898781', '#3d3c38', '#fcfcfb'
BETA_STYLE = {6.4: (C1, 'o'), 45.0: (C2, 's'), 90.0: (C3, '^')}
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'lines.linewidth': 2})


def save(fig, out, name):
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out / f'{name}.png')


def fig1(P, old, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    ks = [r['k'] for r in old['S']]
    axs[0].plot(ks, [r['loco'] for r in old['S']], color=C1, marker='o', ms=5, label='old library (sparse_01)')
    for key, c, m, lab in [('raw_t1000', C2, 's', 'Tier 1, raw basis'), ('rg_t1000', C3, '^', 'Tier 1, (r, g) basis')]:
        rows = P['pooled'][key]
        axs[0].plot([r['k'] for r in rows], [r['loco_rms'] for r in rows], color=c, marker=m, ms=5, label=lab)
    axs[0].set_title('t > 1000 (like for like; RMSE / rms)', fontsize=10)
    for key, c, m, lab in [('raw_full', C2, 's', 'Tier 1, raw basis'), ('rg_full', C3, '^', 'Tier 1, (r, g) basis')]:
        rows = P['pooled'][key]
        axs[1].plot([r['k'] for r in rows], [r['loco_std'] for r in rows], color=c, marker=m, ms=5, label=lab)
    axs[1].set_title('entire record (RMSE / std)', fontsize=10)
    for ax in axs:
        ax.set_xlabel('number of terms'); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel('leave-one-case-out NRMSE of S')
    axs[0].set_ylim(bottom=0)
    save(fig, out, 'fig1_loco_vs_terms')
    best_old = min(r['loco'] for r in old['S'])
    for key in ['raw_t1000', 'rg_t1000', 'raw_full', 'rg_full']:
        rows = P['pooled'][key]; m = 'loco_rms' if 't1000' in key else 'loco_std'
        b = min(rows, key=lambda r: r[m])
        print(f"  fig1 {key}: best LOCO {b[m]:.3f} at k={b['k']} (old library best {best_old:.3f})")


def fig2(P, out, window='full'):
    ids = list(P['theory_coef'])
    fig, axs = plt.subplots(3, 3, figsize=(10, 8.2))
    held = {n for n, r in P['per_case'].items() if n in P['_held']}
    for ax, k in zip(axs.flat, ids):
        for n, r in P['per_case'].items():
            pl = P['placement'][n]; c, m = BETA_STYLE[pl['beta']]
            w = r[window]; v, lo, hi = w['coef'][k], w['lo'][k], w['hi'][k]
            if v is None or not np.isfinite(v):          # column vanishes for this case (cot beta = 0)
                continue
            # percentile interval can exclude the point estimate when the bootstrap is skewed
            ax.errorbar(pl['delta_D2'], v, yerr=[[max(v - lo, 0.)], [max(hi - v, 0.)]], fmt=m, ms=6, color=c,
                        mfc=SURF if n in held else c, mew=1.2, elinewidth=1, capsize=0)
        ax.axhline(P['theory_coef'][k], color=MUTED, ls='--', lw=1.2)
        ax.set_xscale('log'); ax.set_title(f"{k}  [{P['factor'][k]}]   theory {P['theory_coef'][k]:.4g}", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel('δ = Re / 10')
    handles = [plt.Line2D([], [], color=c, marker=m, ls='', ms=6, label=f'β = {b:g}°') for b, (c, m) in BETA_STYLE.items()]
    handles += [plt.Line2D([], [], color=INK, marker='o', mfc=SURF, ls='', ms=6, label='held-out case (open)'),
                plt.Line2D([], [], color=MUTED, ls='--', label='row-1 theory')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Per-case Tier-1 coefficients of S (entire record; bars: 95% block bootstrap)', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, out, 'fig2_coef_vs_delta')
    for k in ids:
        lowd = [P['per_case'][n][window]['coef'][k] for n in P['per_case'] if P['placement'][n]['delta_D2'] <= 2]
        print(f"  fig2 {k}: theory {P['theory_coef'][k]:.4g}, median over delta<=2 cases {np.nanmedian(lowd):.4g} (n={len(lowd)})")


def fig3(P, out, key='raw_full'):
    rows = P['pooled'][key]
    best = min(rows, key=lambda r: r['loco_std'])
    errs = {**best['dev_cases'], **best['heldout_cases']}
    names = sorted(errs, key=lambda n: (P['placement'][n]['delta_D2'], n))
    lav = [P['per_case'][n]['full']['nrmse_lavalle'] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(y + 0.2, lav, height=0.38, color='#c9c7c0', label='Lavalle alone (S = 0)')
    ax.barh(y - 0.2, [errs[n] for n in names], height=0.38, color=C1, label=f"best pooled Tier-1 form ({best['k']} terms)")
    ax.set_yticks(y, [n.replace('_theta', ' β') + (' (H)' if n in best['heldout_cases'] else '') for n in names], fontsize=8)
    ax.set_xlabel('NRMSE of S (RMSE / std, entire record)'); ax.invert_yaxis(); ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_title('Per case, sorted by δ = Re/10; (H) = held-out, others in-sample', fontsize=10)
    save(fig, out, 'fig3_percase_nrmse')
    print(f"  fig3 best {key}: k={best['k']} LOCO {best['loco_std']:.3f} held-out {best['heldout_std']:.3f} beta64 {best['beta64_std']:.3f}")
    print('       terms: ' + ', '.join(f'{t} {c:.4g}' for t, c in zip(best['terms'], best['coef'])))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--phase0', required=True)
    a.add_argument('--sparse', required=True)
    a.add_argument('--config', default=str(Path(__file__).resolve().parents[3] / 'ml' / 'functional_mlp' / 'data_config.json'))
    a.add_argument('--out', required=True)
    a = a.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = json.loads(Path(a.phase0).read_text())
    P['_held'] = set(json.loads(Path(a.config).read_text())['heldout_cases'])
    fig1(P, json.loads(Path(a.sparse).read_text()), out)
    fig2(P, out)
    fig3(P, out)


if __name__ == '__main__':
    main()
