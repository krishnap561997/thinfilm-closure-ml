#!/usr/bin/env python3
"""Save individual QoI/region plots and absolute-correlation peak locations.

Example:
  python plot_statistics.py CASE_statistics.h5 --format png --output-dir figures

Correlation limits enclose the strongest-|R| lobes of BOTH h and q curves.
Each lobe ends at its nearest zero crossing on either side (linearly interpolated).
Without a crossing, the finite-data boundary is used. --zero-tol additionally
accepts sampled |R| <= tolerance. Missing-data gaps are never crossed.

--kind correlations (default), pdfs, or both. Each region gets its own directory,
with pearson/ and pdfs_<type>/ subdirectories. Peak CSVs are saved inside that
region directory. Distances retain source x units.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._') or 'unnamed'


def simulation_name(path, f):
    stem = Path(path).stem
    if stem.endswith('_statistics'):
        stem = stem[:-len('_statistics')]
    stem = stem.removeprefix('thinfilm_closure_data_')
    if stem.startswith('Re'):
        return safe_name(stem)
    try:
        params = json.loads(f.attrs['parameters_json'])
        ca = f"{params['Ca']:.3f}".replace('.', '_')
        beta = f"{params['beta']:g}".replace('.', '_')
        return f"Re{int(params['Re']):03d}_Ca{ca}_theta{beta}"
    except (KeyError, TypeError, ValueError):
        return safe_name(stem)


def peak_lobe(distances, curve, zero_tol=0.0):
    """Return strongest-|R| peak and its zero-bounded lobe; None if undefined."""
    r, y = np.asarray(distances, float), np.asarray(curve, float)
    valid = np.flatnonzero(np.isfinite(y))
    if not len(valid):
        return None
    maximum = np.max(np.abs(y[valid]))
    ties = valid[np.abs(y[valid]) == maximum]
    peak = int(ties[np.argmin(np.abs(r[ties]))])
    if maximum == 0:
        return dict(index=peak, distance=float(r[peak]), correlation=0.0,
                    absolute_correlation=0.0, left=float(r[valid[0]]),
                    right=float(r[valid[-1]]), left_reason='zero_curve',right_reason='zero_curve')

    def boundary(direction):
        i = peak
        while 0 <= i + direction < len(y):
            j = i + direction
            if not np.isfinite(y[j]):
                return float(r[i]), 'finite_data_boundary'
            if y[j] == 0:
                return float(r[j]), 'zero'
            if np.signbit(y[i]) != np.signbit(y[j]):
                crossing = r[i] - y[i]*(r[j]-r[i])/(y[j]-y[i])
                return float(crossing), 'interpolated_zero'
            if abs(y[j]) <= zero_tol:
                return float(r[j]), 'near_zero'
            i = j
        return float(r[i]), 'available_boundary'

    left, lreason = boundary(-1)
    right, rreason = boundary(1)
    return dict(index=peak, distance=float(r[peak]), correlation=float(y[peak]),
                absolute_correlation=float(maximum), left=left, right=right,
                left_reason=lreason, right_reason=rreason)


def visible_extents(r, curves, left, right):
    values = []
    inside = (r >= left) & (r <= right)
    for y in curves:
        values.extend(y[inside & np.isfinite(y)].tolist())
        # Include interpolated values at limits when they fall inside a segment.
        for bound in (left, right):
            j = int(np.searchsorted(r, bound))
            if 0 < j < len(r) and np.isfinite(y[j-1:j+1]).all():
                values.append(float(np.interp(bound,r[j-1:j+1],y[j-1:j+1])))
    if not values:
        return -1.0, 1.0
    lo, hi = min(values), max(values)
    padding = .06*(hi-lo) if hi>lo else max(.02, .06*abs(lo))
    return lo-padding, hi+padding


def save_figure(fig, directory, basename, formats, dpi):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = directory / f'{basename}.{ext}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'SAVED {path.resolve()}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=Path)
    parser.add_argument('--qois', nargs='+')
    parser.add_argument('--regions', nargs='+', default=['All','Highlands','Lowlands'])
    parser.add_argument('--kind', choices=('correlations','pdfs','both'), default='correlations')
    parser.add_argument('--pdf', choices=('raw','standardized'), default='standardized',help='PDF variable if --kind pdfs or both')
    parser.add_argument('--format', choices=('png','pdf','both'), default='png',help='Figure file format')
    parser.add_argument('--output-dir',type=Path,default=Path('figures'))
    parser.add_argument('--zero-tol',type=float,default=0.0,help='Optional sampled |R| threshold treated as zero')
    parser.add_argument('--dpi',type=int,default=300)
    args = parser.parse_args()
    if not np.isfinite(args.zero_tol) or not 0 <= args.zero_tol < 1:
        parser.error('--zero-tol must lie in [0,1)')
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    formats = ('png','pdf') if args.format=='both' else (args.format,)
    rows = []
    with h5py.File(args.file,'r') as f:
        qois = args.qois or list(f['qois'])
        if not qois:
            parser.error('No QoIs selected or stored')
        for name in qois:
            for region in args.regions:
                if f'qois/{name}/{region}' not in f:
                    parser.error(f'No stored results: {name}/{region}')
        simulation = simulation_name(args.file,f)
        r = np.asarray(f['distances'][:],float)
        if r.ndim!=1 or not len(r) or not np.isfinite(r).all() or np.any(np.diff(r)<=0):
            parser.error('Stored distances must be finite and strictly increasing')
        offsets = np.asarray(f['offsets'][:]) if 'offsets' in f else None
        if offsets is not None and offsets.shape!=r.shape:
            parser.error('Stored offsets and distances have different shapes')
        for name in qois:
            for region in args.regions:
                basename = f'{simulation}_{safe_name(name)}_{safe_name(region)}'
                if args.kind in ('correlations','both'):
                    g = f[f'qois/{name}/{region}/pearson']
                    curves = {field:np.asarray(g[field][:],float) for field in ('h','q')}
                    if any(y.shape!=r.shape for y in curves.values()):
                        parser.error(f'Curve shape mismatch: {name}/{region}')
                    lobes = {field:peak_lobe(r,y,args.zero_tol) for field,y in curves.items()}
                    defined = [peak for peak in lobes.values() if peak is not None]
                    for field,peak in lobes.items():
                        row = dict(simulation=simulation,qoi=name,region=region,field=field,
                                   n_points=int(g.attrs.get('n_points',0)),status='ok' if peak else 'undefined',
                                   peak_offset=np.nan,peak_distance=np.nan,peak_correlation=np.nan,
                                   peak_absolute_correlation=np.nan,left_limit=np.nan,right_limit=np.nan,
                                   left_reason='',right_reason='')
                        if peak:
                            row.update(peak_offset=int(offsets[peak['index']]) if offsets is not None else np.nan,
                                       peak_distance=peak['distance'],peak_correlation=peak['correlation'],
                                       peak_absolute_correlation=peak['absolute_correlation'],left_limit=peak['left'],
                                       right_limit=peak['right'],left_reason=peak['left_reason'],right_reason=peak['right_reason'])
                            print(f"PEAK {simulation} | {name} | {region} | {field}: "
                                  f"r={peak['distance']:.10g}, R={peak['correlation']:.8g}, "
                                  f"|R|={peak['absolute_correlation']:.8g}; "
                                  f"bounds=[{peak['left']:.8g}, {peak['right']:.8g}] "
                                  f"({peak['left_reason']}, {peak['right_reason']})")
                        else:
                            print(f'UNDEFINED {simulation} | {name} | {region} | {field}')
                        rows.append(row)
                    if not defined:
                        print(f'SKIP undefined correlations: {name}/{region}')
                    else:
                        # Union keeps both strongest lobes/peaks visible on one panel.
                        left = min(peak['left'] for peak in defined)
                        right = max(peak['right'] for peak in defined)
                        if left==right:
                            pad = .5*np.min(np.diff(r)) if len(r)>1 else .5
                            left,right = left-pad,right+pad
                        fig,ax = plt.subplots(figsize=(8,5),constrained_layout=True)
                        for field,color in (('h','tab:blue'),('q','tab:orange')):
                            ax.plot(r,curves[field],color=color,lw=1.5,label=rf'$R_{{Y{field}}}(r)$')
                            peak = lobes[field]
                            if peak:
                                ax.axvline(peak['distance'],color=color,ls='--',lw=1,
                                           label=f"{field}: max |R|={peak['absolute_correlation']:.3g} at r={peak['distance']:.5g}")
                                ax.plot(peak['distance'],peak['correlation'],'o',color=color,ms=5)
                        ax.axhline(0,color='gray',lw=.7)
                        if left<=0<=right: ax.axvline(0,color='gray',lw=.7)
                        # ax.set_xlim(left,right)
                        ax.set_xlim(r.min(),r.max())
                        #ax.set_ylim(*visible_extents(r,list(curves.values()),left,right))
                        ax.set_ylim(*visible_extents(r,list(curves.values()),r.min(),r.max()))
                        ax.set_xlabel('Separation r (same units as x)')
                        ax.set_ylabel('Pearson correlation')
                        ax.set_title(f"{simulation}\n{name} / {region} — {g.attrs.get('n_points',0):,} observations")
                        ax.grid(alpha=.3);ax.legend(fontsize=9)
                        directory = args.output_dir/safe_name(region)/'pearson'
                        save_figure(fig,directory,basename,formats,args.dpi)
                if args.kind in ('pdfs','both'):
                    g = f[f'qois/{name}/{region}/pdf/{args.pdf}']
                    density,edges = g['density'][:],g['bin_edges'][:]
                    finite = np.isfinite(density)
                    if not finite.any():
                        print(f'SKIP undefined PDF: {name}/{region}')
                        continue
                    fig,ax = plt.subplots(figsize=(8,5),constrained_layout=True)
                    ax.stairs(density,edges,lw=1.5)
                    ax.set_xlim(edges[0],edges[-1])
                    maximum = float(np.max(density[finite]))
                    ax.set_ylim(0,maximum*1.06 if maximum>0 else 1)
                    ax.set_xlabel('(Y - region mean) / region std' if args.pdf=='standardized' else 'QoI value')
                    ax.set_ylabel('Probability density');ax.set_title(f'{simulation}\n{name} / {region}')
                    ax.grid(alpha=.3)
                    directory = args.output_dir/safe_name(region)/f'pdfs_{args.pdf}'
                    save_figure(fig,directory,basename,formats,args.dpi)
    if rows:
        for region in args.regions:
            region_rows = [row for row in rows if row['region'] == region]
            if not region_rows:
                continue
            directory = args.output_dir/safe_name(region)
            directory.mkdir(parents=True,exist_ok=True)
            path = directory/f'{simulation}_correlation_peaks.csv'
            with path.open('w',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=list(region_rows[0]))
                writer.writeheader();writer.writerows(region_rows)
            print(f'SAVED {region} peak locations: {path.resolve()}')

if __name__=='__main__':
    main()
