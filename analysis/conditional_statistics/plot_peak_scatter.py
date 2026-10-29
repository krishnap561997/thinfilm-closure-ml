#!/usr/bin/env python3
"""Combine *_correlation_peaks.csv files and scatter the strongest-|R| peaks.

Default: x=physical peak distance, y=|R|, color=simulation, marker=h or q.
One figure and combined CSV per region, with one panel per QoI.
No simulation HDF5 files are reopened. Region folders are searched recursively.

Examples:
  python plot_peak_scatter.py --root figures_by_region --format both
  python plot_peak_scatter.py --root figures --regions All --color-by Re
  python plot_peak_scatter.py --root figures --x Ca --y peak_distance --color-by Re

Multiple configuration folders are retained, not averaged. Use --config-id HASH
or --root figures/HASH to restrict an analysis. Output is saved, not shown.
Dependencies: NumPy, Matplotlib (plus the Python standard library).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SIMULATION = re.compile(r'Re(?P<Re>\d+)_Ca(?P<Ca>\d+_\d+)_theta(?P<beta>\d+(?:_\d+)?)')
NUMERIC = ('peak_offset','peak_distance','peak_correlation','peak_absolute_correlation',
           'left_limit','right_limit','n_points')
LABELS = {
    'peak_distance':'Peak separation r (same units as x)',
    'peak_offset':r'Peak offset $n$ (distance $n\Delta x$)',
    'peak_correlation':r'Signed correlation at strongest $|R|$',
    'peak_absolute_correlation':r'Peak absolute correlation $|R|$',
    'Re':r'$Re$', 'Ca':r'$Ca$', 'beta':r'Inclination $\beta$ (degrees)',
}


def load_peak_csvs(root, config_id=None):
    """Return all finite, status=ok peaks plus provenance from recursive CSV discovery."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(root.rglob('*_correlation_peaks.csv'))
    rows, skipped = [], 0
    matched_files = 0
    for path in files:
        # New layout: figures/<config hash>/<region>/<simulation>_correlation_peaks.csv.
        ids = [part for part in path.parts[:-1] if re.fullmatch(r'[0-9a-fA-F]{12,64}',part)]
        identifier = ids[-1] if ids else None
        matched_files += 1
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'simulation','qoi','region','field','peak_distance',
                        'peak_correlation','peak_absolute_correlation'}
            missing = required-set(reader.fieldnames or [])
            if missing:
                raise ValueError(f'{path}: missing columns {sorted(missing)}')
            for line,row in enumerate(reader,start=2):
                if row.get('status','ok') != 'ok':
                    skipped += 1
                    continue
                row = dict(row)
                region_scoped = path.parent.name == row['region']
                row_identifier = identifier or (path.parent.parent.name if region_scoped else path.parent.name)
                if config_id is not None and row_identifier != config_id:
                    continue
                for key in NUMERIC:
                    try:
                        row[key] = float(row.get(key,'nan'))
                    except (ValueError,TypeError) as error:
                        raise ValueError(f'{path}:{line}: invalid numeric {key}') from error
                if row['field'] not in ('h','q'):
                    raise ValueError(f'{path}:{line}: unrecognized field {row["field"]!r}')
                if not np.isfinite([row['peak_distance'],row['peak_correlation'],row['peak_absolute_correlation']]).all():
                    skipped += 1
                    continue
                match = SIMULATION.search(row['simulation'])
                for parameter in ('Re','Ca','beta'):
                    row[parameter] = float(match[parameter].replace('_','.')) if match else np.nan
                row['config_id'] = row_identifier
                row['csv_layout'] = 'region_folder' if region_scoped else 'legacy_aggregate'
                row['source_csv'] = str(path)
                rows.append(row)
    if not matched_files:
        raise FileNotFoundError(f'No matching *_correlation_peaks.csv files under {root}')
    # If legacy aggregate CSVs coexist with new region CSVs, prefer the latter
    # for the same config/simulation/QoI/region/field instead of double counting.
    unique = {}
    duplicates = 0
    for row in rows:
        key = tuple(row[k] for k in ('config_id','simulation','qoi','region','field'))
        if key not in unique:
            unique[key] = row
        else:
            duplicates += 1
            old = unique[key]
            priority = lambda r: (r['csv_layout']=='region_folder', Path(r['source_csv']).stat().st_mtime_ns, r['source_csv'])
            if priority(row) > priority(old): unique[key] = row
    rows = list(unique.values())
    if duplicates:
        warnings.warn(f'Removed {duplicates} duplicate peak rows; region-folder CSVs take precedence over legacy aggregates, then newest CSV wins.')
    print(f'Loaded {len(rows):,} unique valid peak rows from {matched_files} CSV files; skipped {skipped} undefined rows.')
    if len({row['config_id'] for row in rows}) > 1:
        warnings.warn('Multiple configurations loaded. Points are retained separately; use --config-id or a narrower --root for like-for-like comparisons.')
    return rows


def write_combined(rows,path):
    # This basename deliberately does not match the input discovery pattern.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path('figures'),help='Recursively search this directory for peak CSVs')
    p.add_argument('--output-dir',type=Path,default=Path('peak_scatter'))
    p.add_argument('--config-id',help='Restrict CSVs to a configuration folder')
    p.add_argument('--qois',nargs='+')
    p.add_argument('--regions',nargs='+')
    p.add_argument('--fields',nargs='+',choices=('h','q'),default=['h','q'])
    p.add_argument('--x',choices=tuple(LABELS),default='peak_distance')
    p.add_argument('--y',choices=tuple(LABELS),default='peak_absolute_correlation')
    p.add_argument('--color-by',choices=('simulation','Re','Ca','beta'),default='simulation')
    p.add_argument('--format',choices=('png','pdf','both'),default='png')
    p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--columns',type=int,default=2,help='Maximum QoI panels per row within each region figure')
    p.add_argument('--label-points',action='store_true',help='Annotate each point with its simulation')
    args = p.parse_args()
    if args.dpi<=0 or args.columns<=0: p.error('--dpi and --columns must be positive')
    rows = load_peak_csvs(args.root,args.config_id)
    qois = args.qois or sorted({r['qoi'] for r in rows})
    regions = args.regions or sorted({r['region'] for r in rows})
    available_qois,available_regions = {r['qoi'] for r in rows},{r['region'] for r in rows}
    if set(qois)-available_qois: p.error(f'QoIs not found: {sorted(set(qois)-available_qois)}')
    if set(regions)-available_regions: p.error(f'Regions not found: {sorted(set(regions)-available_regions)}')
    rows = [r for r in rows if r['qoi'] in qois and r['region'] in regions and r['field'] in args.fields]
    if not rows: p.error('No peak rows remain after selection')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    numeric_color = args.color_by != 'simulation'
    plotted = [r for r in rows if np.isfinite(r[args.x]) and np.isfinite(r[args.y])
               and (not numeric_color or np.isfinite(r[args.color_by]))]
    if len(plotted)!=len(rows):
        warnings.warn(f'{len(rows)-len(plotted)} rows lack finite plotting coordinates or simulation parameters; retained in combined CSV but not plotted.')
    if not plotted: p.error('No finite plotting coordinates; check filenames/selected axes')
    simulations = sorted({r['simulation'] for r in plotted})
    categorical = plt.get_cmap('tab20' if len(simulations)<=20 else 'turbo')
    colors = {sim:categorical(i/max(1,len(simulations)-1)) for i,sim in enumerate(simulations)}
    if numeric_color:
        v = [r[args.color_by] for r in plotted]
        lo,hi = min(v),max(v)
        pad = .5*max(abs(lo),1.) if lo==hi else 0
        norm = matplotlib.colors.Normalize(lo-pad,hi+pad)
        mapper = matplotlib.cm.ScalarMappable(norm=norm,cmap='viridis')
    formats=('png','pdf') if args.format=='both' else (args.format,)
    marker_map = {'h':'o','q':'^'}
    for region in regions:
        safe_region = re.sub(r'[^A-Za-z0-9_.-]+','_',region).strip('._') or 'region'
        directory = args.output_dir/safe_region
        directory.mkdir(parents=True,exist_ok=True)
        region_rows = [r for r in rows if r['region']==region]
        if not region_rows:
            print(f'SKIP {region}: no selected peak rows')
            continue
        combined = directory/'combined_peak_data.csv'
        write_combined(region_rows,combined)
        print(f'SAVED combined {region} peak data: {combined.resolve()}')
        ncols = min(args.columns,len(qois))
        nrows = (len(qois)+ncols-1)//ncols
        fig,axes = plt.subplots(nrows,ncols,squeeze=False,
                                figsize=(5.5*ncols,4*nrows),constrained_layout=True)
        for ax,qoi in zip(axes.flat,qois):
            panel = [r for r in plotted if r['qoi']==qoi and r['region']==region]
            for field in args.fields:
                subset = [r for r in panel if r['field']==field]
                if not subset: continue
                c = [mapper.to_rgba(r[args.color_by]) if numeric_color else colors[r['simulation']] for r in subset]
                ax.scatter([r[args.x] for r in subset],[r[args.y] for r in subset],
                           c=c,marker=marker_map[field],s=48,alpha=.8,edgecolors='black',linewidths=.35)
                if args.label_points:
                    for r in subset:
                        ax.annotate(r['simulation'],(r[args.x],r[args.y]),xytext=(3,3),textcoords='offset points',fontsize=6)
            if not panel: ax.text(.5,.5,'No matching peaks',ha='center',transform=ax.transAxes)
            if args.x in ('peak_distance','peak_offset'): ax.axvline(0,color='gray',lw=.7,ls=':')
            ax.set_title(f'{qoi} ({len(panel)} peaks)')
            ax.set_xlabel(LABELS[args.x]);ax.set_ylabel(LABELS[args.y]);ax.grid(alpha=.3)
            if args.y=='peak_absolute_correlation': ax.set_ylim(-.03,1.03)
            elif args.y=='peak_correlation': ax.set_ylim(-1.03,1.03)
            handles=[Line2D([],[],marker=marker_map[field],linestyle='None',color='gray',label=f'QoI versus {field}') for field in args.fields]
            ax.legend(handles=handles,fontsize=8,loc='best')
        for ax in axes.flat[len(qois):]: ax.set_visible(False)
        visible = list(axes.flat)[:len(qois)]
        fig.suptitle(region)
        if numeric_color:
            fig.colorbar(mapper,ax=visible,label=LABELS[args.color_by])
        else:
            region_sims = sorted({r['simulation'] for r in plotted if r['region']==region})
            if region_sims:
                handles=[Line2D([],[],marker='o',linestyle='None',color=colors[sim],label=sim) for sim in region_sims]
                fig.legend(handles=handles,loc='outside lower center',ncol=min(3,len(handles)),fontsize=8,title='Simulation')
        basename=f'peak_scatter_{safe_region}_{args.x}_vs_{args.y}'
        for ext in formats:
            path=directory/f'{basename}.{ext}'
            fig.savefig(path,dpi=args.dpi,bbox_inches='tight')
            print(f'SAVED {path.resolve()}')
        plt.close(fig)

if __name__=='__main__':main()
