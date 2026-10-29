#!/usr/bin/env python3
"""Conditional PDFs and exact spatial Pearson correlations; NumPy + h5py."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from datetime import datetime, timezone
import numpy as np
import h5py

VERSION = '1.0.0'
REGEX = re.compile(r'^thinfilm_closure_data_Re(?P<Re>\d+)_Ca(?P<Ca>\d+_\d+)_theta(?P<beta>\d+(?:_\d+)?)\.h5$')
LOG = logging.getLogger('closure_stats')


def simulation_parameters(path):
    match = REGEX.fullmatch(path.name)
    if not match:
        raise ValueError(f'Unrecognized case filename: {path.name}')
    return {key: float(value.replace('_', '.')) for key, value in match.groupdict().items()}


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for key in ('data_dir', 'output_dir', 'qois', 'offsets', 'regions', 'pdf'):
        if key not in cfg:
            raise ValueError(f'Missing config key: {key}')
    for key in ('data_dir', 'output_dir'):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    if not cfg['qois'] or len(set(cfg['qois'])) != len(cfg['qois']):
        raise ValueError('qois must be a nonempty list of unique dataset names')
    if any(not isinstance(q, str) or '/' in q or not q for q in cfg['qois']):
        raise ValueError('QoIs must be root-level dataset names')
    off = cfg['offsets']
    if isinstance(off, dict):
        lo, hi, step = off['min'], off['max'], off.get('step', 1)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (lo, hi, step)) or step <= 0 or hi < lo:
            raise ValueError('offset range requires integer min <= max and positive integer step')
        offsets = list(range(lo, hi + 1, step))
    else:
        if not isinstance(off, list) or not off or any(isinstance(v, bool) or not isinstance(v, int) for v in off):
            raise ValueError('offsets must be an integer list or min/max/step object')
        offsets = sorted(set(off))
    cfg['offsets'] = offsets
    if not cfg['regions'] or 'All' not in cfg['regions']:
        raise ValueError('regions must include All')
    for name, rule in cfg['regions'].items():
        if '/' in name or not name or set(rule) - {'h_gt','h_lt','abs_hx_le','require_finite_hx'}:
            raise ValueError(f'Invalid region name or rule: {name}')
        for key in ('h_gt','h_lt','abs_hx_le'):
            if key in rule and (not np.isfinite(rule[key]) or (key == 'abs_hx_le' and rule[key] < 0)):
                raise ValueError(f'Invalid region threshold: {name}/{key}')
        if rule.get('h_gt', -np.inf) >= rule.get('h_lt', np.inf):
            raise ValueError(f'Empty height interval in {name}')
    pdf = cfg['pdf']
    if not isinstance(pdf.get('bins', 100), int) or pdf.get('bins',100) < 1:
        raise ValueError('pdf.bins must be a positive integer')
    if not np.isfinite(pdf.get('n_std',3)) or pdf.get('n_std',3) <= 0:
        raise ValueError('pdf.n_std must be positive')
    for key in ('correlation_use_fluctuations','pdf_use_fluctuations'):
        cfg.setdefault(key, key.startswith('correlation'))
        if not isinstance(cfg[key], bool):
            raise ValueError(f'{key} must be true or false')
    cfg.setdefault('simulation', {'Re':40,'Ca':0.1,'beta':45})
    cfg.setdefault('watch', {})
    return cfg


def read_field(f, name, nx, nt):
    a = np.asarray(f[name][...], dtype=np.float64)
    if a.shape == (nx, nt):
        return a
    if a.shape == (nt, nx):
        return a.T.copy()
    raise ValueError(f'{name}: shape {a.shape}; expected {(nx,nt)} or {(nt,nx)}')


def fluctuations(a):
    finite = np.isfinite(a)
    count = finite.sum(axis=1, keepdims=True)
    mean = np.divide(np.where(finite,a,0).sum(axis=1, keepdims=True), count,
                     out=np.full((a.shape[0],1),np.nan), where=count>0)
    return np.where(finite, a-mean, np.nan), mean[:,0]


def centered_unit(a):
    if a.size < 2:
        return None
    ac = a-a.mean()
    scale = np.max(np.abs(ac))
    if scale == 0:
        return None
    ac = ac/scale
    return ac / np.sqrt(np.dot(ac,ac))


def pearson_prepared(a_unit, b):
    if a_unit is None:
        return np.nan
    b_unit = centered_unit(b)
    if b_unit is None:
        return np.nan
    return float(np.clip(np.dot(a_unit,b_unit),-1,1))


def pearson(a, b):
    return pearson_prepared(centered_unit(a),b)


def signature(path):
    stat = path.stat()
    return {'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns}


def config_hash(cfg):
    # Watching cadence and the single-case selector do not alter per-file statistics.
    calc = {k:v for k,v in cfg.items() if k not in ('watch','simulation','output_dir')}
    calc['program_version'] = VERSION
    return hashlib.sha256(json.dumps(calc, sort_keys=True).encode()).hexdigest()


def write_array(group, name, values):
    a = np.asarray(values)
    kwargs = {'compression':'gzip', 'shuffle':True} if a.ndim and a.size else {}
    return group.create_dataset(name, data=a, **kwargs)


def write_pdf(group, values, raw_edges, n_std, bins):
    n = len(values)
    group.attrs['n_points'] = n
    mean = float(values.mean()) if n else np.nan
    std = float(values.std(ddof=0)) if n else np.nan
    group.attrs.update(mean=mean, std=std, variance=std*std,
                       rms=float(np.sqrt(np.mean(values**2))) if n else np.nan)
    raw = group.create_group('raw')
    write_array(raw, 'bin_edges', raw_edges)
    counts = np.histogram(values, bins=raw_edges)[0]
    write_array(raw, 'counts', counts)
    write_array(raw, 'density', counts / (n*np.diff(raw_edges)) if n else np.full(bins,np.nan))
    raw.attrs['status'] = 'ok' if n else 'empty_region'
    zgroup = group.create_group('standardized')
    edges = np.linspace(-n_std, n_std, bins+1)
    write_array(zgroup, 'bin_edges', edges)
    if n >= 2 and np.isfinite(std) and std > 0:
        z = (values-mean)/std
        counts = np.histogram(z, bins=edges)[0]
        density = counts/(n*np.diff(edges))
        zgroup.attrs.update(status='ok', underflow=int(np.sum(z<edges[0])),
                            overflow=int(np.sum(z>edges[-1])), included_fraction=float(counts.sum()/n))
    else:
        counts, density = np.zeros(bins,dtype=np.int64), np.full(bins,np.nan)
        zgroup.attrs.update(status='empty_or_constant_region', underflow=0, overflow=0, included_fraction=np.nan)
    write_array(zgroup, 'counts', counts)
    write_array(zgroup, 'density', density)
    zgroup.attrs['normalization'] = 'counts / (all_region_points * bin_width); tails not renormalized'


def calculate(source, output, cfg, digest, source_sig):
    offsets = np.asarray(cfg['offsets'],dtype=int)
    K = int(np.max(np.abs(offsets)))
    with h5py.File(source, 'r') as f, h5py.File(output, 'w') as out:
        x = np.asarray(f['x'][...],dtype=float).reshape(-1)
        t = np.asarray(f['t'][...],dtype=float).reshape(-1)
        nx, nt = len(x),len(t)
        if nx < 2 or nt < 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
            raise ValueError('Invalid x or t')
        dx = float(x[1]-x[0])
        if dx <= 0 or not np.allclose(np.diff(x), dx) or 2*K >= nx:
            raise ValueError('Require a uniform increasing x grid and 2*max_offset < nx')
        h, q = (read_field(f,name,nx,nt) for name in ('h','q'))
        need_hx = any(rule.get('require_finite_hx',False) or 'abs_hx_le' in rule for rule in cfg['regions'].values())
        hx = read_field(f,'h_x',nx,nt) if need_hx else None
        hp, hm = fluctuations(h)
        qp, qm = fluctuations(q)
        H, Q = (hp,qp) if cfg['correlation_use_fluctuations'] else (h,q)
        start, stop = K,nx-K
        hc = h[start:stop]
        base = np.isfinite(hc)
        for n in offsets:
            base &= np.isfinite(H[start+n:stop+n]) & np.isfinite(Q[start+n:stop+n])
        masks = {}
        for region,rule in cfg['regions'].items():
            mask = base.copy()
            if 'h_gt' in rule: mask &= hc > rule['h_gt']
            if 'h_lt' in rule: mask &= hc < rule['h_lt']
            if rule.get('require_finite_hx',False) or 'abs_hx_le' in rule:
                mask &= np.isfinite(hx[start:stop])
            if 'abs_hx_le' in rule: mask &= np.abs(hx[start:stop]) <= rule['abs_hx_le']
            masks[region] = mask
        out.attrs.update(program_version=VERSION, config_hash=digest,
                         config_json=json.dumps(cfg,sort_keys=True), source_path=str(source.resolve()),
                         source_signature_json=json.dumps(source_sig), created_utc=datetime.now(timezone.utc).isoformat(),
                         dx=dx, nx=nx, nt=nt, center_start=start, center_stop_exclusive=stop,
                         parameters_json=json.dumps(simulation_parameters(source)), complete=False)
        out.attrs['method'] = 'Equal-weight x,t Pearson; temporal mean removed at each x when enabled; region-conditional recentering'
        write_array(out,'offsets',offsets)
        write_array(out,'distances',offsets*dx)
        write_array(out,'x',x)
        write_array(out,'t',t)
        means = out.create_group('time_means')
        write_array(means,'h',hm); write_array(means,'q',qm)
        results = out.create_group('qois')
        bins = cfg['pdf'].get('bins',100)
        for name in cfg['qois']:
            LOG.info('%s: QoI %s',source.name,name)
            y = read_field(f,name,nx,nt)
            yp, ym = fluctuations(y)
            yc = (yp if cfg['correlation_use_fluctuations'] else y)[start:stop]
            pc = (yp if cfg['pdf_use_fluctuations'] else y)[start:stop]
            # Means/validity computed using all stored times, matching the notebook.
            # Both transformations have the same finite mask.
            selected = {region:mask & np.isfinite(yc) & np.isfinite(pc) for region,mask in masks.items()}
            union = np.logical_or.reduce(list(selected.values()))
            edges = np.histogram_bin_edges(pc[union],bins=bins) if union.any() else np.linspace(-0.5,0.5,bins+1)
            group = results.create_group(name)
            write_array(group,'time_mean',ym)
            for region,mask in selected.items():
                rg = group.create_group(region)
                ix,it = np.nonzero(mask)
                yy = yc[ix,it]
                y_unit = centered_unit(yy)
                rh,rq = np.full(len(offsets),np.nan),np.full(len(offsets),np.nan)
                for i,n in enumerate(offsets):
                    rh[i] = pearson_prepared(y_unit,H[start+ix+n,it])
                    rq[i] = pearson_prepared(y_unit,Q[start+ix+n,it])
                cg = rg.create_group('pearson')
                cg.attrs['n_points'] = len(yy)
                write_array(cg,'h',rh); write_array(cg,'q',rq)
                for field,curve in (('h',rh),('q',rq)):
                    if np.any(np.isfinite(curve)):
                        index = int(np.nanargmax(np.abs(curve)))
                        cg.attrs[f'{field}_strongest_offset'] = int(offsets[index])
                        cg.attrs[f'{field}_strongest_distance'] = float(offsets[index]*dx)
                        cg.attrs[f'{field}_strongest_correlation'] = float(curve[index])
                write_pdf(rg.create_group('pdf'),pc[mask],edges,cfg['pdf'].get('n_std',3),bins)
                LOG.info('  %s: %s points',region,format(len(yy),','))
        out.attrs['complete'] = True


def analyze_one(source,cfg,force=False):
    digest = config_hash(cfg)
    directory = Path(cfg['output_dir']) / digest[:12]
    directory.mkdir(parents=True,exist_ok=True)
    destination = directory / (source.stem + '_statistics.h5')
    # Advisory process lock releases automatically if a process crashes.
    import fcntl
    with open(str(destination)+'.lock','a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info('BUSY %s',source.name)
            return 'busy'
        before = signature(source)
        if destination.exists() and not force:
            try:
                with h5py.File(destination,'r') as f:
                    if (f.attrs.get('complete',False) and f.attrs.get('config_hash')==digest
                        and json.loads(f.attrs['source_signature_json'])==before):
                        LOG.info('SKIP unchanged: %s',source.name)
                        return 'skipped'
            except (OSError,KeyError,ValueError):
                pass
        fd,temp = tempfile.mkstemp(prefix=destination.stem+'.',suffix='.partial.h5',dir=directory)
        os.close(fd)
        try:
            calculate(source,Path(temp),cfg,digest,before)
            if signature(source) != before:
                raise RuntimeError('Source changed while reading; output discarded; retry after writer finishes')
            os.replace(temp,destination)
        finally:
            with contextlib.suppress(FileNotFoundError): os.unlink(temp)
        LOG.info('SAVED %s',destination)
    return 'saved'


def select_files(cfg,mode):
    directory = Path(cfg['data_dir'])
    if not directory.is_dir(): raise FileNotFoundError(directory)
    files = sorted(p for p in directory.glob('thinfilm_closure_data_*.h5') if REGEX.fullmatch(p.name))
    if mode == 'single':
        selection = cfg['simulation']
        files = [p for p in files if all(np.isclose(simulation_parameters(p)[k],float(selection[k]),rtol=1e-9,atol=1e-12) for k in ('Re','Ca','beta'))]
        if len(files)!=1: raise ValueError(f'Expected one matching simulation, found {len(files)}: {selection}')
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=('single','scan'),default='single')
    parser.add_argument('--watch',action='store_true',help='Continuously rescan all cases until Ctrl-C/job termination')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    if args.watch and args.force: parser.error('--watch and --force cannot be combined')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    cfg=load_config(args.config)
    interval=float(cfg['watch'].get('interval_seconds',300))
    stable=float(cfg['watch'].get('stable_seconds',60))
    if interval<=0 or stable<0: raise ValueError('Invalid watch interval/stable_seconds')
    try:
        while True:
            failures=0
            paths=select_files(cfg,'scan' if args.watch else args.mode)
            LOG.info('Found %d matching files',len(paths))
            for path in paths:
                try:
                    if args.watch and time.time()-path.stat().st_mtime<stable:
                        LOG.info('DEFER recently modified: %s',path.name)
                        continue
                    analyze_one(path,cfg,args.force)
                except Exception:
                    failures+=1
                    LOG.exception('FAILED %s',path)
                    if args.mode=='single' and not args.watch: raise
            if not args.watch: return int(failures>0)
            LOG.info('Rescanning in %.0f seconds',interval)
            time.sleep(interval)
    except KeyboardInterrupt:
        LOG.info('Stopped')
        return 0

if __name__=='__main__':
    raise SystemExit(main())
