"""Prepare immutable, shared sample indices and memory-mapped fields."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def read_parameter(f, key):
    group = f['parameters']
    value = group[key][()] if key in group else group.attrs[key]
    a = np.asarray(value)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f'Invalid scalar parameter {key}')
    return float(a.reshape(-1)[0])


def prepare(config):
    cfg = json.loads(Path(config).read_text())
    root = Path(cfg['data_dir']).expanduser().resolve()
    dest = Path(cfg['cache_dir']).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f'{dest} already exists. Reuse it or choose a new cache_dir.')
    dev, held = cfg['development_cases'], cfg['heldout_cases']
    names = dev + held
    if len(names) != len(set(names)) or not dev:
        raise ValueError('Case lists must be nonempty (development), unique and disjoint.')
    tr, va, te = [cfg['fractions'][s] for s in ('train', 'val', 'test')]
    if min(tr, va, te) <= 0 or not np.isclose(tr + va + te, 1):
        raise ValueError('Positive train/val/test fractions must sum to one.')
    R = cfg['max_half_width']
    exclude = cfg['outlet_exclusion_fraction']
    gap = cfg['temporal_gap_snapshots']
    if R < 0 or not 0 <= exclude < 1 or not isinstance(gap, int) or gap < 0:
        raise ValueError('Invalid domain margin, outlet exclusion, or temporal gap.')
    if cfg['flat_height_tolerance'] < 0:
        raise ValueError('Flat height tolerance must be nonnegative.')
    for cap in cfg['sample_caps'].values():
        if cap is not None and (not isinstance(cap, int) or cap <= 0):
            raise ValueError('Sample caps must be positive integers or null.')
    targets = cfg['available_targets']
    if not targets or len(set(targets)) != len(targets) or set(targets) & {'h', 'q', 'curvature'}:
        raise ValueError('Invalid available_targets.')
    paths = {name: root / f'thinfilm_closure_data_{name}.h5' for name in names}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    dest.mkdir(parents=True)
    manifest = {'schema': 1, 'config': cfg, 'cases': [],
                'derived_inputs': {'curvature': {
                    'formula': 'h_xx / (1 + h_x**2)**1.5',
                    'source_fields': ['h_x', 'h_xx'],
                    'grid': 'original DNS grid',
                    'eligibility': 'finite throughout the common maximum-radius window'}}}
    for name in names:
        p = paths[name]
        out = dest / name
        out.mkdir()
        with h5py.File(p, 'r') as f:
            x, t = np.asarray(f['x'][:], float).ravel(), np.asarray(f['t'][:], float).ravel()
            if len(x) < 2 or len(t) < 3 or not np.isfinite(x).all() or not np.isfinite(t).all():
                raise ValueError(f'{name}: invalid coordinates')
            if not (np.diff(x) > 0).all() or not (np.diff(t) > 0).all():
                raise ValueError(f'{name}: coordinates must strictly increase')
            dx = np.diff(x)
            if not np.allclose(dx, dx.mean(), rtol=1e-5, atol=1e-10):
                raise ValueError(f'{name}: index-stride implementation requires a uniform x grid')
            params = [read_parameter(f, key) for key in ('Re', 'Ca', 'angle')]
            if min(params[:2]) <= 0:
                raise ValueError('Re and Ca must be positive')
            extra = {key: read_parameter(f, key) for key in ('Fr2', 'We', 'Lambda')
                     if key in f['parameters'] or key in f['parameters'].attrs}
            fields = {}
            for key in ['h', 'q'] + targets:
                arr = np.asarray(f[key][:], dtype=np.float32)
                if arr.shape != (len(x), len(t)):
                    raise ValueError(f'{name}/{key}: expected (nx, nt), got {arr.shape}')
                # Time-major cache makes same-snapshot windows contiguous on disk.
                np.save(out / f'{key}.npy', np.ascontiguousarray(arr.T))
                fields[key] = arr
            # Compute from original, unnormalized DNS derivatives in time blocks.
            for key in ('h_x', 'h_xx'):
                if key not in f or f[key].shape != (len(x), len(t)):
                    raise ValueError(f'{name}/{key}: required shape (nx, nt) for curvature')
            curvature = np.lib.format.open_memmap(out / 'curvature.npy', mode='w+',
                dtype=np.float32, shape=(len(t), len(x)))
            for start in range(0, len(t), 16):
                stop = min(start + 16, len(t))
                hx = np.asarray(f['h_x'][:, start:stop], dtype=np.float64).T
                hxx = np.asarray(f['h_xx'][:, start:stop], dtype=np.float64).T
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    block = (hxx / np.hypot(1., hx)**3).astype(np.float32)
                block[~(np.isfinite(hx) & np.isfinite(hxx))] = np.nan
                curvature[start:stop] = block
            curvature.flush()
            fields['curvature'] = curvature.T
        np.save(out / 'x.npy', x)
        np.save(out / 't.npy', t)
        end_x = x[0] + (1 - exclude) * (x[-1] - x[0])
        centers = np.flatnonzero((x >= x[0] + R) & (x <= end_x - R))
        if not len(centers):
            raise ValueError(f'{name}: no centers remain inside common margins')
        lo = np.searchsorted(x, x[centers] - R - 1e-10)
        hi = np.searchsorted(x, x[centers] + R + 1e-10, side='right')
        a, b = int(tr * len(t)), int((tr + va) * len(t))
        ranges = {'train': range(0, a), 'val': range(a + gap, b),
                  'test': range(b + gap, len(t))} if name in dev else {'heldout': range(len(t))}
        entry = {'name': name, 'role': 'development' if name in dev else 'heldout',
                 'parameters': params, 'extra_parameters': extra, 'dx': float(dx.mean()),
                 'source': str(p), 'source_size': p.stat().st_size,
                 'source_mtime_ns': p.stat().st_mtime_ns, 'shape': [len(x), len(t)],
                 'splits': {}}
        for split, times in ranges.items():
            rows = []
            for it in times:
                valid = np.ones(len(centers), bool)
                if cfg['flat_mask']:
                    hh = fields['h'][:, it]
                    disturbed = np.flatnonzero((x <= end_x) & np.isfinite(hh) &
                                  (np.abs(hh - cfg['flat_reference_height']) > cfg['flat_height_tolerance']))
                    if not len(disturbed):
                        continue
                    valid &= centers <= disturbed[-1]
                # Common eligibility for every stride/window and every available target.
                bad = ~(np.isfinite(fields['h'][:, it]) & np.isfinite(fields['q'][:, it]))
                bad |= fields['h'][:, it] <= 0
                bad |= ~np.isfinite(fields['curvature'][:, it])
                prefix = np.r_[0, np.cumsum(bad)]
                valid &= (prefix[hi] - prefix[lo]) == 0
                for key in targets:
                    valid &= np.isfinite(fields[key][centers, it])
                ix = centers[valid]
                rows.append(np.column_stack((np.full(len(ix), it, dtype=np.int32), ix)).astype(np.int32))
            indices = np.concatenate(rows) if rows else np.empty((0, 2), np.int32)
            eligible = len(indices)
            if eligible == 0:
                raise ValueError(f'{name}/{split}: no eligible samples; inspect mask/splits')
            cap = cfg['sample_caps'][split]
            seed = int(hashlib.sha256(f'{cfg["sampling_seed"]}/{name}/{split}'.encode()).hexdigest()[:16], 16)
            if cap is not None and eligible > cap:
                pick = np.random.default_rng(seed).choice(eligible, cap, replace=False)
                indices = indices[np.sort(pick)]
            index_path = out / f'{split}_indices.npy'
            np.save(index_path, indices)
            entry['splits'][split] = {'eligible': eligible, 'selected': len(indices),
                'time_indices': list(times), 'sha256': hashlib.sha256(index_path.read_bytes()).hexdigest()}
        manifest['cases'].append(entry)
        print(name, {s: v['selected'] for s, v in entry['splits'].items()}, flush=True)
        del fields
    manifest['id'] = digest(manifest)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Prepared {dest}; cache id {manifest["id"]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='data_config.json')
    prepare(p.parse_args().config)
