"""Prepare full-grid fields and shared candidate-center indices for window MLPs.

Fields are stored once per case, rather than duplicated into sample windows.
The dataset loader later gathers neighbors around each selected (time, space)
index. ``max_half_width`` defines a common eligibility radius in x-coordinate
units, keeping the selected centers identical across window/stride experiments.
DNS derivatives are stored only at selected centers, aligned with split indices.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(obj):
    """Return a reproducible identity for JSON-serializable metadata."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def read_parameter(f, key):
    """Read a finite scalar from a parameter dataset or group attribute."""
    group = f['parameters']
    value = group[key][()] if key in group else group.attrs[key]
    a = np.asarray(value)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f'Invalid scalar parameter {key}')
    return float(a.reshape(-1)[0])


def prepare(config):
    """Validate sources, cache full fields, and freeze split/sample membership."""
    # Read the sampling policy and protect existing caches from overwriting.
    cfg = json.loads(Path(config).read_text())
    root = Path(cfg['data_dir']).expanduser().resolve()
    dest = Path(cfg['cache_dir']).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f'{dest} already exists. Reuse it or choose a new cache_dir.')

    # Development cases are split in time; held-out cases remain separate.
    dev, held = cfg['development_cases'], cfg['heldout_cases']
    names = dev + held
    if len(names) != len(set(names)) or not dev:
        raise ValueError('Case lists must be nonempty (development), unique and disjoint.')

    tr, va, te = [cfg['fractions'][s] for s in ('train', 'val', 'test')]
    if min(tr, va, te) <= 0 or not np.isclose(tr + va + te, 1):
        raise ValueError('Positive train/val/test fractions must sum to one.')

    # R is a spatial distance, not a number of grid points.
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

    local_fields = cfg['local_fields']
    if (not isinstance(local_fields, list)
            or len(local_fields) != 4
            or set(local_fields) != {'h_x', 'h_xx', 'q_x', 'q_xx'}):
        raise ValueError('local_fields must contain h_x, h_xx, q_x, q_xx exactly once.')

    targets = cfg['available_targets']
    if not targets or len(set(targets)) != len(targets) or set(targets) & ({'h', 'q', 'curvature'} | set(local_fields)):
        raise ValueError('Invalid available_targets.')

    # Check that every source exists before creating the output directory.
    paths = {name: root / f'thinfilm_closure_data_{name}.h5' for name in names}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    dest.mkdir(parents=True)
    manifest = {
        'schema': 2,
        'config': cfg,
        'cases': [],
        'local_fields': {
            'names': local_fields,
            'source': 'original HDF5 datasets; no derivative recomputation',
            'layout': '{split}_{field}.npy, shape (selected,), float32',
            'alignment': 'row-for-row with {split}_indices.npy',
            'eligibility': 'finite at the center only',
        },
    }

    for name in names:
        p = paths[name]
        out = dest / name
        out.mkdir()

        with h5py.File(p, 'r') as f:
            # Source arrays use (space, time); coordinates must increase.
            x, t = (
                np.asarray(f['x'][:], float).ravel(),
                np.asarray(f['t'][:], float).ravel(),
            )
            if len(x) < 2 or len(t) < 3 or not np.isfinite(x).all() or not np.isfinite(t).all():
                raise ValueError(f'{name}: invalid coordinates')
            if not (np.diff(x) > 0).all() or not (np.diff(t) > 0).all():
                raise ValueError(f'{name}: coordinates must strictly increase')

            # A fixed index stride must correspond to a fixed spatial spacing.
            dx = np.diff(x)
            if not np.allclose(dx, dx.mean(), rtol=1e-5, atol=1e-10):
                raise ValueError(f'{name}: index-stride implementation requires a uniform x grid')

            # These parameters are metadata here; the loader appends Re/Ca/angle
            # to every model input. Values are copied without unit conversion.
            params = [read_parameter(f, key) for key in ('Re', 'Ca', 'angle')]
            if min(params[:2]) <= 0:
                raise ValueError('Re and Ca must be positive')
            extra = {key: read_parameter(f, key) for key in ('Fr2', 'We', 'Lambda')
                     if key in f['parameters'] or key in f['parameters'].attrs}

            # Save complete fields, including locations not selected as centers.
            # Retain source orientation in `fields` for the checks below.
            fields = {}
            for key in ['h', 'q'] + targets:
                arr = np.asarray(f[key][:], dtype=np.float32)
                if arr.shape != (len(x), len(t)):
                    raise ValueError(f'{name}/{key}: expected (nx, nt), got {arr.shape}')
                # Time-major cache makes same-snapshot windows contiguous on disk.
                np.save(out / f'{key}.npy', np.ascontiguousarray(arr.T))
                fields[key] = arr

            # Read original DNS derivatives without computing or saving windows.
            # These arrays are used only at center indices below; only selected
            # center values are written to disk after sampling.
            for key in local_fields:
                if key not in f or f[key].shape != (len(x), len(t)):
                    raise ValueError(f'{name}/{key}: required shape (nx, nt)')
                fields[key] = np.asarray(f[key][:], dtype=np.float32)

        np.save(out / 'x.npy', x)
        np.save(out / 't.npy', t)

        # Remove the outlet region, then leave room for the full radius R on
        # both sides of every center. The underlying field files stay complete.
        end_x = x[0] + (1 - exclude) * (x[-1] - x[0])
        centers = np.flatnonzero((x >= x[0] + R) & (x <= end_x - R))
        if not len(centers):
            raise ValueError(f'{name}: no centers remain inside common margins')

        # [lo, hi) contains every original-grid point within +/- R of a center.
        # These bounds validate neighborhoods; they do not create stored windows.
        lo = np.searchsorted(x, x[centers] - R - 1e-10)
        hi = np.searchsorted(x, x[centers] + R + 1e-10, side='right')

        # Chronological splits, with optional omitted snapshots at boundaries.
        a, b = int(tr * len(t)), int((tr + va) * len(t))
        ranges = (
            {
                'train': range(0, a),
                'val': range(a + gap, b),
                'test': range(b + gap, len(t)),
            }
            if name in dev else {'heldout': range(len(t))}
        )

        # Record source provenance and the ORIGINAL (nx, nt) shape. Cached field
        # arrays are transposed to (nt, nx).
        entry = {
            'name': name,
            'role': 'development' if name in dev else 'heldout',
            'parameters': params,
            'extra_parameters': extra,
            'dx': float(dx.mean()),
            'source': str(p),
            'source_size': p.stat().st_size,
            'source_mtime_ns': p.stat().st_mtime_ns,
            'shape': [len(x), len(t)],
            'splits': {},
        }

        for split, times in ranges.items():
            rows = []

            for it in times:
                valid = np.ones(len(centers), bool)

                # Keep centers up to the last disturbed height before the outlet.
                # This permits flat points behind that front and flat neighbors
                # ahead of it; it is not a pointwise height-disturbance mask.
                if cfg['flat_mask']:
                    hh = fields['h'][:, it]
                    disturbed = np.flatnonzero(
                        (x <= end_x)
                        & np.isfinite(hh)
                        & (np.abs(hh - cfg['flat_reference_height'])
                           > cfg['flat_height_tolerance'])
                    )
                    if not len(disturbed):
                        continue
                    valid &= centers <= disturbed[-1]

                # Common eligibility for every stride/window and every available target.
                # Reject a center if ANY point in its full-radius neighborhood
                # has invalid h/q or nonpositive h. Prefix sums count
                # bad points in each [lo, hi) interval without explicit windows.
                bad = ~(np.isfinite(fields['h'][:, it]) & np.isfinite(fields['q'][:, it]))
                bad |= fields['h'][:, it] <= 0
                prefix = np.r_[0, np.cumsum(bad)]
                valid &= (prefix[hi] - prefix[lo]) == 0

                # Targets need to be finite only at the center. Check all cached
                # targets so later target-group experiments share sample indices.
                # Derivatives are checked only at the center, never neighbors.
                for key in targets + local_fields:
                    valid &= np.isfinite(fields[key][centers, it])

                ix = centers[valid]
                rows.append(
                    np.column_stack((np.full(len(ix), it, dtype=np.int32), ix))
                    .astype(np.int32)
                )

            # Each row is (time_index, space_index), never a flattened window.
            indices = np.concatenate(rows) if rows else np.empty((0, 2), np.int32)
            eligible = len(indices)
            if eligible == 0:
                raise ValueError(f'{name}/{split}: no eligible samples; inspect mask/splits')

            # Caps apply independently to each case/split. The stable seed makes
            # selection reproducible; sorting preserves time-then-space order.
            cap = cfg['sample_caps'][split]
            seed = int(hashlib.sha256(f'{cfg["sampling_seed"]}/{name}/{split}'.encode()).hexdigest()[:16], 16)
            if cap is not None and eligible > cap:
                pick = np.random.default_rng(seed).choice(eligible, cap, replace=False)
                indices = indices[np.sort(pick)]

            index_path = out / f'{split}_indices.npy'
            np.save(index_path, indices)

            # Store one derivative value per selected local point. No full-grid
            # derivative files or neighbor derivative arrays are emitted.
            for key in local_fields:
                values = fields[key][indices[:, 1], indices[:, 0]]
                np.save(out / f'{split}_{key}.npy', values)
            entry['splits'][split] = {
                'eligible': eligible,
                'selected': len(indices),
                'time_indices': list(times),
                'sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
            }

        manifest['cases'].append(entry)
        print(name, {s: v['selected'] for s, v in entry['splits'].items()}, flush=True)
        del fields

    # The manifest identifies the cache used by training and evaluation.
    manifest['id'] = digest(manifest)
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Prepared {dest}; cache id {manifest["id"]}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='data_config.json')
    prepare(p.parse_args().config)
