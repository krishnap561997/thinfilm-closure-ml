"""Create a new curvature cache, preserving every existing selected sample exactly."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import h5py
import numpy as np


def augment(source_cache, output_cache, data_dir=None):
    src = Path(source_cache).expanduser().resolve()
    dst = Path(output_cache).expanduser().resolve()
    if src == dst or src in dst.parents:
        raise ValueError('Output must be separate from, and not inside, the source cache')
    if dst.exists():
        raise FileExistsError('Choose a new output cache directory')
    manifest = json.loads((src/'manifest.json').read_text())
    if 'curvature' in manifest.get('derived_inputs', {}):
        raise ValueError('Source already has curvature; reuse it')
    # Copy ordinary files, never modify or hard-link the original cache.
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('manifest.json'))
    radius = manifest['config']['max_half_width']
    provenance = {}
    for entry in manifest['cases']:
        name = entry['name']
        raw = (Path(data_dir).expanduser()/f'thinfilm_closure_data_{name}.h5') if data_dir else Path(entry['source'])
        x = np.load(src/name/'x.npy')
        t = np.load(src/name/'t.npy')
        cached_h = np.load(src/name/'h.npy', mmap_mode='r')
        cached_q = np.load(src/name/'q.npy', mmap_mode='r')
        indices = {split: np.load(src/name/f'{split}_indices.npy', mmap_mode='r') for split in entry['splits']}
        nt, nx = len(t), len(x)
        curvature = np.lib.format.open_memmap(dst/name/'curvature.npy', mode='w+', dtype='float32', shape=(nt,nx))
        with h5py.File(raw, 'r') as f:
            for key in ['h','q','h_x','h_xx']:
                if f[key].shape != (nx,nt):
                    raise ValueError(f'{raw}/{key}: expected {(nx,nt)}')
            if not np.array_equal(f['x'][:].ravel(), x) or not np.array_equal(f['t'][:].ravel(), t):
                raise ValueError(f'{name}: raw coordinates differ from source cache')
            # Work in small time blocks; derivatives use their original full-grid values.
            for start in range(0,nt,16):
                stop = min(start+16,nt)
                for key, old in [('h',cached_h),('q',cached_q)]:
                    values = np.asarray(f[key][:,start:stop],np.float32).T
                    if not np.array_equal(values,old[start:stop],equal_nan=True):
                        raise ValueError(f'{name}: raw {key} differs from existing cache')
                hx = np.asarray(f['h_x'][:,start:stop],float).T
                hxx = np.asarray(f['h_xx'][:,start:stop],float).T
                with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                    block = (hxx / np.hypot(1.,hx)**3).astype(np.float32)
                block[~(np.isfinite(hx)&np.isfinite(hxx))] = np.nan
                curvature[start:stop] = block
                for it in range(start,stop):
                    bad = ~np.isfinite(block[it-start])
                    if not bad.any():
                        continue
                    prefix = np.r_[0,np.cumsum(bad)]
                    for split, ids in indices.items():
                        # Original indices are sorted by time then space.
                        left,right = np.searchsorted(ids[:,0],[it,it+1])
                        centers = ids[left:right,1]
                        lo = np.searchsorted(x,x[centers]-radius-1e-10)
                        hi = np.searchsorted(x,x[centers]+radius+1e-10,side='right')
                        if np.any(prefix[hi]-prefix[lo]):
                            raise ValueError(f'{name}/{split}, t index {it}: nonfinite curvature in a selected center neighborhood. No samples were dropped. Inspect derivatives; output cache is incomplete.')
        curvature.flush()
        del curvature
        for split, info in entry['splits'].items():
            p = dst/name/f'{split}_indices.npy'
            if hashlib.sha256(p.read_bytes()).hexdigest() != info['sha256']:
                raise ValueError(f'{name}/{split}: index integrity check failed')
        provenance[name] = str(raw.resolve())
        print(f'{name}: curvature saved; all split indices unchanged',flush=True)
    result = copy.deepcopy(manifest)
    result.pop('id',None)
    result['parent_cache_id'] = manifest['id']
    result['config']['cache_dir'] = str(dst)
    result['derived_inputs'] = {'curvature': {'formula':'h_xx / (1 + h_x**2)**1.5',
        'derivatives':'stored h_x and h_xx on original grid; no recomputation, clipping or smoothing',
        'source_files':provenance}}
    result['id'] = hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    (dst/'manifest.json').write_text(json.dumps(result,indent=2))
    print(f'Prepared {dst}; cache id {result["id"]}')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-cache',required=True)
    p.add_argument('--output-cache',required=True)
    p.add_argument('--data-dir',help='Override original HDF5 directory if the files moved')
    a=p.parse_args()
    augment(a.source_cache,a.output_cache,a.data_dir)
