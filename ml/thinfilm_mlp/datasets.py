"""Lazy, vectorized window construction from shared immutable caches."""
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, cache, split, targets, half_window, stride, input_fields=None):
        self.root = Path(cache)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.input_fields = list(input_fields) if input_fields is not None else ['h', 'q']
        if self.input_fields not in (['h', 'q'], ['h', 'q', 'curvature']):
            raise ValueError("input_fields must be ['h','q'] or ['h','q','curvature']")
        self.channels = len(self.input_fields) + 1  # signed dx is last
        self.targets = list(targets)
        if not self.targets or len(set(targets)) != len(targets):
            raise ValueError('Targets must be nonempty and unique')
        if not set(targets) <= set(self.manifest['config']['available_targets']):
            raise ValueError('Requested targets are not in this cache')
        if half_window < 0 or stride < 1:
            raise ValueError('half_window >= 0 and stride >= 1 required')
        self.offsets = np.arange(-half_window, half_window + 1) * stride
        self.width = len(self.offsets)
        self.split = split
        self.entries = [e for e in self.manifest['cases'] if split in e['splits']]
        if not self.entries:
            raise ValueError(f'No {split} data')
        self.ends = np.cumsum([e['splits'][split]['selected'] for e in self.entries])
        self.starts = np.r_[0, self.ends[:-1]]
        self._maps = {}
        for entry in self.entries:
            for field in self.input_fields:
                if not (self.root / entry['name'] / f'{field}.npy').is_file():
                    raise FileNotFoundError(f'Missing cached input {field}; run augment_curvature.py first')
            x = np.load(self.root / entry['name'] / 'x.npy', mmap_mode='r')
            span = half_window * stride
            if span >= len(x) or (span and np.max(x[span:] - x[:-span]) >
                                  self.manifest['config']['max_half_width'] + 1e-10):
                raise ValueError(f'Window exceeds configured maximum in {entry["name"]}')

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = {}
        return state

    def maps(self, ci):
        if ci not in self._maps:
            folder = self.root / self.entries[ci]['name']
            self._maps[ci] = {k: np.load(folder / f'{k}.npy', mmap_mode='r')
                             for k in self.input_fields + ['x', self.split + '_indices'] + self.targets}
        return self._maps[ci]

    def __getitems__(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        cases = np.searchsorted(self.ends, ids, side='right')
        X = np.empty((len(ids), self.channels * self.width + 3), np.float32)
        Y = np.empty((len(ids), len(self.targets)), np.float32)
        for ci in np.unique(cases):
            where = np.flatnonzero(cases == ci)
            m = self.maps(int(ci))
            it, ix = m[self.split + '_indices'][ids[where] - self.starts[ci]].T
            nb = ix[:, None] + self.offsets
            fields = [m[k][it[:, None], nb] for k in self.input_fields]
            dx = m['x'][nb] - m['x'][ix, None]
            X[where, :self.channels * self.width] = np.stack(fields + [dx], axis=-1).reshape(len(where), -1)
            X[where, -3:] = self.entries[ci]['parameters']
            Y[where] = np.stack([m[k][it, ix] for k in self.targets], axis=1)
        return list(zip(X, Y, cases.astype(np.int64)))

    def __getitem__(self, i):
        return self.__getitems__([i])[0]
