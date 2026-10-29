"""Construct spatial-window or local samples from immutable NumPy caches.

Fields are memory-mapped on demand. Each sample contains a flattened spatial
window, three case parameters, and targets evaluated only at the center point.
LocalDataset instead gathers center fields and split-local DNS derivatives,
without neighbors or signed-distance inputs.
"""

import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Expose one cached split as a PyTorch dataset across multiple cases."""

    def __init__(self, cache, split, targets, half_window, stride, input_fields=None):
        """Validate fields/windows and build a global index over selected centers.

        ``half_window`` counts neighbors on each side; ``stride`` is their
        separation in grid indices. The manifest fixes sample membership.
        """
        self.root = Path(cache)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())

        # The final channel at each window position is signed distance from
        # the center. Input fields retain their specified order.
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
        # Example: half_window=2, stride=3 gives [-6, -3, 0, 3, 6].
        self.offsets = np.arange(-half_window, half_window + 1) * stride
        self.width = len(self.offsets)

        self.split = split
        self.entries = [e for e in self.manifest['cases'] if split in e['splits']]
        if not self.entries:
            raise ValueError(f'No {split} data')

        # Treat all cases as one contiguous dataset. Case ci occupies global
        # indices [starts[ci], ends[ci]); these bounds locate case-local rows.
        self.ends = np.cumsum([e['splits'][split]['selected'] for e in self.entries])
        self.starts = np.r_[0, self.ends[:-1]]

        # Open field maps lazily, rather than loading all cases into memory.
        self._maps = {}

        for entry in self.entries:
            for field in self.input_fields:
                if not (self.root / entry['name'] / f'{field}.npy').is_file():
                    raise FileNotFoundError(f'Missing cached input {field}; run augment_curvature.py first')

            # Check the requested physical radius against the common margin
            # used during preparation. A small tolerance allows roundoff.
            x = np.load(self.root / entry['name'] / 'x.npy', mmap_mode='r')
            span = half_window * stride
            if span >= len(x) or (
                span and np.max(x[span:] - x[:-span])
                > self.manifest['config']['max_half_width'] + 1e-10
            ):
                raise ValueError(f'Window exceeds configured maximum in {entry["name"]}')

    def __len__(self):
        """Return the total number of selected centers across this split."""
        return int(self.ends[-1])

    def __getstate__(self):
        """Clear open maps when serializing for DataLoader worker processes.

        Each worker opens its own read-only maps when first accessing a case.
        """
        state = self.__dict__.copy()
        state['_maps'] = {}
        return state

    def maps(self, ci):
        """Return lazily opened, read-only array maps for case index ``ci``."""
        if ci not in self._maps:
            folder = self.root / self.entries[ci]['name']
            self._maps[ci] = {
                k: np.load(folder / f'{k}.npy', mmap_mode='r')
                for k in self.input_fields + ['x', self.split + '_indices'] + self.targets
            }
        return self._maps[ci]

    def __getitems__(self, ids):
        """Gather a batch of global sample indices while preserving their order.

        Return a list of (input, target, case_index) tuples for PyTorch's default
        collation. Grouping reads by case avoids opening maps per sample.
        """
        ids = np.asarray(ids, dtype=np.int64)
        # side='right' assigns an index at a case boundary to the next case.
        cases = np.searchsorted(self.ends, ids, side='right')

        # Each position contributes input fields plus signed dx; Re, Ca, angle
        # occupy the final three columns. Targets have no neighbor dimension.
        X = np.empty((len(ids), self.channels * self.width + 3), np.float32)
        Y = np.empty((len(ids), len(self.targets)), np.float32)

        for ci in np.unique(cases):
            # `where` identifies this case's positions in the requested batch.
            where = np.flatnonzero(cases == ci)
            m = self.maps(int(ci))

            # Convert global sample IDs to case-local rows, then recover the
            # original DNS time and space indices stored during preparation.
            it, ix = m[self.split + '_indices'][ids[where] - self.starts[ci]].T

            # Broadcasting constructs (batch_for_case, window_width) indices.
            # Every neighbor uses the same snapshot as its center.
            nb = ix[:, None] + self.offsets
            fields = [m[k][it[:, None], nb] for k in self.input_fields]
            dx = m['x'][nb] - m['x'][ix, None]

            # Flatten position-first: [h_left, q_left, dx_left, ..., h_center,
            # q_center, 0, ...] for the default h/q inputs.
            X[where, :self.channels * self.width] = np.stack(
                fields + [dx], axis=-1
            ).reshape(len(where), -1)
            X[where, -3:] = self.entries[ci]['parameters']

            # Targets are always evaluated at the local center (it, ix).
            Y[where] = np.stack([m[k][it, ix] for k in self.targets], axis=1)

        # Case IDs allow training and validation to weight cases equally.
        return list(zip(X, Y, cases.astype(np.int64)))

    def __getitem__(self, i):
        """Fetch one sample through the same implementation used for batches."""
        return self.__getitems__([i])[0]


class LocalDataset(WindowDataset):
    """Load configurable center fields and derivatives for input ablation.

    Reuses the window loader's length, worker serialization, and single-item
    access methods. No window configuration or distance channel is required.
    """

    def __init__(self, cache, split, targets, input_fields=None):
        """Validate selected fields and construct split-wide sample indexing.

        Inputs follow ``input_fields`` order, then Re, Ca, angle. Derivatives
        come from aligned split arrays; h/q and targets use DNS center indices.
        """
        self.root = Path(cache)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.split = split
        self.input_fields = (
            list(input_fields) if input_fields is not None
            else ['h', 'q', 'h_x', 'h_xx', 'q_x', 'q_xx']
        )
        allowed = {'h', 'q', 'h_x', 'h_xx', 'q_x', 'q_xx'}
        if (not self.input_fields
                or len(set(self.input_fields)) != len(self.input_fields)
                or not set(self.input_fields) <= allowed):
            raise ValueError('Input fields must be a nonempty unique subset of h, q, h_x, h_xx, q_x, q_xx')

        self.targets = list(targets)
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError('Targets must be nonempty and unique')
        if not set(self.targets) <= set(self.manifest['config']['available_targets']):
            raise ValueError('Requested targets are not in this cache')

        self.entries = [e for e in self.manifest['cases'] if split in e['splits']]
        if not self.entries:
            raise ValueError(f'No {split} data')
        counts = [e['splits'][split]['selected'] for e in self.entries]
        if any(n <= 0 for n in counts):
            raise ValueError('Each case must have selected samples')
        self.ends = np.cumsum(counts)
        self.starts = np.r_[0, self.ends[:-1]]
        self.input_order = self.input_fields + ['Re', 'Ca', 'angle']
        self.input_dim = len(self.input_order)
        self._maps = {}

        # Check file headers now, without retaining maps or reading full arrays.
        # Derivative lengths must match the frozen sample-index rows exactly.
        for ci, entry in enumerate(self.entries):
            folder = self.root / entry['name']
            n = counts[ci]
            indices = np.load(folder / f'{split}_indices.npy', mmap_mode='r')
            if indices.shape != (n, 2):
                raise ValueError(f'{entry["name"]}: invalid {split} index shape')
            nx, nt = entry['shape']
            for field in dict.fromkeys(self.input_fields + self.targets):
                arr = np.load(self._field_path(folder, field), mmap_mode='r')
                expected = (n,) if field in allowed - {'h', 'q'} else (nt, nx)
                if arr.shape != expected:
                    raise ValueError(f'{entry["name"]}/{field}: expected {expected}, got {arr.shape}')

    def _field_path(self, folder, field):
        """Resolve full-grid fields versus split-aligned derivative files."""
        if field in {'h_x', 'h_xx', 'q_x', 'q_xx'}:
            return folder / f'{self.split}_{field}.npy'
        return folder / f'{field}.npy'

    def maps(self, ci):
        """Open a case's local input/target arrays and coordinates read-only."""
        if ci not in self._maps:
            folder = self.root / self.entries[ci]['name']
            self._maps[ci] = {
                field: np.load(self._field_path(folder, field), mmap_mode='r')
                for field in self.input_fields + self.targets
            }
            # Coordinates are retained for prediction export compatibility.
            for field in ['x', self.split + '_indices']:
                self._maps[ci][field] = np.load(folder / f'{field}.npy', mmap_mode='r')
        return self._maps[ci]

    def __getitems__(self, ids):
        """Gather local samples, preserving requested order and repeated IDs."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError('Sample IDs must be a one-dimensional array in dataset bounds')
        cases = np.searchsorted(self.ends, ids, side='right')
        X = np.empty((len(ids), self.input_dim), dtype=np.float32)
        Y = np.empty((len(ids), len(self.targets)), dtype=np.float32)

        for ci in np.unique(cases):
            where = np.flatnonzero(cases == ci)
            rows = ids[where] - self.starts[ci]
            m = self.maps(int(ci))
            it, ix = m[self.split + '_indices'][rows].T

            # Full-grid h/q use (time, space); derivatives use selected rows.
            for column, field in enumerate(self.input_fields):
                X[where, column] = (
                    m[field][it, ix] if field in {'h', 'q'} else m[field][rows]
                )
            X[where, -3:] = self.entries[ci]['parameters']
            Y[where] = np.stack([m[field][it, ix] for field in self.targets], axis=1)

        return list(zip(X, Y, cases.astype(np.int64)))
