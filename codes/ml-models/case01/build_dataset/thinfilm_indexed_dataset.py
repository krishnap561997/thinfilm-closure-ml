#!/usr/bin/env python3
"""Lazy PyTorch Dataset for the index produced by build_thinfilm_dataset.py."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ThinFilmIndexedDataset(Dataset):
    """Read selected samples lazily from multiple HDF5 simulation files.

    `feature_keys` and `target_keys` are HDF5 dataset paths. Every dataset is
    expected to be 2-D with the same time/x orientation recorded in metadata.
    Scalar parameters Ca, Re, and theta can optionally be appended as features.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        split: str,
        feature_keys: Sequence[str],
        target_keys: Sequence[str],
        include_parameters: bool = True,
    ) -> None:
        self.feature_keys = [self._normalize_key(k) for k in feature_keys]
        self.target_keys = [self._normalize_key(k) for k in target_keys]
        self.include_parameters = include_parameters

        with Path(metadata_csv).open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not rows:
            raise ValueError(f"No simulations found for split={split!r}")

        self.simulations = []
        cumulative = [0]
        for row in rows:
            index_data = np.load(row["index_file"], mmap_mode="r")
            n = int(index_data["time_index"].shape[0])
            index_data.close()
            self.simulations.append(row)
            cumulative.append(cumulative[-1] + n)
        self.cumulative = np.asarray(cumulative, dtype=np.int64)

        # HDF5 handles are opened lazily and separately in each DataLoader worker.
        self._h5_handles: dict[str, h5py.File] = {}
        self._index_cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key if key.startswith("/") else "/" + key

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def _get_indices(self, path: str) -> dict[str, np.ndarray]:
        if path not in self._index_cache:
            with np.load(path) as data:
                self._index_cache[path] = {
                    "time_index": data["time_index"].copy(),
                    "x_index": data["x_index"].copy(),
                    "is_high_hx": data["is_high_hx"].copy(),
                }
        return self._index_cache[path]

    @staticmethod
    def _read_scalar(ds: h5py.Dataset, t: int, x: int, time_axis: int) -> float:
        return float(ds[t, x] if time_axis == 0 else ds[x, t])

    def __getitem__(self, global_index: int):
        sim_pos = int(np.searchsorted(self.cumulative, global_index, side="right") - 1)
        local_index = int(global_index - self.cumulative[sim_pos])
        row = self.simulations[sim_pos]

        indices = self._get_indices(row["index_file"])
        t_idx = int(indices["time_index"][local_index])
        x_idx = int(indices["x_index"][local_index])
        time_axis = int(row["time_axis"])

        h5 = self._get_h5(row["h5_path"])
        features = [self._read_scalar(h5[key], t_idx, x_idx, time_axis)
                    for key in self.feature_keys]
        if self.include_parameters:
            features.extend([float(row["ca"]), float(row["re"]), float(row["theta"])])

        targets = [self._read_scalar(h5[key], t_idx, x_idx, time_axis)
                   for key in self.target_keys]

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "is_high_hx": torch.tensor(bool(indices["is_high_hx"][local_index])),
            "simulation_id": row["simulation_id"],
            "time_index": t_idx,
            "x_index": x_idx,
        }

    def close(self) -> None:
        for h5 in self._h5_handles.values():
            h5.close()
        self._h5_handles.clear()
