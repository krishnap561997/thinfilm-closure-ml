#!/usr/bin/env python3
"""Build simulation metadata and sampled point indices for thin-film ML data.

The script does not duplicate the raw HDF5 fields. It creates:
  1. simulations_metadata.csv   -- one row per simulation
  2. splits.json                -- reproducible train/validation/test membership
  3. indices/<simulation_id>.npz -- sampled (time_index, x_index) pairs

Sampling combines:
  * deterministic coverage: every `x_stride` spatial point
  * targeted coverage: random points among the largest |h_x| values

Example
-------
python build_thinfilm_dataset.py \
  --root /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/ROM_Ca_Re_sweep \
  --output ./dataset_index \
  --hx-key hx \
  --x-stride 4 \
  --time-stride 5 \
  --high-hx-percentile 95 \
  --high-hx-extra-per-time 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

FOLDER_RE = re.compile(
    r"^Ca_(?P<ca>[0-9.]+)_Re_(?P<re>[0-9.]+)_theta_(?P<theta>[0-9.]+)$"
)


@dataclass(frozen=True)
class Simulation:
    simulation_id: str
    ca: float
    re: float
    theta: float
    folder: str
    h5_path: str
    n_time: int
    n_x: int
    time_axis: int
    x_axis: int
    split: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--filename", default="thinfilm_training_data.h5")
    p.add_argument("--hx-key", required=True,
                   help="HDF5 dataset path for h_x, e.g. hx or /fields/hx")
    p.add_argument("--time-key", default=None,
                   help="Optional HDF5 dataset path for time values")
    p.add_argument("--x-key", default=None,
                   help="Optional HDF5 dataset path for x values")
    p.add_argument("--x-stride", type=int, default=4)
    p.add_argument("--time-stride", type=int, default=5,
                   help="Use 1 to retain every saved time; 5 is a practical starting point")
    p.add_argument("--high-hx-percentile", type=float, default=95.0)
    p.add_argument("--high-hx-extra-per-time", type=int, default=200,
                   help="Maximum extra high-|h_x| points per retained time")
    p.add_argument("--train-count", type=int, default=20)
    p.add_argument("--val-count", type=int, default=5)
    p.add_argument("--test-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--split-search-trials", type=int, default=50000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_h5_key(key: str) -> str:
    return key if key.startswith("/") else "/" + key


def infer_axes(shape: tuple[int, ...], n_time_hint: int | None, n_x_hint: int | None) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Expected h_x to be 2-D, received shape {shape}")

    if n_time_hint is not None and n_x_hint is not None:
        if shape == (n_time_hint, n_x_hint):
            return 0, 1
        if shape == (n_x_hint, n_time_hint):
            return 1, 0

    # For the user's data, n_x (~8000) is larger than n_time (~1500).
    x_axis = int(np.argmax(shape))
    time_axis = 1 - x_axis
    return time_axis, x_axis


def discover_simulations(args: argparse.Namespace) -> list[Simulation]:
    simulations: list[Simulation] = []
    hx_key = normalize_h5_key(args.hx_key)
    time_key = normalize_h5_key(args.time_key) if args.time_key else None
    x_key = normalize_h5_key(args.x_key) if args.x_key else None

    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir():
            continue
        match = FOLDER_RE.match(folder.name)
        if not match:
            continue

        h5_path = folder / args.filename
        if not h5_path.exists():
            print(f"WARNING: missing {h5_path}; skipping")
            continue

        with h5py.File(h5_path, "r") as h5:
            if hx_key not in h5:
                available: list[str] = []
                h5.visititems(lambda name, obj: available.append('/' + name)
                              if isinstance(obj, h5py.Dataset) else None)
                raise KeyError(
                    f"Dataset {hx_key!r} not found in {h5_path}.\n"
                    f"Available datasets include:\n  " + "\n  ".join(available[:100])
                )

            hx_shape = tuple(h5[hx_key].shape)
            n_time_hint = int(h5[time_key].shape[0]) if time_key and time_key in h5 else None
            n_x_hint = int(h5[x_key].shape[0]) if x_key and x_key in h5 else None
            time_axis, x_axis = infer_axes(hx_shape, n_time_hint, n_x_hint)
            n_time = int(hx_shape[time_axis])
            n_x = int(hx_shape[x_axis])

        ca = float(match.group("ca"))
        reynolds = float(match.group("re"))
        theta = float(match.group("theta"))
        sim_id = f"Ca_{ca:.4g}_Re_{reynolds:.4g}_theta_{theta:.4g}"

        simulations.append(Simulation(
            simulation_id=sim_id,
            ca=ca,
            re=reynolds,
            theta=theta,
            folder=folder.name,
            h5_path=str(h5_path.resolve()),
            n_time=n_time,
            n_x=n_x,
            time_axis=time_axis,
            x_axis=x_axis,
        ))

    if not simulations:
        raise RuntimeError(f"No simulation folders found under {args.root}")
    return simulations


def level_balance_score(rows: list[Simulation], assignments: np.ndarray) -> float:
    """Score lower when each split contains balanced parameter levels."""
    split_codes = {0: "train", 1: "validation", 2: "test"}
    score = 0.0

    for attr in ("ca", "re", "theta"):
        levels = sorted({getattr(r, attr) for r in rows})
        global_counts = np.array([sum(getattr(r, attr) == lv for r in rows) for lv in levels], dtype=float)
        global_frac = global_counts / global_counts.sum()

        for code in split_codes:
            idx = np.flatnonzero(assignments == code)
            if len(idx) == 0:
                return math.inf
            counts = np.array([
                sum(getattr(rows[i], attr) == lv for i in idx) for lv in levels
            ], dtype=float)
            frac = counts / counts.sum()
            score += float(np.sum((frac - global_frac) ** 2))
            # Strong penalty if a parameter level is absent from validation/test.
            if code in (1, 2):
                score += 2.0 * float(np.count_nonzero(counts == 0))
    return score


def choose_split(rows: list[Simulation], args: argparse.Namespace) -> list[str]:
    n = len(rows)
    if args.train_count + args.val_count + args.test_count != n:
        raise ValueError(
            f"Split counts sum to {args.train_count + args.val_count + args.test_count}, "
            f"but {n} simulations were found."
        )

    rng = np.random.default_rng(args.seed)
    base = np.array(
        [0] * args.train_count + [1] * args.val_count + [2] * args.test_count,
        dtype=np.int8,
    )
    best = None
    best_score = math.inf

    for _ in range(args.split_search_trials):
        candidate = rng.permutation(base)
        score = level_balance_score(rows, candidate)
        if score < best_score:
            best = candidate.copy()
            best_score = score
            if best_score == 0.0:
                break

    assert best is not None
    names = np.array(["train", "validation", "test"])
    print(f"Selected simulation-level split; balance score = {best_score:.6g}")
    return names[best].tolist()


def read_hx_time_slice(dataset: h5py.Dataset, t_index: int, time_axis: int) -> np.ndarray:
    if time_axis == 0:
        return np.asarray(dataset[t_index, :], dtype=np.float64)
    return np.asarray(dataset[:, t_index], dtype=np.float64)


def sample_indices_for_simulation(
    sim: Simulation,
    args: argparse.Namespace,
    output_file: Path,
) -> dict[str, int | float]:
    if output_file.exists() and not args.overwrite:
        with np.load(output_file) as data:
            return {
                "n_base": int(data["is_high_hx"].size - data["is_high_hx"].sum()),
                "n_high_hx": int(data["is_high_hx"].sum()),
                "n_total": int(data["is_high_hx"].size),
                "hx_threshold": float(data["hx_threshold"]),
            }

    hx_key = normalize_h5_key(args.hx_key)
    stable_offset = int.from_bytes(hashlib.sha256(sim.simulation_id.encode()).digest()[:4], "little")
    rng = np.random.default_rng(args.seed + stable_offset)
    retained_times = np.arange(0, sim.n_time, args.time_stride, dtype=np.int32)
    base_x = np.arange(0, sim.n_x, args.x_stride, dtype=np.int32)
    base_mask = np.zeros(sim.n_x, dtype=bool)
    base_mask[base_x] = True

    # Estimate a simulation-wide threshold using all retained times but a spatial stride.
    hx_for_threshold: list[np.ndarray] = []
    with h5py.File(sim.h5_path, "r") as h5:
        hx_ds = h5[hx_key]
        threshold_stride = max(1, args.x_stride)
        for t_idx in retained_times:
            hx = read_hx_time_slice(hx_ds, int(t_idx), sim.time_axis)
            hx_for_threshold.append(np.abs(hx[::threshold_stride]))
        threshold_values = np.concatenate(hx_for_threshold)
        hx_threshold = float(np.nanpercentile(threshold_values, args.high_hx_percentile))
        del hx_for_threshold, threshold_values

        time_parts: list[np.ndarray] = []
        x_parts: list[np.ndarray] = []
        high_parts: list[np.ndarray] = []

        for t_idx in retained_times:
            # Deterministic base sample.
            time_parts.append(np.full(base_x.size, t_idx, dtype=np.int32))
            x_parts.append(base_x)
            high_parts.append(np.zeros(base_x.size, dtype=np.uint8))

            # Random targeted sample from points not already in the base grid.
            hx = read_hx_time_slice(hx_ds, int(t_idx), sim.time_axis)
            candidates = np.flatnonzero((np.abs(hx) >= hx_threshold) & (~base_mask))
            if candidates.size:
                n_extra = min(args.high_hx_extra_per_time, candidates.size)
                chosen = rng.choice(candidates, size=n_extra, replace=False).astype(np.int32)
                time_parts.append(np.full(n_extra, t_idx, dtype=np.int32))
                x_parts.append(chosen)
                high_parts.append(np.ones(n_extra, dtype=np.uint8))

    time_index = np.concatenate(time_parts)
    x_index = np.concatenate(x_parts)
    is_high_hx = np.concatenate(high_parts)

    np.savez_compressed(
        output_file,
        time_index=time_index,
        x_index=x_index,
        is_high_hx=is_high_hx,
        hx_threshold=np.float64(hx_threshold),
        x_stride=np.int32(args.x_stride),
        time_stride=np.int32(args.time_stride),
        high_hx_percentile=np.float64(args.high_hx_percentile),
        high_hx_extra_per_time=np.int32(args.high_hx_extra_per_time),
    )

    return {
        "n_base": int(np.count_nonzero(is_high_hx == 0)),
        "n_high_hx": int(np.count_nonzero(is_high_hx == 1)),
        "n_total": int(is_high_hx.size),
        "hx_threshold": hx_threshold,
    }


def write_metadata(rows: list[Simulation], stats: dict[str, dict], output: Path) -> None:
    fieldnames = [
        "simulation_id", "ca", "re", "theta", "split", "folder", "h5_path",
        "n_time", "n_x", "time_axis", "x_axis", "index_file",
        "n_base_samples", "n_high_hx_samples", "n_total_samples", "hx_threshold",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            st = stats[row.simulation_id]
            writer.writerow({
                "simulation_id": row.simulation_id,
                "ca": row.ca,
                "re": row.re,
                "theta": row.theta,
                "split": row.split,
                "folder": row.folder,
                "h5_path": row.h5_path,
                "n_time": row.n_time,
                "n_x": row.n_x,
                "time_axis": row.time_axis,
                "x_axis": row.x_axis,
                "index_file": st["index_file"],
                "n_base_samples": st["n_base"],
                "n_high_hx_samples": st["n_high_hx"],
                "n_total_samples": st["n_total"],
                "hx_threshold": st["hx_threshold"],
            })


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    indices_dir = args.output / "indices"
    indices_dir.mkdir(exist_ok=True)

    rows = discover_simulations(args)
    print(f"Found {len(rows)} simulations")

    split_names = choose_split(rows, args)
    rows = [Simulation(**{**asdict(row), "split": split})
            for row, split in zip(rows, split_names)]

    stats: dict[str, dict] = {}
    for i, sim in enumerate(rows, start=1):
        index_file = indices_dir / f"{sim.simulation_id}.npz"
        print(f"[{i:02d}/{len(rows):02d}] {sim.simulation_id} -> {sim.split}")
        st = sample_indices_for_simulation(sim, args, index_file)
        st["index_file"] = str(index_file.resolve())
        stats[sim.simulation_id] = st

    metadata_path = args.output / "simulations_metadata.csv"
    write_metadata(rows, stats, metadata_path)

    split_payload = {
        "seed": args.seed,
        "train": [r.simulation_id for r in rows if r.split == "train"],
        "validation": [r.simulation_id for r in rows if r.split == "validation"],
        "test": [r.simulation_id for r in rows if r.split == "test"],
        "sampling": {
            "x_stride": args.x_stride,
            "time_stride": args.time_stride,
            "high_hx_percentile": args.high_hx_percentile,
            "high_hx_extra_per_time": args.high_hx_extra_per_time,
        },
    }
    with (args.output / "splits.json").open("w") as f:
        json.dump(split_payload, f, indent=2)

    print(f"\nWrote metadata: {metadata_path}")
    print(f"Wrote split record: {args.output / 'splits.json'}")
    print(f"Wrote sample-index files under: {indices_dir}")


if __name__ == "__main__":
    main()
