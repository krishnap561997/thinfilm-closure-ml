#!/usr/bin/env python3
"""
Train a pointwise thin-film neural network.

Inputs
------
[h, h_x, h_xx, h_xxx, q_x]

Outputs
-------
[q_t, h_t]

The script uses the dataset index created by build_thinfilm_dataset.py:
    dataset_index/
        simulations_metadata.csv
        splits.json
        indices/*.npz

Architecture
------------
Fully connected MLP with:
    - configurable hidden depth and width
    - tanh activation
    - MSE loss
    - Adam optimizer
    - ReduceLROnPlateau learning-rate scheduler
    - early stopping
    - feature and target standardization based only on training data

Example
-------
python train_thinfilm_mlp.py \
    --index-dir /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/ROM_Ca_Re_sweep/dataset_index \
    --output-dir ./runs/thinfilm_mlp_case01 \
    --epochs 1000 \
    --hidden-layers 6 \
    --hidden-width 100 \
    --batch-size 4096 \
    --learning-rate 1e-3 \
    --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_KEYS = ("/h", "/h_x", "/h_xx", "/h_xxx", "/q_x", "Ca", "Re", "theta")
TARGET_KEYS = ("/q_t", "/h_t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Directory containing simulations_metadata.csv and indices/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thinfilm_mlp_run"),
        help="Directory for checkpoints, normalization statistics, and history.",
    )

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--hidden-width", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    parser.add_argument(
        "--stats-batch-size",
        type=int,
        default=16384,
        help="Batch size used while calculating normalization statistics.",
    )
    parser.add_argument(
        "--stats-max-samples",
        type=int,
        default=2_000_000,
        help=(
            "Maximum number of training samples used for normalization. "
            "Use 0 to process the complete training set."
        ),
    )

    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="Epochs without validation improvement before reducing learning rate.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Stop after this many epochs without validation-loss improvement.",
    )
    parser.add_argument(
        "--minimum-epochs",
        type=int,
        default=100,
        help="Do not early-stop before this epoch.",
    )

    parser.add_argument(
        "--include-parameters",
        action="store_true",
        help="Append Ca, Re, and theta to the five local flow inputs.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=0.0,
        help="Maximum gradient norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint from which to resume training.",
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ThinFilmIndexedDataset(Dataset):
    """
    Lazily read indexed point samples from multiple HDF5 simulations.

    Each item corresponds to one sampled (time_index, x_index) pair listed in
    the simulation's compressed NPZ index file.
    """

    def __init__(
        self,
        metadata_csv: Path,
        split: str,
        feature_keys: Sequence[str],
        target_keys: Sequence[str],
        include_parameters: bool = False,
    ) -> None:
        self.feature_keys = [self._normalize_h5_key(k) for k in feature_keys]
        self.target_keys = [self._normalize_h5_key(k) for k in target_keys]
        self.include_parameters = include_parameters

        with metadata_csv.open(newline="") as stream:
            rows = [
                row for row in csv.DictReader(stream)
                if row["split"].strip().lower() == split.lower()
            ]

        if not rows:
            raise ValueError(
                f"No simulations with split={split!r} were found in {metadata_csv}"
            )

        self.simulations: list[dict[str, str]] = []
        cumulative = [0]

        for row in rows:
            index_path = Path(row["index_file"])
            if not index_path.is_file():
                raise FileNotFoundError(f"Missing index file: {index_path}")

            with np.load(index_path) as index_data:
                number_of_samples = int(index_data["time_index"].shape[0])

            self.simulations.append(row)
            cumulative.append(cumulative[-1] + number_of_samples)

        self.cumulative = np.asarray(cumulative, dtype=np.int64)

        # These caches are created independently inside each DataLoader worker.
        self._h5_handles: dict[str, h5py.File] = {}
        self._index_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _normalize_h5_key(key: str) -> str:
        return key if key.startswith("/") else f"/{key}"

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def _get_indices(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if path not in self._index_cache:
            with np.load(path) as data:
                self._index_cache[path] = (
                    data["time_index"].astype(np.int64, copy=True),
                    data["x_index"].astype(np.int64, copy=True),
                )
        return self._index_cache[path]

    @staticmethod
    def _read_scalar(
        dataset: h5py.Dataset,
        time_index: int,
        x_index: int,
        time_axis: int,
    ) -> float:
        if time_axis == 0:
            return float(dataset[time_index, x_index])
        return float(dataset[x_index, time_index])

    def __getitem__(self, global_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if global_index < 0:
            global_index += len(self)
        if global_index < 0 or global_index >= len(self):
            raise IndexError(global_index)

        simulation_position = int(
            np.searchsorted(self.cumulative, global_index, side="right") - 1
        )
        local_index = int(global_index - self.cumulative[simulation_position])
        row = self.simulations[simulation_position]

        time_indices, x_indices = self._get_indices(row["index_file"])
        time_index = int(time_indices[local_index])
        x_index = int(x_indices[local_index])
        time_axis = int(row["time_axis"])

        h5_file = self._get_h5(row["h5_path"])

        features = [
            self._read_scalar(
                h5_file[key], time_index, x_index, time_axis
            )
            for key in self.feature_keys
        ]

        if self.include_parameters:
            features.extend(
                [
                    float(row["ca"]),
                    float(row["re"]),
                    float(row["theta"]),
                ]
            )

        targets = [
            self._read_scalar(
                h5_file[key], time_index, x_index, time_axis
            )
            for key in self.target_keys
        ]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

    def close(self) -> None:
        for handle in self._h5_handles.values():
            handle.close()
        self._h5_handles.clear()
        self._index_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RunningMoments:
    """Numerically stable streaming mean and variance for vector data."""

    def __init__(self, number_of_variables: int) -> None:
        self.count = 0
        self.mean = torch.zeros(number_of_variables, dtype=torch.float64)
        self.m2 = torch.zeros(number_of_variables, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim != 2:
            raise ValueError(f"Expected a 2-D tensor, received {values.shape}")

        batch_count = values.shape[0]
        if batch_count == 0:
            return

        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean += delta * (batch_count / total_count)
        self.m2 += (
            batch_m2
            + delta.square() * self.count * batch_count / total_count
        )
        self.count = total_count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count < 2:
            raise RuntimeError("At least two samples are required for normalization.")

        variance = self.m2 / (self.count - 1)
        standard_deviation = torch.sqrt(torch.clamp(variance, min=1.0e-24))

        return (
            self.mean.to(torch.float32),
            standard_deviation.to(torch.float32),
        )


@dataclass
class Normalization:
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    def to(self, device: torch.device) -> "Normalization":
        return Normalization(
            feature_mean=self.feature_mean.to(device),
            feature_std=self.feature_std.to(device),
            target_mean=self.target_mean.to(device),
            target_std=self.target_std.to(device),
        )

    def normalize_features(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.feature_mean) / self.feature_std

    def normalize_targets(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.target_mean) / self.target_std

    def denormalize_targets(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.target_std + self.target_mean


def compute_normalization(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    maximum_samples: int,
) -> Normalization:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    feature_moments: RunningMoments | None = None
    target_moments: RunningMoments | None = None
    processed = 0

    for features, targets in loader:
        if maximum_samples > 0:
            remaining = maximum_samples - processed
            if remaining <= 0:
                break
            if features.shape[0] > remaining:
                features = features[:remaining]
                targets = targets[:remaining]

        if feature_moments is None:
            feature_moments = RunningMoments(features.shape[1])
            target_moments = RunningMoments(targets.shape[1])

        feature_moments.update(features)
        target_moments.update(targets)
        processed += features.shape[0]

        if maximum_samples > 0 and processed >= maximum_samples:
            break

    if feature_moments is None or target_moments is None:
        raise RuntimeError("No samples were available to compute normalization.")

    feature_mean, feature_std = feature_moments.finalize()
    target_mean, target_std = target_moments.finalize()

    print(f"Normalization samples: {processed:,}")
    print(f"Feature mean: {feature_mean.tolist()}")
    print(f"Feature std:  {feature_std.tolist()}")
    print(f"Target mean:  {target_mean.tolist()}")
    print(f"Target std:   {target_std.tolist()}")

    return Normalization(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )


class ThinFilmMLP(nn.Module):
    """Fully connected tanh network for local thin-film dynamics."""

    def __init__(
        self,
        number_of_inputs: int,
        number_of_outputs: int = 2,
        hidden_layers: int = 6,
        hidden_width: int = 100,
    ) -> None:
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive.")

        layers: list[nn.Module] = [
            nn.Linear(number_of_inputs, hidden_width),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_width, hidden_width),
                    nn.Tanh(),
                ]
            )

        layers.append(nn.Linear(hidden_width, number_of_outputs))
        self.network = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier initialization is appropriate for tanh activations.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight,
                    gain=nn.init.calculate_gain("tanh"),
                )
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RegressionMetrics:
    """Accumulate physical-space MSE, MAE, and R² without storing predictions."""

    def __init__(self, number_of_outputs: int) -> None:
        self.count = 0
        self.squared_error = torch.zeros(number_of_outputs, dtype=torch.float64)
        self.absolute_error = torch.zeros(number_of_outputs, dtype=torch.float64)
        self.target_sum = torch.zeros(number_of_outputs, dtype=torch.float64)
        self.target_squared_sum = torch.zeros(number_of_outputs, dtype=torch.float64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to(dtype=torch.float64, device="cpu")
        target = target.detach().to(dtype=torch.float64, device="cpu")

        difference = prediction - target

        self.count += target.shape[0]
        self.squared_error += difference.square().sum(dim=0)
        self.absolute_error += difference.abs().sum(dim=0)
        self.target_sum += target.sum(dim=0)
        self.target_squared_sum += target.square().sum(dim=0)

    def compute(self) -> dict[str, list[float]]:
        if self.count == 0:
            raise RuntimeError("No samples were supplied to RegressionMetrics.")

        mse = self.squared_error / self.count
        mae = self.absolute_error / self.count

        total_sum_of_squares = (
            self.target_squared_sum
            - self.target_sum.square() / self.count
        )
        r2 = 1.0 - self.squared_error / torch.clamp(
            total_sum_of_squares, min=1.0e-30
        )

        return {
            "mse": mse.tolist(),
            "mae": mae.tolist(),
            "r2": r2.tolist(),
        }


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    normalization: Normalization,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> tuple[float, dict[str, list[float]]]:
    training = optimizer is not None
    model.train(training)

    normalized_loss_sum = 0.0
    number_of_samples = 0
    physical_metrics = RegressionMetrics(number_of_outputs=2)

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            normalized_features = normalization.normalize_features(features)
            normalized_targets = normalization.normalize_targets(targets)

            normalized_prediction = model(normalized_features)
            loss = loss_function(normalized_prediction, normalized_targets)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if gradient_clip > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()

            batch_size = targets.shape[0]
            normalized_loss_sum += loss.item() * batch_size
            number_of_samples += batch_size

            physical_prediction = normalization.denormalize_targets(
                normalized_prediction
            )
            physical_metrics.update(physical_prediction, targets)

    return (
        normalized_loss_sum / number_of_samples,
        physical_metrics.compute(),
    )


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    normalization: Normalization,
    arguments: argparse.Namespace,
    best_validation_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "feature_keys": list(FEATURE_KEYS),
            "target_keys": list(TARGET_KEYS),
            "feature_mean": normalization.feature_mean.cpu(),
            "feature_std": normalization.feature_std.cpu(),
            "target_mean": normalization.target_mean.cpu(),
            "target_std": normalization.target_std.cpu(),
            "best_validation_loss": best_validation_loss,
            "arguments": vars(arguments),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = args.index_dir / "simulations_metadata.csv"
    splits_json = args.index_dir / "splits.json"

    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    if not splits_json.is_file():
        raise FileNotFoundError(splits_json)

    with splits_json.open() as stream:
        split_record = json.load(stream)

    print("Split summary:")
    for split_name in ("train", "validation", "test"):
        print(f"  {split_name:10s}: {len(split_record.get(split_name, []))} simulations")

    train_dataset = ThinFilmIndexedDataset(
        metadata_csv=metadata_csv,
        split="train",
        feature_keys=FEATURE_KEYS,
        target_keys=TARGET_KEYS,
        include_parameters=args.include_parameters,
    )
    validation_dataset = ThinFilmIndexedDataset(
        metadata_csv=metadata_csv,
        split="validation",
        feature_keys=FEATURE_KEYS,
        target_keys=TARGET_KEYS,
        include_parameters=args.include_parameters,
    )
    test_dataset = ThinFilmIndexedDataset(
        metadata_csv=metadata_csv,
        split="test",
        feature_keys=FEATURE_KEYS,
        target_keys=TARGET_KEYS,
        include_parameters=args.include_parameters,
    )

    print(f"Training samples:   {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")

    device = select_device(args.device)
    print(f"Device: {device}")

    normalization_path = args.output_dir / "normalization.pt"

    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        normalization = Normalization(
            feature_mean=resume_checkpoint["feature_mean"],
            feature_std=resume_checkpoint["feature_std"],
            target_mean=resume_checkpoint["target_mean"],
            target_std=resume_checkpoint["target_std"],
        )
    elif normalization_path.is_file():
        saved_normalization = torch.load(normalization_path, map_location="cpu")
        normalization = Normalization(**saved_normalization)
        print(f"Loaded normalization from {normalization_path}")
    else:
        normalization = compute_normalization(
            dataset=train_dataset,
            batch_size=args.stats_batch_size,
            num_workers=args.num_workers,
            maximum_samples=args.stats_max_samples,
        )
        torch.save(
            {
                "feature_mean": normalization.feature_mean,
                "feature_std": normalization.feature_std,
                "target_mean": normalization.target_mean,
                "target_std": normalization.target_std,
            },
            normalization_path,
        )

    normalization = normalization.to(device)

    number_of_inputs = len(FEATURE_KEYS) + (3 if args.include_parameters else 0)

    model = ThinFilmMLP(
        number_of_inputs=number_of_inputs,
        number_of_outputs=len(TARGET_KEYS),
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
    ).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    start_epoch = 1
    best_validation_loss = math.inf
    epochs_without_improvement = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", math.inf)
        )
        print(f"Resuming from epoch {start_epoch}")

    history: list[dict[str, object]] = []
    best_checkpoint_path = args.output_dir / "best_model.pt"
    last_checkpoint_path = args.output_dir / "last_model.pt"

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            normalization=normalization,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
        )

        validation_loss, validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            normalization=normalization,
            loss_function=loss_function,
            device=device,
            optimizer=None,
            gradient_clip=0.0,
        )

        scheduler.step(validation_loss)
        current_learning_rate = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_normalized_mse": train_loss,
            "validation_normalized_mse": validation_loss,
            "train_q_t_r2": train_metrics["r2"][0],
            "train_h_t_r2": train_metrics["r2"][1],
            "validation_q_t_r2": validation_metrics["r2"][0],
            "validation_h_t_r2": validation_metrics["r2"][1],
            "train_q_t_mse": train_metrics["mse"][0],
            "train_h_t_mse": train_metrics["mse"][1],
            "validation_q_t_mse": validation_metrics["mse"][0],
            "validation_h_t_mse": validation_metrics["mse"][1],
        }
        history.append(row)

        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"lr={current_learning_rate:.3e} | "
            f"train MSE_n={train_loss:.6e} | "
            f"val MSE_n={validation_loss:.6e} | "
            f"val R2(q_t)={validation_metrics['r2'][0]:.5f} | "
            f"val R2(h_t)={validation_metrics['r2'][1]:.5f}",
            flush=True,
        )

        improved = validation_loss < best_validation_loss

        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                normalization,
                args,
                best_validation_loss,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            normalization,
            args,
            best_validation_loss,
        )

        with (args.output_dir / "history.json").open("w") as stream:
            json.dump(history, stream, indent=2)

        if (
            epoch >= args.minimum_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping: validation loss has not improved for "
                f"{epochs_without_improvement} epochs."
            )
            break

    print(f"Loading best model: {best_checkpoint_path}")
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        normalization=normalization,
        loss_function=loss_function,
        device=device,
        optimizer=None,
        gradient_clip=0.0,
    )

    test_results = {
        "normalized_mse": test_loss,
        "q_t": {
            "mse": test_metrics["mse"][0],
            "mae": test_metrics["mae"][0],
            "r2": test_metrics["r2"][0],
        },
        "h_t": {
            "mse": test_metrics["mse"][1],
            "mae": test_metrics["mae"][1],
            "r2": test_metrics["r2"][1],
        },
    }

    with (args.output_dir / "test_metrics.json").open("w") as stream:
        json.dump(test_results, stream, indent=2)

    print("Test results:")
    print(json.dumps(test_results, indent=2))


if __name__ == "__main__":
    main()
