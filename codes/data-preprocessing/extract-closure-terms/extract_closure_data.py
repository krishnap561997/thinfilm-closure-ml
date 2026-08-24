#!/usr/bin/env python3
"""
Extract thin-film closure quantities from a sequence of Basilisk HDF5 snapshots.

The script reads an external JSON configuration file, processes snapshot_*.h5
files one at a time, and writes one consolidated HDF5 file.

Output field datasets have shape (nx, nt), i.e. x by time.

Definitions retained from closure_quantities_nondim.ipynb:
    h          = integral f dy
    q          = integral u f dy
    h_m        = integral f_minus dy
    q_m        = integral u_minus f_minus dy
    shape_factor = integral u^2 f dy
    tau_wall   = (du/dy)|wall with fourth-order cell-centered stencil,
                 using nondimensional mu = 1 by default

Spatial derivatives use the fourth-order finite-difference formulas from the
notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = [
        "snapshot_dir",
        "output_file",
        "start_time",
        "end_time",
        "delta_time",
    ]

    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(
            "Missing required configuration entries: "
            + ", ".join(missing)
        )

    return config


def build_snapshot_values(
    start: float,
    end: float,
    step: float,
) -> np.ndarray:
    """
    Build an inclusive sequence of snapshot labels.

    This is used for file names, not for the physical time stored inside each
    HDF5 file. For integer-labelled files such as snapshot_000230.h5, use
    integer-like start/end/step values in the JSON file.
    """
    if step <= 0:
        raise ValueError("delta_time must be positive.")

    if end < start:
        raise ValueError("end_time must be >= start_time.")

    # Small tolerance prevents floating-point roundoff from dropping end.
    n = int(np.floor((end - start) / step + 1.0e-10)) + 1
    values = start + step * np.arange(n, dtype=float)

    return values


def format_snapshot_filename(
    value: float,
    prefix: str = "snapshot_",
    suffix: str = ".h5",
    width: int = 6,
    decimals: int = 0,
) -> str:
    """
    Format a snapshot filename.

    Examples
    --------
    value=230, width=6, decimals=0
        -> snapshot_000230.h5

    value=2.5, width=8, decimals=2
        -> snapshot_00002.50.h5
    """
    if decimals == 0:
        label = f"{value:0{width}.0f}"
    else:
        label = f"{value:0{width}.{decimals}f}"

    return f"{prefix}{label}{suffix}"


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def compute_cell_centers(
    points: np.ndarray,
    topology: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points)
    topology = np.asarray(topology, dtype=np.int64)

    cell_vertices = points[topology]
    return cell_vertices.mean(axis=1)


def build_cartesian_cell_grid(
    cell_centers: np.ndarray,
    decimals: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_centers = np.asarray(cell_centers)

    x = cell_centers[:, 0]
    y = cell_centers[:, 1]

    x_rounded = np.round(x, decimals=decimals)
    y_rounded = np.round(y, decimals=decimals)

    unique_x = np.unique(x_rounded)
    unique_y = np.unique(y_rounded)

    nx = len(unique_x)
    ny = len(unique_y)

    if nx * ny != len(cell_centers):
        raise ValueError(
            "The cell centers do not form a complete Cartesian grid. "
            f"Found nx={nx}, ny={ny}, nx*ny={nx * ny}, "
            f"n_cells={len(cell_centers)}."
        )

    sorting_indices = np.lexsort((x_rounded, y_rounded))

    x_sorted = x[sorting_indices]
    y_sorted = y[sorting_indices]

    x_matrix = x_sorted.reshape(ny, nx)
    y_matrix = y_sorted.reshape(ny, nx)

    return x_matrix, y_matrix, sorting_indices


def scalar_to_matrix(
    scalar: np.ndarray,
    sorting_indices: np.ndarray,
    matrix_shape: tuple[int, int],
) -> np.ndarray:
    scalar = np.asarray(scalar).squeeze()

    if scalar.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional scalar field after squeezing, "
            f"received shape {scalar.shape}."
        )

    if scalar.size != sorting_indices.size:
        raise ValueError(
            f"Scalar has {scalar.size} values, but mesh has "
            f"{sorting_indices.size} cells."
        )

    return scalar[sorting_indices].reshape(matrix_shape)


def vector_x_to_matrix(
    vector: np.ndarray,
    sorting_indices: np.ndarray,
    matrix_shape: tuple[int, int],
) -> np.ndarray:
    """
    Return only the streamwise component from a cell-based vector dataset.
    """
    vector = np.asarray(vector)

    if vector.ndim == 1:
        # Allows files that store only u_x rather than a full vector.
        ux = vector
    elif vector.ndim == 2:
        ux = vector[:, 0]
    else:
        raise ValueError(
            f"Unexpected velocity dataset shape: {vector.shape}"
        )

    if ux.size != sorting_indices.size:
        raise ValueError(
            f"Velocity has {ux.size} cells, but mesh has "
            f"{sorting_indices.size} cells."
        )

    return ux[sorting_indices].reshape(matrix_shape)


# ---------------------------------------------------------------------------
# Closure utilities
# ---------------------------------------------------------------------------

def depth_integral(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    lower_boundary: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Depth-integrate a cell-centered field along y at each x-location.

    This reproduces the finite-volume integration used in the notebook.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    field = np.asarray(field, dtype=float)

    if x.ndim != 2 or y.ndim != 2 or field.ndim != 2:
        raise ValueError("x, y, and field must be two-dimensional.")

    if x.shape != y.shape or x.shape != field.shape:
        raise ValueError("x, y, and field must have identical shapes.")

    ny, nx = field.shape

    if ny < 2:
        raise ValueError("At least two y cells are required.")

    x_locations = x[0, :].copy()
    integral = np.zeros(nx, dtype=float)

    for i in range(nx):
        y_column = y[:, i]
        field_column = field[:, i]

        order = np.argsort(y_column)
        y_sorted = y_column[order]
        field_sorted = field_column[order]

        if np.any(np.diff(y_sorted) <= 0.0):
            raise ValueError(
                f"Non-increasing y coordinates at x-index {i}."
            )

        y_faces = np.empty(ny + 1, dtype=float)
        y_faces[0] = lower_boundary
        y_faces[1:ny] = 0.5 * (
            y_sorted[:-1] + y_sorted[1:]
        )
        y_faces[ny] = (
            y_sorted[-1]
            + 0.5 * (y_sorted[-1] - y_sorted[-2])
        )

        cell_heights = np.diff(y_faces)

        if np.any(cell_heights <= 0.0):
            raise ValueError(
                f"Non-positive cell height at x-index {i}."
            )

        integral[i] = np.sum(field_sorted * cell_heights)

    return x_locations, integral


def compute_wall_shear(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    mu: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fourth-order lower-wall shear stencil used in the notebook.
    """
    if x.shape != y.shape or x.shape != u.shape:
        raise ValueError("x, y, and u must have identical shapes.")

    if u.ndim != 2:
        raise ValueError("x, y, and u must be two-dimensional.")

    if u.shape[0] < 4:
        raise ValueError(
            "Fourth-order wall shear requires at least four y cells."
        )

    dy = y[1, 0] - y[0, 0]

    if dy <= 0.0:
        raise ValueError("y must increase away from the lower wall.")

    u1 = u[0, :]
    u2 = u[1, :]
    u3 = u[2, :]
    u4 = u[3, :]

    dudy_wall = (
        3675.0 * u1
        - 1225.0 * u2
        + 441.0 * u3
        - 75.0 * u4
    ) / (840.0 * dy)

    return x[0, :], mu * dudy_wall


def first_derivative_x(
    x: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """
    Fourth-order first derivative, including one-sided boundary formulas.
    """
    x = np.asarray(x, dtype=float)
    field = np.asarray(field, dtype=float)

    if x.ndim != 1 or field.ndim != 1:
        raise ValueError("x and field must be one-dimensional.")

    if x.shape != field.shape:
        raise ValueError("x and field must have identical shapes.")

    if x.size < 5:
        raise ValueError("At least five x points are required.")

    dx_values = np.diff(x)
    dx = dx_values[0]

    if dx <= 0.0 or not np.allclose(dx_values, dx):
        raise ValueError("x must be monotonically increasing and uniform.")

    derivative = np.empty_like(field, dtype=float)

    derivative[2:-2] = (
        field[:-4]
        - 8.0 * field[1:-3]
        + 8.0 * field[3:-1]
        - field[4:]
    ) / (12.0 * dx)

    derivative[0] = (
        -25.0 * field[0]
        + 48.0 * field[1]
        - 36.0 * field[2]
        + 16.0 * field[3]
        - 3.0 * field[4]
    ) / (12.0 * dx)

    derivative[1] = (
        -3.0 * field[0]
        - 10.0 * field[1]
        + 18.0 * field[2]
        - 6.0 * field[3]
        + field[4]
    ) / (12.0 * dx)

    derivative[-2] = (
        -field[-5]
        + 6.0 * field[-4]
        - 18.0 * field[-3]
        + 10.0 * field[-2]
        + 3.0 * field[-1]
    ) / (12.0 * dx)

    derivative[-1] = (
        3.0 * field[-5]
        - 16.0 * field[-4]
        + 36.0 * field[-3]
        - 48.0 * field[-2]
        + 25.0 * field[-1]
    ) / (12.0 * dx)

    return derivative


def second_derivative_x(
    x: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """
    Fourth-order second derivative, including one-sided boundary formulas.
    """
    x = np.asarray(x, dtype=float)
    field = np.asarray(field, dtype=float)

    if x.ndim != 1 or field.ndim != 1:
        raise ValueError("x and field must be one-dimensional.")

    if x.shape != field.shape:
        raise ValueError("x and field must have identical shapes.")

    if x.size < 6:
        raise ValueError("At least six x points are required.")

    dx_values = np.diff(x)
    dx = dx_values[0]

    if dx <= 0.0 or not np.allclose(dx_values, dx):
        raise ValueError("x must be monotonically increasing and uniform.")

    derivative = np.empty_like(field, dtype=float)

    derivative[2:-2] = (
        -field[4:]
        + 16.0 * field[3:-1]
        - 30.0 * field[2:-2]
        + 16.0 * field[1:-3]
        - field[:-4]
    ) / (12.0 * dx**2)

    derivative[0] = (
        45.0 * field[0]
        - 154.0 * field[1]
        + 214.0 * field[2]
        - 156.0 * field[3]
        + 61.0 * field[4]
        - 10.0 * field[5]
    ) / (12.0 * dx**2)

    derivative[1] = (
        10.0 * field[0]
        - 15.0 * field[1]
        - 4.0 * field[2]
        + 14.0 * field[3]
        - 6.0 * field[4]
        + field[5]
    ) / (12.0 * dx**2)

    derivative[-2] = (
        field[-6]
        - 6.0 * field[-5]
        + 14.0 * field[-4]
        - 4.0 * field[-3]
        - 15.0 * field[-2]
        + 10.0 * field[-1]
    ) / (12.0 * dx**2)

    derivative[-1] = (
        -10.0 * field[-6]
        + 61.0 * field[-5]
        - 156.0 * field[-4]
        + 214.0 * field[-3]
        - 154.0 * field[-2]
        + 45.0 * field[-1]
    ) / (12.0 * dx**2)

    return derivative


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

REQUIRED_SNAPSHOT_DATASETS = (
    "Geometry/Points",
    "Topology",
    "Cells/f",
    "Cells/f_minus",
    "Cells/u.x",
    "Cells/u_minus.x",
    "Parameters/time",
    "Parameters/time_minus",
)


def validate_snapshot(h5: h5py.File, snapshot_path: Path) -> None:
    missing = [
        dataset
        for dataset in REQUIRED_SNAPSHOT_DATASETS
        if dataset not in h5
    ]

    if missing:
        raise KeyError(
            f"{snapshot_path} is missing required datasets:\n  "
            + "\n  ".join(missing)
        )


def scalar_value(dataset: h5py.Dataset) -> float:
    value = np.asarray(dataset[...]).squeeze()

    if value.size != 1:
        raise ValueError(
            f"Expected scalar dataset {dataset.name}, "
            f"received shape {dataset.shape}."
        )

    return float(value)


def copy_static_parameters(
    source: h5py.File,
    destination: h5py.Group,
) -> None:
    """
    Copy every dataset below /Parameters except snapshot-dependent time values.

    This automatically preserves Re, Ca, h0, u0, t0, angle, We, Fr, etc.,
    whatever is actually present in the source snapshot.
    """
    if "Parameters" not in source:
        return

    time_names = {
        "time",
        "time_minus",
        "t",
        "tm",
        "dt",
    }

    def recurse(src_group: h5py.Group, dst_group: h5py.Group) -> None:
        for name, item in src_group.items():
            if name in time_names:
                continue

            if isinstance(item, h5py.Dataset):
                dst_group.create_dataset(name, data=item[...])

                for key, value in item.attrs.items():
                    dst_group[name].attrs[key] = value

            elif isinstance(item, h5py.Group):
                child = dst_group.create_group(name)
                recurse(item, child)

    recurse(source["Parameters"], destination)

    for key, value in source["Parameters"].attrs.items():
        destination.attrs[key] = value


def create_output_datasets(
    output: h5py.File,
    nx: int,
    nt: int,
    compression: str | None,
    compression_level: int | None,
) -> dict[str, h5py.Dataset]:
    field_names = (
        "h",
        "q",
        "h_m",
        "q_m",
        "shape_factor",
        "shape_factor_m",
        "h_x",
        "q_x",
        "h_xx",
        "q_xx",
        "tau_wall",
        "tau_wall_m",
    )

    kwargs: dict[str, Any] = {
        "shape": (nx, nt),
        "dtype": np.float64,
        "chunks": (min(nx, 1024), 1),
    }

    if compression is not None:
        kwargs["compression"] = compression
        if compression == "gzip" and compression_level is not None:
            kwargs["compression_opts"] = compression_level

    datasets = {
        name: output.create_dataset(name, **kwargs)
        for name in field_names
    }

    datasets["t"] = output.create_dataset(
        "t", shape=(nt,), dtype=np.float64
    )
    datasets["tm"] = output.create_dataset(
        "tm", shape=(nt,), dtype=np.float64
    )
    datasets["dt"] = output.create_dataset(
        "dt", shape=(nt,), dtype=np.float64
    )

    return datasets


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def extract_snapshots(config: dict[str, Any]) -> Path:
    snapshot_dir = Path(config["snapshot_dir"]).expanduser()
    output_path = Path(config["output_file"]).expanduser()

    snapshot_values = build_snapshot_values(
        float(config["start_time"]),
        float(config["end_time"]),
        float(config["delta_time"]),
    )

    prefix = str(config.get("snapshot_prefix", "snapshot_"))
    suffix = str(config.get("snapshot_suffix", ".h5"))
    width = int(config.get("filename_width", 6))
    decimals = int(config.get("filename_decimals", 0))

    wall_mu = float(config.get("wall_shear_mu", 1.0))
    lower_boundary = float(config.get("lower_boundary", 0.0))

    compression = config.get("compression", "gzip")
    if compression in ("none", "None", ""):
        compression = None

    compression_level = int(config.get("compression_level", 4))

    filenames = [
        format_snapshot_filename(
            value,
            prefix=prefix,
            suffix=suffix,
            width=width,
            decimals=decimals,
        )
        for value in snapshot_values
    ]

    snapshot_paths = [snapshot_dir / name for name in filenames]

    missing = [path for path in snapshot_paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        extra = ""
        if len(missing) > 20:
            extra = f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(
            "The following snapshot files were not found:\n"
            f"{preview}{extra}"
        )

    first_path = snapshot_paths[0]

    # Build mesh mapping once. The workflow assumes every snapshot uses the
    # same fixed Cartesian mesh, consistent with the notebook.
    with h5py.File(first_path, "r") as first:
        validate_snapshot(first, first_path)

        points = first["Geometry/Points"][...]
        topology = first["Topology"][...]

        cell_centers = compute_cell_centers(points, topology)
        x_matrix, y_matrix, sorting_indices = build_cartesian_cell_grid(
            cell_centers
        )

    xc = x_matrix[0, :].copy()
    nx = xc.size
    nt = len(snapshot_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as output:
        output.attrs["description"] = (
            "Consolidated thin-film closure data extracted from Basilisk DNS."
        )
        output.attrs["matrix_layout"] = "x_by_time"
        output.attrs["nx"] = nx
        output.attrs["nt"] = nt
        output.attrs["source_snapshot_dir"] = str(snapshot_dir)
        output.attrs["wall_shear_mu"] = wall_mu
        output.attrs["lower_boundary"] = lower_boundary

        output.create_dataset("x", data=xc)

        datasets = create_output_datasets(
            output,
            nx,
            nt,
            compression,
            compression_level,
        )

        with h5py.File(first_path, "r") as first:
            parameter_group = output.create_group("parameters")
            copy_static_parameters(first, parameter_group)

        for time_index, snapshot_path in enumerate(snapshot_paths):
            print(
                f"[{time_index + 1:>{len(str(nt))}}/{nt}] "
                f"{snapshot_path.name}"
            )

            with h5py.File(snapshot_path, "r") as snapshot:
                validate_snapshot(snapshot, snapshot_path)

                f = scalar_to_matrix(
                    snapshot["Cells/f"][...],
                    sorting_indices,
                    x_matrix.shape,
                )
                f_minus = scalar_to_matrix(
                    snapshot["Cells/f_minus"][...],
                    sorting_indices,
                    x_matrix.shape,
                )

                u = vector_x_to_matrix(
                    snapshot["Cells/u.x"][...],
                    sorting_indices,
                    x_matrix.shape,
                )
                u_minus = vector_x_to_matrix(
                    snapshot["Cells/u_minus.x"][...],
                    sorting_indices,
                    x_matrix.shape,
                )

                t = scalar_value(snapshot["Parameters/time"])
                tm = scalar_value(snapshot["Parameters/time_minus"])
                dt = t - tm

                if dt <= 0.0:
                    raise ValueError(
                        f"Non-positive dt in {snapshot_path}: "
                        f"time={t}, time_minus={tm}."
                    )

                _, h = depth_integral(
                    x_matrix,
                    y_matrix,
                    f,
                    lower_boundary=lower_boundary,
                )
                _, q = depth_integral(
                    x_matrix,
                    y_matrix,
                    u * f,
                    lower_boundary=lower_boundary,
                )
                _, h_m = depth_integral(
                    x_matrix,
                    y_matrix,
                    f_minus,
                    lower_boundary=lower_boundary,
                )
                _, q_m = depth_integral(
                    x_matrix,
                    y_matrix,
                    u_minus * f_minus,
                    lower_boundary=lower_boundary,
                )
                _, shape_factor = depth_integral(
                    x_matrix,
                    y_matrix,
                    u**2 * f,
                    lower_boundary=lower_boundary,
                )

                _, tau_wall = compute_wall_shear(
                    x_matrix,
                    y_matrix,
                    u,
                    mu=wall_mu,
                )

                _, shape_factor_m = depth_integral(
                    x_matrix,
                    y_matrix,
                    u_minus**2 * f,
                    lower_boundary=lower_boundary,
                )

                _, tau_wall_m = compute_wall_shear(
                    x_matrix,
                    y_matrix,
                    u_minus,
                    mu=wall_mu,
                )
                h_x = first_derivative_x(xc, h)
                q_x = first_derivative_x(xc, q)
                h_xx = second_derivative_x(xc, h)
                q_xx = second_derivative_x(xc, q)

                datasets["h"][:, time_index] = h
                datasets["q"][:, time_index] = q
                datasets["h_m"][:, time_index] = h_m
                datasets["q_m"][:, time_index] = q_m
                datasets["shape_factor"][:, time_index] = shape_factor
                datasets["shape_factor_m"][:, time_index] = shape_factor_m
                datasets["h_x"][:, time_index] = h_x
                datasets["q_x"][:, time_index] = q_x
                datasets["h_xx"][:, time_index] = h_xx
                datasets["q_xx"][:, time_index] = q_xx
                datasets["tau_wall"][:, time_index] = tau_wall
                datasets["tau_wall_m"][:, time_index] = tau_wall_m

                datasets["t"][time_index] = t
                datasets["tm"][time_index] = tm
                datasets["dt"][time_index] = dt

        output.flush()

    print(f"\nWrote consolidated file:\n  {output_path}")
    print(f"Field matrix shape: ({nx}, {nt}) = (x, t)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract h, q, previous-step quantities, closure terms, "
            "and spatial derivatives from Basilisk snapshot HDF5 files."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to JSON configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    extract_snapshots(config)


if __name__ == "__main__":
    main()
