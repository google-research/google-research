# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
import datetime
import itertools
import json

import pandas as pd


@dataclass
class GridPaths:
    """Resolved output paths for a grid run."""

    out_dir: Path
    csv_path: Path


def timestamp_now():
    """Return a timestamp string suitable for directory names."""
    return datetime.datetime.now().strftime("%m%d-%H%M%S")


def ensure_dir(path):
    """Create a directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_grid_paths(
    *,
    out_root,
    experiment_name,
    append_csv_path = None,
    force_append = False,
):
    """
    Create output directories and resolve the CSV path.

    Args:
        out_root: Base directory for all grid outputs.
        experiment_name: Name prefix for the run directory.
        append_csv_path: Optional existing CSV to append to.
        force_append: If True, always append to append_csv_path.

    Returns:
        GridPaths with the resolved output directory and CSV path.
    """
    out_dir = ensure_dir(Path(out_root) / f"{experiment_name}_{timestamp_now()}")
    if force_append and append_csv_path:
        csv_path = Path(append_csv_path)
        ensure_dir(csv_path.parent)
    else:
        csv_path = out_dir / "runs.csv"
    return GridPaths(out_dir=out_dir, csv_path=csv_path)


def write_json(path, payload):
    """
    Write a JSON payload to disk with indentation.

    Args:
        path: File path to write.
        payload: JSON-serializable dictionary.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def csv_has_rows(path):
    """Return True if a CSV file exists and has data rows."""
    return path.exists() and path.stat().st_size > 0


def append_csv_row(path, row, columns):
    """
    Append a single row to a CSV file, creating it with a header if needed.

    Args:
        path: CSV file path.
        row: Row dictionary to append.
        columns: Column order for the CSV.
    """
    ensure_dir(path.parent)
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, mode="a", header=not csv_has_rows(path), index=False)


def grid_product(grid):
    """
    Iterate over the cartesian product of a grid dictionary.

    Args:
        grid: Mapping of parameter name to iterable of values.

    Yields:
        Dictionary of one concrete hyperparameter setting.
    """
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))
