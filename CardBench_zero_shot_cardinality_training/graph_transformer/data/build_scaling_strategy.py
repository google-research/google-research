# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Collect global statistics for numerical features to build scaling strategy."""

from collections.abc import Mapping, Sequence
import json
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np
import tqdm

from CardBench_zero_shot_cardinality_training.graph_transformer import constants
from sparse_deferred.structs import graph_struct

_DATASET_NAMES = flags.DEFINE_list(
    "dataset_names",
    None,
    "Comma-separated list of dataset names.",
    required=True,
)

_INPUT_DATASET_PATH = flags.DEFINE_string(
    "input_dataset_path",
    None,
    "Input dataset path.",
    required=True,
)

_DATASET_TYPE = flags.DEFINE_enum(
    "dataset_type",
    None,
    ["binary_join", "single_table"],
    "Dataset type.",
    required=True,
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Output path.",
    required=True,
)


def _collect_data(
    dbs,
):
  """Collect raw data for numerical features.

  Args:
    dbs: A list of InMemoryDBs.

  Returns:
    A dictionary of raw data for numerical features.
  """
  raw_data = {}
  for node_name, features in constants.SCALING_NUMERICAL_FEATURES.items():
    raw_data[node_name] = {}
    for feature_name in features:
      raw_data[node_name][feature_name] = []

  for db in dbs:
    for i in tqdm.tqdm(range(db.size)):
      graph = db.get_item(i)
      for node_name, features in constants.SCALING_NUMERICAL_FEATURES.items():
        for feature_name in features:
          raw_data[node_name][feature_name].extend(
              list(graph.nodes[node_name][feature_name].astype(np.float32))
          )

  return raw_data


def build_scaling_strategy(
    dbs,
):
  """Create scaling strategy for numerical features.

  Args:
    dbs: A list of InMemoryDBs.

  Returns:
    A dictionary of scaling strategy for numerical features.
  """
  raw_data = _collect_data(dbs)
  scaling_strategy = {}
  for node_name, features in constants.SCALING_NUMERICAL_FEATURES.items():
    scaling_strategy[node_name] = {}
    for feature_name in features:
      value = raw_data[node_name][feature_name]
      scaling_strategy[node_name][feature_name] = {
          "mean": np.mean(value),
          "std": np.std(value),
          "median": np.median(value),
          "min": np.min(value),
          "max": np.max(value),
          "log_mean": np.mean(np.log10(np.clip(value, a_min=1e-9, a_max=None))),
          "log_std": np.std(np.log10(np.clip(value, a_min=1e-9, a_max=None))),
          "log_median": np.median(
              np.log10(np.clip(value, a_min=1e-9, a_max=None))
          ),
          "log_max": np.max(np.log10(np.clip(value, a_min=1e-9, a_max=None))),
          "log_min": np.min(np.log10(np.clip(value, a_min=1e-9, a_max=None))),
      }

      # Always use log scaling before standardization
      scaling_strategy[node_name][feature_name]["log_scale"] = True

  return scaling_strategy


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  dbs = []

  for dataset_name in _DATASET_NAMES.value:
    dbs.append(
        graph_struct.InMemoryDB.from_sharded_files(
            os.path.join(
                _INPUT_DATASET_PATH.value,
                _DATASET_TYPE.value,
                f"{dataset_name}_{_DATASET_TYPE.value}.npz",
            )
        )
    )

  scaling_strategy = build_scaling_strategy(dbs)

  with open(
      os.path.join(
          _OUTPUT_PATH.value, _DATASET_TYPE.value, "scaling_strategy.json"
      ),
      "w",
  ) as f:
    json.dump(scaling_strategy, f, indent=2, default=float)


if __name__ == "__main__":
  app.run(main)
