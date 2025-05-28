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

"""Run clustering code for paper.

Clustering code of "Scalable Differentially Private Clustering via
Hierarchically Separated Trees" KDD'22 https://arxiv.org/abs/2206.08646
"""

from collections.abc import Sequence
import os
from typing import Optional
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import numpy as np
from hst_clustering import io
# pylint: disable=g-bad-import-order
from hst_clustering import dynamic_program
from hst_clustering import dp_hst

MakeDirs = lambda x: os.makedirs(x, exist_ok=True)
ReadRawData = io.ReadRawData
LoadFilesIntoDataFrame = io.LoadFilesIntoDataFrame


_DIRECT_RUNNER = "DirectRunner"
_DATAFLOW_RUNNER = "DataflowRunner"


# Tab separated file with the first column indicating the example number
# All other columns represent the coordinates of a point in R^d.
RAW_DATA = flags.DEFINE_string(
    "raw_data", "", "File path where raw data is stored"
)
OUTPUT_DIR = flags.DEFINE_string("output_dir", "", "Output directory")

K_PARAMS = flags.DEFINE_integer("k_params", 10, "Number of centers")

DIMENSIONS = flags.DEFINE_integer(
    "dimensions", None, "Must specify the input dimensions of the points."
)
LAYERS = flags.DEFINE_integer("layers", 10, "Number of layers for the hst.")
EPSILON = flags.DEFINE_float("epsilon", 1, "The epsilon parameter in DP.")
DELTA = flags.DEFINE_float("delta", 0.00001, "The delta parameter in DP.")
SEED = flags.DEFINE_integer(
    "seed", 0, "Seed for the random numbers used in the HST."
)
MIN_VALUE_ENTRY = flags.DEFINE_float(
    "min_value_entry", None, "Min value for entry in vectors."
)
MAX_VALUE_ENTRY = flags.DEFINE_float(
    "max_value_entry", None, "Max value for entry in vectors."
)

runner_choices = [_DIRECT_RUNNER, _DATAFLOW_RUNNER]


RUNNER = flags.DEFINE_enum(
    "runner",
    None,
    runner_choices,
    "The underlying runner; if not specified, use the default runner.",
)
NUM_BUCKETS_BEAM = flags.DEFINE_integer(
    "num_buckets_beam", 100, "Uses this many buckets in beam the reshuffling " +
    "of the data."
)

LLOYD_ITERS = 0


def create_beam_runner(
    runner_name,
):
  """Creates appropriate runner."""
  if runner_name == _DIRECT_RUNNER:
    runner = beam.runners.DirectRunner()
  elif runner_name == _DATAFLOW_RUNNER:
    runner = beam.runners.DataflowRunner()
  else:
    runner = None
  return runner


def main(argv):
  pipeline_args = argv[1:]
  logging.info("Additional pipeline args: %s", pipeline_args)
  pipeline_options = PipelineOptions(pipeline_args)

  # Running the DP HST pipeline using beam.
  # Make sure remote workers have access to variables/imports in the global
  # namespace.
  if RUNNER.value == _DATAFLOW_RUNNER:
    pipeline_options.view_as(SetupOptions).save_main_session = True

  assert RAW_DATA.value
  assert DIMENSIONS.value > 0
  assert OUTPUT_DIR.value
  assert K_PARAMS.value > 0
  assert LAYERS.value > 0
  assert EPSILON.value > 0
  assert DELTA.value > 0
  assert MIN_VALUE_ENTRY.value < MAX_VALUE_ENTRY.value
  assert NUM_BUCKETS_BEAM.value > 0

  base_dir = OUTPUT_DIR.value
  MakeDirs(base_dir)
  hst_output = base_dir + "/hst.csv"

  dp_hst.run_hst_pipeline(
      input_points=RAW_DATA.value,
      output_hst=hst_output,
      dimensions=DIMENSIONS.value,
      min_value_entry=MIN_VALUE_ENTRY.value,
      max_value_entry=MAX_VALUE_ENTRY.value,
      layers=LAYERS.value,
      seed=SEED.value,
      epsilon=EPSILON.value,
      delta=DELTA.value,
      num_buckets=NUM_BUCKETS_BEAM.value,
      runner=create_beam_runner(RUNNER.value),
      pipeline_options=pipeline_options,
  )
  logging.info("Pipeline completed")

  # Running the dynamic program.
  raw_data = ReadRawData(RAW_DATA.value)
  data_frame, feature_columns = LoadFilesIntoDataFrame(
      glob_string=hst_output + "*", dimensions=DIMENSIONS.value
  )
  tree = io.DataFrameToTree(data_frame)
  tree.validate_tree(True)
  k = K_PARAMS.value
  tree.solve_dp(k)
  logging.info("Finished solving the tree for k=%d", k)
  score, centers = dynamic_program.eval_hst_solution(
      raw_data, data_frame, tree, feature_columns, LLOYD_ITERS
  )
  logging.info("Score %.3f", score)

  results_output = base_dir + "/results.txt"
  centers_output = base_dir + "/centers.npy"
  with open(results_output, "w") as f:
    f.write(str(score))
  with open(centers_output, "wb") as f:
    np.save(f, centers)


if __name__ == "__main__":
  app.run(main)
