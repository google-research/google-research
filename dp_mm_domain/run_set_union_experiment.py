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

"""Script for running a single set union experiment and saving its plots."""

from collections.abc import Sequence
import pickle

from absl import app
from absl import flags
import numpy as np

from dp_mm_domain import set_union_experiment


_L0_RANGE = flags.DEFINE_list(
    "l0_range",
    None,
    "A comma-separated list of l0_range to try.",
    required=True,
)
_EPSILON = flags.DEFINE_float(
    "epsilon",
    1.0,
    "Epsilon to use for the experiment.",
)
_DELTA = flags.DEFINE_float(
    "delta",
    1e-5,
    "Delta to use for the experiment.",
)

_NUM_TRIALS = flags.DEFINE_integer(
    "num_trials",
    5,
    "The number of trials to run for each l0_bound.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_file",
    "./set_union_results",
    "File path for the output plots.",
)

_DATA_PATH = flags.DEFINE_string(
    "data_path",
    None,
    "File path for the input data.",
    required=True,
)

_SEED = flags.DEFINE_integer(
    "seed",
    None,
    "Random seed to use for the experiment.",
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  l0_range = _L0_RANGE.value
  epsilon = _EPSILON.value
  delta = _DELTA.value
  num_trials = _NUM_TRIALS.value
  output_path = _OUTPUT_PATH.value
  data_path = _DATA_PATH.value

  input_methods = [
      set_union_experiment.SetUnionMethod.WGM,
      set_union_experiment.SetUnionMethod.POLICY_GAUSSIAN,
      set_union_experiment.SetUnionMethod.POLICY_GREEDY,
  ]

  with open(data_path, "rb") as f:
    input_data = pickle.load(f)

  if _SEED.value is not None:
    np.random.seed(_SEED.value)
  np.random.shuffle(input_data)

  results = set_union_experiment.compare_methods(
      input_data, input_methods, l0_range, epsilon, delta, num_trials
  )
  set_union_experiment.plot_results(
      results, l0_range, epsilon, delta, output_path
  )


if __name__ == "__main__":
  app.run(main)
