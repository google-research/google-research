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

"""Script for running a single top-k experiment and saving its plots."""

from collections.abc import Sequence
import pickle

from absl import app
from absl import flags
import numpy as np

from dp_mm_domain import topk_experiment


_K_RANGE = flags.DEFINE_list(
    "k_range",
    None,
    "A comma-separated list of k values to try.",
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

_L0_BOUND = flags.DEFINE_float(
    "l0_bound",
    100,
    "The l0_bound to use for the experiment.",
)

_K_BAR_MULTIPLIER_RANGE = flags.DEFINE_list(
    "k_bar_multiplier_range",
    None,
    "A comma-separated list of k_bar_multiplier values to try. Only used for"
    " LIMITED_DOMAIN method.",
)

# TODO: output path needs be changed before moving out of experimental.
_OUTPUT_PATH = flags.DEFINE_string(
    "output_file",
    "./topk_results",
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

  k_range = [int(k) for k in _K_RANGE.value]
  epsilon = _EPSILON.value
  delta = _DELTA.value

  k_bar_multiplier_range = None
  if _K_BAR_MULTIPLIER_RANGE.value is not None:
    k_bar_multiplier_range = [
        float(k_bar_multiplier)
        for k_bar_multiplier in _K_BAR_MULTIPLIER_RANGE.value
    ]

  num_trials = _NUM_TRIALS.value
  output_path = _OUTPUT_PATH.value
  data_path = _DATA_PATH.value
  l0_bound = _L0_BOUND.value

  input_methods = [
      topk_experiment.TopKMethod.WGM_THEN_PEEL,
      topk_experiment.TopKMethod.LIMITED_DOMAIN,
  ]

  with open(data_path, "rb") as f:
    input_data = pickle.load(f)

  if _SEED.value is not None:
    np.random.seed(_SEED.value)

  results = topk_experiment.compare_methods(
      input_data,
      input_methods,
      k_range,
      epsilon,
      delta,
      l0_bound,
      num_trials,
      k_bar_multiplier_range,
  )

  topk_experiment.plot_results(
      results, k_range, epsilon, delta, output_path
  )


if __name__ == "__main__":
  app.run(main)
