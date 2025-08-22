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

"""Computes the errors of algorithms computing reach directly."""

import itertools
import multiprocessing
import pathlib
from typing import Sequence

from absl import app
from absl import flags
from numpy import random
import pandas as pd

from privacy_sandbox.reach_whitepaper import constants
from privacy_sandbox.reach_whitepaper import sample

_CAPS = flags.DEFINE_multi_integer("caps", range(1, 5), "Caps to use.")
_REPETITIONS = flags.DEFINE_integer(
    "repetitions", 100, "Number of repetitions."
)
_OUTPUT = flags.DEFINE_string("output", None, "Output file.", required=True)


def sample_estimated_cardinality(epsilon, count, cap):
  """Compute estimate of the cardinality given true cardinality."""

  return random.laplace(count, 1.0 * cap / epsilon)


def _estimate(input_):
  return sample_estimated_cardinality(*input_)


def _compute_direct_errors(
    caps,
    repetitions,
    output,
):
  """Compute the errors of the sketching algorithm."""
  errors = []

  for window_size, cap, slicing_granularity in itertools.product(
      sample.WindowSize, caps, sample.SlicingGranularity
  ):
    slices = sample.sample_slices(
        window_size,
        cap,
        n_samples=repetitions,
        slicing_granularity=slicing_granularity,
    )
    with multiprocessing.Pool() as pool:
      estimates = pool.map(
          _estimate,
          [
              (
                  constants.EPSILON
                  / (len(sample.WindowSize) * len(sample.SlicingGranularity)),
                  slice_size.observed_size,
                  cap,
              )
              for slice_size in slices
          ],
      )
    for estimate, slice_size in zip(estimates, slices):
      errors.append([
          window_size,
          cap,
          slicing_granularity,
          slice_size.observed_size,
          slice_size.true_size,
          estimate,
      ])
  df = pd.DataFrame(
      data=errors,
      columns=[
          "window_size",
          "cap",
          "observed_size",
          "true_size",
          "estimate",
      ],
  )
  df.to_csv(output)

  print(f"The errors are saved to {output}")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  _compute_direct_errors(
      _CAPS.value,
      _REPETITIONS.value,
      pathlib.Path(_OUTPUT.value).resolve(),
  )


if __name__ == "__main__":
  app.run(main)
