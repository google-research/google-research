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

"""Computes the errors of the sketching algorithm."""

import multiprocessing
import pathlib
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
from numpy import random
import pandas as pd

from privacy_sandbox.reach_whitepaper import constants
from privacy_sandbox.reach_whitepaper import sample
from privacy_sandbox.reach_whitepaper import sketches


SKETCH_SIZE = 10_000

_CAPS = flags.DEFINE_multi_integer("caps", range(1, 5), "Caps to use.")
_REPETITIONS = flags.DEFINE_integer(
    "repetitions", 100, "Number of repetitions."
)
_WINDOW_SIZES = flags.DEFINE_multi_integer(
    "window_sizes", range(1, 362, 10), "Window sizes."
)
_OUTPUT = flags.DEFINE_string("output", None, "Output file.", required=True)


class LaplaceNoiser:
  """Noiser adding Laplace noiser to each sketch bucket.

  Attributes:
    epsilon: The privacy budget.
    cap: The sensitivity of the Laplace noise.
  """

  def __init__(self, epsilon, cap):
    self.epsilon = epsilon
    self.cap = cap

  def __call__(
      self, n_slices_and_window_size, n_registers
  ):
    """Generates Laplace noise for the given number of repetitions, window size, and registers.

    Args:
      n_slices_and_window_size: A tuple of the number of repetitions and the
        window size.
      n_registers: The number of registers.

    Returns:
      A numpy array of Laplace noise.
    """
    return random.laplace(
        0,
        1.0 * self.cap / self.epsilon,
        size=(
            n_slices_and_window_size[0],
            n_slices_and_window_size[1],
            n_registers,
        ),
    )


def uniform_distribution(n_registers):
  """Uniform distribution over registers."""
  return [1.0 / n_registers] * n_registers


def _estimate(input_):
  """This is a utility function to use together with multiprocessing.map."""

  return sketches.sample_estimated_cardinality(
      input_[0],
      input_[1],
      input_[2],
      input_[3],
  )


def _compute_sketching_errors(
    caps,
    repetitions,
    window_sizes,
    output,
):
  """Computes the errors of the sketching algorithm."""
  errors = []
  register_probabilities = uniform_distribution(SKETCH_SIZE)

  for cap in caps:
    noiser = LaplaceNoiser(constants.EPSILON, cap)
    for window_size in window_sizes:
      slices = sample.sample_slices(window_size, cap, n_samples=repetitions)
      with multiprocessing.Pool() as pool:
        estimates = pool.map(
            _estimate,
            [
                (
                    slice_size.observed_size,
                    window_size,
                    noiser,
                    register_probabilities,
                )
                for slice_size in slices
            ],
        )
      for estimate, slice_size in zip(estimates, slices):
        errors.append([
            SKETCH_SIZE,
            window_size,
            cap,
            slice_size.observed_size,
            slice_size.true_size,
            estimate,
        ])
  df = pd.DataFrame(
      data=errors,
      columns=[
          "sketch_size",
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

  _compute_sketching_errors(
      _CAPS.value,
      _REPETITIONS.value,
      _WINDOW_SIZES.value,
      pathlib.Path(_OUTPUT.value).resolve(),
  )


if __name__ == "__main__":
  app.run(main)
