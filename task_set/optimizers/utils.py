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

"""Utilities for sampling optimizers."""

import numpy as np


def sample_log_float(rng, low, high):
  """Sample a float value logrithmically between `low` and `high`.

  Args:
    rng: A NumPy random number generator or `RandomState`.
    low: One end of the range. Must be strictly positive.
    high: The other end of the range. Must be strictly positive.

  Returns:
    A float in the closed interval [min(low, high), max(low, high)].

  Raises:
    ValueError: If `low` or `high` is not strictly positive.
  """
  low = float(low)
  high = float(high)
  if low <= 0.0 or high <= 0.0:
    raise ValueError(
        "sample_log_float requires strictly positive bounds, but got "
        f"low={low!r}, high={high!r}."
    )
  log_low, log_high = np.log(low), np.log(high)
  if log_low > log_high:
    # Accept the bounds in either order. The legacy `RandomState.uniform`
    # silently samples the interval when they are reversed, but
    # `Generator.uniform` raises "high - low < 0".
    log_low, log_high = log_high, log_low
  return float(np.exp(rng.uniform(log_low, log_high)))


def sample_bool(rng, p):
  """Sample a boolean that is True `p` percent of time."""
  if not 0.0 <= p <= 1.0:
    raise ValueError("p must be between 0 and 1.")
  return rng.uniform(0.0, 1.0) < p
