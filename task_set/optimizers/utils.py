# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# python3
"""Utilities for sampling optimizers."""

import numpy as np


def sample_log_float(rng, low, high):
  """Sample a float value logrithmically between `low` and `high`."""
  return float(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))


def sample_bool(rng, p):
  """Sample a boolean that is True `p` percent of time."""
  if not 0.0 <= p <= 1.0:
    raise ValueError("p must be between 0 and 1.")
  return rng.uniform(0.0, 1.0) < p
