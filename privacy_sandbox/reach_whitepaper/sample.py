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

"""The synthetic data generation for reach."""

import dataclasses
import enum
import functools

import numpy as np
import pandas as pd

from privacy_sandbox.reach_whitepaper import synthetic_dataset


class WindowSize(enum.Enum):
  """Standard sizes of query windows, the value is in days."""

  DAY = 1
  WEEK = 7
  MONTH = 30


class SlicingGranularity(enum.Enum):
  """Standard slicing granularities."""

  LEVEL_1 = 1.01
  """Slicing is over advertiser."""

  LEVEL_2 = 1.1
  """Slicing is over advertiser and campaign."""

  LEVEL_3 = 1.5
  """Slicing is over advertiser, campaign, and geo."""

  LEVEL_4 = 1.6
  """Slicing is over advertiser, campaign, geo, line item, and creative."""


@dataclasses.dataclass
class SliceSize:
  """Observed and true sizes of the slice."""

  true_size: int
  observed_size: int


@functools.cache
def _cap_discount_rate(
    cap,
    cap_discount_shape_b = 1.18,
    cap_discount_scale = 2.14,
):
  """The probability of a user being removed assuming the contribution are capped by cap."""
  if cap < 1:
    raise ValueError("cap must be positive.")

  if cap > 1000:
    raise ValueError("cap must be less than 1000.")

  sample = synthetic_dataset.sample_discrete_power_law(
      b=cap_discount_shape_b,
      x_min=1,
      x_max=1000,
      n_samples=10000,
  )
  dist = pd.Series(sample).value_counts()
  dist = dist.reindex(range(1001), fill_value=0)
  dist /= dist.sum()
  dist *= cap_discount_scale
  return dist[cap]


def _window_discount_rate(n_days, discount_shape=-0.06, scale=1):
  """Return discount rate for a window of n_days days."""
  return scale * np.power(n_days, discount_shape)


def sample_slices(
    window_size,
    cap,
    n_samples,
    slicing_granularity = SlicingGranularity.LEVEL_4,
):
  """Sample sizes of slices.

  Args:
    window_size: Window (in days) that the query is using.
    cap: The maximal number of slices a user might contribute.
    n_samples: Number of samples to draw.
    slicing_granularity: Granular of the slicing.

  Returns:
    A list of size n_samples with the true and observed size of each slice.
  """
  sample_dataset_shape_b = slicing_granularity.value

  reach_1_day = synthetic_dataset.sample_discrete_power_law(
      b=sample_dataset_shape_b,
      x_min=1,
      x_max=1000,
      n_samples=n_samples,
  )

  # stage 1: Apply impact of capping.
  # Compute probability of being reported (success, not impacted by capped)
  # after capping.
  p_reported = 1 - _cap_discount_rate(cap=cap)
  # For a slice of true size n, each contribution has p_reported probability to
  # be reported; thus after capping, observed contributions become
  # np.random.binomial(n, p) (there may be faster approximations).

  reach_1_day_capped = (
      (np.random.binomial(reach_1_day, p_reported)).round().astype(int)
  )

  if isinstance(window_size, WindowSize):
    number_of_days = window_size.value
  else:
    number_of_days = window_size

  # stage 2: Apply impact of time window.
  reach_n_days = np.round(
      reach_1_day * number_of_days * _window_discount_rate(number_of_days)
  ).astype(int)
  reach_n_days_capped = np.round(
      reach_1_day_capped
      * number_of_days
      * _window_discount_rate(number_of_days)
  ).astype(int)

  return [
      SliceSize(uncapped, capped)
      for uncapped, capped in zip(reach_n_days, reach_n_days_capped)
  ]
