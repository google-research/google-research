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

"""Tests functions for synthetic dataset generation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from privacy_sandbox.reach_whitepaper import synthetic_dataset


class SyntheticDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product([1.5, 2, 3], range(1, 10), range(1, 10), range(1, 10))
  )
  def test_produces_correct_number_of_samples(
      self, b, x_min, length, n_samples
  ):
    x_max = x_min + length

    assert (
        len(
            synthetic_dataset.sample_discrete_power_law(
                b, x_min, x_max, n_samples
            )
        )
        == n_samples
    )

  @parameterized.parameters(
      itertools.product([1.5, 2, 3], range(1, 10), range(1, 10), range(1, 10))
  )
  def test_all_values_are_in_range(
      self, b, x_min, length, n_samples
  ):
    x_max = x_min + length

    samples = synthetic_dataset.sample_discrete_power_law(
        b, x_min, x_max, n_samples
    )

    assert (
        min(samples) >= x_min
    ), f"min(samples) = {min(samples)} and x_min = {x_min}"
    assert (
        max(samples) <= x_max
    ), f"max(samples) = {max(samples)} and x_max = {x_max}"


if __name__ == "__main__":
  absltest.main()
