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

"""Tests for task_set.optimizers.utils."""

from absl.testing import absltest
import numpy as np

from task_set.optimizers import utils


class SampleLogFloatTest(absltest.TestCase):

  def test_samples_within_range_ascending(self):
    rng = np.random.RandomState(0)
    for _ in range(1000):
      value = utils.sample_log_float(rng, 1e-6, 1e-3)
      self.assertGreaterEqual(value, 1e-6)
      self.assertLessEqual(value, 1e-3)

  def test_samples_within_range_descending(self):
    """Bounds given the other way round sample the same interval."""
    rng = np.random.RandomState(0)
    for _ in range(1000):
      value = utils.sample_log_float(rng, 1e-3, 1e-6)
      self.assertGreaterEqual(value, 1e-6)
      self.assertLessEqual(value, 1e-3)

  def test_works_with_generator(self):
    """np.random.Generator rejects reversed bounds; the helper must not."""
    rng = np.random.default_rng(0)
    value = utils.sample_log_float(rng, 1e-3, 1e-6)
    self.assertGreaterEqual(value, 1e-6)
    self.assertLessEqual(value, 1e-3)

    ascending = utils.sample_log_float(rng, 1e-6, 1e-3)
    self.assertGreaterEqual(ascending, 1e-6)
    self.assertLessEqual(ascending, 1e-3)

  def test_rejects_non_positive_bounds(self):
    """A zero or negative bound is reported, not an obscure OverflowError."""
    rng = np.random.RandomState(0)
    for low, high in [(0.0, 1e-3), (-1.0, 1e-3), (1e-6, 0.0), (1e-6, -1.0)]:
      with self.subTest(f'low={low} high={high}'):
        with self.assertRaises(ValueError):
          utils.sample_log_float(rng, low, high)

  def test_accepts_integer_bounds(self):
    rng = np.random.RandomState(0)
    value = utils.sample_log_float(rng, 1, 100)
    self.assertGreaterEqual(value, 1.0)
    self.assertLessEqual(value, 100.0)

  def test_single_point_range(self):
    rng = np.random.RandomState(0)
    self.assertAlmostEqual(utils.sample_log_float(rng, 1e-3, 1e-3), 1e-3)


class SampleBoolTest(absltest.TestCase):

  def test_fraction_true(self):
    rng = np.random.RandomState(0)
    draws = [utils.sample_bool(rng, 0.25) for _ in range(20000)]
    self.assertAlmostEqual(sum(draws) / len(draws), 0.25, delta=0.02)

  def test_rejects_probability_outside_unit_interval(self):
    rng = np.random.RandomState(0)
    for p in (-0.1, 1.1):
      with self.subTest(f'p={p}'):
        with self.assertRaises(ValueError):
          utils.sample_bool(rng, p)


if __name__ == '__main__':
  absltest.main()
