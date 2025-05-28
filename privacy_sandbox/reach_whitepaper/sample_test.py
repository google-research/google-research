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

"""Test the function for sampling slices."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from privacy_sandbox.reach_whitepaper import sample


class SampleTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          itertools.chain(range(1, 361, 10), sample.WindowSize),
          range(1, 11),
          range(1, 1001, 100),
      )
  )
  def test_produces_correct_number_of_samples(
      self, window_size, cap, n_samples
  ):
    assert len(sample.sample_slices(window_size, cap, n_samples)) == n_samples

  @parameterized.parameters(
      itertools.product(
          itertools.chain(range(1, 361, 10), sample.WindowSize),
          range(1, 1001, 100),
      )
  )
  def test_zero_cap_fails(self, window_size, n_samples):
    with self.assertRaises(ValueError):
      assert sample.sample_slices(window_size, 0, n_samples)

  @parameterized.parameters(
      itertools.product(
          itertools.chain(range(1, 361, 10), sample.WindowSize),
          range(1001, 1101),
          range(1, 1001, 100),
      )
  )
  def test_large_cap_fails(self, window_size, cap, n_samples):
    with self.assertRaises(ValueError):
      assert sample.sample_slices(window_size, cap, n_samples)


if __name__ == "__main__":
  absltest.main()
