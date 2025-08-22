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

"""Tests for random_utils."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import numpy as np

from jaxraytrace import random_utils


class TestRandomUtils(parameterized.TestCase):

  @parameterized.product(
      rng_seed=[0, 1],
      num_samples=[100],
      radius=[1, 10],
  )
  def test_random_point_in_sphere(self, rng_seed, num_samples,
                                  radius):
    rng = jax.random.PRNGKey(rng_seed)
    random_points = random_utils.random_points_in_sphere(
        radius, num=num_samples, rng=rng)

    for random_point in random_points:
      self.assertLessEqual(
          np.sum(np.square(random_point)) + 1e-4, np.square(radius))

  @parameterized.product(
      rng_seed=[0, 1],
      num_samples=[100],
      radius=[1, 10],
  )
  def test_random_point_on_sphere(self, rng_seed, num_samples,
                                  radius):
    rng = jax.random.PRNGKey(rng_seed)
    random_points = random_utils.random_points_on_sphere(
        radius, num=num_samples, rng=rng)

    for random_point in random_points:
      self.assertLessEqual(
          np.sum(np.square(random_point)) - 1e-4, np.square(radius))
      self.assertGreaterEqual(
          np.sum(np.square(random_point)) + 1e-4, np.square(radius))


if __name__ == '__main__':
  absltest.main()
