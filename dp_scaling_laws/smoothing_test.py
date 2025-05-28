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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from dp_scaling_laws import smoothing

# pylint: disable=invalid-name


class SmoothingTest(parameterized.TestCase):

  @parameterized.parameters(
      smoothing.cumulative_min,
      smoothing.isotonic,
      smoothing.smooth_second_order,
  )
  def test_monotonic(self, smooth_fn):
    x = np.random.rand(16)
    y = smooth_fn(x)
    np.testing.assert_allclose(y, np.sort(y)[::-1])

  @parameterized.parameters(
      smoothing.cumulative_min,
      smoothing.isotonic,
      smoothing.smooth_second_order,
      smoothing.none,
  )
  def test_idempotent(self, smooth_fn):
    x = np.random.rand(16)
    y = smooth_fn(x)
    np.testing.assert_allclose(y, smooth_fn(y))

  @parameterized.parameters(
      smoothing.rolling_mean,
      smoothing.smooth_second_order,
      smoothing.none,
      smoothing.cumulative_min,
      smoothing.isotonic,
  )
  def test_shapes(self, smooth_fn):
    x = np.random.rand(16)
    y = smooth_fn(x)
    self.assertEqual(y.shape, x.shape)

  @parameterized.parameters(*range(16))
  def test_2d_smooth(self, _):
    Y = np.random.rand(5, 8)
    X = smoothing.smooth_second_order_2d(Y)
    D = X[:-1] - X[1:]
    self.assertEqual(X.shape, Y.shape)
    np.testing.assert_array_less(X[1:], X[:-1] + 1e-10)
    np.testing.assert_array_less(X[:, 1:], X[:, :-1] + 1e-10)
    np.testing.assert_array_less(D[1:], D[:-1] + 1e-10)
    np.testing.assert_allclose(X, smoothing.smooth_second_order_2d(X))


if __name__ == "__main__":
  absltest.main()
