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

"""Tests for utilities.ml_utils."""
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from eq_mag_prediction.utilities import ml_utils


class StandardScalerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='one_axis', shape=(2, 3, 4, 5), feature_axes=1),
      dict(testcase_name='two_axes', shape=(2, 3, 4), feature_axes=(1, 2)),
      dict(
          testcase_name='two_nonconsecutive_axes',
          shape=(2, 3, 4),
          feature_axes=(0, 2),
      ),
  )
  def test_standard_scaler(self, shape, feature_axes):
    for seed in [4, 42, 1337]:
      rs = np.random.RandomState(seed=seed)
      x = rs.rand(*shape)
      scaler = ml_utils.StandardScaler(feature_axes)
      self.assertTrue(np.allclose(scaler.transform(x), x))

      scaler.fit(x)
      x_scaled = scaler.transform(x)

      self.assertEqual(x_scaled.shape, x.shape)
      self.assertTrue(np.allclose(x_scaled.mean(feature_axes), 0))
      self.assertTrue(np.allclose(x_scaled.std(feature_axes), 1))

  def test_fit_transform(self):
    shape, axes, seed = (2, 3, 4, 5), (0, 2), 42

    rs = np.random.RandomState(seed=seed)
    x = rs.rand(*shape)
    scaler = ml_utils.StandardScaler(axes)
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    rs = np.random.RandomState(seed=seed)
    x = rs.rand(*shape)
    new_scaler = ml_utils.StandardScaler(axes)
    np.testing.assert_array_almost_equal(x_scaled, new_scaler.fit_transform(x))
    np.testing.assert_array_almost_equal(scaler.mean, new_scaler.mean)
    np.testing.assert_array_almost_equal(scaler.std, new_scaler.std)

  @parameterized.parameters(((0, 1),), (1,))
  def test_standard_scaler_with_zero_std(self, feature_axes):
    array = np.ones((5, 6)) * 888
    scaler = ml_utils.StandardScaler(feature_axes)
    scaler.fit(array)
    scaled = scaler.transform(array)

    self.assertEqual(scaled.shape, array.shape)
    self.assertTrue(np.allclose(scaled.mean(feature_axes), 0))
    # The standard deviation is 0, because the array is constant. Still, the
    # scaling does not raise an exception, and returns a 0-mean array.
    self.assertTrue(np.allclose(scaled.std(feature_axes), 0))

  def test_no_transformation(self):
    shape, seed = (2, 3, 4, 5), 1905
    rs = np.random.RandomState(seed=seed)
    loc = rs.rand(1)*100
    std = rs.rand(1)*100
    x = rs.normal(loc, std, size=shape)
    scaler = ml_utils.StandardScaler(None)
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    np.testing.assert_array_almost_equal(x_scaled, x)


if __name__ == '__main__':
  absltest.main()
