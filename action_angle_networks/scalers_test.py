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

"""Tests for scalers."""

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from action_angle_networks import scalers


class TrainTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'arr': [[1., 2., 3.], [3., 4., 5.]],
          'arr_scaled': [[-1, -1, -1], [1, 1, 1]],
          'mean': [2., 3., 4.],
          'std': [1., 1., 1.],
      },
      {
          'arr': [[11., 2., 3.], [3., 4., 5.]],
          'arr_scaled': [[1, -1, -1], [-1, 1, 1]],
          'mean': [7., 3., 4.],
          'std': [4., 1., 1.],
      },
  )
  def test_standard_scaler(self, arr,
                           arr_scaled,
                           mean, std):
    arr = np.asarray(arr)
    arr_scaled = np.asarray(arr_scaled)

    scaler = scalers.StandardScaler()
    scaler = scaler.fit(arr)

    self.assertTrue(np.allclose(scaler.transform(arr), arr_scaled))
    self.assertTrue(np.allclose(scaler.inverse_transform(arr_scaled), arr))

    self.assertTrue(np.allclose(scaler.mean(), mean))
    self.assertTrue(np.allclose(scaler.std(), std))

  @parameterized.parameters(
      {
          'arr': [[1., 2., 3.], [3., 4., 5.]],
          'arr_scaled': [[1., 2., 3.], [3., 4., 5.]],
      },
      {
          'arr': [[11., 2., 3.], [3., 4., 5.]],
          'arr_scaled': [[11., 2., 3.], [3., 4., 5.]],
      },
  )
  def test_identity_scaler(self, arr,
                           arr_scaled):
    arr = np.asarray(arr)
    arr_scaled = np.asarray(arr_scaled)

    scaler = scalers.IdentityScaler()
    scaler = scaler.fit(arr)

    self.assertTrue(np.allclose(scaler.transform(arr), arr_scaled))
    self.assertTrue(np.allclose(scaler.inverse_transform(arr_scaled), arr))


if __name__ == '__main__':
  absltest.main()
