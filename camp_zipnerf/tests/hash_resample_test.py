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

"""Tests for hash_resample."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from internal import hash_resample
import numpy as np


class Resample3dTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_centered_fp16',
          half_pixel_center=True,
          dtype=np.float16,
      ),
      dict(
          testcase_name='_centered_fp32',
          half_pixel_center=True,
          dtype=np.float32,
      ),
      dict(
          testcase_name='_uncentered_fp16',
          half_pixel_center=False,
          dtype=np.float16,
      ),
      dict(
          testcase_name='_uncentered_fp32',
          half_pixel_center=False,
          dtype=np.float32,
      ),
  )
  def test_hash_resample_3d_nearest_neighbor_correct(
      self, half_pixel_center, dtype
  ):
    # Generate some sample locations inside and outside of the grid.
    shape = [5, 5, 8]
    data = np.random.uniform(low=0.0, high=1.0, size=[1024, 3]).astype(dtype)
    sample_locations = np.array(shape) * np.random.uniform(
        low=-1, high=2, size=[10000, 3]
    ).astype(dtype)

    fn = functools.partial(
        hash_resample.hash_resample_3d,
        data=data,
    )

    # Nearest neighbor interpolation must match trilinear with rounded inputs.
    np.testing.assert_allclose(
        fn(
            locations=np.floor(sample_locations)
            if half_pixel_center
            else np.round(sample_locations),
            method='TRILINEAR',
            half_pixel_center=False,
        ),
        fn(
            locations=sample_locations,
            method='NEAREST',
            half_pixel_center=half_pixel_center,
        ),
    )


if __name__ == '__main__':
  absltest.main()