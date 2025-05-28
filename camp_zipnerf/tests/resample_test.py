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

"""Tests for resample."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from internal import resample
import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate


class Resample3dTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data = np.random.uniform(low=0.0, high=1.0, size=[5, 5, 8, 3]).astype(
        np.float32
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_xyz_no_flatten',
          coordinate_order='xyz',
          flatten=False,
      ),
      dict(testcase_name='_xyz_flatten', coordinate_order='xyz', flatten=True),
      dict(
          testcase_name='_zyx_no_flatten',
          coordinate_order='zyx',
          flatten=False,
      ),
      dict(testcase_name='_zyx_flatten', coordinate_order='zyx', flatten=True),
  )
  def test_resample_3d(self, coordinate_order, flatten):
    sample_locations = np.array(
        [
            [
                [[2.1, 0.2, 0.4], [1.1, 2.1, 3.2]],
                [[0.7, 1.1, 3.1], [2.7, 3.3, 2.1]],
                [[2.5, 2.1, 1.4], [3.1, 3.1, 1.3]],
            ],
            [
                [[0.8, 2.3, 1.25], [2.9, 1.6, 0.3]],
                [[1.4, 1.5, 1.6], [0.3, 2.1, 2.2]],
                [[3.3, 0.1, 0.7], [2.4, 1.25, 1.5]],
            ],
        ],
        dtype=np.float32,
    )

    def _maybe_flatten(loc):
      if flatten:
        return jnp.reshape(loc, [loc.shape[0], -1, loc.shape[-1]])
      else:
        return loc

    actual_resampled = resample.resample_3d(
        self.data,
        _maybe_flatten(sample_locations),
        edge_behavior='CONSTANT_OUTSIDE',
        constant_values=0.0,
        coordinate_order=coordinate_order,
    )
    if flatten:
      actual_resampled = actual_resampled.reshape(
          sample_locations.shape[:-1] + self.data.shape[-1:]
      )
    if coordinate_order == 'xyz':
      sample_locations = np.flip(sample_locations, axis=-1)
    fn = interpolate.RegularGridInterpolator(
        (
            np.arange(0, self.data.shape[0]),
            np.arange(0, self.data.shape[1]),
            np.arange(0, self.data.shape[2]),
        ),
        self.data,
    )
    expected_resampled = fn(sample_locations)

    np.testing.assert_allclose(
        expected_resampled, actual_resampled, rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(testcase_name='_constant_outside', edge_behavior='CONSTANT_OUTSIDE'),
      dict(testcase_name='_clamp', edge_behavior='CLAMP'),
  )
  def test_resample_3d_edges_zero_outside(self, edge_behavior):
    d, h, w = (
        self.data.shape[0] - 1,
        self.data.shape[1] - 1,
        self.data.shape[2] - 1,
    )
    df, hf, wf = float(d), float(h), float(w)

    sample_locations = np.array(
        [
            [
                [[-0.5, 0.0, 0.0], [wf + 0.5, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, hf + 0.5, 0.0]],
            ],
            [
                [[-0.5, 0.0, -0.5], [wf + 0.5, 0.0, -0.5]],
                [[0.0, -0.5, -0.5], [0.0, hf + 0.5, -0.5]],
            ],
            [
                [[-0.5, 0.0, df + 0.5], [wf + 0.5, 0.0, df + 0.5]],
                [[0.0, -0.5, df + 0.5], [0.0, hf + 0.5, df + 0.5]],
            ],
            [
                [[-0.5, 0.0, df + 0.5], [wf + 0.5, 0.0, df + 0.5]],
                [[0.0, -0.5, df + 0.5], [0.0, hf + 0.5, df + 0.5]],
            ],
        ],
        dtype=np.float32,
    )
    constant = 42.0
    actual_resampled = resample.resample_3d(
        self.data,
        sample_locations,
        edge_behavior=edge_behavior,
        constant_values=constant,
        coordinate_order='xyz',
    )
    if edge_behavior == 'CONSTANT_OUTSIDE':
      # Pad the input with the constant value.
      padded_data = np.pad(
          self.data,
          ((1, 1), (1, 1), (1, 1), (0, 0)),
          mode='constant',
          constant_values=constant,
      )
    elif edge_behavior == 'CLAMP':
      # Duplicate the input with the edge pixels. This works because the
      # "outside" values are outside by < 1 pixel.
      padded_data = np.pad(
          self.data, ((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge'
      )
    fn = interpolate.RegularGridInterpolator(
        (
            np.arange(0, self.data.shape[0] + 2),
            np.arange(0, self.data.shape[1] + 2),
            np.arange(0, self.data.shape[2] + 2),
        ),
        padded_data,
    )

    zyx_samples = np.flip(sample_locations, axis=-1)
    expected_resampled = fn(zyx_samples + 1.0)

    np.testing.assert_allclose(
        expected_resampled, actual_resampled, rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_constant_outside_centered',
          edge_behavior='CONSTANT_OUTSIDE',
          half_pixel_center=True,
      ),
      dict(
          testcase_name='_constant_outside_uncentered',
          edge_behavior='CONSTANT_OUTSIDE',
          half_pixel_center=False,
      ),
      dict(
          testcase_name='_clamp_centered',
          edge_behavior='CLAMP',
          half_pixel_center=True,
      ),
      dict(
          testcase_name='_clamp_uncentered',
          edge_behavior='CLAMP',
          half_pixel_center=False,
      ),
  )
  def test_resample_3d_nearest_neighbor_matches_quantized_trilinear(
      self, edge_behavior, half_pixel_center
  ):
    # Generate some sample locations inside and outside of the grid.
    sample_locations = np.array(self.data.shape[:-1]) * np.random.uniform(
        low=-1, high=2, size=[10000, 3]
    ).astype(np.float32)

    fn = functools.partial(
        resample.resample_3d,
        data=self.data,
        edge_behavior=edge_behavior,
        constant_values=42.0,
        coordinate_order='xyz',
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

  @parameterized.named_parameters(
      dict(
          testcase_name='_constant_outside_centered',
          edge_behavior='CONSTANT_OUTSIDE',
          half_pixel_center=True,
      ),
      dict(
          testcase_name='_constant_outside_uncentered',
          edge_behavior='CONSTANT_OUTSIDE',
          half_pixel_center=False,
      ),
      dict(
          testcase_name='_clamp_centered',
          edge_behavior='CLAMP',
          half_pixel_center=True,
      ),
      dict(
          testcase_name='_clamp_uncentered',
          edge_behavior='CLAMP',
          half_pixel_center=False,
      ),
  )
  def test_resample_3d_nearest_neighbor_matches_trilinear_convolution(
      self, edge_behavior, half_pixel_center
  ):
    """Test that conv(nearest, tent) matches conv(trilinear, box)."""
    # Construct a 3D grid with a delta spike at (d, d, d).
    d = 4
    shape = [2 * d] * 3 + [1]
    data = np.zeros(shape)
    data[d, d, d, 0] = 1.0

    # Linspace the area around the spike.
    xx = jnp.arange(d - 2, d + 2, 0.1)
    locations = jnp.stack(jnp.meshgrid(xx, xx, d * jnp.ones([1])), axis=-1)

    # Construct a box filter.
    f_box = np.zeros([17, 17])
    f_box[4:-4, 4:-4] = 1.0

    # Construct a tent filter.
    f_tent = 1 - jnp.abs(jnp.arange(-8, 9) / 9)
    f_tent = f_tent[:, None] * f_tent[None, :]

    # Interpolate with both methods.
    fn = functools.partial(
        resample.resample_3d,
        data=data,
        locations=locations,
        edge_behavior=edge_behavior,
        constant_values=42.0,
        coordinate_order='xyz',
        half_pixel_center=half_pixel_center,
    )

    trilerp = fn(method='TRILINEAR')[Ellipsis, 0, 0]
    nearest = fn(method='NEAREST')[Ellipsis, 0, 0]

    # Convolve each interpolation result with the other filter.
    trilerp_conv = jax.scipy.signal.convolve2d(trilerp, f_box)
    nearest_conv = jax.scipy.signal.convolve2d(nearest, f_tent)

    # Normalize to get rid of scale factors.
    trilerp_conv /= jnp.sum(trilerp_conv)
    nearest_conv /= jnp.sum(nearest_conv)

    np.testing.assert_allclose(trilerp_conv, nearest_conv, atol=1e-3)


if __name__ == '__main__':
  absltest.main()