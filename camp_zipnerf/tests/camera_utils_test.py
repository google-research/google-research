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

"""Tests for camera_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from internal import camera_utils
from internal import utils
from jax import random
import jax.numpy as jnp
import numpy as np
import scipy


def _create_test_camera_tuple(height=48, width=64, focal=50.0):
  rng = random.PRNGKey(0)

  # Set the resolution and focal length.
  intrinsic = camera_utils.intrinsic_matrix(
      focal, focal, width / 2.0, height / 2.0
  )
  inv_intrinsic = np.linalg.inv(intrinsic)

  # Randomized camera orientation (camera-to-world).
  key, rng = random.split(rng)
  extrinsic = camera_utils.viewmatrix(*random.normal(key, (3, 3)))

  # Randomized distortion parameters.
  key, rng = random.split(rng)
  distortion_params_list = random.uniform(key, (5,), minval=-0.01, maxval=0.01)
  distortion_params_list = np.array(distortion_params_list)
  distortion_params_names = ['k1', 'k2', 'k3', 'p1', 'p2']
  distortion_params = {
      k: x for k, x in zip(distortion_params_names, distortion_params_list)
  }
  return inv_intrinsic, extrinsic, distortion_params


class CameraUtilsTest(parameterized.TestCase):

  def test_convert_to_ndc(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      # Random pinhole camera intrinsics.
      key, rng = random.split(rng)
      focal, width, height = random.uniform(
          key, (3,), minval=100.0, maxval=200.0
      )
      camtopix = camera_utils.intrinsic_matrix(
          focal,
          focal,
          width / 2.0,
          height / 2.0,
      )
      pixtocam = np.linalg.inv(camtopix)
      near = 1.0

      # Random rays, pointing forward (negative z direction).
      num_rays = 1000
      key, rng = random.split(rng)
      origins = jnp.array([0.0, 0.0, 1.0])
      origins += random.uniform(key, (num_rays, 3), minval=-1.0, maxval=1.0)
      directions = jnp.array([0.0, 0.0, -1.0])
      directions += random.uniform(key, (num_rays, 3), minval=-0.5, maxval=0.5)

      # Project world-space points along each ray into NDC space.
      t = jnp.linspace(0.0, 1.0, 10)
      pts_world = origins + t[:, None, None] * directions
      pts_ndc = jnp.stack(
          [
              -focal / (0.5 * width) * pts_world[Ellipsis, 0] / pts_world[Ellipsis, 2],
              -focal / (0.5 * height) * pts_world[Ellipsis, 1] / pts_world[Ellipsis, 2],
              1.0 + 2.0 * near / pts_world[Ellipsis, 2],
          ],
          axis=-1,
      )

      # Get NDC space rays.
      origins_ndc, directions_ndc = camera_utils.convert_to_ndc(
          origins, directions, pixtocam, near
      )

      # Ensure that the NDC space points lie on the calculated rays.
      directions_ndc_norm = jnp.linalg.norm(
          directions_ndc, axis=-1, keepdims=True
      )
      directions_ndc_unit = directions_ndc / directions_ndc_norm
      projection = ((pts_ndc - origins_ndc) * directions_ndc_unit).sum(axis=-1)
      pts_ndc_proj = origins_ndc + directions_ndc_unit * projection[Ellipsis, None]

      # pts_ndc should be close to their projections pts_ndc_proj onto the rays.
      np.testing.assert_allclose(pts_ndc, pts_ndc_proj, atol=1e-5, rtol=1e-5)

  def test_points_to_pixels(self):
    """Check that points_to_pixels() is the inverse of pixels_to_rays()."""
    height, width, focal = 48, 64, 50.0
    inv_intrinsic, extrinsic, distortion_params = _create_test_camera_tuple(
        height=height, width=width, focal=focal
    )

    # Compute our rays.
    pix_x_int, pix_y_int = np.meshgrid(
        np.arange(width), np.arange(height), indexing='xy'
    )
    origins, directions = camera_utils.pixels_to_rays(
        pix_x_int,
        pix_y_int,
        inv_intrinsic,
        extrinsic,
        distortion_params,
        xnp=jnp,
    )[:2]

    # Project out to 3D points at random depths.
    key = random.PRNGKey(1)
    depths = random.uniform(key, (height, width, 1), minval=1.0, maxval=10.0)
    points = origins + directions * depths

    # Reproject into original camera frame and check coordinates are the same.
    coordinates, _ = camera_utils.points_to_pixels(
        points, inv_intrinsic, extrinsic, distortion_params, xnp=jnp
    )
    np.testing.assert_allclose(
        coordinates[Ellipsis, 0], pix_x_int, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        coordinates[Ellipsis, 1], pix_y_int, atol=1e-5, rtol=1e-5
    )

  @parameterized.product(
      projection_type=[
          camera_utils.ProjectionType.PERSPECTIVE,
          camera_utils.ProjectionType.FISHEYE,
      ],
  )
  def test_tuple_jax_camera_conversion_round_trip(
      self,
      projection_type,
  ):
    """Tests that the camera tuple survives a round trip conversion."""
    height, width = 48, 64
    camera_tuple = _create_test_camera_tuple(height=48, width=64)
    jax_camera = camera_utils.jax_camera_from_tuple(
        camera_tuple,
        jnp.array([width, height]),
        projection_type=projection_type,
    )
    camera_tuple_rt = camera_utils.tuple_from_jax_camera(jax_camera)
    # Set 4th distortion coefficient since JAX camera will set to zero if not
    # present.
    camera_tuple[2]['k4'] = 0.0

    chex.assert_trees_all_close(camera_tuple, camera_tuple_rt, rtol=1e-5)

  def test_safe_interpolate_1d(self):
    """Tests that safe_interpolate_1d works when n >= k+1."""
    x = np.array([1, 2, 3, 2, 1, 2, 3], dtype=np.float32)
    t_input = np.array([0, 1, 2, 8, 9, 10, 100], dtype=np.float32)
    t_output = t_input + 0.5
    y = camera_utils.safe_interpolate_1d(x, 5, 20, t_input, t_output)

    # Mimic scipy.
    tck = scipy.interpolate.splrep(t_input, x, s=20, k=5)
    y_expected = scipy.interpolate.splev(t_output, tck).astype(x.dtype)
    np.testing.assert_allclose(y, y_expected)

  def test_safe_interpolate_1d_too_few_points(self):
    """Tests that safe_interpolate_1d works when n < k+1."""
    x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    t_input = np.arange(len(x), dtype=np.float32)
    t_output = t_input / 2.
    y = camera_utils.safe_interpolate_1d(x, 5, 20, t_input, t_output)

    # Mimic scipy with a lower polynomial degree.
    tck = scipy.interpolate.splrep(t_input, x, s=20, k=4)
    y_expected = scipy.interpolate.splev(t_output, tck).astype(x.dtype)
    np.testing.assert_allclose(y, y_expected)

  def test_safe_interpolate_1d_empty_input(self):
    """Tests that safe_interpolate_1d works when n = 0."""
    x = t_input = np.array([], dtype=np.float32)
    t_output = np.array([1, 2, 3], dtype=np.float32)
    y = camera_utils.safe_interpolate_1d(x, 5, 20, t_input, t_output)

    # Expect constant value
    y_expected = np.array([0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(y, y_expected)

  @parameterized.named_parameters(
      ('inside_box', [-1.0] * 3, [1.0] * 3, 0.1, 2.0, True),
      ('outside_box', [-1.0, -1.0, 0.5], [1.0, 1.0, 2.0], 1.0, 4.0, True),
      ('box_behind', [-1.0] * 3, [-0.5] * 3, 0.0, 0.0, False),
      ('box_miss', [5.0] * 3, [6.0] * 3, 0.0, 0.0, False),
  )
  def test_modify_rays_with_bbox(self, cmin, cmax, near, far, valid):
    """Tests that modify_rays_with_bbox works."""
    # Test on a "4x4" image to check if shapes come out ok.
    batch = lambda x: np.tile(np.array(x).reshape((1, 1, -1)), (4, 4, 1))

    rays = utils.Rays(
        origins=batch([0.0, 0.0, 0.0]),
        # nb: expected results are in units of direction, so twice as long as
        # you might think!
        directions=batch([0.0, 0.0, 0.5]),
        viewdirs=batch([0.0, 0.0, 1.0]),
        near=batch([0.1]),
        far=batch([10.0]),
    )
    corners = np.array([cmin, cmax])
    rays = camera_utils.modify_rays_with_bbox(rays, corners)
    np.testing.assert_allclose(rays.near, batch(near))
    np.testing.assert_allclose(rays.far, batch(far))
    np.testing.assert_allclose(rays.lossmult, batch(valid))

if __name__ == '__main__':
  absltest.main()
