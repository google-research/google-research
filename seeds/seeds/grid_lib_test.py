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
import numpy as np
import xarray as xr

from seeds.seeds import grid_lib


class GridLibTest(absltest.TestCase):

  def test_xyz_to_lonlat(self):
    lon, lat = grid_lib.xyz_to_lonlat(
        np.array([[0, -1, 0], [0, 1e-16, 0], [-1e-16, 0, 0], [0, 0, 1e-16]])
    )

    np.testing.assert_allclose(lon, np.array([270, 90, 180, 0]))
    np.testing.assert_allclose(lat, np.array([0, 0, 0, 90]))

  def test_xyz_lonlat_invariance_on_the_sphere(self):
    lat = np.arange(-90, 90, 7)
    lon = np.linspace(0, 360, len(lat))

    xyz = grid_lib.lonlat_to_xyz(lon, lat)
    output_lon, output_lat = grid_lib.xyz_to_lonlat(xyz)

    np.testing.assert_allclose(output_lat, lat)
    np.testing.assert_allclose(output_lon, lon)

  def test_cubedsphere_num_points_is_correct(self):
    grid = grid_lib.CubedSphere(96)
    lon, _ = grid.grid_points

    self.assertLen(lon, grid.num_points)

  def test_cubedsphere_on_raises_on_wrong_size(self):
    data = np.arange(6 * 7 * 7 + 1)

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'No cubedsphere has 295 grid points.'
    ):
      grid_lib.CubedSphere.on(data)

  def test_cubedsphere_on_correct_size(self):
    n = 7
    data = np.arange(6 * n**2)

    grid = grid_lib.CubedSphere.on(data)

    self.assertEqual(grid.size, n)

  def test_cubedsphere_equiangularity(self):
    n = 17

    lon, _ = grid_lib.CubedSphere(n).grid_points
    one_face = lon.reshape((6, n, n))[0]
    angle_spacing = np.diff(one_face[0, :n])

    np.testing.assert_almost_equal(angle_spacing, angle_spacing[0])

  def test_bilinear_interpolate(self):
    def f(x, y):
      """This is a bilinear function where the interpolation is exact."""
      return x * 3 + x * y - y * 2

    # The source rectangular grid.
    source_x = np.linspace(-1, 1, 6)
    source_y = np.linspace(-1, 1, 4)
    (x, y) = np.meshgrid(source_x, source_y)
    data = f(x, y)
    # Some unstructured points inside the square.
    theta, r = np.meshgrid(np.linspace(-np.pi, np.pi, 10), np.linspace(0, 1, 4))
    target_x = (r * np.cos(theta)).flatten()
    target_y = (r * np.sin(theta)).flatten()

    cache = grid_lib._compute_bilinear_interpolator_cache(
        source_x, source_y, target_x, target_y
    )
    interpolant = grid_lib._eval_bilinear_interpolate(cache, data)
    exact_values = f(target_x, target_y)

    np.testing.assert_allclose(interpolant, exact_values, atol=1e-7)

  def test_equirectangular_on_correct_np_array(self):
    data = np.zeros((33, 64))

    grid = grid_lib.Equirectangular.on(data)

    self.assertEqual(grid.num_points, 33 * 64)

  def test_equirectangular_on_correct_xr_array(self):
    data = xr.DataArray(
        np.zeros((32, 64)),
        dims=('latitude', 'longitude'),
        coords={
            'latitude': np.linspace(-89, 89, 32),
            'longitude': np.linspace(0, 360, 64, endpoint=False),
        },
    )

    grid = grid_lib.Equirectangular.on(data)

    self.assertEqual(grid.num_points, 32 * 64)

  def test_equirectangular_cubedsphere_round_trip_has_small_error(self):
    cubedsphere = grid_lib.CubedSphere(128)
    equirectangular = grid_lib.Equirectangular.from_shape((257, 512))
    lon, lat = np.meshgrid(equirectangular.longitude, equirectangular.latitude)
    x = np.cos(lat / 180 * np.pi) ** 2
    data = x * (1 - x) ** 0.5 * np.cos(3 * lon / 180 * np.pi)

    to = equirectangular.to(cubedsphere)(data)
    back = cubedsphere.to(equirectangular)(to)
    l2_error = np.sqrt(np.sum((data - back) ** 2)) / np.sqrt(np.sum(data**2))

    self.assertLess(l2_error, 0.005)

  def test_vectorized_bilinear_interpolate(self):
    source = grid_lib.Equirectangular.from_shape((33, 64))
    target = grid_lib.CubedSphere(12)
    data = np.zeros((2, 3, 4, *source.data_shape))
    interpolant = source.to(target)(data)

    self.assertEqual(interpolant.shape, (2, 3, 4, target.num_points))

  def test_plot_gridder(self):
    grid = grid_lib.CubedSphere(16)
    data = np.ones(grid.num_points)

    gridder = grid_lib.CubedSphere.on(data).plot_gridder()
    lonlat = gridder(data)
    self.assertEqual(lonlat.shape, (33, 64))
    self.assertTrue(np.allclose(lonlat, 1, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
  absltest.main()
