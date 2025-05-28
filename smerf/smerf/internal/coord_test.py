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

"""tests for coord.py."""

import jax.numpy as jnp
import numpy as np

from smerf.internal import configs
from smerf.internal import coord
from smerf.internal import grid_utils
from smerf.internal import utils


def test_dist_transforms():
  # Ensure that tdist <> smdist <> sdist transforms are reversible.
  batch_shape = (3, 4)

  near = 1.0
  far = 1e5
  tdist = np.geomspace(near * 1.01, far * 0.99, num=12).reshape(
      *batch_shape
  )
  sdist = coord.tdist_to_sdist(tdist, near, far)
  assert sdist.shape == tdist.shape
  np.testing.assert_array_less(sdist, 1)
  np.testing.assert_array_less(0, sdist)

  sdist_v2 = sdist
  tdist_v2 = coord.sdist_to_tdist(sdist_v2, near, far)
  np.testing.assert_allclose(tdist, tdist_v2, rtol=1e-2)


def test_submodel_to_world_transforms():
  # Ensure that submodel <> world transforms work as expected.
  config = configs.Config(submodel_grid_resolution=2)  # 2^3 = 8 voxels
  grid_config = grid_utils._calculate_grid_config(config)
  sm_idxs = jnp.arange(8).reshape(8, 1)

  # Each point is at the center of a submodel's coordinate system.
  sm = jnp.zeros((8, 3), dtype=jnp.float32)
  t = coord.submodel_to_world(sm_idxs, sm, config, grid_config)
  sm_v2 = coord.world_to_submodel(sm_idxs, t, config, grid_config)
  assert sm.shape == t.shape == sm_v2.shape

  # pylint: disable=bad-whitespace
  np.testing.assert_allclose(
      t,
      np.array([  # Centers of each submodel coordinate system in t-coordinates.
          [-0.5, -0.5, -0.5],
          [-0.5, -0.5,  0.5],
          [-0.5,  0.5, -0.5],
          [-0.5,  0.5,  0.5],

          [ 0.5, -0.5, -0.5],
          [ 0.5, -0.5,  0.5],
          [ 0.5,  0.5, -0.5],
          [ 0.5,  0.5,  0.5],
      ]),
  )
  # pylint: enable=bad-whitespace
  np.testing.assert_allclose(sm, sm_v2, rtol=1e-2)


def test_world_to_submodel_scale():
  # Ensure that submodel <> world transforms scale the scene as intended.
  config = configs.Config(submodel_grid_resolution=2)  # 2^3 = 8 voxels
  grid_config = grid_utils._calculate_grid_config(config)

  sm_idxs = jnp.array([0], dtype=jnp.int32)
  t = jnp.array([0.1, -0.2, -0.6], dtype=jnp.float32)
  sm = coord.world_to_submodel(sm_idxs, t, config, grid_config)
  t_v2 = coord.submodel_to_world(sm_idxs, sm, config, grid_config)
  np.testing.assert_allclose(
      sm, np.array([1.2, 0.6, -0.2], dtype=np.float32), atol=1e-5
  )
  np.testing.assert_allclose(t, t_v2, atol=1e-2)


def test_rays_to_sm_idxs():
  config = configs.Config(submodel_grid_resolution=2)  # 2^3 = 8 voxels
  grid_config = grid_utils._calculate_grid_config(config)

  def dummy_array(last_dim=1, dtype=np.float32):
    return np.zeros((8, last_dim), dtype=dtype)

  rays = utils.Rays(
      origins=np.array([
          [-0.5, -0.5, -0.5],
          [-0.5, -0.5, 0.5],
          [-0.5, 0.5, -0.5],
          [-0.5, 0.5, 0.5],
          [0.5, -0.5, -0.5],
          [0.5, -0.5, 0.5],
          [0.5, 0.5, -0.5],
          [0.5, 0.5, 0.5],
      ]),
      directions=dummy_array(3),
      viewdirs=dummy_array(3),
      radii=dummy_array(1),
      imageplane=dummy_array(1),
      lossmult=dummy_array(1),
      near=dummy_array(1),
      far=dummy_array(1),
      cam_idx=dummy_array(1, dtype=np.int32),
  )
  sm_idxs = coord.rays_to_sm_idxs(rays, config, grid_config)
  np.testing.assert_allclose(sm_idxs, np.arange(8).reshape(8, 1))


def test_contract_and_uncontract():
  # Ensure that contract and uncontract are reversible and correct.

  # Ensure contract() is correct.
  x = np.array([[
      [0, 0, 0],
      [-0.1, 0.2, 0.3],
      [0.5, -1.0, 2.0],
      [5, 10, -20],
      [50, 100, -200],
      [50, 100, -1_000],
      [50, -1_000, -1_000],
      [50, -999.99, -1_000],
  ]])
  y = coord.contract(x)
  y_expected = np.array([[
      # ||x|| < 1 means no change.
      [0, 0, 0],
      [-0.1, 0.2, 0.3],
      # Divide by ||x|| < 1
      [0.25, -0.5, 1.5],
      # Larger ||x||
      [0.25, 0.5, -1.95],
      [0.25, 0.5, -1.995],
      [0.05, 0.1, -1.999],
      # If |x_i| = |x_j|, then |y_i| = |y_j|
      [0.05, -1.999, -1.999],
      # If |x_i| = |x_j|-eps, then |y_i| != |y_j|
      [0.05, -0.99999, -1.999],
  ]])
  np.testing.assert_allclose(y, y_expected, rtol=1e-4)

  # Ensure uncontract() is the inverse of contract().
  np.testing.assert_allclose(x, coord.uncontract(y), rtol=1e-4)


def test_uncontract_on_large_values():
  # Ensure that uncontract doesn't produce nan and inf values.
  x = np.array([0, 1_000_000_000, -1_000_000_000])
  x2 = coord.uncontract(coord.contract(x), allow_inf=False)
  assert np.all(np.sign(x) == np.sign(x2))
  assert not np.any(np.isnan(x2))
  assert not np.any(np.isinf(x2))
