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

"""Test models.py."""

import gin
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from smerf.internal import configs
from smerf.internal import grid_utils
from smerf.internal import models
from smerf.internal import utils


@pytest.fixture
def gin_config():
  """Sets up and tears down a Gin config."""
  gin.parse_config("""
      MultiHashEncoding.hash_map_size = 1024
      MultiPropHashEncoding.hash_map_size = 512
      """)
  yield
  gin.clear_config()


@pytest.mark.parametrize(
    "cls, net_kernels",
    [
        (models.MultiDensityAndFeaturesMLP, 1),
        (models.MultiDensityAndFeaturesMLP, 4),
        (models.MultiPropMLP, 1),
        (models.MultiPropMLP, 4),
    ],
)
def test_mlp(cls, net_kernels, gin_config):  # pylint: disable=redefined-outer-name, unused-argument
  b = 8 * 20
  k = 16

  model = cls(net_kernels=net_kernels, net_hash_kernels=4)

  # Initialize model.
  sm_idxs = jnp.mod(jnp.arange(b), 4).reshape((b // k, k, 1))
  xs = jnp.zeros((b // k, k, 3))
  rng = jax.random.PRNGKey(0)
  params = jax.jit(model.init)(rng, sm_idxs[0:1], xs[0:1])

  # Apply model.
  model_apply_jit = jax.jit(model.apply)
  y = model_apply_jit(params, sm_idxs, xs)

  # Verify shapes.
  if cls == models.MultiDensityAndFeaturesMLP:
    features, density = y
    assert features.shape == (b // k, k, model.num_output_channels)
  elif cls == models.MultiPropMLP:
    density = y
  else:
    raise ValueError(f"Unrecognized model type: {model}")

  assert density.shape == (b // k, k, 1)


def test_query_representation():
  config = configs.Config(
      triplane_resolution=8,
      sparse_grid_resolution=4,
      submodel_grid_resolution=2,
  )
  grid_config = grid_utils._calculate_grid_config(config)

  n = 16  # Number of rays
  k = 3  # Number of points per ray
  s = 8 + 12  # Number of query positions per point.
  sm_idxs = np.arange(n*k).reshape(n, k, 1)
  s_positions = np.random.randn(n, k, 3)

  def density_and_features_mlp(query_sm_idxs, query_s_positions):
    assert query_sm_idxs.shape == (n*k*s, 1)
    assert query_s_positions.shape == (n*k*s, 3)

    # Verify that the same value was broadcasted to all positions, all query
    # points.
    query_sm_idxs = query_sm_idxs.reshape(n*k, s, 1)
    for i in range(n*k):
      np.testing.assert_allclose(query_sm_idxs[i], sm_idxs.flatten()[i])

    features = np.zeros((n*k*s, 7))
    density = np.zeros((n*k*s, 1))
    return features, density

  features, density = models.query_representation(
      sm_idxs, s_positions, config, grid_config, density_and_features_mlp
  )

  assert features.shape == (n, k, 7)
  assert density.shape == (n, k, 1)


def test_one_round_of_hierarchical_sampling():
  config = configs.Config(
      triplane_resolution=8,
      sparse_grid_resolution=4,
      submodel_grid_resolution=2,  # 2^3 = 8 choices
  )
  grid_config = grid_utils._calculate_grid_config(config)

  b = 1
  k = 7

  broadcast_scalar = lambda x: jnp.broadcast_to(x, (b, 1))
  normalize = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
  rays = utils.Rays(
      origins=jnp.array([[0.25, 0.25, -0.25]]),  # [b, 3]
      directions=normalize(jnp.array([[1.0, 0.0, -1.0]])),  # [b, 3]
      viewdirs=normalize(jnp.array([[1.0, 0.0, -1.0]])),  # [b, 3]
      radii=broadcast_scalar(1.0),  # [b, 1]
      imageplane=jnp.array([[0.5, 0.5]]),  # [b, 2]
      lossmult=broadcast_scalar(1.0),  # [b, 1]
      near=broadcast_scalar(1.0),  # [b, 1]
      far=broadcast_scalar(100.0),  # [b, 1]
      cam_idx=broadcast_scalar(0),  # [b, 1]
      exposure_idx=None,
      exposure_values=None,
  )
  sdist = jnp.concatenate(
      [broadcast_scalar(0.0), broadcast_scalar(100.0)], axis=-1
  )  # [b, 2]
  weights = broadcast_scalar(1.0)  # [b, 1]

  # Arguments that won't be changed.
  kwargs = dict(
      rng=jax.random.PRNGKey(0),
      i_level=0,
      num_samples=k,
      prod_num_samples=1,
      train_frac=0.5,
      init_s_near=0.0,
      init_s_far=1.0,
      rays=rays,
      sdist=sdist,
      weights=weights,
      config=config,
      grid_config=grid_config,
  )

  # Test #1: Sample sdist values.
  _, t_positions, _, new_sdist, new_tdist = (
      models.one_round_of_hierarchical_sampling(
          tdist_override=None, **kwargs
      )
  )

  assert t_positions.shape == (b, k, 3)
  assert new_sdist.shape == (b, k+1)
  assert new_tdist.shape == (b, k+1)

  # Test #2: Use tdist_override.
  tdist_override = jnp.linspace(0.0, rays.far[0, 0], num=b * (k + 1))
  tdist_override = tdist_override.reshape(b, k + 1)
  _, t_positions, _, new_sdist, new_tdist = (
      models.one_round_of_hierarchical_sampling(
          tdist_override=tdist_override, **kwargs
      )
  )

  assert t_positions.shape == (b, k, 3)
  assert new_sdist.shape == (b, k+1)
  assert new_tdist.shape == (b, k+1)


def test_maybe_replace_sm_idxs():
  config = configs.Config(
      submodel_grid_resolution=2,  # 2^3 = 8 choices
      submodel_idx_replace_percent=0.1,
  )
  grid_config = grid_utils._calculate_grid_config(config)

  rng = jax.random.PRNGKey(0)
  sm_idxs = np.zeros((256, 4, 1), dtype=np.int32)
  _, new_sm_idxs = models._resample_sm_idxs(
      rng, sm_idxs, 0.5, config, grid_config
  )
  assert sm_idxs.shape == new_sm_idxs.shape

  observed = np.mean(sm_idxs != new_sm_idxs)
  expected = config.submodel_idx_replace_percent
  assert (
      abs(observed - expected) < 0.05
  )
