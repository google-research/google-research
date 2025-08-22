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

"""tests for grid_utils.py."""

import numpy as np
import pytest
from smerf.internal import configs
from smerf.internal import grid_utils


@pytest.mark.parametrize(
    "test_fn, k",
    [
        (grid_utils.triplane_get_eval_positions_and_local_coordinates, 12),
        (grid_utils.sparse_grid_get_eval_positions_and_local_coordinates, 8),
    ],
)
def test_get_eval_positions_and_local_coordinates_helpers(test_fn, k):
  n = 16
  sm_idxs = np.arange(n).reshape(n, 1)
  s_positions = np.random.randn(n, 3)
  voxel_size = 0.5
  axis = 1

  query_sm_idxs, query_s_positions, s_positions_local = (
      test_fn(sm_idxs, s_positions, voxel_size, axis)
  )

  assert query_sm_idxs.shape == (n, k, 1)
  assert query_s_positions.shape == (n, k, 3)
  assert s_positions_local.shape == (n, 3)

  # Ensure that sm_idxs is broadcasted.
  for i in range(n):
    np.testing.assert_allclose(query_sm_idxs[i], i)


def test_get_eval_positions_and_local_coordinates():
  config = configs.Config(triplane_resolution=4, sparse_grid_resolution=2)
  grid_config = grid_utils._calculate_grid_config(config)

  n = 16
  k = 8 + 12
  sm_idxs = np.arange(n).reshape(n, 1)
  s_positions = np.random.randn(n, 3)

  (
      query_sm_idxs,
      query_s_positions,
      triplane_s_positions_local,
      sparse_grid_s_positions_local,
  ) = grid_utils.get_eval_positions_and_local_coordinates(
      sm_idxs, s_positions, config, grid_config
  )

  assert query_sm_idxs.shape == (n*k, 1)
  assert query_s_positions.shape == (n*k, 3)
  assert triplane_s_positions_local.shape == (n, 3)
  assert sparse_grid_s_positions_local.shape == (n, 3)

  # Ensure that sm_idxs is broadcasted.
  for i in range(n):
    np.testing.assert_allclose(query_sm_idxs[i*k:i*(k+1)], i)
