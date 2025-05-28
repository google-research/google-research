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

"""Functions to extract a grid/atlas of colors and features from a SNeRG MLP."""

import functools
import gc
import itertools

from internal import coord
from internal import datasets
from internal import grid_utils
from internal import math
from internal import models
from internal import render
from internal import train_utils
import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage
import tqdm


def create_mlp_p(state):
  mlp = models.DensityAndFeaturesMLP()
  params = state.params['params']['DensityAndFeaturesMLP_0']
  params = jax.tree.map(lambda x: jnp.array(x), params)  # pylint: disable=unnecessary-lambda

  def mlp_fn(positions):
    return mlp.apply({'params': params}, positions)

  return jax.pmap(mlp_fn, in_axes=(0))


def evaluate_mlp(mlp_p, positions):
  """Evaluates the DensityAndFeaturesMLP.

  Args:
    mlp_p: Pmapped function for MLP evalution.
    positions: 3D coordinates.

  Returns:
    8-dimensional vector consisting of RGB color, density and 4-dimnensional
    appearance feature vector.

  Does not apply any activation functions on the outputs.
  """
  num_devices = jax.device_count()
  sh = positions.shape[:-1]

  # Prepare for multi GPU evaluation.
  positions = positions.reshape((-1, 3))
  actual_num_inputs = positions.shape[0]
  rounded_num_inputs = (
      np.ceil(actual_num_inputs / num_devices).astype(int) * num_devices)

  # Adds padding.
  positions = positions.copy()
  positions.resize((rounded_num_inputs, 3))
  positions = positions.reshape((num_devices, -1, 3))

  features, density = mlp_p(positions)

  def remove_padding_and_reshape(x):
    x = x.reshape((-1, x.shape[-1]))[:actual_num_inputs]
    return x.reshape(sh + (x.shape[-1],))

  features = remove_padding_and_reshape(features)
  density = remove_padding_and_reshape(density)
  return features, density


def compute_alive_voxels(
    state,
    dataset,
    config,
    grid_config,
    alpha_threshold,
    weight_threshold,
    use_alpha_culling,
    subsampling_factor,
    use_only_first_image,
):
  """Performs visiblity culling.

  Args:
    state: Model state.
    dataset: Dataset.
    config: Config parameters.
    grid_config: Grid config parameters.
    alpha_threshold: Threshold on alpha value.
    weight_threshold: Threshold on volume rendering weight.
    use_alpha_culling: Whether to use alpha culling.
    subsampling_factor: Which fraction of the training dataset to use.
    use_only_first_image: If true, only uses the first image for debugging.

  Returns:
    A grid that indicates which voxels are alive.

  We only consider a voxel as occupied if it contributed siginficantly to the
  rendering of any training image. We therefore render the entire training set
  and record the sampling positions predicted by the ProposalMLP. For any
  sampled point with a sufficently high volume rendering weight and alpha value
  we mark the 8 adjacent voxels as occupied.

  We are able to instantiate a full-res occupancy grid that is stored on the
  CPU.
  """

  voxel_size = grid_config['voxel_size_to_use']
  grid_size = [grid_config['resolution_to_use']] * 3

  alive_voxels = np.zeros(grid_size, dtype=bool)
  cpu = jax.local_devices(backend='cpu')[0]
  to_cpu = lambda x: jax.device_put(x, cpu)

  _, _, render_eval_pfn, _, _ = train_utils.setup_model(
      config, jax.random.PRNGKey(20200823), dataset, return_ray_results=True
  )
  train_frac = 1.0
  rngs = jax.random.split(jax.random.PRNGKey(0), jax.local_device_count())

  if use_only_first_image:
    train_indices = [0]
  else:
    train_indices = np.random.RandomState(42).choice(
        len(dataset.images),
        size=(len(dataset.images) // subsampling_factor,),
        replace=False,
    )
  for train_idx in tqdm.tqdm(list(train_indices)):
    tqdm.tqdm.write(f'\nRendering training view {train_idx}')

    rays = datasets.cam_to_rays(dataset, train_idx)
    rendering = models.render_image(
        functools.partial(render_eval_pfn, state.params, train_frac),
        rays,
        rngs[0],
        config,
        verbose=False,
        transfer_to_cpu=True,
    )

    density = to_cpu(rendering['density'])
    tdist = to_cpu(rendering['tdist'])
    weights = to_cpu(rendering['weights'])
    del rendering
    gc.collect()

    positions = render.get_sample_positions_along_ray(
        tdist, to_cpu(rays.origins), to_cpu(rays.directions), to_cpu(rays.radii)
    )  # BxSx3.
    num_samples = positions.shape[-2]

    # Filter points based on volume rendering weight (computed based on
    # step size used during training).
    alive_samples = weights > weight_threshold  # BxS.
    positions_alive = positions[alive_samples].reshape(-1, 3)  # Ax3.
    viewdirs_alive = np.repeat(
        to_cpu(rays.viewdirs)[Ellipsis, np.newaxis, :], repeats=num_samples, axis=-2
    )  # BxSx3.
    viewdirs_alive = viewdirs_alive[alive_samples].reshape(-1, 3)  # Ax3.

    # Step sizes used for density to alpha computation. The deeper we are in
    # squashed space the larger the steps become during rendering.
    if use_alpha_culling:
      step_sizes = coord.stepsize_in_squash(
          positions_alive, viewdirs_alive, voxel_size)  # A

    positions_alive = coord.contract(positions_alive)

    # Filter points based on alpha values (computed based on step size used
    # during rendering).
    if use_alpha_culling:
      density_alive = density[alive_samples].reshape(-1)  # A.
      alpha = math.density_to_alpha(density_alive, step_sizes)  # A.
      positions_alive = positions_alive[alpha > alpha_threshold]

    positions_alive = grid_utils.world_to_grid(positions_alive, voxel_size, np)
    positions_alive_0 = np.floor(positions_alive).astype(np.int32)
    positions_alive_1 = np.ceil(positions_alive).astype(np.int32)

    def remove_out_of_bound_points(x):
      mask = (x >= 0).all(1) & (x < grid_config['resolution_to_use']).all(1)
      return x[mask]

    # Splat to eight voxel corners.
    corner_coords = [[False, True] for _ in range(3)]
    for z in itertools.product(*corner_coords):
      l = []
      for i, b in enumerate(z):
        l.append(positions_alive_1[Ellipsis, i] if b else positions_alive_0[Ellipsis, i])
      positions_corner = np.stack(l, axis=-1)
      positions_corner = remove_out_of_bound_points(positions_corner)
      alive_voxels[
          positions_corner[:, 0], positions_corner[:, 1], positions_corner[:, 2]
      ] = True

  return alive_voxels


def get_atlas_block_size(data_block_size):
  """Add 1-voxel apron for native trilerp in the WebGL renderer."""
  return data_block_size + 1


def bake_triplane(
    state,
    config,
    grid_config,
    batch_size,
    alive_voxels,
):
  """Bakes triplanes."""
  triplane_voxel_size = grid_config['triplane_voxel_size']
  triplane_resolution = config.triplane_resolution
  planes_features = []
  planes_density = []

  # Setup MLP (doing this inside the loop may cause re-jitting).
  mlp_p = create_mlp_p(state)

  for plane_idx in range(3):
    print('baking plane', plane_idx)
    spans = [
        np.arange(triplane_resolution) if plane_idx != c else np.array([0.0])
        for c in range(3)
    ]
    positions_grid = np.stack(np.meshgrid(*spans, indexing='ij'), axis=-1)
    positions = grid_utils.grid_to_world(
        positions_grid, triplane_voxel_size, np).reshape(-1, 3)
    num_texels = positions.shape[0]
    plane = np.zeros((num_texels, models.NUM_CHANNELS + 1), dtype=np.float32)
    for batch_start in tqdm.tqdm(list(range(0, num_texels, batch_size))):
      batch_end = min(batch_start + batch_size, num_texels)
      features, density = evaluate_mlp(mlp_p, positions[batch_start:batch_end])
      plane[batch_start:batch_end, : models.NUM_CHANNELS] = features
      plane[batch_start:batch_end, [-1]] = density
      del features, density
      gc.collect()
    plane = plane.reshape((triplane_resolution, triplane_resolution, -1))
    # Project alive voxels on plane and set dead texels to zero. Otherwise
    # dead texels would contain random values that cannot be compressed well
    # with PNG.
    mask_2d = alive_voxels.any(axis=plane_idx)
    mask_2d = ~scipy.ndimage.maximum_filter(mask_2d, size=25)  # dilate heavily
    plane[mask_2d] = -1000000.0
    planes_features.append(plane[Ellipsis, : models.NUM_CHANNELS])
    planes_density.append(plane[Ellipsis, [-1]])
  return planes_features, planes_density


def bake_sparse_grid(
    state,
    grid_config,
    alive_macroblocks,
    data_block_size,
    batch_size_in_blocks,
):
  """Bakes sparse grid."""
  sparse_grid_voxel_size = grid_config['sparse_grid_voxel_size']

  atlas_block_size = get_atlas_block_size(data_block_size)
  num_occupied_blocks = alive_macroblocks.sum()

  def create_atlas_1d(n):
    sh = (
        num_occupied_blocks,
        atlas_block_size,
        atlas_block_size,
        atlas_block_size,
        n,
    )
    return np.zeros(sh, dtype=np.float32)

  sparse_grid_features_1d = create_atlas_1d(models.NUM_CHANNELS)
  sparse_grid_density_1d = create_atlas_1d(1)

  # Setup MLP (doing this inside the loop would cause re-jitting).
  mlp_p = create_mlp_p(state)

  # Baking.
  min_voxel = (
      np.stack(np.nonzero(alive_macroblocks), axis=-1).astype(np.float32)
      * data_block_size
  )
  for block_start in tqdm.tqdm(
      list(range(0, num_occupied_blocks, batch_size_in_blocks))
  ):
    block_end = min(block_start + batch_size_in_blocks, num_occupied_blocks)

    # Build AxAxA grid for each block.
    min_voxel_batch = min_voxel[block_start:block_end]  # Bx3.
    span = np.arange(atlas_block_size)  # A.
    x_grid = np.stack(
        np.meshgrid(span, span, span, indexing='ij'), axis=-1
    )  # AxAxAx3.
    x_grid = (
        min_voxel_batch[:, np.newaxis, np.newaxis, np.newaxis, :]
        + x_grid[np.newaxis]
    )  # Bx1x1x1x3 + 1xAxAxAx3 = BxAxAxAx3.

    # Evaluate MLP.
    positions = grid_utils.grid_to_world(
        x_grid, sparse_grid_voxel_size, np
    )  # BxAxAxAx3.
    features, density = evaluate_mlp(mlp_p, positions)  # BxAxAxAx7, BxAxAxAx3.
    sparse_grid_features_1d[block_start:block_end] = features
    sparse_grid_density_1d[block_start:block_end] = density
    del features, density, min_voxel_batch, positions
    gc.collect()
  return sparse_grid_features_1d, sparse_grid_density_1d


def reshape_into_3d_atlas_and_compute_indirection_grid(
    sparse_grid_features_1d,
    sparse_grid_density_1d,
    data_block_size,
    alive_macroblocks,
):
  """Reshapes into 3D atlas and computes indirection grid."""
  atlas_block_size = get_atlas_block_size(data_block_size)
  num_occupied_blocks = sparse_grid_features_1d.shape[0]

  # Find 3D texture dimensions with lowest storage impact with a brute
  # force search.
  def compute_az(ax, ay):
    num_blocks_per_atlas_unit = ax * ay
    az = int(np.ceil(num_occupied_blocks / num_blocks_per_atlas_unit))
    return az, ax * ay * az

  best_num_occupied_blocks_padded = np.inf
  for ax_cand, ay_cand in itertools.product(range(1, 255), range(1, 255)):
    az, num_occupied_blocks_padded = compute_az(ax_cand, ay_cand)

    # Make sure that the volume texture does not exceed 2048^3 since
    # some devices do not support abitrarily large textures (although the limit
    # is usally higher than 2048^3 for modern devices). Also make sure that
    # resulting indices will be smaller than 255 since we are using a single
    # byte to encode the indices and the value 255 is reserved for empty blocks.
    if (
        num_occupied_blocks_padded < best_num_occupied_blocks_padded
        and az < 255
        and ax_cand * atlas_block_size <= 2048
        and ay_cand * atlas_block_size <= 2048
        and az * atlas_block_size <= 2048
    ):
      ax, ay = ax_cand, ay_cand
      best_num_occupied_blocks_padded = num_occupied_blocks_padded
  az, num_occupied_blocks_padded = compute_az(ax, ay)

  # Make sure that last dim is smallest. During .png export we slice this volume
  # along the last dim and sorting ensures that not too many .pngs are created.
  ax, ay, az = sorted([ax, ay, az], reverse=True)

  # Add padding, if necessary.
  required_padding = num_occupied_blocks_padded - num_occupied_blocks
  if required_padding > 0:

    def add_padding(x):
      padding = np.zeros((required_padding,) + x.shape[1:])
      return np.concatenate([x, padding], axis=0)

    sparse_grid_features_1d = add_padding(sparse_grid_features_1d)
    sparse_grid_density_1d = add_padding(sparse_grid_density_1d)

  # Reshape into 3D texture.
  def reshape_into_3d_texture(x):
    x = x.reshape(
        ax, ay, az,
        atlas_block_size, atlas_block_size, atlas_block_size, x.shape[-1])
    x = x.swapaxes(2, 3).swapaxes(1, 2).swapaxes(3, 4)
    return x.reshape(
        ax * atlas_block_size, ay * atlas_block_size, az * atlas_block_size,
        x.shape[-1],
    )

  sparse_grid_features = reshape_into_3d_texture(sparse_grid_features_1d)
  sparse_grid_density = reshape_into_3d_texture(sparse_grid_density_1d)

  # Compute indirection grid.
  block_indices_compact = np.arange(num_occupied_blocks)
  block_indices_compact = np.unravel_index(block_indices_compact, [ax, ay, az])
  block_indices_compact = np.stack(block_indices_compact, axis=-1)
  index_grid_size = alive_macroblocks.shape
  sparse_grid_block_indices = -1 * np.ones(
      (index_grid_size[0], index_grid_size[1], index_grid_size[2], 3), np.int16
  )
  sparse_grid_block_indices[alive_macroblocks] = block_indices_compact

  return sparse_grid_features, sparse_grid_density, sparse_grid_block_indices
