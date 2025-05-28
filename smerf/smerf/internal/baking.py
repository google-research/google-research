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
# pylint: disable=logging-fstring-interpolation

import dataclasses
import functools
import gc
import itertools

from absl import logging
import chex
from etils import etqdm
import flax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import scipy.ndimage
import skimage.measure
from smerf.internal import coord
from smerf.internal import datasets
from smerf.internal import distill
from smerf.internal import grid_utils
from smerf.internal import math
from smerf.internal import models
from smerf.internal import quantize
from smerf.internal import render
from smerf.internal import train_utils
from smerf.internal import utils


def _create_mlp_pmap(pparams, config, grid_config):
  """Creates state for _evaluate_mlp_pmap()."""
  # Build model.
  # TODO(duckworthd): Unify with simliar logic in models.py.
  model = models.MultiDensityAndFeaturesMLP(
      net_kernels=grid_config['num_submodel_mlp_kernels'],
      net_hash_kernels=grid_config['num_submodel_hash_encoding_kernels'],
  )

  # Extract model's parameters.
  pparams = {'params': pparams['params']['MultiDensityAndFeaturesMLP_0']}

  return (model, pparams, config, grid_config)


def _evaluate_mlp_pmap(state, sm_idxs, s_positions):
  """Evaluates the DensityAndFeaturesMLP in batches.

  Args:
    state: See _create_mlp().
    sm_idxs: i32[..., 1]. Which submodels to query.
    s_positions: f32[..., 3]. Positions in squash coordinates.

  Returns:
    features: f32[..., 7]. 7-dimensional vector consisting of RGB color
      and 4-dimensional appearance feature vector. Sigmoid not yet applied.
    density: f32[..., 1]. 1-dimensional vector consisting of density. Density
      activation function not yet applied.
  """
  assert sm_idxs.shape[:-1] == s_positions.shape[:-1]

  # Unpack model and its state.
  model, pparams, config, grid_config = state

  # Convert sm_idxs to param_idxs.
  param_idxs = coord.sm_idxs_to_params_idxs(sm_idxs, config, grid_config)

  # Prepare inputs for pmap.
  param_idxs, batch_info = utils.pre_pmap(param_idxs, 1)
  s_positions, _ = utils.pre_pmap(s_positions, 1)

  # Apply model.
  features, density = _evaluate_mlp_pmap_helper(
      pparams, param_idxs, s_positions, model
  )

  # Remove padding and reshape.
  features = utils.post_pmap(features, batch_info)
  density = utils.post_pmap(density, batch_info)

  return features, density


@functools.partial(
    jax.pmap,
    in_axes=(0, 0, 0),
    static_broadcasted_argnums=(3,)
)
def _evaluate_mlp_pmap_helper(
    # Array arguments
    params,  # 0
    param_idxs,  # 1
    s_positions,  # 2
    # Constant arguments
    model,  # 3
):
  """Helper function for _evaluate_mlp_pmap()."""
  return model.apply(params, param_idxs, s_positions)


@functools.partial(
    jax.pmap,
    in_axes=(0, None, 0),
    static_broadcasted_argnums=range(3, 8),
)
def _compute_alive_voxels_grid_positions_pmap(
    # Array arguments
    rays,  # 0
    sm_idx,  # 1
    rendering,  # 2
    # Constant arguments
    use_alpha_culling,  # 3
    alpha_threshold,  # 4
    weight_threshold,  # 5
    config,  # 6
    grid_config,  # 7
):
  """Computes voxel positions for use in compute_alive_voxels()."""
  voxel_size = grid_config['voxel_size_to_use']

  density = rendering['density']  # BxS
  tdist = rendering['tdist']  # Bx(S+1)
  weights = rendering['weights']  # BxS

  # Save shape for later.
  batch_shape = density.shape

  # Determine points sampled along in world coordinates. These are the points
  # proposed by a PropMLP.
  t_positions = render.get_sample_positions_along_ray(
      tdist, rays.origins, rays.directions, rays.radii
  )  # BxSx3.

  # Broadcast directions along 'S' dimension.
  t_directions = jnp.broadcast_to(
      rays.directions[Ellipsis, jnp.newaxis, :], t_positions.shape
  )  # BxSx3
  t_viewdirs = jnp.broadcast_to(
      rays.viewdirs[Ellipsis, jnp.newaxis, :], t_positions.shape
  )  # BxSx3

  # Filter points based on volume rendering weight (computed based on
  # step size used during training). This will be enforced at the very
  # end of this computation.
  is_valid = (weights > weight_threshold).reshape(-1)  # A
  t_positions = t_positions.reshape(-1, 3)  # Ax3
  t_directions = t_directions.reshape(-1, 3)  # Ax3
  t_viewdirs = t_viewdirs.reshape(-1, 3)  # Ax3
  chex.assert_equal_shape_prefix(
      (t_positions, t_directions, t_viewdirs, is_valid), 1
  )

  # Convert to submodel coordinates, then squash coordinates.
  sm_positions = coord.world_to_submodel(
      sm_idxs=sm_idx,
      t=t_positions,
      config=config,
      grid_config=grid_config,
  )  # Ax3

  # World and submodel coordinates share the same direction vectors.
  sm_viewdirs = t_viewdirs  # Ax3.

  # Step sizes used for density to alpha computation. The deeper we are in
  # squashed space the larger the steps become during rendering.
  t_step_sizes = None
  if use_alpha_culling:
    # Compute step sizes in submodel coordinates used by the real-time
    # rendered. A fixed step size in squash coordinates is used.
    sm_step_sizes = coord.sm_stepsize_from_s_stepsize(
        sm_positions, sm_viewdirs, voxel_size
    )  # A

    # Convert from step sizes in submodel coordinates to step sizes in
    # world coordinates.
    t_step_sizes = coord.sm_dist_to_world_dist(
        sm_idxs=sm_idx,
        sm_dist=sm_step_sizes,
        config=config,
        grid_config=grid_config,
    )  # A
    chex.assert_equal_shape((sm_step_sizes, t_step_sizes))

  # Compute positions in squash coordinates.
  s_positions = coord.contract(sm_positions)  # Ax3
  chex.assert_equal_shape((t_positions, sm_positions, s_positions))

  # Filter points based on alpha values (computed based on step size used
  # during realtime rendering).
  if use_alpha_culling:
    density = density.reshape(-1)  # A.
    alpha = math.density_to_alpha(density, t_step_sizes)  # A.
    chex.assert_equal_shape_prefix(
        (s_positions, t_step_sizes, density, alpha), 1
    )
    is_valid = is_valid & (alpha > alpha_threshold)  # A

  # Calculate positions in voxel space.
  grid_positions = grid_utils.world_to_grid(
      s_positions, voxel_size, jnp
  ).reshape(*batch_shape, 3)  # BxSx3
  is_valid = is_valid.reshape(*batch_shape)  # BxS

  return grid_positions, is_valid


@functools.partial(jax.pmap, in_axes=(0,), static_broadcasted_argnums=(1,))
def _median_filter_chunk_pmap(x, s):
  """Applies median filter to a chunk of inputs."""
  # Compute offsets.
  o = jnp.arange(-s, s + 1, dtype=jnp.int32)
  o = jnp.meshgrid(o, o, o, indexing='ij')
  o = jnp.stack(o, axis=-1).reshape(1, -1, 3)

  # Get size of our filter_size x filter_size x filter_size window.
  n = o.shape[1]

  # Compute indices.
  sh = x.shape
  p = [jnp.arange(s, q - s) for q in sh]
  p = jnp.meshgrid(*p, indexing='ij')
  p = jnp.stack(p, axis=-1).reshape(-1, 1, 3)
  p = o + p
  p = p.reshape(-1, 3)

  # Fetch filter_size x filter_size x filter_size windows.
  v = x[p[:, 0], p[:, 1], p[:, 2]]
  v = v.reshape(-1, n)
  v = jnp.median(v, axis=-1).astype(x.dtype)

  # Unflatten the result.
  osh = np.array(x.shape) - 2 * s
  v = v.reshape(osh)
  return v


def _itertools_batched(elems, n):
  """Mimics itertools.batched."""
  result = []
  for elem in elems:
    result.append(elem)
    if len(result) == n:
      yield tuple(result)
      result = []
  if result:
    yield tuple(result)


@jax.jit
def _apply_3x3x3_maximum_filter(x):
  """Behaves like scipy.ndimage.maximum_filter with mode='constant' and size=3'.

  Computes the 3x3x3 dilation of the input volume using the ML accelerators
  instead of on the CPU.

  Args:
    x: bool[n,n,n].  Array to apply maximum filter to.

  Returns:
    y: bool[n,n,n]. Result of applying a 3x3x3 maximum filter.
  """
  # The filter is separable, do one dimension at a time.
  dilated_x = jnp.maximum(
      x,
      jnp.pad(x, ((1, 0), (0, 0), (0, 0)), mode='constant')[:-1, :, :],
  )
  dilated_x = jnp.maximum(
      dilated_x,
      jnp.pad(x, ((0, 1), (0, 0), (0, 0)), mode='constant')[1:, :, :],
  )

  dilated_y = jnp.maximum(
      dilated_x,
      jnp.pad(dilated_x, ((0, 0), (1, 0), (0, 0)), mode='constant')[:, :-1, :],
  )
  dilated_y = jnp.maximum(
      dilated_y,
      jnp.pad(dilated_x, ((0, 0), (0, 1), (0, 0)), mode='constant')[:, 1:, :],
  )

  dilated_z = jnp.maximum(
      dilated_y,
      jnp.pad(dilated_y, ((0, 0), (0, 0), (1, 0)), mode='constant')[:, :, :-1],
  )
  dilated_z = jnp.maximum(
      dilated_z,
      jnp.pad(dilated_y, ((0, 0), (0, 0), (0, 1)), mode='constant')[:, :, 1:],
  )
  return dilated_z


def _apply_median_filter(x, filter_size, chunk_size=None):
  """Behaves like scipy.ndimage.median_filter with mode='reflect'.

  Computes the median of each [filter_size, filter_size, filter_size] patch of
  `x` using all devices available. The input is split into blocks of shape
  [chunk_size, chunk_size, chunk_size] to avoid OOMs.

  Args:
    x: bool[n,n,n].  Array to apply median filter to.
    filter_size: int. f for a filter of size [f,f,f].
    chunk_size: int. k for chunks of size [k,k,k].

  Returns:
    y: bool[n,n,n]. Result of applying a median filter.
  """
  if chunk_size is None:
    chunk_size = x.shape[0]
  assert x.ndim == 3, 'x must be 3D array'
  assert (
      np.array(x.shape) % chunk_size == 0
  ).all(), 'chunk_size must divide x.shape'
  assert (
      filter_size % 2 == 1 and filter_size >= 3
  ), 'filter_size must be be an uneven integer greater or equal to 3'

  k = chunk_size

  # Output has the same size and dtype as input.
  y = np.empty_like(x)

  s = int((filter_size - 1) / 2)
  pad_width = [(s, s) for _ in range(3)]

  # The 'symmetric' mode corresponds to 'reflect' in
  # scipy.ndimage.median_filter.
  x = np.pad(x, pad_width, mode='symmetric')

  # Setup chunks.
  r = [range(0, u, chunk_size) for u in y.shape]

  # Iterate over batches of m <= num_devices chunks such that each device has
  # something to work on.
  num_devices = jax.local_device_count()
  for cl_batch in _itertools_batched(itertools.product(*r), num_devices):
    # Lower bound for this batch.
    cl_batch = np.array(cl_batch, dtype=np.int32)  # i32[m,3]
    m, _ = cl_batch.shape

    # Upper bound for this batch
    ch_batch = cl_batch + chunk_size
    chex.assert_shape(ch_batch, (m, 3))

    # Fetch chunks from the input.
    cx_batch = []
    for cl, ch in zip(cl_batch, ch_batch):
      # Fetch a padded chunk.
      ch_s = ch + 2 * s
      cx = x[cl[0] : ch_s[0], cl[1] : ch_s[1], cl[2] : ch_s[2]]
      cx_batch.append(cx)
    cx_batch = np.stack(cx_batch, axis=0)  # bool[m,k,k,k]
    chex.assert_shape(cx_batch, (m, k+2*s, k+2*s, k+2*s))

    # Filter the chunk.
    cx_batch = _median_filter_chunk_pmap(cx_batch, s)  # bool[m,k,k,k]
    chex.assert_shape(cx_batch, (m, k, k, k))

    # Transfer back to CPU
    cx_batch = jax.device_get(cx_batch)

    # Write into result tensor.
    for cl, ch, cx in zip(cl_batch, ch_batch, cx_batch):
      y[cl[0] : ch[0], cl[1] : ch[1], cl[2] : ch[2]] = cx

  return y


def apply_median_filter(alive_voxels, config):
  """Applies a median filter to alive_voxels if needed.

  Args:
    alive_voxels: bool[n,n,n] array. True if a pixel is eligible for querying
       at baked rendering time.
    config: configs.Config instance.

  Returns:
    alive_voxels: bool[n,n,n]. Alive voxels to use for all further computations.
  """
  if config.baking_alive_voxels_median_filter_size is None:
    return alive_voxels

  # Applies a median filter to alive_voxels. This should suppress floaters
  # and fill holes.
  filter_size = config.baking_alive_voxels_median_filter_size
  chunk_size = config.baking_alive_voxels_median_chunk_size
  logging.info(
      'Applying median filter with filter_size=%s, chunk_size=%s',
      filter_size,
      chunk_size,
  )
  alive_voxels = _apply_median_filter(
      alive_voxels, filter_size=filter_size, chunk_size=chunk_size
  )
  return alive_voxels


def _compute_alive_voxels(
    render_fn,
    dataset,
    config,
    grid_config,
    sm_idx,
    *,
    max_images=None,
    use_alpha_culling=False,
    alpha_threshold=None,
    weight_threshold=None,
    dtype=None,
    xnp=None,
):
  """Performs visiblity culling.

  Args:
    render_fn: Function to pass to models.render_image() for rendering.
    dataset: Training dataset.
    config: MERF's Config parameters.
    grid_config: Grid config parameters derived from config.
    sm_idx: int. Which submodel to compute alive voxels for.
    max_images: int. Maximum number of images to render.
    use_alpha_culling: If True, apply alpha culling logic.
    alpha_threshold: Threshold on alpha value.
    weight_threshold: Threshold on volume rendering weight.
    dtype: dtype for result.
    xnp: numpy or jax.numpy.

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
  if alpha_threshold is None or not use_alpha_culling:
    alpha_threshold = 0.

  if weight_threshold is None or not use_alpha_culling:
    weight_threshold = 0.

  if dtype is None:
    dtype = bool

  if xnp is None:
    xnp = jnp

  grid_size = [grid_config['resolution_to_use']] * 3

  # Instantiate dense voxel grid. Because this grid can be large, we store it
  # in host memory with regular numpy.
  alive_voxels = np.zeros(grid_size, dtype=dtype)

  def update_alive_voxels(idxs, values):
    """Updates alive_voxels."""
    i = idxs[Ellipsis, 0]
    j = idxs[Ellipsis, 1]
    k = idxs[Ellipsis, 2]
    alive_voxels[i, j, k] += values

  def is_in_bounds(x):
    """True if an index is in the bounds of the voxel grid."""
    lower = xnp.all(x >= 0, axis=-1)
    upper = xnp.all(x < grid_config['resolution_to_use'], axis=-1)
    return lower & upper

  # Decide on the number of images to render.
  if isinstance(max_images, (tuple, list)):
    train_idxs = list(max_images)
  else:
    train_idxs = _compute_alive_voxels_choose_views(
        dataset, sm_idx, max_images, config, grid_config
    )

  logging.info(
      f'Rendering {len(train_idxs)} images for alive_voxels computation.'
  )

  num_images_processed = 0
  for train_idx in train_idxs:
    # Increment number of processed images.
    logging.info(
        f'Processing view #{train_idx} ({num_images_processed} of'
        f' {len(train_idxs)})...'
    )
    num_images_processed += 1

    # Prepare rays for rendering. Ray casting happens here.
    # TODO(duckworthd): Cast rays with GPUs.
    rays = datasets.cam_to_rays(dataset, train_idx, xnp=jnp)  # f32[H, W, C]
    rays = datasets.preprocess_rays(
        rays=rays, mode='test', merf_config=config, dataset=dataset
    )

    if config.baking_subsample_factor is not None:
      # Subsample rays.
      # pylint: disable=cell-var-from-loop
      step = config.baking_subsample_factor
      every_nth = lambda x: x[::step, ::step]
      rays = jax.tree_util.tree_map(every_nth, rays)
      # pylint: enable=cell-var-from-loop

    if config.baking_enable_ray_jitter:
      # Insert dummy patch dims.
      insert_patch_dims = lambda x: x.reshape(*x.shape[0:-1], 1, 1, x.shape[-1])
      rays = jax.tree_util.tree_map(insert_patch_dims, rays)

      # Jitter rays.
      # TODO(duckworthd): Use pjitter_rays to leverage all devices.
      rng = jax.random.PRNGKey(train_idx)
      rays = train_utils.jitter_rays(rng, rays, config, strict=False)

      # Remove patch dims.
      remove_patch_dims = lambda x: x.reshape(*x.shape[0:-3], x.shape[-1])
      rays = jax.tree_util.tree_map(remove_patch_dims, rays)

    # Override sm_idx for all rays.
    sm_idxs = jnp.full_like(rays.sm_idxs, sm_idx)
    rays = dataclasses.replace(rays, sm_idxs=sm_idxs)

    # Rays are processed in batches. Iterate over the output batches.
    batches = _compute_alive_voxels_grid_positions_batched(
        render_fn=render_fn,
        rng=jax.random.PRNGKey(42),
        rays=rays,
        sm_idx=sm_idx,
        use_alpha_culling=use_alpha_culling,
        alpha_threshold=alpha_threshold,
        weight_threshold=weight_threshold,
        config=config,
        grid_config=grid_config,
    )
    # grid_positions  : f32[D,R,S,3]
    # is_valid        : bool[D,R,S]
    # batch_info      : See utils.pre_pmap().
    for grid_positions, is_valid, batch_info in batches:
      # Remove buffer entries. This will move arrays to device 0.
      #   i32[B,S,3], bool[B,S] for B <= D*R
      # pylint: disable=cell-var-from-loop
      post_pmap = lambda x: utils.post_pmap(x, batch_info, xnp=xnp)
      grid_positions = post_pmap(grid_positions)
      is_valid = post_pmap(is_valid)
      # pylint: enable=cell-var-from-loop

      # Splat to eight voxel corners. Each living point marks its 8
      # corresponding corners as active.
      corner_coords = [[False, True] for _ in range(3)]
      for corner in itertools.product(*corner_coords):
        # Construct voxel indices to update.
        grid_positions_corner = _compute_alive_voxels_grid_positions_corner(
            grid_positions, corner, xnp
        )  # i32[D,R,S,3]

        # Some coordinates may lie outside of the voxel boundaries. If so, mark
        # these updates as invalid. This operation will be distributed across
        # GPUs if its inputs are.
        #   bool[D,R,S].
        is_valid_corner = is_valid & is_in_bounds(grid_positions_corner)

        # Move to host RAM. For some reason, performing the gather on device
        # with a boolean array triggers a JIT compile.
        grid_positions_corner = jax.device_get(grid_positions_corner)
        is_valid_corner = jax.device_get(is_valid_corner)

        # Drop invalid entries. This isn't strictly necessary, but it reduces
        # the number of locations to reference when updating alive_voxels.
        #   f32[K,3], bool[K] where K = sum(is_valid_corner)
        grid_positions_corner = grid_positions_corner[is_valid_corner]
        is_valid_corner = is_valid_corner[is_valid_corner]

        # Update alive_voxels. All valid voxel corners will be marked as alive.
        update_alive_voxels(grid_positions_corner, is_valid_corner)
        del grid_positions_corner, is_valid_corner

  return alive_voxels


def _compute_alive_voxels_grid_positions_batched(
    render_fn,
    rng,
    rays,
    sm_idx,
    use_alpha_culling,
    alpha_threshold,
    weight_threshold,
    config,
    grid_config,
):
  """Computes grid positions for points along camera rays.

  Computes positions for samples along camera rays, as proposed by a PropMLP,
  in the coordinate system of alive_voxels.

  Rays are processed in batches up to a maximum size defined in 'config'.
  Each position is returned, alongside a boolean value indicating whether or
  not it should be used to mark a position as 'alive' in an occupancy grid.

  An effort is made to keep JAX arrays partitioned across all available GPUs.

  In the following docstring, the following shape annotations are used,
    C: Number of channels.
    D: number of local devices.
    R: Per-device batch size.
    S: Number of samples per ray.

  **WARNING!!** Yielded outputs may contain buffer entries!! Apply
  utils.post_pmap(x, batch_info) before interpreting output quantities.

  Args:
    render_fn: Function for rendering rays. Matches signature from
      models.render_image().
    rng: Random seed for rendering. Passed to render_fn.
    rays: Rays with entries of shape [..., C].
    sm_idx: int. Which submodel to use for rendering.
    use_alpha_culling: If True, apply alpha culling logic.
    alpha_threshold: Threshold on alpha value.
    weight_threshold: Threshold on volume rendering weight.
    config: MERF's Config parameters.
    grid_config: Grid config parameters derived from config.

  Yields:
    grid_positions_batch: f32[D, R, S, 3]. Grid positions corresponding to
      samples along a camera ray. Values are in [0, resolution_to_use).
    is_valid_batch: bool[D, R, S]. If the corresponding grid position should
      be used to mark an alive_voxel.
    batch_info: Use with utils.post_pmap() to remove buffer entries.
  """
  batch_dims = rays.origins.shape[:-1]
  num_rays = np.prod(batch_dims)
  num_rays_per_batch = (
      config.render_chunk_size // config.gradient_accumulation_steps
  )

  # Merge batch dims for all inputs
  flatten_fn = lambda x: x.reshape(-1, x.shape[-1])
  rays = jax.tree_util.tree_map(flatten_fn, rays)  # f32[N,C]

  for start in range(0, num_rays, num_rays_per_batch):
    # Get a batch of rays.
    # pylint: disable=cell-var-from-loop
    stop = min(num_rays, start + num_rays_per_batch)
    rays_batch = jax.tree_util.tree_map(
        lambda x: x[start:stop], rays
    )  # f32[B,C]
    # pylint: enable=cell-var-from-loop

    # Prepare for pmap'd render.
    _, batch_info = utils.pre_pmap(rays_batch.origins, ndims=1)
    pre_pmap = lambda x: utils.pre_pmap(x, ndims=1)[0]
    rays_batch = jax.tree_util.tree_map(pre_pmap, rays_batch)  # f32[D,R,C]

    # Render batch of rays. These outputs will be partitioned across devices.
    # render_fn() applies jax.lax.all_gather() at the end of its computation.
    # {str: f32[D,D,R,C]}
    rendering_batch, _ = render_fn(rng, rays_batch)

    # Undo the all-gather. Each device now has the rays it calculated.
    # {str: f32[D,R,C]}
    rendering_batch = {
        k: undo_all_gather(v)
        for k, v in rendering_batch.items()
        if k in ['density', 'tdist', 'weights']
    }

    # Compute voxel indices to update. These outputs will be partitioned across
    # devices.
    # f32[D,R,S,3], bool[D,R,S]
    grid_positions_batch, is_valid_batch = (
        _compute_alive_voxels_grid_positions_pmap(
            rays_batch,
            sm_idx,
            rendering_batch,
            bool(use_alpha_culling),
            float(alpha_threshold),
            float(weight_threshold),
            config,
            grid_config,
        )
    )

    # Yield result
    yield grid_positions_batch, is_valid_batch, batch_info


def undo_all_gather(x):
  """The inverse of jax.lax.all_gather()."""
  # Leading two dimensions must equal #devices. After all_gather(),
  # x[i] == x[j] for all i, j.
  n = jax.local_device_count()
  chex.assert_shape(x, (n, n, Ellipsis))

  # Fetch x[i, i] for each device i.
  i = jnp.arange(n)
  return _undo_all_gather(x, i)


@functools.partial(jax.pmap, in_axes=(0, 0))
def _undo_all_gather(x, i):
  return x[i]


def _compute_alive_voxels_grid_positions_corner(
    grid_positions,  # 0
    corner,  # 1
    xnp=jnp,  # 2
):
  """Computes voxel corner indices for grid positions.

  Which corner is specified by three boolean values, indicating whether to use
  the ceiling (True) or floor (False).

  Args:
    grid_positions: f32[..., 3]. Positions in voxel coordinates. Values in
      [0, max)
    corner: (bool, bool, bool). Specifies which corner to return.
    xnp: numpy or jax.numpy

  Returns:
    i32[..., 3]. Integer coordinates of the target voxel corner corresponding
      to each grid position. Values in {0, 1, ..., max-1}.
  """
  result = []
  for i, b in enumerate(corner):
    floor_or_ceil = xnp.ceil if b else xnp.floor
    idx = floor_or_ceil(grid_positions[Ellipsis, i]).astype(xnp.int32)
    result.append(idx)
  return xnp.stack(result, axis=-1)


def _compute_alive_voxels_choose_views(
    dataset, sm_idx, max_images, config, grid_config
):
  """Chooses which views to render for alive_voxels computation.

  Returns up to `max_images` image idxs for rendering. If there are more images
  available than desired, the images closest to the submodel origin are
  returned. Distance is measured wrt a camera's origin.

  Args:
    dataset: Dataset instance.
    sm_idx: int. Which submodel to choose images for.
    max_images: int. Maximum number of images to return.
    config: ...
    grid_config: ...

  Returns:
    i32[max_images]. image idxs to render in compute_alive_voxels()
  """
  dataset_size = len(dataset.images)
  is_allowed = np.ones((dataset_size,), dtype=bool)

  # Determine the location of each camera in world coordinates.
  t_origins = dataset.camtoworlds[:, :3, -1]  # [n, 3]

  # Measure distance to this submodel's origin in world coordinates.
  def compute_distance(norm):
    sm_idxs = np.array(sm_idx).reshape(1, 1)
    distances = coord.distance_to_submodel_volumes(
        t_origins, sm_idxs, config, grid_config, ord=norm
    )  # [n, 1]
    distances = jax.device_get(distances)  # [n, 1]
    distances = np.reshape(distances, (-1,))  # [n]
    return distances

  # Ignore cameras that are far from the origin in inf-norm.
  if config.baking_max_distance_to_submodel_origin is not None:
    is_close_to_origin = (
        compute_distance(norm=np.inf)
        <= config.baking_max_distance_to_submodel_origin
    )
    is_allowed = is_allowed & is_close_to_origin

  # Keep the closest k images.
  if max_images is None:
    max_images = config.baking_max_images_to_render
  if max_images is not None and max_images < dataset_size:
    # Order train_idxs by distance, nearest to farthest.
    distances = compute_distance(norm=2)
    # Compute the distance of the k-th closest camera.
    kth_lowest_dist = np.partition(distances, max_images)[max_images]
    # Keep only cameras closer than that. There wil lbe exactly k results.
    is_closer_than_kth = distances < kth_lowest_dist
    is_allowed = is_allowed & is_closer_than_kth

  train_idxs = np.arange(dataset_size)  # [n]
  train_idxs = train_idxs[is_allowed]
  return train_idxs


def get_atlas_block_size(data_block_size):
  """Add 1-voxel apron for native trilerp in the WebGL renderer."""
  return data_block_size + 1


def _construct_alive_triplane_positions(alive_voxels, axis, max_filter_size=25):
  """Calculates alive positions and texel coordinates for a single triplane.

  Args:
    alive_voxels: bool[K,K,K]. Occupancy grid with same spatial resolution as
      triplanes.
    axis: int. Which axis to construct a plane for.
    max_filter_size: int. Size for maximum filter. Used to determine which
      neighboring texels should be considered alive.

  Returns:
    triplane_positions: f32[B,3]. 3D positions to query the model at in
      triplane voxel coordinates.
    texel_idxs: (f32[B], f32[B]). Indices in texture map corresponding to
      triplane_positions.
  """
  n = alive_voxels.ndim
  other_axes = tuple(d for d in range(n) if d != axis)

  # A texel is alive if it intersects at least one alive voxel along 'axis'.
  mask_2d = np.any(alive_voxels, axis=axis, keepdims=True)

  # A neighboring texel is alive if it's near an alive voxel.
  mask_2d = scipy.ndimage.maximum_filter(mask_2d, size=max_filter_size)

  # 3D coordinates of alive texels. Includes 0s for 'axis'.
  triplane_positions = np.stack(np.nonzero(mask_2d), axis=-1)  # Bx3

  # 2D coordinates of alive texels.
  texel_idxs = tuple(triplane_positions[:, d] for d in other_axes)

  return triplane_positions.astype(np.float32), texel_idxs


def compute_triplane_features(
    pparams,
    config,
    batch_size,
    alive_voxels,
    sm_idx,
):
  """Bakes triplanes.

  Args:
    pparams: Replicated MERF model parameters.
    config: Config instance.
    batch_size: Number of positions to evaluate at once.
    alive_voxels: bool[k,k,k]. High resolution occupancy grid.
    sm_idx: int. Which submodel is being processed.

  Returns:
    planes_features: [f32[k,k,7]]. List of triplane feature maps before
      activation.
    planes_density: [f32[k,k,1]. List of triplane density maps before
      activation.
  """
  grid_config = config.grid_config
  use_triplanes = config.triplane_resolution > 0
  if not use_triplanes:
    return None, None

  triplane_voxel_size = grid_config['triplane_voxel_size']
  triplane_resolution = config.triplane_resolution
  planes_features = []
  planes_density = []

  assert alive_voxels.shape == (triplane_resolution,) * 3

  # Setup MLP (doing this inside the loop may cause re-jitting).
  mlp_p = _create_mlp_pmap(pparams, config, grid_config)

  ndims = len(alive_voxels.shape)
  for plane_idx in range(ndims):
    logging.info(f'baking triplane {plane_idx=}')

    # Positions in voxel coordiantes. Only texels corresponding to an alive
    # voxel will be evaluated.
    max_filter_size = _compute_triplane_filter_size(config)
    logging.info(
        'Dilating triplane feature map with max_filter_size=%d',
        max_filter_size,
    )

    triplane_positions, texel_idxs = _construct_alive_triplane_positions(
        alive_voxels, plane_idx, max_filter_size=max_filter_size
    )
    logging.info(
        f'Evaluating {len(triplane_positions)} live texels for'
        f' plane={plane_idx}.'
    )

    # Positions in squash space.
    s_positions = grid_utils.grid_to_world(
        triplane_positions, triplane_voxel_size, np
    )

    # Submodel indices.
    sm_idxs = np.broadcast_to(sm_idx, s_positions[Ellipsis, :1].shape)

    # Dense feature map.
    plane = np.full(
        (triplane_resolution, triplane_resolution, models.NUM_CHANNELS + 1),
        -1000000.0,  # a large negative number ensures alpha=0
        dtype=np.float32,
    )

    # Evaluate model at locations in s_positions.
    num_alive_texels = len(triplane_positions)
    for batch_start in etqdm.tqdm(list(range(0, num_alive_texels, batch_size))):
      batch_end = min(batch_start + batch_size, num_alive_texels)

      # Evaluate model.
      features, density = _evaluate_mlp_pmap(
          mlp_p,
          sm_idxs[batch_start:batch_end],
          s_positions[batch_start:batch_end],
      )

      # Update coordinates in 'plane'.
      batch_i, batch_j = [v[batch_start:batch_end] for v in texel_idxs]
      plane[batch_i, batch_j, : models.NUM_CHANNELS] = features
      plane[batch_i, batch_j, -1:] = density

      del features, density

    # Save feature planes for later.
    planes_features.append(plane[Ellipsis, : models.NUM_CHANNELS])
    planes_density.append(plane[Ellipsis, -1:])

  return planes_features, planes_density


def _compute_triplane_filter_size(config):
  """Computes amount of overrun when baking triplanes.

  When rendering, there is not enough information in a low-resolution occupancy
  grid or the sparse voxel grid to determine if a triplane texel entry is
  real or not. To avoid using placeholder texel entries, we populate a buffer
  around texels marked by alive_voxels. This buffer is large enough to cover
  the length of a sparse grid block as measured in texels.

  Args:
    config: ...

  Returns:
    Size of max filter to apply to occupancy grid projected onto a single plane.
  """
  if config.baking_triplane_features_buffer is None:
    triplane_texels_per_sparse_grid_block = (
        config.triplane_resolution / config.sparse_grid_resolution
    ) * get_atlas_block_size(config.data_block_size)
    buffer_size = int(np.ceil(triplane_texels_per_sparse_grid_block))
  else:
    buffer_size = config.baking_triplane_features_buffer
  return buffer_size * 2 + 1


def _bake_sparse_grid(
    pparams,
    config,
    grid_config,
    alive_macroblocks,
    data_block_size,
    batch_size_in_blocks,
    sm_idx,
):
  """Bakes sparse grid for a submodel."""
  sparse_grid_voxel_size = grid_config['sparse_grid_voxel_size']

  # Number of voxels per macroblock+1. The "+1" ensures that each macroblock
  # has features for all eight corners within the macroblock.
  atlas_block_size = get_atlas_block_size(data_block_size)

  # Number of occupied macroblocks. Features will be extracted for each.
  num_occupied_blocks = alive_macroblocks.sum()

  def create_atlas_1d(n):
    shape = (
        num_occupied_blocks,
        atlas_block_size,
        atlas_block_size,
        atlas_block_size,
        n,
    )
    return np.zeros(shape, dtype=np.float32)

  sparse_grid_features_1d = create_atlas_1d(models.NUM_CHANNELS)
  sparse_grid_density_1d = create_atlas_1d(1)

  # Setup MLP (doing this inside the loop would cause re-jitting).
  mlp_p = _create_mlp_pmap(pparams, config, grid_config)

  # Find the lower corner of each AABB corresponding to each occupied
  # macroblock in sparse grid voxel coordinates.
  lower_sparse_grid_positions = (
      np.stack(np.nonzero(alive_macroblocks), axis=-1).astype(np.float32)
      * data_block_size
  )  # Mx3
  for block_start in etqdm.tqdm(
      range(0, num_occupied_blocks, batch_size_in_blocks)
  ):
    block_end = min(block_start + batch_size_in_blocks, num_occupied_blocks)

    # Choose macroblocks to assemble.
    lower_sparse_grid_positions_batch = lower_sparse_grid_positions[
        block_start:block_end
    ]  # Bx3.

    # Build AxAxA grid for each macroblock.
    # Voxel coordinates within a macroblock.
    span = np.arange(atlas_block_size)  # A.
    macroblock_positions = np.stack(
        np.meshgrid(span, span, span, indexing='ij'), axis=-1
    )  # AxAxAx3.

    # Find positions for these macroblocks in sparse grid coordinates.
    sparse_grid_positions = (
        lower_sparse_grid_positions_batch[
            :, np.newaxis, np.newaxis, np.newaxis, :
        ]
        + macroblock_positions[np.newaxis]
    )  # Bx1x1x1x3 + 1xAxAxAx3 = BxAxAxAx3.

    # Convert to squash coordinates. BxAxAxAx3.
    s_positions = grid_utils.grid_to_world(
        sparse_grid_positions, sparse_grid_voxel_size, np
    )

    # Broadcast submodel indices.
    sm_idxs = np.broadcast_to(sm_idx, s_positions[Ellipsis, :1].shape)

    # BxAxAxAx7, BxAxAxAx3.
    features, density = _evaluate_mlp_pmap(mlp_p, sm_idxs, s_positions)
    sparse_grid_features_1d[block_start:block_end] = features
    sparse_grid_density_1d[block_start:block_end] = density

    del features, density, lower_sparse_grid_positions_batch, s_positions
    gc.collect()

  return sparse_grid_features_1d, sparse_grid_density_1d


def _reshape_into_3d_atlas_and_compute_indirection_grid(
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
  ax = ay = az = None
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


def sm_idx_for_camera(dataset, camidx, config, grid_config):
  """Computes sm_idx for a single camera."""
  # Ray origins are constructed as follows in  pixels_to_rays(),
  origin = dataset.camtoworlds[camidx, :3, -1]
  ray = utils.Rays(origins=origin)
  sm_idx = coord.rays_to_sm_idxs(ray, config, grid_config, ignore_override=True)
  return sm_idx[0]


def find_cam_idx_for_submodel(dataset, sm_idxs, config, grid_config):
  """Finds a camera assigned to a given submodel."""
  if not sm_idxs:
    return None

  for cam_idx in range(len(dataset.images)):
    cam_sm_idx = sm_idx_for_camera(dataset, cam_idx, config, grid_config)
    if cam_sm_idx in sm_idxs:
      return cam_idx

  return None


def final_alpha_threshold(config):
  """alpha_threshold at the end of training."""
  alpha_threshold = 0.0
  if config.alpha_threshold is not None:
    alpha_threshold = config.alpha_threshold(
        config.max_steps
    )
  return alpha_threshold


def compute_alive_voxels(
    render_fn,
    dataset,
    config,
    use_alpha_culling,
    sm_idx,
    *,
    max_images=None,
    load_alive_voxels_from_disk=False,
    save_alive_voxels_to_disk=False,
    alive_voxels_path=None,
    dtype=None,
    xnp=None,
):
  """Computes alive voxels for a submodel by rendering views in a dataset.

  Uses tdist values produced by MERF's PropMLP to construct a high-resolution
  occupancy grid. Rays are sourced from views in `dataset`. If alpha culling is
  enabled, only points with non-trivial density and weights (as predicted by
  MERF) will be considered.

  Args:
    render_fn: Function for rendering MERF. See model.render_image().
    dataset: Dataset to render.
    config: Config instance.
    use_alpha_culling: If True, ignore voxels with small alpha or weights.
    sm_idx: int. Which submodel to use.
    max_images: int or list of ints. Maximum number of images to render when
      to build alive_voxels or list of image idxs to use.
    load_alive_voxels_from_disk: If True, load alive_voxels from disk if
      possible.
    save_alive_voxels_to_disk: If True, save alive_voxels to disk after
      computing.
    alive_voxels_path: epath.Path. Where to save alive_voxels.
    dtype: dtype for result.
    xnp: numpy or jax.numpy

  Returns:
    alive_voxels: dtype[k,k,k]. An occupancy grid at the finest resolution
      specified by a grid_config derived from config.
  """
  grid_config = config.grid_config
  alpha_threshold = weight_threshold = final_alpha_threshold(config)
  if (
      load_alive_voxels_from_disk
      and alive_voxels_path is not None
      and alive_voxels_path.exists()
  ):
    logging.info(f'Loading alive_voxels from {alive_voxels_path}...')
    alive_voxels = utils.load_np(alive_voxels_path)
  else:
    logging.info('Calculating alive_voxels...')
    alive_voxels = _compute_alive_voxels(
        render_fn=render_fn,
        dataset=dataset,
        config=config,
        grid_config=grid_config,
        sm_idx=sm_idx,
        max_images=max_images,
        use_alpha_culling=use_alpha_culling,
        alpha_threshold=alpha_threshold,
        weight_threshold=weight_threshold,
        dtype=dtype,
        xnp=xnp,
    )
    logging.info('Finished calculating alive_voxels!')
  if save_alive_voxels_to_disk and alive_voxels_path is not None:
    logging.info(f'Saving alive_voxels to {alive_voxels_path}...')
    alive_voxels_path.parent.mkdir(parents=True, exist_ok=True)
    try:
      utils.save_np(alive_voxels, alive_voxels_path)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(
          'Failed to cache alive_voxels to disk. Original error below.'
      )

  logging.info(
      '{:.2f}% voxels are occupied.'.format(
          100 * alive_voxels.sum() / alive_voxels.size
      )
  )

  return alive_voxels


def compute_alive_macroblocks(alive_voxels, config):
  """Computes alive macroblocks with np.max.

  Determines which macroblocks should be instantiated. Rather than storing
  features in a dense, high-resolution voxel grid, macroblocks with non-trivial
  density are identified using a high-resolution occupancy grid. If a macroblock
  contains at least one non-empty occupancy voxel, it is instantiated.

  Args:
    alive_voxels: bool[k,k,k]. High resolution occupancy grid.
    config: Config instance.

  Returns:
    alive_macroblocks: bool[k',k',k']. Low-resolution occupancy grid for
      macroblocks.
  """
  # Compute alive data blocks. Macroblocks are strictly lower spatial
  # resolution than alive voxels.
  use_sparse_grid = config.sparse_grid_resolution > 0
  use_triplanes = config.triplane_resolution > 0

  # Number of occupancy grid voxels per side for a single macroblock.
  data_block_size = config.data_block_size

  alive_macroblocks = None
  logging.info(
      f'Calculating alive macroblocks for {data_block_size=} with'
      f' {use_sparse_grid=} and {use_triplanes=}'
  )
  if use_sparse_grid:
    if use_triplanes:
      downsampling_ratio_3d_grid = int(
          config.triplane_resolution / config.sparse_grid_resolution
      )
      alive_voxels_3d_grid = skimage.measure.block_reduce(
          alive_voxels,
          (
              downsampling_ratio_3d_grid,
              downsampling_ratio_3d_grid,
              downsampling_ratio_3d_grid,
          ),
          np.max,
      )
    else:
      alive_voxels_3d_grid = alive_voxels

    alive_macroblocks = skimage.measure.block_reduce(
        alive_voxels_3d_grid,
        (data_block_size, data_block_size, data_block_size),
        np.max,
    )

    # logging.info out metrics.
    num_alive_macroblocks = alive_macroblocks.sum()
    percent_alive_macroblocks = (
        100 * num_alive_macroblocks / alive_macroblocks.size
    )
    logging.info(
        f'Sparse grid: {num_alive_macroblocks} out of'
        f' {alive_macroblocks.size} ({percent_alive_macroblocks:.1f}%)'
        ' macroblocks are occupied.'
    )
  return alive_macroblocks


def compute_sparse_grid_features(
    config, pparams, alive_macroblocks, sm_idx, batch_size
):
  """Computes features for sparse grid.

  Extracts pre-activation feature vectors for the sparse voxel grid. Only voxel
  corners corresponding within an alive macroblock are considered.

  Args:
    config: Config instance.
    pparams: Replicated MERF model parameters.
    alive_macroblocks: bool[k',k',k']. Occupancy grid for macroblocks.
    sm_idx: int. Which submodel is being considered.
    batch_size: Number of macroblocks to process at once.

  Returns:
    sparse_grid_features: Feature vectors.
    sparse_grid_density: Density.
    sparse_grid_block_indices: Maps locations to macroblocks.
  """
  use_sparse_grid = config.sparse_grid_resolution > 0
  if not use_sparse_grid:
    return None, None, None

  data_block_size = config.data_block_size
  grid_config = config.grid_config

  # Bake sparse grid.
  sparse_grid_features_1d, sparse_grid_density_1d = (
      _bake_sparse_grid(
          pparams=pparams,
          config=config,
          grid_config=grid_config,
          alive_macroblocks=alive_macroblocks,
          data_block_size=data_block_size,
          batch_size_in_blocks=batch_size,
          sm_idx=sm_idx,
      )
  )

  # Reshape sparse grid into 3D volume atlas texture (for OpenGL) and
  # compute the indirection grid.
  sparse_grid_features, sparse_grid_density, sparse_grid_block_indices = (
      _reshape_into_3d_atlas_and_compute_indirection_grid(
          sparse_grid_features_1d,
          sparse_grid_density_1d,
          data_block_size,
          alive_macroblocks,
      )
  )
  return sparse_grid_features, sparse_grid_density, sparse_grid_block_indices


def estimate_vram_consumption(
    config,
    sparse_grid_features,
    sparse_grid_density,
    sparse_grid_block_indices,
    planes_features,
    planes_density,
    *,
    sm_export_dir=None,
):
  """Estimates VRAM consumption.

  Args:
    config: Config instance.
    sparse_grid_features:
    sparse_grid_density:
    sparse_grid_block_indices:
    planes_features:
    planes_density:
    sm_export_dir:

  Returns:
    vram_consumption:
  """
  logging.info('Estimating VRAM consumption...')
  use_sparse_grid = config.sparse_grid_resolution > 0
  use_triplanes = config.triplane_resolution > 0

  vram_consumption = {}
  if use_sparse_grid:
    vram_consumption.update(
        dict(
            sparse_3d_grid=(
                math.as_mib(sparse_grid_features)
                + math.as_mib(sparse_grid_density)
            ),
            indirection_grid=math.as_mib(sparse_grid_block_indices),
        )
    )
  if use_triplanes:
    # Assume that all three planes have the same size.
    vram_consumption['triplanes'] = 3 * (
        math.as_mib(planes_features[0])
        + math.as_mib(planes_density[0])
    )
  vram_consumption['total'] = sum(vram_consumption.values())
  logging.info('VRAM consumption:')
  for k in sorted(vram_consumption):
    logging.info(f'{k}: {vram_consumption[k]:.2f} MiB')

  if sm_export_dir is not None:
    sm_export_dir.mkdir(parents=True, exist_ok=True)
    utils.save_json(vram_consumption, sm_export_dir / 'vram.json')

  return vram_consumption


def _apply_filter_chunk(x, factor, filter_fn):
  """Applies a filter function to patches of shape [f, f]."""
  w = x.shape
  f = factor
  x = x.reshape(w[0] // f, f, w[1] // f, f, w[2] // f, f)
  x = x.transpose(0, 2, 4, 1, 3, 5)
  x = x.reshape(*x.shape[:3], -1)
  x = filter_fn(x, axis=-1)
  return x


# Same as _apply_filter_chunk, but apply pmap to the 0th argument.
_apply_filter_chunk_pmap = jax.pmap(
    _apply_filter_chunk, in_axes=(0,), static_broadcasted_argnums=(1, 2)
)


def _apply_filter(x, factor, filter_fn, chunk_size=None):
  """Behaves like skimage.measure.block_reduce."""
  if chunk_size is None:
    chunk_size = x.shape

  assert (np.array(x.shape) % factor == 0).all(), 'factor must divide x.shape'
  assert (
      np.array(x.shape) % chunk_size == 0
  ).all(), 'chunk_size must divide x.shape'
  assert chunk_size % factor == 0, 'factor must divide chunk_size'

  # Filter is a no-op for filter_size=1.
  if factor == 1:
    return x

  # Compute output shape.
  sh = np.array(x.shape) // factor
  y = np.empty(sh, dtype=x.dtype)

  # Setup chunks.
  r = [range(0, u, chunk_size // factor) for u in sh]
  num_devices = jax.local_device_count()
  for cl_batch in _itertools_batched(itertools.product(*r), num_devices):
    cl_batch = np.array(cl_batch, dtype=np.int32)  # i32[m,3]
    ch_batch = cl_batch + chunk_size // factor

    # Fetch chunks
    l_batch = cl_batch * factor
    h_batch = l_batch + chunk_size

    cx_batch = []
    for l, h in zip(l_batch, h_batch):
      cx = x[l[0]:h[0], l[1]:h[1], l[2]:h[2]]
      cx_batch.append(cx)
    cx_batch = np.stack(cx_batch, axis=0)

    # Apply max filter.
    cx_batch = _apply_filter_chunk_pmap(cx_batch, factor, filter_fn)
    cx_batch = jax.device_get(cx_batch)

    # Update result.
    for cl, ch, cx in zip(cl_batch, ch_batch, cx_batch):
      y[cl[0]:ch[0], cl[1]:ch[1], cl[2]:ch[2]] = cx

  return y


def downsample_alive_voxels(alive_voxels, config):
  """Downsamples occupancy grid 'alive_voxels'."""
  results = []
  chunk_size = config.baking_alive_voxels_max_chunk_size
  downsample_factors = config.baking_occupancy_grid_downsample_factors
  for factor in downsample_factors:
    result = _apply_filter(alive_voxels, factor, jnp.max, chunk_size=chunk_size)
    results.append((factor, result))
  return results


def pack_occupancy_grids(occupancy_grids):
  """Bit-pack occupancy grids.

  Args:
    occupancy_grids: List of (int, bool[?,?,?]) values. Each entry includes a
      downsampling factor and an occupancy grid.

  Returns:
    List of (int, int64[?,?,?]) values. Each entry includes a downsampling
      factor and a bit-packed occupancy grid.
  """
  packed_occupancy_grids = []
  for i in range(len(occupancy_grids) - 1):
    factor = occupancy_grids[i + 1][0]
    packed_occupancy_grid = np.zeros_like(occupancy_grids[i + 1][1]).astype(
        np.int64
    )
    for dz in range(2):
      for dy in range(2):
        for dx in range(2):
          bit = dx + 2 * (dy + 2 * dz)
          bit_value = 2**bit
          sliced_grid = occupancy_grids[i][1][dz::2, dy::2, dx::2].astype(
              np.int64
          )
          packed_occupancy_grid += bit_value * sliced_grid
    packed_occupancy_grids.append((factor, packed_occupancy_grid))
  return packed_occupancy_grids


def _compute_distance_grid(occupancy_grid):
  """Computes distance grid corresponding to an occupancy grid."""
  # Construct a distance grid such that distance_grid[i,j,k] <= number of
  # voxels between the current location and the closest occupied location in
  # occupancy_grid.
  distance_grid = jnp.zeros(occupancy_grid.shape, dtype=np.int32)
  dilated_occupancy = jax.device_put(occupancy_grid)

  # Since we pack our distance field into an 8-bit texture, we only need to
  # compute the distance field up to 255 voxel widths. Note this loops
  # constructs a _conservative_ distance field which is valid for any sample
  # location within a voxel: voxels >255 units away from the nearest occupied
  # voxel will be assigned a distance of 255.
  max_distance = min(255, occupancy_grid.shape[0])
  logging.info(
      f'Computing distance grid of shape {distance_grid.shape} with'
      f' {max_distance=}'
  )
  for _ in range(max_distance):
    dilated_occupancy = _apply_3x3x3_maximum_filter(dilated_occupancy)
    distance_grid += jnp.where(dilated_occupancy, 0, 1)

  return jax.device_get(distance_grid)


def compute_distance_grids(occupancy_grids, config):
  """Computes distance grids at varying spatial resolutions.

  Args:
    occupancy_grids: list of (int, bool[?,?,?]). An entry is True if it is
      occupied.
    config: configs.Config instance.

  Returns:
    [(int, int32[m,m,m])] for m <= 512. Distance, in voxels, to the
      nearest occupied location based on an occupancy grid at the
      target_resolution.
  """
  factor_to_occupancy_grid = {factor: grid for factor, grid in occupancy_grids}
  result = []
  for factor in config.baking_distance_grid_downsample_factors:
    if factor not in factor_to_occupancy_grid:
      raise ValueError(
          f'Cannot compute distance grid with {factor=} unless an occupancy'
          ' grid of the same resolution is also available.'
      )
    occupancy_grid = factor_to_occupancy_grid[factor]
    distance_grid = _compute_distance_grid(occupancy_grid)
    result.append((factor, distance_grid))
  return result


def bake_submodel(
    sm_idx,
    merf_model,
    merf_pstate,
    dataset,
    merf_render_fn,
    merf_config,
    *,
    sm_temp_dir=None,
):
  """Bakes a single submodel.

  Computes baked representation for a target submodel.

  If sm_temp_dir is specified, this function will save baked parameters in an
  orbax checkpoint.  Subsequent calls to this function will return the
  checkpoint's contents.

  Args:
    sm_idx: int. Which submodel to bake.
    merf_model: MERF Model instance. Model architecture.
    merf_pstate: PyTree representing MERF model state. State should be
      replicated across all devices.
    dataset: mipnerf360 Dataset instance. Represents training set.
    merf_render_fn: Function for rendering MERF. API should match the API for
      models.render_image().
    merf_config: Config instance.
    sm_temp_dir: epath.Path instance. Where to read and write temporary outputs.

  Returns:
    sparse_grid_features: ...
    sparse_grid_density: ...
    sparse_grid_block_indices: ...
    planes_features: ...
    planes_density: ...
    deferred_mlp_vars: ...
    occupancy_grids: ...
    distance_grid: ...
  """
  gc.collect()

  # TODO(duckworthd): Implement baking for these features. They're needed by
  # the DeferredMLP.
  if (
      merf_config.num_viewdir_features > 0
      or merf_config.num_origin_features > 0
  ):
    raise NotImplementedError(
        'Baking for extra DeferredMLP inputs is not yet implemented.'
    )

  # Enable alpha culling if the alpha_threshold at the end of training is
  # non-trivial.
  use_alpha_culling = merf_config.baking_force_alpha_culling
  if use_alpha_culling is None:
    use_alpha_culling = final_alpha_threshold(merf_config) > 0

  # Try to load baked params from a checkpoint. This will return None if there
  # are no checkpoints available.
  try:
    baked_params = load_baked_params(sm_temp_dir=sm_temp_dir)
  except Exception:  # pylint: disable=broad-exception-caught
    baked_params = None

  # Exit early if the scene is already baked.
  if baked_params is not None:
    logging.info(
        f'Submodel {sm_idx} has already been exported. Loading baked params'
        ' from disk.'
    )
    return baked_params

  # Estimate batch size for various baking operations.
  num_rays_per_subbatch = (
      merf_config.batch_size // merf_config.gradient_accumulation_steps
  )
  num_samples_per_ray = merf_model.num_final_samples
  num_mlp_evals_per_sample = (
      grid_utils.calculate_num_evaluations_per_sample(merf_config)
  )
  num_mlp_evals_per_subbatch = (
      num_rays_per_subbatch * num_samples_per_ray * num_mlp_evals_per_sample
  )

  # Determine which voxels have non-trivial density.
  logging.info('Computing alive_voxels...')
  alive_voxels_path = None
  if sm_temp_dir is not None:
    alive_voxels_path = sm_temp_dir / 'alive_voxels.npy'

  alive_voxels = compute_alive_voxels(
      render_fn=merf_render_fn,
      dataset=dataset,
      config=merf_config,
      use_alpha_culling=use_alpha_culling,
      sm_idx=sm_idx,
      load_alive_voxels_from_disk=False,
      save_alive_voxels_to_disk=True,
      alive_voxels_path=alive_voxels_path,
  )

  # Apply median filter to remove floaters.
  logging.info('Applying median filter (if needed)...')
  alive_voxels = apply_median_filter(alive_voxels, merf_config)

  # Compute occupancy grids based on downsampling factors. A median filter
  # is (optionally) applied here to eliminate floaters in rendering without
  # affecting which triplane and sparse grid locations are baked.
  logging.info('Downsampling occupancy grids...')
  occupancy_grids = downsample_alive_voxels(
      alive_voxels=alive_voxels,
      config=merf_config,
  )

  # Compute alive data blocks. Data blocks are strictly lower spatial
  # resolution than alive voxels.
  logging.info('Computing alive macroblocks...')
  alive_macroblocks = compute_alive_macroblocks(
      alive_voxels, merf_config
  )

  # Bake features for sparse voxel grid.
  logging.info('Computing sparse voxel feature grid...')
  sparse_grid_features, sparse_grid_density, sparse_grid_block_indices = (
      compute_sparse_grid_features(
          config=merf_config,
          pparams=merf_pstate.params,
          alive_macroblocks=alive_macroblocks,
          sm_idx=sm_idx,
          batch_size=(
              num_mlp_evals_per_subbatch // (merf_config.data_block_size**3)
          ),
      )
  )

  # Bake triplanes.
  logging.info('Computing triplane feature maps...')
  planes_features, planes_density = compute_triplane_features(
      pparams=merf_pstate.params,
      config=merf_config,
      batch_size=num_mlp_evals_per_subbatch,
      alive_voxels=alive_voxels,
      sm_idx=sm_idx,
  )

  # Compute VRAM consumption.
  estimate_vram_consumption(
      config=merf_config,
      sparse_grid_features=sparse_grid_features,
      sparse_grid_density=sparse_grid_density,
      sparse_grid_block_indices=sparse_grid_block_indices,
      planes_features=planes_features,
      planes_density=planes_density,
      sm_export_dir=sm_temp_dir,
  )

  # Load Deferred MLP.
  deferred_mlp_vars = flax.jax_utils.unreplicate(
      merf_pstate.params['params']['DeferredMLP_0']
  )

  # Quantize model parameters.
  logging.info('Quantizing parameters...')
  (
      planes_features,
      planes_density,
      sparse_grid_features,
      sparse_grid_density,
  ) = quantize.map_quantize(
      planes_features,
      planes_density,
      sparse_grid_features,
      sparse_grid_density,
  )

  # Pack the occupancy grids so we can use the 8 bits in a byte to fit 2x the
  # resolution into the same memory footprint.
  packed_occupancy_grids = pack_occupancy_grids(occupancy_grids)

  # Compute a distance field, so that we can do coarse empty space skipping
  # without any ray/bbox intersection tests.
  distance_grids = compute_distance_grids(occupancy_grids, merf_config)

  # Package baked parameters into a collection.
  baked_params = (
      sparse_grid_features,
      sparse_grid_density,
      sparse_grid_block_indices,
      planes_features,
      planes_density,
      deferred_mlp_vars,
      occupancy_grids,
      packed_occupancy_grids,
      distance_grids,
  )

  # Serialize baked parameters to disk.
  if sm_temp_dir is not None:
    logging.info(f'Saving baked parameters to {sm_temp_dir}...')
    save_baked_params(baked_params=baked_params, sm_temp_dir=sm_temp_dir)

  return baked_params


def save_baked_params(baked_params, sm_temp_dir):
  """Writes baked parameters to disk."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  checkpointer.save(sm_temp_dir / 'baked_params', baked_params, force=True)


def load_baked_params(sm_temp_dir):
  """Restores baked parameters from disk."""
  checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  baked_params = checkpointer.restore(sm_temp_dir / 'baked_params')
  return baked_params


def construct_sm_temp_dir(sm_idx, merf_config):
  """Constructs path to a temporary directory for this submodel."""
  baked_log_dir = distill.baked_log_dir(merf_config)
  sm_temp_dir = baked_log_dir / 'temp' / f'sm_{sm_idx:03d}'
  return sm_temp_dir
