# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utilites for rendering."""

import jax
import jax.numpy as jnp

from light_field_neural_rendering.src.utils import data_types
from light_field_neural_rendering.src.utils import data_utils


def normalize_disp(dataset_name):
  """Function that specifies if disparity should be normalized"""
  return dataset_name in ["forward_facing"]


def get_render_function(model, config, randomized):
  """Construct the rendering function."""

  def render_fn(variables, key_0, key_1, batch):
    return jax.lax.all_gather(
        model.apply(variables, key_0, key_1, batch, randomized),
        axis_name="batch")

  render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),  # Only distribute the data input.
      donate_argnums=(3,),
      axis_name="batch",
  )

  return render_pfn


def preprocess_eval_batch(batch, num_rays):
  """Function to preprocess the evaluation batch like flattening the rays for chunking.

  Args:
    batch: data_types.Batch
    num_rays: Number of rays in the batch

  Returns:
    batch: with reshaped rays
  """

  batch.target_view.rays = jax.tree_map(
      lambda r: r.reshape((num_rays, r.shape[-1])), batch.target_view.rays)

  # Remove rgb as it is not required for rendering
  batch.target_view.rgb = None
  return batch


def convert_to_chunk(batch, start_idx, chunk, host_id):
  """Return a chunk from a batch.

  Args:
    batch: The batch to convert to chunks
    start_idx: starting index for the chunk
    chunk: (int) size of chunk
    host_id: jax host id

  Returns:
    chunk_batch: chunk form the batch
    padding: specify how much padding was done to distribute evenly to all
    devices.
  """
  chunk_rays = jax.tree_map(lambda r: r[start_idx:start_idx + chunk],
                            batch.target_view.rays)
  chunk_size = chunk_rays.batch_shape[0]
  rays_remaining = chunk_size % jax.device_count()
  if rays_remaining != 0:
    padding = jax.device_count() - rays_remaining
    chunk_rays = jax.tree_map(
        lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays)
  else:
    padding = 0

  # After padding the number of chunk_rays is always divisible by
  # host_count.
  # Distribute rays to hosts
  rays_per_host = chunk_rays.batch_shape[0] // jax.process_count()
  start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
  chunk_rays = jax.tree_map(lambda r: data_utils.shard(r[start:stop]),
                            chunk_rays)
  chunk_batch = data_types.Batch(target_view=data_types.Views(rays=chunk_rays))

  return chunk_batch, padding


def render_image(render_fn, batch, rng, normalize_disp, chunk=8192):
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function.
    batch: a `Rays` data_types.Batch, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    normalize_disp: bool, if true then normalize `disp` to [0, 1].
    chunk: int, the size of chunks to render sequentially.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  #--------------------------------------------------------------------------
  # Preprocess batch
  height, width = batch.target_view.rays.origins.shape[:2]
  num_rays = height * width
  batch = preprocess_eval_batch(batch, num_rays)

  unused_rng, key_0, key_1 = jax.random.split(rng, 3)
  host_id = jax.host_id()
  results = []

  reference_batch = data_utils.shard(batch.reference_views)

  for i in range(0, num_rays, chunk):
    # After padding the number of chunk_rays is always divisible by
    # host_count.
    chunk_batch, padding = convert_to_chunk(batch, i, chunk, host_id)
    chunk_batch.reference_views = reference_batch
    chunk_results = render_fn(key_0, key_1, chunk_batch)[-1]
    results.append([
        data_utils.unshard(x[0], padding)
        for x in chunk_results
        if x is not None
    ])
    # pylint: enable=cell-var-from-loop
  # For Neural LightField model the NN return only the rgb values,
  # the disparity and acc fields are None. Where as for NeRF there
  # are three return variable. The if condition below handles the
  # two cases
  if len(results[0]) == 3:
    rgb, disp, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
    # Normalize disp for visualization for ndc_rays in llff front-facing scenes.
    if normalize_disp:
      disp = (disp - disp.min()) / (disp.max() - disp.min())
    ret = (rgb.reshape((height, width, -1)), disp.reshape(
        (height, width, -1)), acc.reshape((height, width, -1)))
  elif len(results[0]) == 2:
    #return the rgb values and the depth
    rgb, disp = [jnp.concatenate(r, axis=0) for r in zip(*results)]
    # Normalize disp for visualization for ndc_rays in llff front-facing scenes.
    if normalize_disp:
      disp = (disp - disp.min()) / (disp.max() - disp.min())
    ret = (rgb.reshape((height, width, -1)), disp.reshape(height, width,
                                                          -1), None)
  elif len(results[0]) == 1:
    rgb = [jnp.concatenate(r, axis=0) for r in zip(*results)][0]
    ret = (rgb.reshape((height, width, -1)), None, None)

  return ret
