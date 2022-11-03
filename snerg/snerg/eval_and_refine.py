# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tools that evaluate (and refine) the quality of a baked SNeRG model."""

import functools
import math

import flax
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from snerg.nerf import datasets
from snerg.nerf import utils
from snerg.snerg import model_utils
from snerg.snerg import rendering



class ImageQualityEvaluator:
  """Evaluates image quality metrics between an image pair."""

  def __init__(self):
    """Initializes the stateful (e.g. network-based) quality metrics."""
    # Compiling to the CPU because it's faster and more accurate.
    self._ssim_fn = jax.jit(
        functools.partial(utils.compute_ssim, max_val=1.), backend="cpu")



  def ssim(self, image1, image2):
    """Compute the SSIM metric."""
    return np.array(self._ssim_fn(image1, image2))

  def psnr(self, image1, image2):
    return -10. * np.log(np.mean((image1 - image2)**2)) / np.log(10.)

  def eval_image_list(self, images1, images2):
    """Returns the average PSNR, SSIM and LPIPS between each pair of images."""
    psnrs = [
        self.psnr(image1, image2) for image1, image2 in zip(images1, images2)
    ]
    ssims = [
        self.ssim(image1, image2) for image1, image2 in zip(images1, images2)
    ]

    return np.array(psnrs).mean(), np.array(ssims).mean()  # pylint: disable=unreachable


def build_sharded_dataset_for_view_dependence(source_dataset, atlas_t,
                                              atlas_block_indices_t,
                                              atlas_params, scene_params,
                                              grid_params):
  """Builds a dataset that we can run the view-dependence MLP on.

  We ray march through a baked SNeRG model to generate images with RGB colors
  and features. These serve as the input for the view-dependence MLP which adds
  back the effects such as highlights.

  To make use of multi-host parallelism provided by JAX, this function shards
  the dataset, so that each host contains only a slice of the data.

  Args:
    source_dataset: The nerf.datasets.Dataset we should compute data for.
    atlas_t: A tensorflow tensor containing the texture atlas.
    atlas_block_indices_t: A tensorflow tensor containing the indirection grid.
    atlas_params: A dict with params for building and rendering with
      the 3D texture atlas.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    grid_params: A dict with parameters describing the high-res voxel grid which
      the atlas is representing.

  Returns:
    rgb_data: The RGB (+ features) input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 7) numpy array.
    alpha_data: The alpha channel of the input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 1) numpy array.
    direction_data: The direction vectors for the input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 3) numpy array.
    ref_data: The reference RGB colors for each input data sample, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 3) numpy array.
  """

  num_hosts = jax.host_count()
  num_local_devices = jax.local_device_count()
  host_id = jax.host_id()
  num_images = source_dataset.camtoworlds.shape[0]
  num_batches = math.ceil(num_images / num_hosts)
  num_batches = num_local_devices * math.ceil(num_batches / num_local_devices)

  rgb_list = []
  alpha_list = []
  viewdir_list = []
  ref_list = []
  for i in range(num_batches):
    base_index = i * num_hosts
    dataset_index = base_index + host_id

    rgb = np.zeros(
        (source_dataset.h, source_dataset.w, scene_params["_channels"]),
        dtype=np.float32)
    alpha = np.zeros((source_dataset.h, source_dataset.w, 1), dtype=np.float32)
    viewdirs = np.zeros((source_dataset.h, source_dataset.w, 3),
                        dtype=np.float32)

    if dataset_index < num_images:
      rgb, alpha = rendering.atlas_raymarch_image_tf(
          source_dataset.h, source_dataset.w, source_dataset.focal,
          source_dataset.camtoworlds[dataset_index], atlas_t,
          atlas_block_indices_t, atlas_params, scene_params, grid_params)
      _, _, viewdirs = datasets.rays_from_camera(
          scene_params["_use_pixel_centers"], source_dataset.h,
          source_dataset.w, source_dataset.focal,
          np.expand_dims(source_dataset.camtoworlds[dataset_index], 0))

    np_rgb = np.array(rgb).reshape(
        (source_dataset.h, source_dataset.w, scene_params["_channels"]))
    np_alpha = np.array(alpha).reshape((source_dataset.h, source_dataset.w, 1))
    np_viewdirs = viewdirs.reshape((np_rgb.shape[0], np_rgb.shape[1], 3))
    if scene_params["white_bkgd"]:
      np_rgb[Ellipsis, 0:3] = np.ones_like(
          np_rgb[Ellipsis, 0:3]) * (1.0 - np_alpha) + np_rgb[Ellipsis, 0:3]

    rgb_list.append(np_rgb)
    alpha_list.append(np_alpha)
    viewdir_list.append(np_viewdirs)
    ref_list.append(source_dataset.images[dataset_index % num_images])

  rgb_data = np.stack(rgb_list, 0).reshape(
      (-1, num_local_devices, source_dataset.h, source_dataset.w,
       scene_params["_channels"]))
  alpha_data = np.stack(alpha_list, 0).reshape(
      (-1, num_local_devices, source_dataset.h, source_dataset.w, 1))
  viewdir_data = np.stack(viewdir_list, 0).reshape(
      (-1, num_local_devices, source_dataset.h, source_dataset.w, 3))
  ref_data = np.stack(ref_list, 0).reshape(
      (-1, num_local_devices, source_dataset.h, source_dataset.w, 3))

  return rgb_data, alpha_data, viewdir_data, ref_data


def eval_dataset_and_unshard(viewdir_mlp_model, viewdir_mlp_params,
                             rgb_features, directions, source_dataset,
                             scene_params):
  """Evaluates view-dependence on a sharded dataset and unshards the result.

  This function evaluates the view-dependence MLP on a dataset, adding back
  effects such as highlights.

  To make use of multi-host parallelism provided by JAX, this function takes as
  input a shardeds dataset, so each host only evaluates a slice of the data.
  Note that this function unshards the data before returning, which broadcasts
  the results back to all JAX hosts.

  Args:
    viewdir_mlp_model: A nerf.model_utils.MLP that predicts the per-ray
      view-dependent residual color.
    viewdir_mlp_params: A dict containing the MLP parameters for the per-ray
      view-dependence MLP.
    rgb_features: The RGB (+ features) input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 7) numpy array.
    directions: he direction vectors for the input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 3) numpy array.
    source_dataset: The nerf.datasets.Dataset we are evaluating.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).

  Returns:
    A list of color images, each stored as a (H, W, 3) numpy array.
  """

  @functools.partial(jax.pmap, in_axes=(0, 0), axis_name="batch")
  def pmap_eval_fn(rgb_and_feature_chunk, direction_chunk):
    """We need an inner function as only JAX types can be passed to a pmap."""
    residual = model_utils.viewdir_fn(viewdir_mlp_model, viewdir_mlp_params,
                                      rgb_and_feature_chunk, direction_chunk,
                                      scene_params)
    output = jnp.minimum(1.0, rgb_and_feature_chunk[Ellipsis, 0:3] + residual)
    return jax.lax.all_gather(output, axis_name="batch")

  num_hosts = jax.host_count()
  num_local_devices = jax.local_device_count()
  num_images = source_dataset.camtoworlds.shape[0]
  num_batches = math.ceil(num_images / num_hosts)
  num_batches = num_local_devices * math.ceil(num_batches / num_local_devices)

  outputs = []
  for i in range(len(rgb_features)):
    # First, evaluate the loss in parallel across all devices.
    output_batch = pmap_eval_fn(rgb_features[i], directions[i])
    output_batch = np.reshape(
        output_batch[0],
        (num_hosts, num_local_devices, source_dataset.h, source_dataset.w, 3))

    # Then, make sure to populate the output array in the same order
    # as the original dataset.
    for j in range(num_local_devices):
      base_index = (i * num_local_devices + j) * num_hosts
      for k in range(num_hosts):
        gathered_dataset_index = base_index + k
        if gathered_dataset_index >= num_images:
          break

        outputs.append(np.array(output_batch[k][j]).reshape(
            (source_dataset.h, source_dataset.w, 3)))

  return outputs


def refine_view_dependence_mlp(rgbs,
                               directions,
                               refs,
                               viewdir_mlp_model,
                               viewdir_mlp_params,
                               scene_params,
                               learning_rate=2e-4,
                               num_epochs=500):
  """Refines the view-dependence MLP ona sharded dataset.

  To make use of multi-host parallelism provided by JAX, this function takes as
  input a shardeds dataset, so each host only evaluates a slice of the data.

  Args:
    rgbs: The RGB (+ features) input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 7) numpy array.
    directions: he direction vectors for the input data, stored as an
      (N/NUM_HOSTS, NUM_LOCAL_DEVICES, H, W, 3) numpy array.
    refs: The nerf.datasets.Dataset we are evaluating.
    viewdir_mlp_model: A nerf.model_utils.MLP that predicts the per-ray
      view-dependent residual color.
    viewdir_mlp_params: A dict containing the MLP parameters for the per-ray
      view-dependence MLP.
    scene_params: A dict for scene specific params (bbox, rotation, resolution).
    learning_rate: The learning rate for the Adam optimizer.
    num_epochs: The number of epochs we will optimize for.

  Returns:
    A dict with the refined parameters for the per-ray view-dependence MLP.
  """

  def train_step(model, state, ref, rgb_features, directions, lr):
    """One optimization step for the view-dependence MLP.

    Args:
      model: The linen model for the view-dependence MLP.
      state: utils.TrainState, state of the model/optimizer.
      ref: reference image.
      rgb_features: diffuse rgb image, with latent features for view-dependence.
      directions: direction vectors for each pixel.
      lr: float, real-time learning rate.

    Returns:
      new_state: utils.TrainState, new training state.
      loss: The training loss (L2) for the current minibatch.
    """

    def loss_fn(variables):
      residual = model_utils.viewdir_fn(model, variables, rgb_features,
                                        directions, scene_params)
      final_rgb = jnp.minimum(1.0, rgb_features[Ellipsis, 0:3] + residual)
      loss = ((final_rgb - ref[Ellipsis, :3])**2).mean()
      return loss

    loss, grad = jax.value_and_grad(loss_fn)(state.optimizer.target)
    grad = jax.lax.pmean(grad, axis_name="batch")
    loss = jax.lax.psum(loss, axis_name="batch")

    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(optimizer=new_optimizer)
    return new_state, loss

  optimized_viewdir_params = viewdir_mlp_params
  optimizer = flax.optim.Adam(learning_rate).create(optimized_viewdir_params)
  state = utils.TrainState(optimizer=optimizer)
  train_pstep = jax.pmap(
      functools.partial(train_step, viewdir_mlp_model),
      axis_name="batch",
      in_axes=(0, 0, 0, 0, None),
      donate_argnums=(1, 2, 3,))
  state = flax.jax_utils.replicate(state)

  # Our batch size automatically changes to match the number of ML accelerators.
  # To keep the result somewhat consistent between hardware platforms, we
  # compensate for this by scaling the learning rate accordingly.
  num_local_devices_used_for_tuning = 8
  batch_aware_learning_rate = (
      learning_rate * jax.device_count()) / num_local_devices_used_for_tuning
  for _ in range(num_epochs):
    for i in range(rgbs.shape[0]):
      state, _ = train_pstep(state, refs[i], rgbs[i], directions[i],
                             batch_aware_learning_rate)

  return flax.jax_utils.unreplicate(state.optimizer.target)
