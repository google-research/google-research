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

"""Library that contains useful aligment primitives."""

# pytype: disable=attribute-error
import functools
from typing import Any, Union

from absl import logging
import flax
from flax.training.train_state import TrainState
import gin
from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image_utils
from internal import models
from internal import rigid_body
from internal import spin_math
import jax
from jax import random
import jax.numpy as jnp
import jaxcam
import numpy as np
import optax


_Array = Union[np.ndarray, jnp.ndarray]


def align_test_camera(
    model,
    model_state,
    cam_idx,
    dataset,
    config,
):
  """Align test camera via SGD optimization based on reprojection error.

  Args:
    model: NeRF model.
    model_state: Completed training sate of NeRF model.
    cam_idx: Test camera index.
    dataset: Test dataset to sample from.
    config: Config file.

  Returns:
    Aligned test camera.
  """
  with gin.config_scope('test'):
    camera_delta = config.test_camera_delta_cls()

  def _loss_fn(params, jax_camera, batch, model, model_state):
    jax_camera = camera_delta.apply(params, jax_camera)
    transformed_cameras = jax.vmap(camera_utils.tuple_from_jax_camera)(
        jax_camera
    )

    rays = camera_utils.cast_ray_batch(
        (*transformed_cameras, None, None), batch.rays, dataset.camtype, xnp=jnp
    )

    renderings, _ = model.apply(
        model_state.params,
        None,
        rays=rays,
        train_frac=1.0,
        compute_extras=False,
        zero_glo=False,
    )

    loss = jnp.mean(optax.l2_loss(batch.rgb, renderings[-1]['rgb']))

    return loss, {'loss': loss}

  def _train_step(camera_state, jax_camera, batch, model_state, model):
    loss_fn = functools.partial(
        _loss_fn,
        jax_camera=jax_camera,
        batch=batch,
        model=model,
        model_state=model_state,
    )
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grad, aux = grad_fn(camera_state.params)
    grad = jax.lax.pmean(grad, axis_name='batch')
    aux = jax.lax.pmean(aux, axis_name='batch')
    updates, opt_state = camera_state.tx.update(
        grad, camera_state.opt_state, camera_state.params
    )
    params = optax.apply_updates(camera_state.params, updates)
    camera_state = camera_state.replace(
        params=params, opt_state=opt_state, step=camera_state.step + 1
    )

    return camera_state, aux

  logging.info('Optimize test camera idx %d', cam_idx)

  test_cameras = dataset.cameras
  pixtocams, poses, distortion_params = test_cameras[:3]
  if not distortion_params:
    distortion_params = {
        'k1': 0.0,
        'k2': 0.0,
        'k3': 0.0,
    }
    distortion_params = jax.tree_util.tree_map(
        lambda x: np.zeros(test_cameras[0].shape[0]), distortion_params
    )
  test_cameras = (pixtocams, poses, distortion_params, *test_cameras[3:])

  key = random.PRNGKey(config.jax_rng_seed)

  train_pstep = jax.pmap(
      functools.partial(_train_step, model=model),
      axis_name='batch',
      in_axes=(0, 0, None, None),
  )

  camera = jax.tree_util.tree_map(
      lambda x: x[cam_idx : cam_idx + 1], test_cameras
  )

  image_sizes = np.array(
      [(x.shape[1], x.shape[0]) for x in dataset.images[cam_idx : cam_idx + 1]]
  )
  jax_camera = jax.vmap(dataset.jax_camera_from_tuple_fn)(camera, image_sizes)

  camera_params = camera_delta.init({'params': key}, jax_camera)
  logging.info('learning rate %f', config.optimize_test_cameras_lr)
  tx = optax.adam(config.optimize_test_cameras_lr)
  tx = optax.chain(tx, optax.zero_nans())
  camera_state = TrainState.create(apply_fn=None, params=camera_params, tx=tx)
  camera_state_replicated = flax.jax_utils.replicate(camera_state)

  jax_camera_replicated = flax.jax_utils.replicate(jax_camera)

  for idx in range(config.optimize_test_cameras_for_n_steps):
    batch = dataset.generate_flattened_ray_batch(
        cam_idx, config.optimize_test_cameras_batch_size
    )
    camera_state_replicated, aux = train_pstep(
        camera_state_replicated, jax_camera_replicated, batch, model_state
    )
    aux = flax.jax_utils.unreplicate(aux)
    mse = aux['loss']
    psnr = image_utils.mse_to_psnr(mse)
    logging.info('Step: %d, mse %f, psnr %f', idx, mse, psnr)

  camera_state = flax.jax_utils.unreplicate(camera_state_replicated)
  jax_camera = camera_delta.apply(camera_state.params, jax_camera)
  optimized_camera = jax.vmap(camera_utils.tuple_from_jax_camera)(jax_camera)

  # Add last two tuple items as these do not change and are not supported by
  # jax cameras.
  return (*optimized_camera, *camera[3:])


def procrustes(p1, p2):
  """Compute orthogonal procrustes alignment."""
  p1 = np.array(p1)
  p2 = np.array(p2)
  p1_mean = p1.mean(axis=0)
  p2_mean = p2.mean(axis=0)
  s1 = np.sqrt(np.sum((p1 - p1_mean) ** 2))
  s2 = np.sqrt(np.sum((p2 - p2_mean) ** 2))
  x1 = (p1 - p1_mean) / s1
  x2 = 1.0 / s2 * (p2 - p2_mean)
  u, _, vt = np.linalg.svd(x1.T @ x2)
  r = np.dot(u, vt)
  if r[0, 0] < 0:
    r = r @ np.diag(np.array([-1, 1, 1]))
  if r[1, 1] < 0:
    r = r @ np.diag(np.array([1, -1, 1]))
  if r[2, 2] < 0:
    r = r @ np.diag(np.array([1, 1, -1]))

  # Another option would be to only flip if det < 1
  # if np.linalg.det(r) < 0:
  #   r = np.diag(np.array([1,1,-1])) @ r
  return s1, s2, p1_mean, p2_mean, r


def translation_transform(t):
  """Compute translation transform."""
  result = np.eye(4)
  result[:3, 3] = t
  return result


def rotation_transform(r):
  """Compute rotation transform given rotation matrix."""
  result = np.eye(4)
  result[:3, :3] = r
  return result


def scale_transform(s):
  """Compute scale transform."""
  return np.diag([s, s, s, 1.0])


def transform_camera(camera, transform):
  rotation, _, _ = rigid_body.sim3_to_rts(transform)
  aligned_orientation = camera.orientation @ rotation.T
  aligned_position = spin_math.apply_homogeneous_transform(
      transform, camera.position
  )
  return camera.replace(
      orientation=aligned_orientation,
      position=aligned_position,
  )


def compute_procrusted_aligned_cameras(
    train_jax_cameras_opt,
    train_jax_cameras_gt,
    test_jax_cameras,
):
  """Align test cameras based on global rigid transform and scale."""
  logging.info('Align cameras with procrustes.')

  # We must fetch the ground truth train cameras separately since the actual
  # train cameras may have been perturbed etc.

  # Procrustes transform that maps train coordinates to optimized coordinates.
  opt_scale, train_scale, opt_mean, train_mean, rotation = procrustes(
      train_jax_cameras_opt.position, train_jax_cameras_gt.position
  )
  transform_opt_from_train = (
      translation_transform(opt_mean)
      @ scale_transform(opt_scale)
      @ rotation_transform(rotation)
      @ scale_transform(1 / train_scale)
      @ translation_transform(-train_mean)
  )

  test_jax_cameras_procrustes = jax.vmap(transform_camera, in_axes=(0, None))(
      test_jax_cameras, transform_opt_from_train
  )
  train_jax_cameras_procrustes = jax.vmap(transform_camera, in_axes=(0, None))(
      train_jax_cameras_gt, transform_opt_from_train)

  # Apply transform.
  return train_jax_cameras_procrustes, test_jax_cameras_procrustes
