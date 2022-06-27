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

"""Utilities for training and loading distance prediction models.
"""

import functools
from typing import Callable, Dict, Sequence

from acme.jax.types import PRNGKey, Variables  # pylint: disable=g-multiple-import
import flax
from flax import struct
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow.io import gfile
from func_dist.agents.dist_regression import networks


class TrainingState(struct.PyTreeNode):
  """Training state for distance model."""
  step: int
  epoch: int
  best_loss: Dict[str, float]
  encoder_fn: Callable[[Variables, jnp.ndarray], jnp.ndarray] = (
      struct.field(pytree_node=False))
  distance_fn: Callable[[Variables, jnp.ndarray], jnp.ndarray] = (
      struct.field(pytree_node=False))
  domain_discriminator_fn: Callable[[Variables, jnp.ndarray], jnp.ndarray] = (
      struct.field(pytree_node=False))
  distance_optimizer: optax.OptState
  domain_optimizer: optax.OptState


def create_train_state(
    distance_model,
    domain_discriminator,
    key,
    learning_rate):
  """Initialize a TrainingState."""

  key1, key2, key3 = jax.random.split(key, 3)
  dummy_img = jnp.zeros((1, 48, 48, 3))
  dummy_distance = jnp.zeros((1, 1))
  dummy_domain = jnp.zeros((1, 1))

  encoder_params = distance_model.init(
      key1, dummy_img, method=distance_model.encode)
  dummy_emb = distance_model.apply(
      encoder_params, dummy_img, method=distance_model.encode)
  concat_dummy_emb = jnp.concatenate([dummy_emb, dummy_emb], axis=1)

  distance_params = distance_model.init(
      key2, concat_dummy_emb, method=distance_model.predict_distance)
  domain_discriminator_params = domain_discriminator.init(key3, dummy_emb)

  dummy_dist_pred = distance_model.apply(
      distance_params, concat_dummy_emb,
      method=distance_model.predict_distance)
  dummy_domain_pred = domain_discriminator.apply(
      domain_discriminator_params, dummy_emb)

  optimizer_def = flax.optim.Adam(learning_rate=learning_rate)
  distance_model_params = {
      'params': {**encoder_params['params'], **distance_params['params']}}
  distance_optimizer = optimizer_def.create(distance_model_params['params'])
  domain_optimizer = optimizer_def.create(domain_discriminator_params['params'])

  # Loss keys need to be included in the dictionary in order to be restored.
  loss_names = [
      'adversarial_domain_loss', 'distance_error', 'distance_loss',
      'goal_augm_distance_error', 'goal_augm_distance_loss',
      'paired_loss', 'total_distance_model_loss']
  loss_types = (
      [f'val_{loss}' for loss in loss_names]
      + [f'test_{loss}' for loss in loss_names])
  loss_types.extend(['val_domain_discriminator_loss'])
  mse_losses = [
      'test_affine_mse_1', 'test_affine_mse_10', 'test_affine_mse_all',
      'test_mse_1', 'test_mse_10', 'test_mse_all']
  loss_types.extend(mse_losses)
  loss_types.extend([l.replace('test_', 'test_goal_augm_') for l in mse_losses])
  best_loss = {k: np.inf for k in loss_types}

  state = TrainingState(
      encoder_fn=functools.partial(
          distance_model.apply, method=distance_model.encode),
      distance_fn=functools.partial(
          distance_model.apply, method=distance_model.predict_distance),
      domain_discriminator_fn=domain_discriminator.apply,
      distance_optimizer=distance_optimizer,
      domain_optimizer=domain_optimizer,
      step=0,
      epoch=0,
      best_loss=best_loss)
  return state


def save_checkpoint(
    ckpt_dir, state, label = '', keep=10):
  if isinstance(ckpt_dir, str):
    prefix = f'{label}_checkpoint_' if label else 'checkpoint_'
    checkpoints.save_checkpoint(
        ckpt_dir,
        {'distance_optimizer': state.distance_optimizer,
         'domain_optimizer': state.domain_optimizer,
         'step': state.step,
         'epoch': state.epoch,
         'best_loss': state.best_loss,
         },
        state.epoch,
        prefix=prefix,
        keep=keep)


def create_networks(
    encoder_conv_filters, encoder_conv_size):
  distance_model = networks.CNN(
      conv_features=encoder_conv_filters,
      kernel_size=encoder_conv_size,
      dense_features=[64, 64, 64, 1],
      )
  domain_discriminator = networks.FullyConnectedNet([64, 64, 64, 1])
  return distance_model, domain_discriminator


def restore_or_initialize(
    encoder_conv_filters, encoder_conv_size, key,
    ckpt_path, learning_rate, ckpt_label = ''
    ):
  """Restore state from latest checkpoint, if any; otherwise initialize state.

  Args:
    encoder_conv_filters: number of filters per convolutional layer in the
        distance model's encoder network.
    encoder_conv_size: size of convolutional kernels in the distance model's
        encoder network.
    key: jax PRNG key for initializing parameters.
    ckpt_path: path to file or directory from which to load an existing
        checkpoint. If it is a directory, the most recent checkpoint matching
        ckpt_label is loaded.
    learning_rate: learning rate for optimizers in the training state.
    ckpt_label: label to distinguish between checkpoints for different use
        cases, e.g. 'best' for the best validation loss. Defaults to the most
        recent checkpoint.

  Returns:
    state: newly initialized or restored TrainingState.
  """
  distance_model, domain_discriminator = create_networks(
      [int(i) for i in encoder_conv_filters], encoder_conv_size)
  state = create_train_state(
      distance_model, domain_discriminator, key, learning_rate)
  if (gfile.IsDirectory(ckpt_path)
      and checkpoints.latest_checkpoint(ckpt_path) is None):
    save_checkpoint(ckpt_path, state)
    save_checkpoint(ckpt_path, state, label='init')  # Keep initialization.
  else:
    if gfile.IsDirectory(ckpt_path):
      print('Found existing checkpoints:', gfile.ListDir(ckpt_path))
    prefix = f'{ckpt_label}_checkpoint_' if ckpt_label else 'checkpoint_'
    try:
      state = checkpoints.restore_checkpoint(ckpt_path, state, prefix=prefix)
    except KeyError:
      # Needed for backwards compatibility of older checkpoints.
      keys_to_remove = [
          'val_goal_augm_distance_error',
          'val_goal_augm_distance_loss',
          'test_goal_augm_distance_error',
          'test_goal_augm_distance_loss',
          'test_goal_augm_affine_mse_1',
          'test_goal_augm_affine_mse_10',
          'test_goal_augm_affine_mse_all',
          'test_goal_augm_mse_1',
          'test_goal_augm_mse_10',
          'test_goal_augm_mse_all',
      ]
      for key in keys_to_remove:
        state.best_loss.pop(key, None)
      state = checkpoints.restore_checkpoint(ckpt_path, state, prefix=prefix)
    print(f'Restored from epoch {state.epoch}\n')
  return state
