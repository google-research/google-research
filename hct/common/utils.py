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

"""Generic helper utilities."""

import functools
from typing import Any, Optional

import chex
from flax import core
from flax.training import checkpoints
import flax.training.train_state as ts
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hct.common import typing


def param_count(params):
  return sum(x.size for x in jax.tree_util.tree_leaves(params))


def check_params_finite(params):
  return jnp.array(
      [jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(params)]).all()


class TrainStateBN(ts.TrainState):
  """Train-state with batchnorm batch-stats."""
  batch_stats: core.FrozenDict[str, Any]


def make_optax_adam(learning_rate,
                    weight_decay):
  if weight_decay > 0:
    return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
  else:
    return optax.adam(learning_rate=learning_rate)


@functools.partial(jax.jit, static_argnums=(0,))
def split_across_devices(num, x):
  """Split batch across devices."""
  return jnp.reshape(x, (num, x.shape[0] // num) + x.shape[1:])


def insert_batch_axis(x):
  return jax.tree_util.tree_map(lambda leaf: leaf[None, Ellipsis], x)


def remove_batch_axis(x):
  return jax.tree_util.tree_map(lambda leaf: jnp.squeeze(leaf, axis=0), x)


def unbatch_flax_fn(fn, has_params = True):
  """Unbatch flax fn."""
  # assumes all args are passed in as batchified
  # and all kwargs are to be broadcasted
  if has_params:
    def unbatched_fn(params, *args, **kwargs):
      batched_args = map(insert_batch_axis, args)
      return remove_batch_axis(fn(params, *batched_args, **kwargs))
    return unbatched_fn
  else:
    def unbatched_fn(*args, **kwargs):
      batched_args = map(insert_batch_axis, args)
      return remove_batch_axis(fn(*batched_args, **kwargs))
    return unbatched_fn


class BatchManager:
  """A simple batch manager."""

  def __init__(self,
               key,
               dataset,
               batch_size):
    self._prng = hk.PRNGSequence(key)
    self._num = len(dataset['images'])
    assert len(dataset['hf_obs']) == self._num
    assert len(dataset['actions']) == self._num

    # Ensure saved copy off accelerator device
    self._dataset = {k: jax.device_get(arr) for k, arr in dataset.items()}
    self._batch_size = min(batch_size, self._num)
    self._num_batches = self._num // self._batch_size
    self._epochs_completed = 0

    self._permutation = None
    self._current_batch_idx = None
    self._resample()

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_batches(self):
    return self._num_batches

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def _resample(self):
    self._permutation = jax.random.permutation(next(self._prng), self._num)
    total_points = self._batch_size * self._num_batches
    self._permutation = self._permutation[:total_points].reshape(
        (self._num_batches, self._batch_size))
    self._current_batch_idx = 0

  def _select(self, inds):
    return {k: arr[inds] for k, arr in self._dataset.items()}

  def next_batch(self):
    """Get the next batch."""
    assert self._permutation is not None
    inds = self._permutation[self._current_batch_idx]
    ret = self._select(inds)
    self._current_batch_idx += 1
    if self._current_batch_idx >= len(self._permutation):
      self._epochs_completed += 1
      self._resample()
    return ret

  def next_pmapped_batch(self, num_devices):
    # assert that num_devices is compatible with batch size
    assert self._batch_size % num_devices == 0
    ret = self.next_batch()
    return {k: split_across_devices(num_devices, arr) for k, arr in ret.items()}


def compute_norm_stats(
    dataset):
  """Compute mean and std pytrees."""
  means = {k: np.mean(arr, axis=0) for k, arr in dataset.items()}
  stds = {k: np.std(arr, axis=0) for k, arr in dataset.items()}

  # Force normalization for images to be zero mean, std = 255.
  means['images'] = np.zeros_like(means['images'])
  stds['images'] = 255. * np.ones_like(stds['images'])
  return means, stds


def normalize(dataset,
              means,
              stds):
  """Normalize dataset."""
  return jax.tree_util.tree_map(
      lambda leaf, leaf_mean, leaf_std: (leaf - leaf_mean) / leaf_std,
      dataset, means, stds)


def unnormalize(arr,
                arr_mean,
                arr_std):
  """Unormalize array."""
  return arr * arr_std + arr_mean


def save_model(checkpoint_dir, step,
               keep_every, state):
  """Checkpoints and saves models."""
  checkpoints.save_checkpoint(
      ckpt_dir=checkpoint_dir,
      target=state,
      step=step,
      overwrite=True,
      keep_every_n_steps=keep_every)


def restore_model(checkpoint_dir, state,
                  step = None):
  """Restore all models."""
  # NOTE: Assumes states have pre-defined structures.
  return checkpoints.restore_checkpoint(checkpoint_dir, state, step=step)
