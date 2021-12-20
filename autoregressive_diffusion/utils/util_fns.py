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

"""Contains util functions that are needed often.

This contains often used transformations / functions that are very general, but
complicated enough that they warrant an implementation in this file.
"""
import io
import os
from typing import Optional

import flax
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


Array = jnp.ndarray


def get_iterator(
    ds,
    prefetch = 3):
  """Make a prefetch-to-device JAX iterator for the dataset.

  Args:
    ds: Dataset to obtain an iterator for.
    prefetch: Number of dataset entries to pre-fetch to device.

  Returns:
    Dataset iterator.
  """
  # Convert to numpy.
  it = map(lambda x: jax.tree_map(lambda y: y._numpy(), x), iter(ds))  # pylint: disable=protected-access
  if prefetch:
    it = flax.jax_utils.prefetch_to_device(it, prefetch)
  return it


def apply_weight(x, weight):
  """Apply weights an array. Broadcast if necessary."""
  if len(x.shape) < len(weight.shape):
    raise ValueError(f'Incompatible number of dimensions for {x.shape} and '
                     f'{weight.shape}.')
  for i, (dx, dw) in enumerate(zip(x.shape, weight.shape)):
    if dx != dw and dw != 1:
      raise ValueError(f'Unable to brodcast shapes {x.shape} and {weight.shape}'
                       f'in dimension {i}.')
  weight = jnp.reshape(
      weight, weight.shape + (1,) * (len(x.shape) - len(weight.shape)))
  return x * weight


def global_norm(tree, eps=1e-10):
  return jnp.sqrt(eps + jnp.sum(jnp.asarray(
      [jnp.sum(jnp.square(x)) for x in jax.tree_leaves(tree)])))


def clip_by_global_norm(tree, clip_norm, eps=1e-10):
  norm = global_norm(tree)
  scale = jnp.minimum(1.0, clip_norm / norm + eps)
  return jax.tree_map(lambda x: x * scale, tree), norm


def batch_permute(array, permutation):
  """Permutes an input array using permutations, batched.

  This function permutes the array using the permutation array as indexing.
  Importantly, this is done in a batchwise fashion, so each array has its own
  individual permutation.

  Args:
     array: The array to permute with size (batch_size, length, ...).
     permutation: The permutations with size (batch_size, length)
  Returns:
     The array permuted.
  """
  assert array.shape[:2] == permutation.shape, (f'{array.shape} does not '
                                                f'match {permutation.shape}')
  batch_size = permutation.shape[0]

  return array[jnp.arange(batch_size)[:, None], permutation, Ellipsis]


def compute_batch_inverse_permute(permutation):
  """Permutes an inverses of permutations, batched.

  Args:
     permutation: The permutations with size (batch_size, length)
  Returns:
     The inverse permutation, also with size (batch_size, length).
  """
  batch_size, num_steps = permutation.shape
  temp = jnp.full_like(permutation, fill_value=-1)
  arange = jnp.arange(num_steps)[None, :].repeat(batch_size, axis=0)

  # In numpy this would read approximately as follows:
  #   inv_permute[jnp.arange(batch_size)[:, None], permutation] = arange
  # and it essentially inverts the permutation operation by writing the original
  # index to the permutation destinations.
  inv_permute = temp.at[jnp.arange(batch_size)[:, None], permutation].set(
      arange)
  return inv_permute


def sum_except_batch(x):
  return x.reshape(x.shape[0], -1).sum(-1)


def batch(x, num):
  batch_size = x.shape[0]
  assert batch_size % num == 0

  if len(x.shape) == 1:
    return x.reshape(num, batch_size // num)
  else:
    return x.reshape(num, batch_size // num, *x.shape[1:])


def unbatch(x):
  return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def add_empty_axes_behind(x, number):
  new_shape = x.shape + (1,) * number
  return x.reshape(new_shape)


def plot_loss_components(kl_history, save_path, num_stages, max_plots=20):
  """Plots an area plot for the KLs over time."""
  if jax.process_index() == 0:
    assert len(kl_history) >= 1
    num_kls = len(kl_history[0])

    if len(kl_history) < max_plots:
      ts = range(len(kl_history))
    else:
      ts = np.asarray((np.linspace(0, len(kl_history)-1, max_plots)), np.int32)

    num_timesteps = len(ts)

    for i, t in enumerate(ts):
      kls = kl_history[t]
      linear_scale = 1. - (i+1) / float(num_timesteps)
      color = (0.9 * linear_scale, 0.9 * linear_scale, 1.0)
      # kls are multiplied with num_timesteps, to see there contribution better
      # in relation to average bpd values.
      plt.fill_between(
          np.arange(num_kls), kls * len(kls), color=color, alpha=0.9)

    if num_stages > 1:
      steps_per_stage = num_kls // num_stages

      for i in range(1, num_stages):
        x_value = steps_per_stage * i - 0.5
        # draw black vertical line:
        plt.plot(
            np.array([x_value, x_value]),
            np.array([0., 1.]),
            'k--',
            linewidth=1)

    plt.ylim((0, 2 * len(kl_history[-1]) * np.max(kl_history[-1])))

    tf.io.gfile.makedirs(os.path.dirname(save_path))
    with tf.io.gfile.GFile(save_path, 'wb') as out_file:
      plt.savefig(out_file, dpi=200)

    plt.show()
    plt.close()


def plot_batch_images(batch_imgs, n_rows, n_classes):
  grid = make_grid(batch_imgs, n_rows)
  plt.imshow(grid / (n_classes - 1.))
  plt.show()
  plt.close()


def make_grid(batch_imgs, n_rows):
  """Makes grid of images."""
  batch_imgs = np.array(batch_imgs)
  assert len(batch_imgs.shape) == 4, f'Invalid shape {batch_imgs.shape}'

  batchsize, height, width, channels = batch_imgs.shape

  n_cols = (batchsize + n_rows - 1) // n_rows
  grid = np.zeros((n_rows * height, n_cols * width, channels))

  for i, img in enumerate(batch_imgs):
    y = i // n_cols
    x = i % n_cols
    grid[y*height:(y+1)*height, x*width:(x+1)*width, :] = img

  if channels == 1:
    grid = np.concatenate([grid, grid, grid], axis=-1)

  # Upsample if low res to avoid visualization artifacts.
  if height <= 32:
    upsample_factor = 2
    grid = grid.repeat(upsample_factor, axis=0).repeat(upsample_factor, axis=1)

  return grid


class KLTracker():
  """Tracks KL divergences per timestep."""

  def __init__(self, num_steps, momentum=0.95):
    self.history = np.zeros(num_steps)
    # Ensured int64 to avoid overflow
    self.n_updates = np.zeros(num_steps, dtype=np.int64)
    self.momentum = momentum

    # Bit penalty if KL unknown. Hardcoded for 8-bit images.
    self.bit_penalty = 8. / num_steps

  def update(self, t_batch, nelbo_batch):
    """Updates buffers with KL divergences per timestep."""
    assert len(t_batch.shape) == 1 and len(nelbo_batch.shape) == 1
    assert len(t_batch) == len(
        nelbo_batch), f'{len(t_batch)} != {len(nelbo_batch)}'

    for t, nelbo in zip(t_batch, nelbo_batch):
      if self.n_updates[t] == 0:
        self.history[t] = nelbo
      else:
        self.history[t] = self.momentum * self.history[t] + (
            1 - self.momentum) * nelbo

      self.n_updates[t] += 1

  def has_history_forall_t(self):
    return np.alltrue(self.n_updates >= 5)

  def get_kl_per_t(self):
    kl_per_t = self.history + np.where(self.n_updates == 0,
                                       self.bit_penalty,
                                       np.zeros_like(self.history))

    return kl_per_t


def onehot(labels, num_classes):
  x = (labels[Ellipsis, None].astype(jnp.int32) == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def save_chain_to_gif(chain, path, n_rows, max_steps=100):
  """Saves list of batches of images to a gif."""
  if jax.process_index() == 0:
    if len(chain) > max_steps:
      idcs = np.linspace(0, len(chain) - 1, max_steps, dtype=np.int32)
    else:
      idcs = np.arange(0, len(chain))

    chain_grid = [make_grid(chain[i], n_rows) for i in idcs]

    # Extend with last frame 10 times to see results better.
    chain_grid.extend(10 * [make_grid(chain[-1], n_rows=n_rows)])

    chain_grid = [np.asarray(x, dtype=np.uint8) for x in chain_grid]

    # Checks if dir already available and creates if not.
    tf.io.gfile.makedirs(os.path.dirname(path))

    with tf.io.gfile.GFile(path, 'wb') as out_file:
      io_buffer = io.BytesIO()
      imageio.mimwrite(io_buffer, chain_grid, format='gif', duration=.01)
      out_file.write(io_buffer.getvalue())
