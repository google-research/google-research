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

"""Contains some util functions that are useful for multiple ARDM model classes.

This file contains several useful functions that are typically used in an ARDM.
For instance, it contains a method to get a batch of permutations, and another
method to get selection masks based on a permutation and a step.
"""
import functools
from typing import Tuple

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from autoregressive_diffusion.utils import distribution_utils


Array = jnp.ndarray


@functools.partial(jax.jit, static_argnums=(1, 2))
def get_batch_permutations(key, batch_size,
                           n_steps):
  """Gets a batch of permutations."""

  @functools.partial(jax.vmap)
  def vmap_get_permutation(key):
    """Retrieves a single permutation, is vmapped to get batch."""
    return jax.random.permutation(key, n_steps)

  batch_permutations = vmap_get_permutation(jax.random.split(key, batch_size))
  return batch_permutations


@functools.partial(jax.jit, static_argnums=(2,))
def get_selection_for_sigma_and_t(
    sigmas, t, target_shape):
  """Sample time and get selection masks based on a random order (sigma).

  Args:
      sigmas: permutation of a range of integers, shape [batch_size, num_steps]
      t: the current time/step, shape [batch_size]
      target_shape: the shape that the selections will be reshaped to.

  Returns:
    prev_selection: the selection for tokens previous to t w.r.t sigma.
    current_selection: the selection for tokens previous to t w.r.t sigma.

  sigma gives an order, so if sigmas has shape [batchsize, K], then sigmas is a
  permutation of the integers 0, ..., K-1. Using sampled timesteps t [batchsize]
  it is determined which selection is the current token (sigmas == t) and which
  is its past (sigmas < t).


  """
  assert t.shape[0] == sigmas.shape[0]
  assert len(sigmas.shape) == 2
  batch_size = t.shape[0]
  less_than_selection_flat = (sigmas < t[:, None]).astype(jnp.int32)
  equal_to_selection_flat = (sigmas == t[:, None]).astype(jnp.int32)

  less_than_selection_flat = less_than_selection_flat.reshape(
      batch_size, *target_shape)
  equal_to_selection = equal_to_selection_flat.reshape(batch_size,
                                                       *target_shape)
  return less_than_selection_flat, equal_to_selection


def get_selections_for_sigma_and_range(
    sigmas, left_t,
    right_t, target_shape):
  """Sample time and get selection masks based on a random order (sigma).

  Args:
      sigmas: [batch_size, num_steps] permutation of a range of integers.
      left_t: [batch_size] the current time/step.
      right_t: [batch_size] until which time/step is generated.
      target_shape: the shape that the selections will have.

  Returns:
    prev_selection: the selection for tokens previous to t w.r.t sigma.
    current_selection: the selection for tokens previous to t w.r.t sigma.

  sigma gives an order, so if sigmas has shape [batchsize, K], then sigmas is a
  permutation of the integers 0, ..., K-1. Using sampled timesteps t [batchsize]
  it is determined which selection is the current token (sigmas == t) and which
  is its past (sigmas < t).
  """
  assert sigmas.shape[0] == left_t.shape[0], (
      f'{sigmas.shape} does not match {left_t.shape}')
  assert left_t.shape == right_t.shape
  assert len(sigmas.shape) == 2
  batch_size = left_t.shape[0]
  prev_selection_flat = (sigmas < left_t[:, None]).astype(jnp.int32)

  # Current selection is between left_t (inclusive) and right_t (exclusive).
  current_selection_flat = (sigmas >= left_t[:, None]).astype(
      jnp.int32) * (sigmas < right_t[:, None]).astype(jnp.int32)

  prev_selection = prev_selection_flat.reshape(batch_size, *target_shape)
  current_selection = current_selection_flat.reshape(batch_size, *target_shape)
  return prev_selection, current_selection


def integer_linspace(start, stop, steps):
  """Creates a linspace but rounded to integers, practical if skipping steps."""
  assert steps <= stop - start + 1, f'with {start}, {stop}, {steps}'
  linspace = jnp.linspace(start, stop, steps)
  return jnp.asarray(jnp.round(linspace), dtype=jnp.int32)


def prune_chain(chain, max_frames):
  """Prunes a chain for a maximum number of frames."""
  if len(chain) <= max_frames:
    return chain
  else:
    keep_frames = integer_linspace(0, len(chain) - 1, max_frames)
    return chain[keep_frames]


def sample_antithetic(rng, batch_size, max_val):
  """Samples linspaces for antithetic sampling."""
  # Antithetic sampling essentially samples linspaces of integers to reduce
  # the variance of the estimator.
  sample_rng, perm_rng = jax.random.split(rng)

  # The greatest common divisor between the model num_steps and batch_size
  # is found. From this a spacing (for the linspace) and the number of
  # actual samples we have to draw is derived.
  gcd = np.gcd(batch_size, max_val)
  spacing = max_val // gcd
  num_samples = batch_size // gcd

  msg = f'Antithetic sampling using gcd {gcd}, spacing {spacing} num_samples {num_samples} for a batch_size of {batch_size}'
  logging.info(msg)

  # t is sampled between num_samples and repeated 'gcd' times.
  t = jax.random.randint(
      sample_rng, shape=(num_samples,), minval=0, maxval=spacing)
  t = jnp.repeat(t, repeats=gcd, axis=0)  # size: [batch_size]

  # The linspace [0, spacing, 2 * spacing, ..., (gcd - 1) * spacing] is
  # added. It is subsequently tiles so that there is a linspace for each
  # sample in num_samples.
  spacing_range = jnp.arange(gcd) * spacing
  spacing_range = jnp.tile(spacing_range, num_samples)

  # The two are added together, creating num_samples linspaces.
  t = t + spacing_range

  # t will now always increase over parts of the array. We want to avoid
  # alignment when deterministically going over minibatches. So, we randomly
  # permute.
  t = jax.random.permutation(perm_rng, t)

  # Actually we need the weight 1 / num_samples / gcd == 1 / num_steps, this
  # may technically not be prob_qt, depending on your perspective.
  prob_qt = 1. / max_val
  return t, prob_qt


@jax.jit
def get_probs_coding(logits, selection, x=None):
  """Retrieves coding probabilities from logits and selection.

  This function retrieves probabilities from a logits array, only for the
  specified selection. Optionally it also selects the relevant x.

  Args:
    logits: Logits array where the last axis represents the categories.
    selection: Selection mask matching the shape of logits.
    x: Optional input x to select in the same manner as logits.

  Returns:
    The (safe) probabilities to encode x with. Optionally also the selected x.
  """
  assert logits.shape[:-1] == selection.shape
  reduce_axes = tuple(range(1, len(selection.shape)))

  # Selects the relevant variable via selection mask.
  logits = jnp.sum(logits * selection[Ellipsis, None], axis=reduce_axes,
                   keepdims=True)

  probs = distribution_utils.get_safe_probs(logits)

  if x is not None:
    x = jnp.sum(x * selection, axis=reduce_axes, keepdims=True)
    return probs, x
  else:
    return probs


def encode(streams, x, probs):
  """Encodes the variable x at location selection using the distribution.

  Args:
    streams: Batch of bitstream objects to encode to.
    x: Variable to encode.
    probs: The probabilities for x.

  Returns:
    The bitstream objects.
  """
  batch_size = x.shape[0]
  num_classes = probs.shape[-1]

  # Convert to numpy, flatten.
  x_flat = np.asarray(x, dtype=np.uint64).reshape(batch_size)
  probs_flat = np.asarray(
      probs, dtype=np.float64).reshape(batch_size, num_classes)

  for i in range(batch_size):
    streams[i].encode_cat(x=x_flat[i:i+1], probs=probs_flat[i:i+1])

  return streams


def decode(streams, probs):
  """Decodes a variable using the distribution.

  Args:
    streams: Batch of bitstream objects to decode from.
    probs: Distribution parameters for x.

  Returns:
    A tuple with the decoded variable, and the bitstream objects.
  """
  batch_size = probs.shape[0]
  num_classes = probs.shape[-1]

  probs_flat = np.asarray(
      probs, dtype=np.float64).reshape(batch_size, num_classes)

  x_decoded = np.zeros(batch_size, dtype=np.int32)
  for i in range(batch_size):
    x_decoded[i] = streams[i].decode_cat(probs=probs_flat[i:i+1])

  # Inflate x to match the probs shape
  empty_axes = (1,) * (len(probs.shape) - 2)
  x_decoded = x_decoded.reshape(batch_size, *empty_axes)
  x_decoded = x_decoded.astype(np.int32)

  return x_decoded, streams
