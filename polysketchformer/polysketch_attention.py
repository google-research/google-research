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

"""Implements Polysketch Attention in the causal setting.

This file includes the implementation of the polysketch attention mechanism
from https://arxiv.org/abs/2310.01655.
"""

import enum
from typing import Callable

import jax
import jax.numpy as jnp
import scipy.linalg

from polysketchformer import lower_triangular_multiplication
from polysketchformer import mixed_lower_triangular_multiplication


@enum.unique
class SketchType(enum.Enum):
  """Defines the different types of sketches that can be used."""

  GAUSSIAN = enum.auto()
  HADAMARD = enum.auto()


def _sample_randomized_hadamard(
    key, head_size, sketch_size
):
  """Samples a randomized hadamard transform."""
  assert head_size & (head_size - 1) == 0  # head_size has to be a power of 2
  key1, key2 = jax.random.split(key)
  diag = jax.random.randint(
      key1, shape=(head_size,), minval=0, maxval=2, dtype=jnp.float32
  )
  diag = 2 * diag - 1
  subset_of_cols = jax.random.randint(
      key2, shape=(sketch_size,), minval=0, maxval=head_size
  )
  hadamard_matrix = scipy.linalg.hadamard(head_size)
  sampled_hadamard = hadamard_matrix[:, subset_of_cols]
  randomized_sampled_hadamard = diag[:, None] * sampled_hadamard
  return randomized_sampled_hadamard


def make_polysketch_attn_fn(
    feature_dimension,
    sketch_key,
    sketch_size,
    is_causal = True,
    grain_size = 256,
    sketch_type = SketchType.GAUSSIAN,
    precision = jax.lax.Precision.DEFAULT,
):
  """Returns a function which computes the output of the attention mechanism.

  Using the sketch_key, this function samples a sketch of type sketch_type and
  then constructs a function that takes query, key, value matrices and applies
  the polysketch attention mechanism.

  If is_causal is set to True, then the returned function applies the causal
  attention mechanism to the query, key, value matrices using the lower
  triangular multiplication algorithm with a block size = grain_size.

  If is_causal is set to False, then the returned function simply applies the
  attention mechanism without using the causal mask.

  Args:
    feature_dimension: Number of features in the query, key matrices for which
      we want to apply attention mechanism.
    sketch_key: Random key that is to be used to sample the sketching matrix.
    sketch_size: Size of the sketch that is to be applied to the query, key
      matrices.
    is_causal: If set to True, then a causal mask is applied to the attention
      matrix.
    grain_size: Relevant only when is_causal=True. Denotes the grain_size to be
      used when calling the lt_tensor_multiply function.
    sketch_type: The type of sketch to be used.
    precision: The precision to be used for lt_multiply algorithm.

  Returns:
    A function that takes query, key and value matrices as inputs and computes
    the output of the polysketch attention mechanism. When is_causal=True, the
    returned function applies only to query, key and value matrices with context
    length that is divisible by grain_size.
  """
  if sketch_type == SketchType.GAUSSIAN:
    key1, key2 = jax.random.split(sketch_key)
    sketch1 = jax.random.normal(key1, shape=(feature_dimension, sketch_size))
    sketch2 = jax.random.normal(key2, shape=(feature_dimension, sketch_size))
    sketch1 = sketch1 / jnp.sqrt(sketch_size)
    sketch2 = sketch2 / jnp.sqrt(sketch_size)
  elif sketch_type == SketchType.HADAMARD:
    key1, key2 = jax.random.split(sketch_key)
    sketch1 = _sample_randomized_hadamard(key1, feature_dimension, sketch_size)
    sketch2 = _sample_randomized_hadamard(key2, feature_dimension, sketch_size)
    sketch1 = sketch1 / jnp.sqrt(sketch_size)
    sketch2 = sketch2 / jnp.sqrt(sketch_size)

  # We define two functions instead of using is_causal inside the returned
  # function to not increase the size of the returned function unncessarily.
  if is_causal:

    def attention_function(
        query, key, value
    ):
      """The function applies the polysketch attention w/ causal mask.

      Args:
        query: An array of shape [batch, ..., context_length, qk_dim]
        key: An array of shape [batch, ..., context_length, qk_dim]
        value: An array of shape [batch, ..., context_length, v_dim]

      Returns:
        An array of shape [batch, ..., context_length, v_dim]
      """
      query_prime = (query @ sketch1) * (query @ sketch2)
      key_prime = (key @ sketch1) * (key @ sketch2)

      pad_list = [(0, 0)] * len(query.shape)
      pad_list[-1] = (0, 1)
      value_padded = jnp.pad(
          value, pad_list, mode='constant', constant_values=1.0
      )  # Concatenates a column of 1.0s to all the value arrays

      result, _ = lower_triangular_multiplication.tensor_lt_multiply(
          query_prime,
          key_prime,
          value_padded,
          grain_size,
          precision,
      )

      inverse_scalings = result[Ellipsis, -1]
      inverse_scalings = 1.0 + inverse_scalings
      # The normalization factor is added to make the avoid numerical issues
      # during division and make the training stable.

      scalings = jnp.reciprocal(inverse_scalings)
      result = result[Ellipsis, :-1] * scalings[Ellipsis, None]
      return result

  else:

    def attention_function(
        query, key, value
    ):
      """The function applies the polysketch attention w/o causal mask.

      Args:
        query: An array of shape [batch, ..., q_length, qk_dim]
        key: An array of shape [batch, ..., kv_length, qk_dim]
        value: An array of shape [batch, ..., kv_length, v_dim]

      Returns:
        An array of shape [batch, ..., q_length, v_dim]
      """
      query_prime = (query @ sketch1) * (query @ sketch2)
      key_prime = (key @ sketch1) * (key @ sketch2)

      pad_list = [(0, 0)] * len(query.shape)
      pad_list[-1] = (0, 1)
      # Appends a column of 1s to the value matrices before running multiply
      value_padded = jnp.pad(
          value, pad_list, mode='constant', constant_values=1.0
      )

      # Computes (key_prime tensor key_prime).T @ value_padded
      first_product = jnp.einsum(
          '...ki, ...kj, ...kv-> ...ijv',
          key_prime,
          key_prime,
          value_padded,
          precision=precision,
      )
      # Computes (query_prime tensor query_prime) @ first_product
      result = jnp.einsum(
          '...qi, ...qj, ...ijv->...qv',
          query_prime,
          query_prime,
          first_product,
          precision=precision,
      )

      # A normalization factor is added to make the avoid numerical issues
      # during division and make the training stable.
      inverse_scalings = result[Ellipsis, -1]
      inverse_scalings = 1.0 + inverse_scalings
      scalings = jnp.reciprocal(inverse_scalings)
      result = result[Ellipsis, :-1] * scalings[Ellipsis, None]
      return result

  return attention_function


def make_mixed_polysketch_attn_fn(
    feature_dimension,
    sketch_key,
    sketch_size,
    power = 4,
    is_causal = True,
    grain_size = 256,
    sketch_type = SketchType.GAUSSIAN,
    normalization_factor = 1.0,
    precision = jax.lax.Precision.DEFAULT,
):
  """Returns a function which computes the the mixed attention mechanism.

  Using the sketch_key, this function samples a sketch of type sketch_type and
  then constructs a function that takes query, key, value matrices and applies
  the local + learned polysketch attention mechanism. Within a block of size
  grain_size, it applies exact polynomial attention of degree 'power' and
  globally it applies polysketch attention using the sketched sketch matrices.

  If is_causal is set to True, then the returned function applies the causal
  attention mechanism to the query, key, value matrices using the lower
  triangular multiplication algorithm with a block size = grain_size.

  If is_causal is set to False, then the returned function simply applies the
  attention mechanism without using the causal mask.

  Args:
    feature_dimension: Number of features in the query, key matrices for which
      we want to apply attention mechanism.
    sketch_key: Random key that is to be used to sample the sketching matrix.
    sketch_size: Size of the sketch that is to be applied to the query, key
      matrices.
    power: degree of polynomial attention within blocks
    is_causal: If set to True, then a causal mask is applied to the attention
      matrix.
    grain_size: Relevant only when is_causal=True. Denotes the grain_size to be
      used when calling the lt_tensor_multiply function.
    sketch_type: The type of sketch to be used.
    normalization_factor: Constant added in the denominator for stability.
    precision: The precision to be used for lt_multiply algorithm.

  Returns:
    A function that takes query, key and value matrices as inputs and computes
    the output of the polysketch attention mechanism. When is_causal=True, the
    returned function applies only to query, key and value matrices with context
    length that is divisible by grain_size.
  """
  if sketch_type == SketchType.GAUSSIAN:
    key1, key2 = jax.random.split(sketch_key)
    sketch1 = jax.random.normal(key1, shape=(feature_dimension, sketch_size))
    sketch2 = jax.random.normal(key2, shape=(feature_dimension, sketch_size))
    sketch1 = sketch1 / jnp.sqrt(sketch_size)
    sketch2 = sketch2 / jnp.sqrt(sketch_size)
  elif sketch_type == SketchType.HADAMARD:
    key1, key2 = jax.random.split(sketch_key)
    sketch1 = _sample_randomized_hadamard(key1, feature_dimension, sketch_size)
    sketch2 = _sample_randomized_hadamard(key2, feature_dimension, sketch_size)
    sketch1 = sketch1 / jnp.sqrt(sketch_size)
    sketch2 = sketch2 / jnp.sqrt(sketch_size)

  # We define two functions instead of using is_causal inside the returned
  # function to not increase the size of the returned function unncessarily.
  if is_causal:

    def attention_function(
        query, key, value
    ):
      """The function applies the polysketch attention w/ causal mask.

      Args:
        query: An array of shape [batch, ..., context_length, qk_dim]
        key: An array of shape [batch, ..., context_length, qk_dim]
        value: An array of shape [batch, ..., context_length, v_dim]

      Returns:
        An array of shape [batch, ..., context_length, v_dim]
      """
      query_prime = (query @ sketch1) * (query @ sketch2)
      key_prime = (key @ sketch1) * (key @ sketch2)

      pad_list = [(0, 0)] * len(query.shape)
      pad_list[-1] = (0, 1)
      value_padded = jnp.pad(
          value, pad_list, mode='constant', constant_values=1.0
      )  # Concatenates a column of 1.0s to all the value arrays

      result, _ = (
          mixed_lower_triangular_multiplication.mixed_tensor_lt_multiply(
              query_prime,
              key_prime,
              value_padded,
              query,
              key,
              grain_size,
              power=power,
              precision=precision,
          )
      )

      # A normalization factor is added to make the avoid numerical issues
      # during division and make the training stable.
      inverse_scalings = result[Ellipsis, -1]
      inverse_scalings = normalization_factor + inverse_scalings

      scalings = jnp.reciprocal(inverse_scalings)
      result = result[Ellipsis, :-1] * scalings[Ellipsis, None]
      return result

  else:
    raise NotImplementedError(
        'Mixed attention is not implemented in non-causal setting.'
    )

  return attention_function
