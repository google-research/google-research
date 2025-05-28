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

"""Stochastic modules."""

import flax.linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from imp.max.core import utils
from imp.max.utils import sharding
from imp.max.utils import typing


def get_gaussian_orth_rand_mat(
    rng,
    nb_rows,
    nb_columns,
    scaling = False,
    dot_general = lax.dot_general,
    precision = None):
  """Method for constructing 2D Gaussian orthogonal arrays.

  Method for constructing structured block-orthogonal Gaussian matrix.

  Args:
    rng: the key used to generate randomness for the construction of the random
      matrices,
    nb_rows: number of rows of the Gaussian matrix to be constructed,
    nb_columns: number of columns of the Gaussian matrix to be constructed,
    scaling: boolean indicating whether the rows of the Gaussian matrix should
      be normalized to the deterministic length sqrt(nb_rows)
    dot_general: the function that performs dot product.
    precision: the precision with which the dot product is performed.
  Returns:
    The Gaussian matrix of <nb_rows> rows and <nb_columns> columns.
  """
  nb_full_blocks = int(nb_rows / nb_columns)
  block_list = []
  for _ in range(nb_full_blocks):
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(q)
  remaining_rows = nb_rows - nb_full_blocks * nb_columns
  if remaining_rows > 0:
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(q[:remaining_rows])
  final_matrix = jnp.vstack(block_list)

  if scaling:
    multiplier = jnp.sqrt(float(nb_columns)) * jnp.ones((nb_rows))
  else:
    _, rng_input = jax.random.split(rng)
    multiplier = jnp.linalg.norm(
        random.normal(rng_input, (nb_rows, nb_columns)), axis=1)
  multiplier_diag = jnp.diag(multiplier)
  return dot_general(
      multiplier_diag, final_matrix,
      (((multiplier_diag.ndim - 1,), (0,)), ((), ())),
      precision=precision,
  )


def get_gaussian_simplex_rand_mat(
    rng,
    nb_rows,
    nb_columns,
    scaling = False,
    dot_general = lax.dot_general,
    precision = None,
):
  """Method for constructing 2D Gaussian simplex arrays.

  Method for constructing Gaussian matrix that is block-wise simplex, i.e.
  it consists of square-blocks, where the rows within each block form a simplex.

  Args:
    rng: the key used to generate randomness for the construction of the random
      matrices,
    nb_rows: number of rows of the Gaussian matrix to be constructed,
    nb_columns: number of columns of the Gaussian matrix to be constructed,
    scaling: boolean indicating whether the rows of the Gaussian matrix should
      be normalized to the deterministic length sqrt(nb_rows)
    dot_general: the function that performs dot product.
    precision: the precision with which the dot product is performed.
  Returns:
    The Gaussian matrix of <nb_rows> rows and <nb_columns> columns.
  """
  sim_vectors = []
  all_ones_but_last = (
      jnp.ones(nb_columns) - jnp.identity(nb_columns)[nb_columns - 1]
  )
  first_mult = (jnp.sqrt(nb_columns) + 1.0) / jnp.power(nb_columns - 1, 1.5)
  second_mult = 1.0 / jnp.sqrt(nb_columns - 1)
  for i in range(nb_columns - 1):
    sim_vector = (
        jnp.sqrt(nb_columns / (nb_columns - 1)) * jnp.identity(nb_columns)[i]
        - first_mult * all_ones_but_last
    )
    sim_vectors.append(sim_vector)
  sim_vectors.append(second_mult * all_ones_but_last)
  sim_matrix = jnp.transpose(jnp.array(sim_vectors))

  nb_full_blocks = int(nb_rows / nb_columns)
  block_list = []
  for _ in range(nb_full_blocks):
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(jnp.transpose(utils.matmul(q, sim_matrix,
                                                 dot_general=dot_general,
                                                 precision=precision)))
  remaining_rows = nb_rows - nb_full_blocks * nb_columns
  if remaining_rows > 0:
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(
        jnp.transpose(utils.matmul(q, sim_matrix[:, :remaining_rows],
                                   dot_general=dot_general,
                                   precision=precision))
    )
  final_matrix = jnp.vstack(block_list)

  if scaling:
    multiplier = jnp.sqrt(float(nb_columns)) * jnp.ones((nb_rows))
  else:
    _, rng_input = jax.random.split(rng)
    multiplier = jnp.linalg.norm(
        random.normal(rng_input, (nb_rows, nb_columns)), axis=1
    )

  return utils.matmul(jnp.diag(multiplier), final_matrix,
                      dot_general=dot_general,
                      precision=precision)


class DropToken(nn.Module):
  """DropToken Module as in https://arxiv.org/abs/2104.11178."""

  rate: float
  activation_shardings: typing.ShardingAxes = ()
  spmd_enabled: bool = True
  dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None

  @nn.compact
  def __call__(self,
               inputs,
               deterministic = True,
               rng = None):
    """Randomly drops tokens of a given sequence.

    Args:
      inputs: A tensor with shape [..., length, embed]. The minimum and maximum
        supported ranks are 2 and 4, respectively. DropToken is always
        performed on the `length` dimension.
      deterministic: A bool indicating stochastic/deterministic behavior.
        DropToken is only performed when this flag is False.
      rng: JAX's pseudo-rnadom number generator key (jax.random.key).
        This will override the default key if given.

    Returns:
      An array with shape [..., int((1 - rate) * length), embed]

    Raises:
      ValueError: if any inputs with 4 < inputs.ndim < 2 is given.
      ValueError: if rate outside [0, 1] is provided.
    """

    if self.rate < 0 or self.rate > 1:
      raise ValueError('Please provide a valid rate between [0, 1]')

    if (self.rate == 0. or deterministic):
      return inputs

    rank = inputs.ndim
    if not 2 <= rank <= 4:
      raise ValueError(
          'Input must have 2 <= rank <= 4. Instead, received a tensor with '
          f'shape {inputs.shape}')

    if rng is None:
      rng = self.make_rng('droptoken')
    length = inputs.shape[-2]
    cap = int((1 - self.rate) * length)
    iota = lax.iota(jnp.int32, length)
    idx = jax.random.choice(key=rng, a=iota, shape=(cap,), replace=False)

    if self.spmd_enabled:
      outputs = utils.take_along_axis(inputs, idx, -2,
                                      dot_general=self.dot_general,
                                      precision=self.precision)
    else:
      outputs = jnp.take(inputs, idx, -2)

    return sharding.shard_array(outputs, self.activation_shardings)


class MaskToken(DropToken):
  """Token sequence masking in continuous space."""

  embedding_init: nn.initializers.Initializer = nn.initializers.normal(
      stddev=0.02
  )
  embedding_shardings: typing.ShardingAxes = ()
  param_dtype: jax.typing.DTypeLike = jnp.float32

  @nn.compact
  def __call__(
      self,  # pytype: disable=signature-mismatch
      inputs,
      deterministic = True,
      rng = None,
  ):
    """Randomly replaces tokens in a sequence with a learnable variable.

    Args:
      inputs: A tensor with shape [..., length, embed]. The minimum and maximum
        supported ranks are 2 and 4, respectively. Token masking is always
        performed on the `length` dimension.
      deterministic: A bool indicating stochastic/deterministic behavior. Token
        masking is only performed when this flag is False.
      rng: JAX's pseudo-rnadom number generator key (jax.random.key). This
        will override the default key if given.

    Returns:
      dropped_outputs: An array with shape [..., num_keep, embed] where
        num_keep = int((1 - rate) * length)
      masked_outputs: An array with shape [..., length, embed]
      mask_drop: A 0/1 mask with shape [length,] indicating where token has
        been randomly masked and/or dropped.
      idx_keep: An array with shape [(1 - rate) * length,] which contains
        the indices where the outputs are untouched.

    Raises:
      ValueError: if any inputs with 4 < inputs.ndim < 2 is given.
      ValueError: if rate outside [0, 1] is provided.
    """

    if self.rate < 0 or self.rate > 1:
      raise ValueError('Please provide a valid rate between [0, 1]')

    rank = inputs.ndim
    if not 2 <= rank <= 4:
      raise ValueError(
          'Input must have 2 <= rank <= 4. Instead, received a tensor with '
          f'shape {inputs.shape}')

    # although we only use mask embeddings when deterministic=False, we
    # instantiate it before checking for determinism to make sure it is
    # initialized properly when calling model.init w/o specifying determinism
    dim = inputs.shape[-1]
    length = inputs.shape[-2]
    embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.embedding_shardings)
    mask_embedding = self.param(
        name='mask_embedding',
        init_fn=embedding_init,
        shape=(dim,),
        dtype=self.param_dtype,
        unbox=True,
    )

    if (self.rate == 0. or deterministic):
      return inputs, inputs, jnp.zeros(length), lax.iota(jnp.int32, length)

    # create random indices to mask (drop) tokens
    if rng is None:
      rng = self.make_rng('masktoken')
    num_drop = int(self.rate * length)
    num_keep = length - num_drop
    iota = lax.iota(jnp.int32, length)
    idx_keep = jnp.sort(
        jax.random.choice(key=rng, a=iota, shape=(num_keep,), replace=False))
    mask_keep = utils.index_to_mask(idx_keep, length)
    mask_drop = 1 - mask_keep
    idx_drop = utils.mask_to_index(mask_drop, num_drop)

    # make mask embeddings broadcastable to the inputs
    mask_embedding = jnp.asarray(mask_embedding, dtype=inputs.dtype)
    repeat_dims = [1] * (rank-2) +[num_drop, 1]
    broadcastable_mask_embedding = jnp.tile(mask_embedding, repeat_dims)

    if self.spmd_enabled:
      outputs_dropped = utils.take_along_axis(
          inputs=inputs,
          indices=idx_keep,
          axis=-2,
          dot_general=self.dot_general,
          precision=self.precision)
      outputs_masked = utils.scatter_along_axis(
          inputs=inputs,
          updates=broadcastable_mask_embedding,
          indices=idx_drop,
          axis=-2,
          batch_dims=(),
          dot_general=self.dot_general,
          precision=self.precision)

    else:
      raise NotImplementedError

    outputs_dropped = self._shard_outputs(outputs_dropped)
    outputs_masked = self._shard_outputs(outputs_masked)

    return outputs_dropped, outputs_masked, mask_drop, idx_keep
