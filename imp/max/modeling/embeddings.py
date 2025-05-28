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

"""Modality-specific tokenization layers implemented in Jax/Flax."""

import abc
import dataclasses
import functools
import math
import typing as pytyping
from typing import Any, Sequence

import flax.linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

from imp.max.core import constants
from imp.max.core import utils
from imp.max.modeling import normalization
from imp.max.utils import sharding
from imp.max.utils import typing


_default_special_token_init = jax.nn.initializers.glorot_normal()
_default_mask_token_init = jax.nn.initializers.normal()


def get_relative_position_bucket(relative_position,
                                 bidirectional,
                                 num_buckets,
                                 max_distance):
  """Translate relative position to a bucket number for relative attention.

  The relative position is defined as memory_position - query_position, i.e.
  the distance in tokens from the attending position to the attended-to
  position.  If bidirectional=False, then positive relative positions are
  invalid.
  We use smaller buckets for small absolute relative_position and larger
  buckets for larger absolute relative_positions. For all positions within [0,
  num_buckets//2] (unidirectional) and [0, num_buckets//4] (bidirectional)
  the position is used as their bucket (with a shift of num_buckets//2 for
  positive positions in the bidirectional case). For positions farther than the
  above boundary, the position is mapped logarithmically to the remaining
  buckets.
  All relative positions >=max_distance  map to the same bucket.  All relative
  positions <=-max_distance map to the same bucket.  This should allow for
  more graceful generalization to longer sequences than the model has been
  trained on.

  Args:
    relative_position: The relative distance betwerren pairs of tokens.
    bidirectional: A boolean flag indicating whether the attention is
      bidirectional.
    num_buckets: The maximum number of buckets.
    max_distance: The maximum relative distance of the token pairs. In
      max_distance has to be larger than the num_buckets//2 for unidirectional
      and num_buckets//4 for bidirectional bucketing.

  Returns:
    bucket: A tensor with the same shape as relative_position, containing int32
      bucket values in the range [0, num_buckets).
  """
  # TODO(b/217469019) Revisit bucketing behavior.
  bucket = 0
  position = -relative_position
  if bidirectional:
    num_buckets //= 2
    bucket += np.less(position, 0).astype(np.int32) * num_buckets
    position = np.absolute(position)
  else:
    position = np.maximum(position, 0)
  # now position is in the range [0, inf)
  max_exact = num_buckets // 2
  if max_exact >= max_distance:
    raise ValueError(
        f'max_distance has to be greater than {max_exact}, but is '
        f'{max_distance}.'
    )
  epsilon = np.finfo(np.float32).eps
  val_if_large = max_exact + (
      np.log(position.astype(np.float32) / max_exact + epsilon) /
      np.log(max_distance / max_exact) *
      (num_buckets - max_exact)).astype(np.int32)
  val_if_large = np.minimum(val_if_large, num_buckets - 1)
  bucket += np.where(np.less(position, max_exact), position, val_if_large)
  return bucket


class Embed(nn.Module):
  """An SPMD-friendly embedding lookup module.

  A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    embedding_init: embedding initializer.
    spmd_enabled: enables an SPMD-friendly lookup.
    lookup_dot_general: the function with which the SPMD-friendly lookup is
      performed.
    precision: numerical precision of the jax ops.
    embedding: a field to give access to the underlying embedding in some
      downstream applications (e.g. reverse lookup for logit reconstruction).
  """
  num_embeddings: int
  features: int
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  embedding_init: nn.initializers.Initializer = nn.linear.default_embed_init
  spmd_enabled: bool = True
  shardings: typing.ShardingAxes = ()
  lookup_dot_general: typing.DotGeneral = lax.dot_general
  precision: typing.Precision = None
  embedding: jax.Array = dataclasses.field(init=False)

  def setup(self):
    if self.shardings and (len(self.shardings) != 2):
      raise ValueError(
          f'Sharding annotations `{self.shardings}` do not match '
          f'embedding shape {(self.num_embeddings, self.features)}.'
      )
    embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.shardings)
    self.embedding = pytyping.cast(jax.Array, self.param(
        name='embedding',
        init_fn=embedding_init,
        shape=(self.num_embeddings, self.features),
        dtype=self.param_dtype,
        unbox=True,
    ))

  def __call__(self, inputs):
    """Embeds the inputs along the last dimension.

    Args:
      inputs: Input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """

    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')

    # Convert embeddings to the desired dtype
    embedding = jnp.asarray(self.embedding, self.dtype)

    if self.spmd_enabled:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[Ellipsis, jnp.newaxis] == iota, dtype=self.dtype)
      output = self.lookup_dot_general(
          one_hot, embedding,
          (((one_hot.ndim - 1,), (0,)), ((), ())),
          precision=self.precision)
    else:
      output = jnp.take(embedding, inputs, axis=0)

    return output

  def attend(self, query):
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.
    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    query = jnp.asarray(query, self.dtype)
    embedding = jnp.asarray(self.embedding, self.dtype)
    output = self.dot_general(
        query, embedding,
        (((query.ndim - 1,), (embedding.ndim - 1,)), ((), ())),
        precision=self.precision)
    return output


class PosBiasEmbed(nn.Module):
  """An SPMD-friendly Position Bias lookup module."""

  num_buckets: int
  num_heads: int
  dtype: jax.typing.DTypeLike
  param_dtype: jax.typing.DTypeLike = jnp.float32
  embedding_init: nn.initializers.Initializer = nn.linear.default_embed_init
  embedding_shardings: typing.ShardingAxes = ()
  lookup_dot_general: typing.DotGeneral = lax.dot_general

  @nn.compact
  def __call__(self, rp_bucket):
    if self.embedding_shardings and len(self.embedding_shardings) != 2:
      raise ValueError(
          f'Sharding annotations `{self.embedding_shardings}` do not match '
          f'embedding shape {(self.num_heads, self.num_buckets)}.'
      )

    embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.embedding_shardings)
    relative_attention_bias = self.param(
        name='embedding',
        init_fn=embedding_init,
        shape=(self.num_heads, self.num_buckets),
        dtype=self.param_dtype,
        unbox=True,
    )
    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)

    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, Ellipsis] == bcast_iota, dtype=self.dtype)
    # --> shape (num_heads, qlen, klen)
    values = self.lookup_dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims

    return values


class TemporalPosEncode(nn.Module):
  """Temporal Position Encoding Module."""

  hidden_size: int
  pos_buckets: int
  dropout_rate: float = 0.1
  dtype: jax.typing.DTypeLike = jnp.float32
  lookup_dot_general: typing.DotGeneral = lax.dot_general
  embedding_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  embedding_name: str = 'temporal_postition_embeddings'

  def setup(self):
    self.temporal_position_embeddings = Embed(
        num_embeddings=self.pos_buckets,
        features=self.hidden_size,
        dtype=self.dtype,
        shardings=self.embedding_shardings,
        lookup_dot_general=self.lookup_dot_general,
        name=self.embedding_name,
    )
    self.layer_norm = normalization.LayerNorm(
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm'
    )
    self.dropout = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

  def __call__(self,
               inputs,
               deterministic = True,
               token_coordinate = None,
               coordinate_scale = None):
    """Get temporal token embeddings of inputs.

    The positional encoding in this class holds one of the following forms:
      1. If num_tokens < max_buckets, we have more position buckets than the
         requested positions. In this case, if `token_coordinate` is not
         provided, we raise an error asking for proper configuration. Otherwise,
         we scale the coordinates to [0, min(coordinate_scale, max_buckets))
         and construct positional IDs in that range and fetch the embeddings
         using those IDs. Below are two examples:
           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = 8
           max_buckets = 12
           --> position_ids = [0, 2, 4, 6]

           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = None
           max_buckets = 12
           --> position_ids = [0, 3, 6, 9]

      2. If num_tokens > max_buckets, we have less available position buckets
         than the requested token positions. In this case, it is required to
         provide `token_coordinate`. Similar to the configuration above, we
         we scale the coordinates to [0, min(coordinate_scale, max_buckets)).
         In this case, we simply pad the missing buckets by replication.
         Below are two examples:
           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = 8
           max_buckets = 2
           --> position_ids = [0, 0, 1, 1]

           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = None
           max_buckets = 2
           --> position_ids = [0, 0, 1, 1]

      3. If num_tokens == max_buckets, all buckets will be fetched no matter
         `token_coordinate` is provided or not. However, if `token_coordinate`
         is provided, the order in which the embeddings are fetched completely
         relies on the corresponding coordinates.

    Args:
        inputs: Input tensor with shape [batch_size, n_instance, length, dim].
        deterministic: A bool flag for deterministic behavior.
        token_coordinate: An optional array that contains position indices.
          If not provided, the indices are computed based on `self.pos_buckets`.
        coordinate_scale: An optional integer used for scaling
          `token_coordinate` before fetching the position buckets. If provided,
          scale = min(coordinate_scale, self.max_buckets).
    Returns:
        embeddings: Output embedding tensor, float32 with
          shape [batch_size, n_instance, length, dim].
    Raises:
        ValueError if the resulting position embeddings have different length
          than the inputs.
        ValueError if coordinate_scale `coordinate_scale > self.pos_buckets`.
    """
    if token_coordinate is None:
      temporal_position_ids = lax.broadcasted_iota(
          jnp.int32, [1, 1, self.pos_buckets], 2)
    else:
      if coordinate_scale is not None:
        if coordinate_scale > self.pos_buckets:
          raise ValueError(
              '`coordinate_scale` could not exceed the existing pos_buckets.')
        temporal_coordinate_scale = coordinate_scale
      else:
        temporal_coordinate_scale = self.pos_buckets
      temporal_position_ids = (
          token_coordinate * temporal_coordinate_scale).astype(jnp.int32)

    position_embeddings = self.temporal_position_embeddings(
        inputs=temporal_position_ids,
    )
    position_embeddings = self.layer_norm(position_embeddings)

    if inputs.shape[-2] != position_embeddings.shape[-2]:
      if token_coordinate is None:
        solution_msg = (
            'Please either provide `token_coordinate`, or configure this '
            'module with proper `pos_buckets`.')
      else:
        solution_msg = (
            'Please provide a `token_coordinate` that exactly matches the '
            'inputs and is normalized in (0, 1).')
      raise ValueError(
          'The inputs do not contain the same number of tokens as the '
          f'available buckets. num_input_tokens={inputs.shape[-2]} while '
          f'num_available_buckets={position_embeddings.shape[-2]}. '
          f'{solution_msg}')

    embeddings = inputs + position_embeddings
    embeddings = self.dropout(embeddings, deterministic)

    return embeddings


class SpectroTemporalPosEncode(nn.Module):
  """Spectro-Temporal Position Encoding Module."""

  hidden_size: int
  pos_buckets: tuple[int, int]
  dropout_rate: float
  dtype: jax.typing.DTypeLike = jnp.float32
  take_dot_general: typing.DotGeneral = lax.dot_general
  lookup_dot_general: typing.DotGeneral = lax.dot_general
  embedding_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()

  def setup(self):
    self.temporal_position_embeddings = Embed(
        num_embeddings=self.pos_buckets[0],
        features=self.hidden_size,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        shardings=self.embedding_shardings,
        name='temporal_postition_embeddings',
    )
    self.spectoral_position_embeddings = Embed(
        num_embeddings=self.pos_buckets[1],
        features=self.hidden_size,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        shardings=self.embedding_shardings,
        name='spectoral_postition_embeddings',
    )
    self.layer_norm = normalization.LayerNorm(
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm',
    )
    self.dropout = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

  def _build_2d_pos_ids(self, t, s):
    """Creates and returns 3d positional ids (if token_coordinate not provided).

    Args:
      t: Temoporal length of the spatio-temporal tokens.
      s: Spectoral length of the spectro-temporal tokens.


    Returns:
      temporal_ids: Spatio-temporal position IDs of the temporal axis with a
        shape of [1, 1, t*s].
      spectoral_ids: Spatio-temporal position IDs of the vertical axis with a
        shape of [1, 1, t*s].
    """

    # define pos_ids - a fixed tensor which is a function of input shape
    #
    #     **** we use broadcasted_iota as below
    #     lax.broadcasted_iota(jnp.int32, [t, s], 0)
    #
    #     **** which is an SPMD-friendly equivalent to the following:
    #     jnp.arange(0, t)[:, jnp.newaxis] -> (t, 1)
    #     jnp.tile(temporal_ids, [1, s]) -> (t, s)
    #
    temporal_ids = lax.broadcasted_iota(jnp.int32, [t, s], 0)  # (t, s)
    spectoral_ids = lax.broadcasted_iota(jnp.int32, [t, s], 1)  # (t, s)

    # add batch and instance dimensions and flatten all ids to one axis
    # resulting in position ids with shape [1, 1, t*s]
    temporal_ids = jnp.reshape(temporal_ids, [1, 1, -1])
    spectoral_ids = jnp.reshape(spectoral_ids, [1, 1, -1])

    return temporal_ids, spectoral_ids

  def _fetch_embeddings(self,
                        temporal_position_ids,
                        spectoral_position_ids):
    """Performs embedding lookup given the position IDs."""
    temporal_position_embeddings = self.temporal_position_embeddings(
        inputs=temporal_position_ids,
    )

    spectoral_position_embeddings = self.spectoral_position_embeddings(
        inputs=spectoral_position_ids,
    )

    position_embeddings = (
        temporal_position_embeddings + spectoral_position_embeddings)

    position_embeddings = self.layer_norm(position_embeddings)

    return position_embeddings

  def __call__(
      self,
      inputs,
      deterministic = True,
      token_coordinate = None,
      coordinate_scale = None,
      ):
    """Get spectro-temporal token embeddings of inputs.

    The positional encoding in this class holds one of the following forms:
      1. If num_tokens < max_buckets, we have more position buckets than the
         requested positions. In this case, if `token_coordinate` is not
         provided, we raise an error asking for proper configuration. Otherwise,
         we scale the non-temporal coordinates to max_buckets and the temporal
         coordinates to [0, min(coordinate_scale, max_buckets)) and construct
         positional IDs in their corresponding ranges and fetch the embeddings
         using those IDs. Below is an example:
           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = [8, None]
           max_buckets = [12, 4]
           --> temporal_position_ids = [0, 2, 4, 6]
           --> spectoral_position_ids = [0, 1, 2, 3]

      2. If num_tokens > max_buckets, we have less available position buckets
         than the requested token positions. In this case, it is required to
         provide `token_coordinate`. Similar to the configuration above, we
         we scale the non-temporal coordinates to max_buckets and the temporal
         coordinates to [0, min(coordinate_scale, max_buckets)).
         In this case, we simply pad the missing buckets by replication.
         Below are two examples:
           Below is an example:
           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = [8, None]
           max_buckets = [2, 2]
           --> temporal_position_ids = [0, 0, 1, 1]
           --> spectoral_position_ids = [0, 0, 1, 1]

      3. If num_tokens == max_buckets, all buckets will be fetched no matter
         `token_coordinate` is provided or not. However, if `token_coordinate`
         is provided, the order in which the embeddings are fetched completely
         relies on the corresponding coordinates.

    Args:
        inputs: Input tensor with shape [batch_size, n_instance, length, dim].
        deterministic: A bool flag for deterministic behavior.
        token_coordinate: An optional array that contains position indices.
          If not provided, the indices are computed based on `self.pos_buckets`.
        coordinate_scale: An optional integer used for scaling
          `token_coordinate` before fetching the position buckets. If provided,
          scale = min(coordinate_scale, self.max_buckets).
    Returns:
        embeddings: Output embedding tensor, float32 with
          shape [batch_size, n_instance, length, dim].
    """
    if token_coordinate is None:
      (temporal_position_ids,
       spectoral_position_ids) = self._build_2d_pos_ids(*self.pos_buckets)
    else:
      if coordinate_scale is not None:
        temporal_coordinate_scale = coordinate_scale[0] or self.pos_buckets[0]
        spectoral_coordinate_scale = coordinate_scale[1] or self.pos_buckets[1]
      else:
        temporal_coordinate_scale, spectoral_coordinate_scale = self.pos_buckets

      if (temporal_coordinate_scale > self.pos_buckets[0]
          or spectoral_coordinate_scale > self.pos_buckets[1]):
        raise ValueError(
            '`coordinate_scale` could not exceed the existing pos_buckets. '
            f'Received {coordinate_scale=} while pos_buckets={self.pos_buckets}'
            )

      take_fn = functools.partial(
          utils.take_along_axis,
          axis=-1,
          precision='bfloat16',
          dot_general=self.take_dot_general)
      temporal_position_ids = (
          take_fn(token_coordinate, 0) * temporal_coordinate_scale
          ).astype(jnp.int32)
      spectoral_position_ids = (
          take_fn(token_coordinate, 1) * spectoral_coordinate_scale
          ).astype(jnp.int32)

    position_embeddings = self._fetch_embeddings(
        temporal_position_ids, spectoral_position_ids)

    if inputs.shape[-2] != position_embeddings.shape[-2]:
      if token_coordinate is None:
        solution_msg = (
            'Please either provide `token_coordinate`, or configure this '
            'module with proper `pos_buckets`.')
      else:
        solution_msg = (
            'Please provide a `token_coordinate` that exactly matches the '
            'inputs and is normalized in (0, 1).')
      raise ValueError(
          'The inputs do not contain the same number of tokens as the '
          f'available buckets. {solution_msg}')

    embeddings = inputs + position_embeddings
    embeddings = self.dropout(embeddings, deterministic)

    return embeddings


class SpatioTemporalPosEncode(nn.Module):
  """Spatio-Temporal Position Encoding Module."""

  hidden_size: int
  pos_buckets: tuple[int, int, int]
  dropout_rate: float
  dtype: jax.typing.DTypeLike = jnp.float32
  embedding_shardings: typing.ShardingAxes = ()
  layernorm_shardings: typing.ShardingAxes = ()
  take_dot_general: typing.DotGeneral = lax.dot_general
  lookup_dot_general: typing.DotGeneral = lax.dot_general

  def setup(self):
    self.temporal_position_embeddings = Embed(
        num_embeddings=self.pos_buckets[0],
        features=self.hidden_size,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        shardings=self.embedding_shardings,
        name='temporal_postition_embeddings',
    )
    self.vertical_position_embeddings = Embed(
        num_embeddings=self.pos_buckets[1],
        features=self.hidden_size,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        shardings=self.embedding_shardings,
        name='vertical_postition_embeddings',
    )
    self.horizontal_position_embeddings = Embed(
        num_embeddings=self.pos_buckets[2],
        features=self.hidden_size,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        shardings=self.embedding_shardings,
        name='horizontal_postition_embeddings',
    )
    self.layer_norm = normalization.LayerNorm(
        dtype=self.dtype,
        shardings=self.layernorm_shardings,
        name='layer_norm',
    )
    self.dropout = nn.Dropout(self.dropout_rate, broadcast_dims=(-2,))

  def _build_3d_pos_ids(self, t, h, w):
    """Creates and returns 3d positional ids (if token_coordinate not provided).

    Args:
      t: Temoporal length of the spatio-temporal tokens.
      h: Vertical length of the spatio-temporal tokens.
      w: Horizontal length of the spatio-temporal tokens.


    Returns:
      temporal_ids: Spatio-temporal position IDs of the temporal axis with a
        shape of [1, 1, t*h*w].
      vertical_ids: Spatio-temporal position IDs of the vertical axis with a
        shape of [1, 1, t*h*w].
      horizontal_ids: Spatio-temporal position IDs of the horizontal axis with
        a shape of [1, 1, t*h*w].
    """

    # define pos_ids - a fixed tensor which is a function of input shape
    #
    #     **** we use broadcasted_iota as below
    #     lax.broadcasted_iota(jnp.int32, [t, h, w], 0)
    #
    #     **** which is an SPMD-friendly equivalent to the following:
    #     jnp.arange(0, t)[:, jnp.newaxis, jnp.newaxis] -> (t, 1, 1)
    #     jnp.tile(temporal_ids, [1, h, w]) -> (t, h, w)
    #
    temporal_ids = lax.broadcasted_iota(jnp.int32, [t, h, w], 0)  # (t, h, w)
    vertical_ids = lax.broadcasted_iota(jnp.int32, [t, h, w], 1)  # (t, h, w)
    horizontal_ids = lax.broadcasted_iota(jnp.int32, [t, h, w], 2)  # (t, h, w)

    # add batch and instance dimensions and flatten all ids to one axis
    # resulting in position ids with shape [1, 1, t*h*w]
    temporal_ids = jnp.reshape(temporal_ids, [1, 1, -1])
    vertical_ids = jnp.reshape(vertical_ids, [1, 1, -1])
    horizontal_ids = jnp.reshape(horizontal_ids, [1, 1, -1])

    return temporal_ids, vertical_ids, horizontal_ids

  def _fetch_embeddings(self,
                        temporal_position_ids,
                        vertical_position_ids,
                        horizontal_position_ids):
    """Performs embedding lookup given the position IDs."""
    temporal_position_embeddings = self.temporal_position_embeddings(
        inputs=temporal_position_ids,
    )

    vertical_position_embeddings = self.vertical_position_embeddings(
        inputs=vertical_position_ids,
    )

    horizontal_position_embeddings = self.horizontal_position_embeddings(
        inputs=horizontal_position_ids,
    )

    position_embeddings = (
        temporal_position_embeddings +
        vertical_position_embeddings +
        horizontal_position_embeddings
    )

    position_embeddings = self.layer_norm(position_embeddings)

    return position_embeddings

  def __call__(
      self,
      inputs,
      deterministic = True,
      token_coordinate = None,
      coordinate_scale = None,
      ):
    """Get spatio-temporal token embeddings of inputs.

    The positional encoding in this class holds one of the following forms:
      1. If num_tokens < max_buckets, we have more position buckets than the
         requested positions. In this case, if `token_coordinate` is not
         provided, we raise an error asking for proper configuration. Otherwise,
         we scale the non-temporal coordinates to max_buckets and the temporal
         coordinates to [0, min(coordinate_scale, max_buckets)) and construct
         positional IDs in their corresponding ranges and fetch the embeddings
         using those IDs. Below is an example:
           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = [8, 8, None]
           max_buckets = [12, 8, 4]
           --> temporal_position_ids = [0, 2, 4, 6]
           --> vertical_position_ids = [0, 2, 4, 6]
           --> horizontal_position_ids = [0, 1, 2, 3]

      2. If num_tokens > max_buckets, we have less available position buckets
         than the requested token positions. In this case, it is required to
         provide `token_coordinate`. Similar to the configuration above, we
         we scale the non-temporal coordinates to max_buckets and the temporal
         coordinates to [0, min(coordinate_scale, max_buckets)).
         In this case, we simply pad the missing buckets by replication.
         Below are two examples:
           Below is an example:
           token_coordinate = [0., 0.25, 0.5, 0.75]
           coordinate_scale = [8, None, None]
           max_buckets = [2, 3, 2]
           --> temporal_position_ids = [0, 0, 1, 1]
           --> vertical_position_ids = [0, 0, 1, 2]
           --> horizontal_position_ids = [0, 0, 1, 1]

      3. If num_tokens == max_buckets, all buckets will be fetched no matter
         `token_coordinate` is provided or not. However, if `token_coordinate`
         is provided, the order in which the embeddings are fetched completely
         relies on the corresponding coordinates.

    Args:
        inputs: Input tensor with shape [batch_size, n_instance, length, dim].
        deterministic: A bool flag for deterministic behavior.
        token_coordinate: An optional array that contains position indices.
          If not provided, the indices are computed based on `self.pos_buckets`.
        coordinate_scale: An optional integer used for scaling
          `token_coordinate` before fetching the position buckets. If provided,
          scale = min(coordinate_scale, self.max_buckets).
    Returns:
        embeddings: Output embedding tensor, float32 with
          shape [batch_size, n_instance, length, dim].
    """
    if token_coordinate is None:
      (temporal_position_ids,
       vertical_position_ids,
       horizontal_position_ids) = self._build_3d_pos_ids(*self.pos_buckets)
    else:
      if coordinate_scale is not None:
        temporal_coordinate_scale = coordinate_scale[0] or self.pos_buckets[0]
        vertical_coordinate_scale = coordinate_scale[1] or self.pos_buckets[1]
        horizontal_coordinate_scale = coordinate_scale[2] or self.pos_buckets[2]
      else:
        (temporal_coordinate_scale,
         vertical_coordinate_scale,
         horizontal_coordinate_scale) = self.pos_buckets

      if (temporal_coordinate_scale > self.pos_buckets[0]
          or vertical_coordinate_scale > self.pos_buckets[1]
          or horizontal_coordinate_scale > self.pos_buckets[2]):
        raise ValueError(
            '`coordinate_scale` could not exceed the existing pos_buckets. '
            f'Received {coordinate_scale=} while pos_buckets={self.pos_buckets}'
            )

      take_fn = functools.partial(
          utils.take_along_axis,
          axis=-1,
          precision='bfloat16',
          dot_general=self.take_dot_general)
      temporal_position_ids = (
          take_fn(token_coordinate, 0) * temporal_coordinate_scale
      ).astype(jnp.int32)
      vertical_position_ids = (
          take_fn(token_coordinate, 1) * vertical_coordinate_scale
      ).astype(jnp.int32)
      horizontal_position_ids = (
          take_fn(token_coordinate, 2) * horizontal_coordinate_scale
      ).astype(jnp.int32)

    position_embeddings = self._fetch_embeddings(
        temporal_position_ids, vertical_position_ids, horizontal_position_ids)

    if inputs.shape[-2] != position_embeddings.shape[-2]:
      if token_coordinate is None:
        solution_msg = (
            'Please either provide `token_coordinate`, or configure this '
            'module with proper `pos_buckets`.')
      else:
        solution_msg = (
            'Please provide a `token_coordinate` that exactly matches the '
            'inputs and is normalized in (0, 1).')
      raise ValueError(
          'The inputs do not contain the same number of tokens as the '
          f'available buckets. {solution_msg}')

    embeddings = inputs + position_embeddings
    embeddings = self.dropout(embeddings, deterministic)

    return embeddings


# TODO(b/236524615): make relative pos encoding modules efficient using iota
class PositionBias1D(nn.Module):
  """Relative Temporal Position Bias Encoding Module."""

  num_heads: int
  num_relative_buckets: int
  max_relative_distance: int
  bidirectional: bool = True
  dtype: jax.typing.DTypeLike = jnp.float32
  lookup_dot_general: typing.DotGeneral = lax.dot_general
  embedding_shardings: typing.ShardingAxes = ()

  def setup(self):
    self.relative_temporal_attention_bias = PosBiasEmbed(
        num_buckets=self.num_relative_buckets,
        num_heads=self.num_heads,
        dtype=self.dtype,
        embedding_shardings=self.embedding_shardings,
        lookup_dot_general=self.lookup_dot_general,
        name='relative_temporal_attention_bias',
    )

  def __call__(self, qlen, klen=None):
    """Get spatio-temporal attention bias w.r.t inputs.

    Args:
      qlen: length of the input query
      klen: length of the input key
    Returns:
      attenion_bias: the attention bias with shape [num_heads, qlen, klen]
    """

    if klen is None:
      klen = qlen

    context_position = np.arange(qlen)[:, None]
    memory_position = np.arange(klen)[None, :]
    relative_position = memory_position - context_position  # (qlen, klen)

    rp_bucket = get_relative_position_bucket(
        relative_position,
        bidirectional=self.bidirectional,
        num_buckets=self.num_relative_buckets,
        max_distance=self.max_relative_distance,
    )

    # shape (num_heads, qlen, klen)
    attention_bias = self.relative_temporal_attention_bias(rp_bucket)

    return attention_bias


class PositionBias3D(nn.Module):
  """Relative Spatio-Temporal Position Bias Encoding Module."""

  num_heads: int
  num_relative_buckets: tuple[int, int, int]
  max_relative_distance: tuple[int, int, int]
  bidirectional: bool = True
  dtype: jax.typing.DTypeLike = jnp.float32
  lookup_dot_general: typing.DotGeneral = lax.dot_general
  embedding_shardings: typing.ShardingAxes = ()

  def setup(self):
    self.relative_temporal_attention_bias = PosBiasEmbed(
        num_buckets=self.num_relative_buckets[0],
        num_heads=self.num_heads,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        embedding_shardings=self.embedding_shardings,
        name='relative_temporal_attention_bias',
    )
    self.relative_vertical_attention_bias = PosBiasEmbed(
        num_buckets=self.num_relative_buckets[1],
        num_heads=self.num_heads,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        embedding_shardings=self.embedding_shardings,
        name='relative_vertical_attention_bias',
    )
    self.relative_horizontal_attention_bias = PosBiasEmbed(
        num_buckets=self.num_relative_buckets[2],
        num_heads=self.num_heads,
        dtype=self.dtype,
        lookup_dot_general=self.lookup_dot_general,
        embedding_shardings=self.embedding_shardings,
        name='relative_horizontal_attention_bias',
    )

  @staticmethod
  def _build_vid_pos_ids(t, h, w):
    """Creates and returns 3d positional ids.

    Args:
      t: time length
      h: height
      w: width


    Returns:
      pos_ids: outputs with shape [t * h * w, 1],
        where 3 = 1 + 1 + 1; 1 for temporal id, 1 for vertical id, and 1 for
        horizontal id, with the following order: [t, h, w]
    """

    # define pos_ids - a fixed tensor which is a function of input shape
    temporal_ids = np.arange(0, t)[:, None, None]  # (t, 1, 1)
    vertical_ids = np.arange(0, h)[None, :, None]  # (1, h, 1)
    horizontal_ids = np.arange(0, w)[None, None, :]  # (1, 1, w)

    temporal_ids = np.tile(temporal_ids, [1, h, w])  # (t, h, w)
    vertical_ids = np.tile(vertical_ids, [t, 1, w])  # (t, h, w)
    horizontal_ids = np.tile(horizontal_ids, [t, h, 1])  # (t, h, w)

    # (t, h, w, 3)
    pos_ids = np.stack([temporal_ids, vertical_ids, horizontal_ids], axis=3)
    pos_ids = np.reshape(pos_ids, [-1, 3])  # (t*h*w, 3)

    return pos_ids

  def __call__(self, tlen, vlen, hlen):
    """Get spatio-temporal attention bias w.r.t inputs.

    Args:
      tlen: length of the temporal dimension
      vlen: length of the vertical dimension
      hlen: length of the horizontal dimension
    Returns:
      attenion_bias: the attention bias with shape
        [num_heads, tlen*vlen*hlen, tlen*vlen*hlen]
    """

    pos_ids = self._build_vid_pos_ids(tlen, vlen, hlen)

    context_position = pos_ids[:, None, :]
    memory_position = pos_ids[None, :, :]
    relative_position = memory_position - context_position  # (qlen, klen, 3)

    bucket_fn = functools.partial(
        get_relative_position_bucket,
        bidirectional=self.bidirectional,
    )

    rtp_bucket = bucket_fn(relative_position[:, :, 0],
                           num_buckets=self.num_relative_buckets[0],
                           max_distance=self.max_relative_distance[0])
    rvp_bucket = bucket_fn(relative_position[:, :, 1],
                           num_buckets=self.num_relative_buckets[1],
                           max_distance=self.max_relative_distance[1])
    rhp_bucket = bucket_fn(relative_position[:, :, 2],
                           num_buckets=self.num_relative_buckets[2],
                           max_distance=self.max_relative_distance[2])

    # shape (num_heads, qlen, klen)
    temporal_bias = self.relative_temporal_attention_bias(rtp_bucket)
    vertical_bias = self.relative_vertical_attention_bias(rvp_bucket)
    horizontal_bias = self.relative_horizontal_attention_bias(rhp_bucket)

    # add all biases together
    attention_bias = temporal_bias + vertical_bias + horizontal_bias

    return attention_bias


class SpecialToken(nn.Module):
  """Appends special token to a sequence of vectors."""

  features: int
  extension: str
  embedding_init: nn.initializers.Initializer = _default_special_token_init
  embedding_shardings: typing.ShardingAxes = ()
  activation_shardings: typing.ShardingAxes = ()
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32

  def setup(self):
    if self.extension not in [
        constants.Extension.APPEND,
        constants.Extension.PREPEND,
    ]:
      raise ValueError('Wrong extension position!')

    embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.embedding_shardings
    )
    self.spc_token = self.param(
        name='embedding',
        init_fn=embedding_init,
        shape=(1, self.features),
        dtype=self.param_dtype,
        unbox=True,
    )

  def _append_special_token(self,
                            inputs,
                            special_embd):
    """Concatenates a special token to a sequence of tokens."""
    batch_size, n_instance = list(inputs.shape)[0:2]

    # (batch_size, n_instance, 1, d_model)
    special_embd = jnp.tile(special_embd, [batch_size, n_instance, 1, 1])
    special_embd = sharding.shard_array(special_embd, self.activation_shardings)

    if self.extension == constants.Extension.PREPEND:
      return jnp.concatenate([special_embd, inputs], axis=2)
    else:
      return jnp.concatenate([inputs, special_embd], axis=2)

  def __call__(
      self,
      inputs,
      token_mask = None,
      attention_bias = None
  ):
    """Appends special token to a sequence of vectors.

    Args:
      inputs: `[batch, n_instance, q_length, num_heads * d_head]`.
      token_mask: A 0/1 token mask with shape `[batch, n_instance, q_length]`.
      attention_bias: The bias for the attention scores. `[num_heads, q_length,
        kv_length]`.

    Returns:
      Extended inputs, extended token_mask and extended attention_bias.

    Raises:
      ValueError: If token_mask or attention_bias shape don't meet
        requirements.
    """

    utils.verify_attention_shapes(attention_bias, token_mask, inputs.shape)

    # append special token
    spc_token = jnp.asarray(self.spc_token, dtype=self.dtype)
    special_embd = spc_token[jnp.newaxis, jnp.newaxis, :, :]
    inputs = self._append_special_token(inputs, special_embd)

    # extend token_mask and attention_bias accordingly
    if token_mask is not None:
      token_mask = utils.extend_token_mask(token_mask, self.extension)

    if attention_bias is not None:
      attention_bias = utils.extend_attention_bias(attention_bias,
                                                   self.extension)

    return inputs, token_mask, attention_bias


class MaskFiller(nn.Module):
  """Learnable mask embeddings to fill in certain given positions."""

  dim: int
  embedding_init: nn.initializers.Initializer = _default_mask_token_init
  embedding_shardings: typing.ShardingAxes = ()
  dtype: jax.typing.DTypeLike = jnp.float32
  param_dtype: jax.typing.DTypeLike = jnp.float32
  precision: typing.Precision = None
  scatter_dot_general: typing.DotGeneral = lax.dot_general

  def setup(self):
    mask_embedding_init = sharding.modulate_param_init(
        self.embedding_init, self.embedding_shardings
    )
    self.mask_embedding = self.param(
        name='mask_embedding',
        init_fn=mask_embedding_init,
        shape=(self.dim,),
        dtype=self.param_dtype,
        unbox=True,
    )

  def __call__(self,
               inputs,
               mask_position_ids,
               keep_position_ids,
               axis):
    rank = inputs.ndim
    if not 2 <= rank <= 4:
      raise ValueError(
          'Input must have 2 <= rank <= 4. Instead, received a tensor with '
          f'shape {inputs.shape}')
    inputs = inputs.astype(self.dtype)

    # Fetch the mask embedding
    mask_embedding = jnp.asarray(self.mask_embedding, dtype=self.dtype)

    # Tile mask embeddings to reflect the non-dim axes (e.g. batch/instance/pos)
    num_mask_positions = mask_position_ids.shape[-1]
    tile_dims = inputs.shape[:-2] + (num_mask_positions, 1)
    mask_embedding_updates = jnp.tile(mask_embedding, tile_dims)

    if keep_position_ids is None:
      inputs_masked = utils.scatter_along_axis(
          inputs=inputs,
          updates=mask_embedding_updates,
          indices=mask_position_ids,
          axis=-2,
          batch_dims=tuple(range(mask_position_ids.ndim - 1)),
          precision=self.precision,
          dot_general=self.scatter_dot_general)

    else:
      inputs_masked = utils.fill_by_scatter(
          inputs=inputs,
          updates=mask_embedding_updates,
          keep_indices=keep_position_ids,
          fill_indices=mask_position_ids,
          axis=-2,
          length=keep_position_ids.shape[-1] + mask_position_ids.shape[-1],
          keep_batch_dims=tuple(range(keep_position_ids.ndim - 1)),
          fill_batch_dims=tuple(range(mask_position_ids.ndim - 1)),
          precision=self.precision,
          dot_general=self.scatter_dot_general)

    return inputs_masked

# ------------------------------------------------------------------------------
# Performers-compatible Relative Positional Encoding mechanism.
#
# The implementation is taken from the following paper: 'Relative Positional
# Encoding for Transformers with Linear Complexity'
# (github code: https://cifkao.github.io/spe/)
# ------------------------------------------------------------------------------


def sinespe(
    rng_key,
    key_shape,
    num_realizations = 64,
    num_sines = 10,
    dot_general = lax.dot_general,
    precision = None,
):
  """Sinusoidal stochastic positional encoding.

  Args:
    rng_key: A PRNGKey.
    key_shape: The shape of keys and queries of the form [B, L, H, D],
      where B stands for batch dimensions (potentially more than one), L
      is the number of tokens, H stands for the number of heads and D is the
      dimensionality per head.
    num_realizations: The number of realizations of the stochastic process (R).
    num_sines: The number of sin and cos components (K).
    dot_general: the function that performs dot product in the matmuls.
    precision: the precision with which the dot product is performed.

  Returns:
    sinusoidal encoding.
  """
  length = key_shape[-3]
  in_features = key_shape[-1]
  num_heads = key_shape[-2]
  params_shape = (num_heads, in_features, num_sines)
  functor = lambda *args: jax.random.normal(*args) - 4.0
  freqs = functor(rng_key, params_shape)
  offsets = jax.random.normal(rng_key, params_shape)

  def init_gains(rng_key, shape):
    gains = jax.random.normal(rng_key, shape)
    return gains / (
        jnp.sqrt(jnp.linalg.norm(gains, axis=-1, keepdims=True)) / 2
    )

  gains = init_gains(rng_key, params_shape)

  # build omega_q and omega_k,
  # with shape (num_heads, keys_dim, length, 2*num_sines)
  indices = jnp.linspace(0, length - 1, length)

  # making sure the frequencies are in [0, 0.5]
  freqs = jax.nn.sigmoid(freqs[:, :, jnp.newaxis, :]) / 2.0

  phases_q = (
      2 * math.pi * freqs * indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
      + offsets[:, :, jnp.newaxis, :]
  )
  omega_q = jnp.stack([jnp.cos(phases_q), jnp.sin(phases_q)], axis=-1).reshape(
      num_heads, in_features, length, 2 * num_sines
  )

  phases_k = (
      2 * math.pi * freqs * indices[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
  )
  omega_k = jnp.stack([jnp.cos(phases_k), jnp.sin(phases_k)], axis=-1).reshape(
      num_heads, in_features, length, 2 * num_sines
  )

  # gains is (num_heads, keys_dim, num_sines). Making them softplus-nonnegat.
  gains = jax.nn.softplus(gains)

  # now upsample it to 2 * num_sines
  gains = jnp.stack([gains, gains], axis=-1).reshape(
      num_heads, in_features, 2 * num_sines
  )

  # draw noise of appropriate shape
  z = jax.random.normal(
      rng_key,
      (1, num_heads, in_features, 2 * num_sines, num_realizations),
  ) / jnp.sqrt(num_sines * 2)

  # scale each of the 2*num_sines by the appropriate gain
  # z is still (1, num_heads, keys_dim, 2*num_sines, num_realizations)
  z = z * gains[None, Ellipsis, None]

  # computing the sum over the sines.
  # gets (1, num_heads, keys_dim, length, num_realizations)
  qbar = utils.matmul(omega_q[None], z,
                      dot_general=dot_general, precision=precision)
  kbar = utils.matmul(omega_k[None], z,
                      dot_general=dot_general, precision=precision)

  # permuting them to be (1, length, num_heads, keys_dim, num_realizations)
  qbar = jnp.transpose(qbar, (0, 3, 1, 2, 4))
  kbar = jnp.transpose(kbar, (0, 3, 1, 2, 4))

  scale = jnp.sqrt(jnp.sqrt(jnp.reciprocal(num_realizations * in_features)))
  return scale * qbar, scale * kbar


def spegate(rng_key, spe_code):
  """Stochastic Positional Encoding gating mechanism.

  Args:
    rng_key: A PRNGKey.
    spe_code: the code of the stochastic positional encoding mechanism.

  Returns:
    qbar and kbar positional encodings.
  """
  qbar, kbar = spe_code

  ### gate = self.param('gate', kbar.shape[-3:-1], jax.random.normal)
  gate = jax.random.normal(rng_key, kbar.shape[-3:-1])

  # incorporate the constant bias for Pd if required. First draw noise
  # such that noise noise^T = 1, for each head, feature, realization.
  in_features = kbar.shape[-2]
  num_realizations = kbar.shape[-1]
  noise = jax.random.normal(rng_key, kbar.shape[-3:])
  noise = noise / jnp.sqrt(jnp.sqrt(in_features * num_realizations))
  # constrain the gate parameter to be in [0 1]
  gate = jax.nn.sigmoid(gate[Ellipsis, None])
  # add to queries and keys.
  pe_coef, noise_coef = jnp.sqrt(gate), jnp.sqrt(1.0 - gate)
  qbar = pe_coef * qbar + noise_coef * noise
  kbar = pe_coef * kbar + noise_coef * noise

  return qbar, kbar


def apply_spe(keys, spe):
  """Function applying stochastic positional encoding.

  Args:
    keys: keys tensor.
    spe: stochastic poositional encoding tensor.

  Returns:
    Keys tensor modulated by the stochastic positional encoding mechanism.
  """
  # sum over the keys_dim after multiplying by queries and keys
  # spe is (1, max_len, ...), truncating and broadcasting over the batch
  if len(keys.shape) == 5:
    return (spe[None, :, : keys.shape[-3]] * keys[Ellipsis, None]).sum(axis=-2)
  else:
    return (spe[:, : keys.shape[-3]] * keys[Ellipsis, None]).sum(axis=-2)


# ------------------------------------------------------------------------------
# Auxiliary functions for the RPE-masked Performer from:
# https://arxiv.org/abs/2106.12566 and https://arxiv.org/abs/2107.07999.
# ------------------------------------------------------------------------------


class Mask(abc.ABC):
  """API for the scalable attention masking mechanism.

  The API for the masking mechanism used to efficiently modulate attention with
  no explicit materialization of the attention matrix.
  """

  @abc.abstractmethod
  def act(
      self, mask, input_tensor
  ):
    """Multiplies the stack of H masks M (shape [L, L] each) by the inp. tensor.

    We denote by L the length of the input sequence and by H the number of
    heads). Each mask of the stack is element-wise multiplied with the regular
    attention matrix in the brute-force masked attention model.

    The method implements the algorithm of multiplying each matrix M of the
    stack by a given input tensor of the shape [B..., L,H,F]. F stands for the
    feature/embedding dimension. The resulting tensor is of the shape
    [B..., L,H,F]. The stack of the masks is encoded by <mask>.
    The slice corresponding to fixed batch indices (B...) and a head index (H)
    of the resulting tensor is obtained my multiplying corresponding mask M
    with the matrix given by the corresponding slice of the input tensor
    (of shape [L, H]) (standard matrix-matrix multiplication, not element-wise).
    The masks M are usually not explicitly materialized to avoid quadratic in L
    time complexity, but are instead encoded in a compact way.

    Args:
      mask: a compact encoding of the masking mechanism.
      input_tensor: <float>[batch_dims, length, head_dims, emb_dim] array.

    Returns:
      <float>[batch_dims, length, head_dims, emb_dim] result of the
        multiplication.
    """
    raise NotImplementedError


class RPEMask(Mask):
  # TODO(kchoro): support a variant with the first CLS token which is 'special'
  # in a sense that its weight is always constant (e.g. 1) regardless of the
  # relative position.
  """Relative Positional Encoding masking mechanism.

  Relative Positional Encoding masking mechanism for which the corresponding
  mask is Toeplitz (not necessarily symmetric).

  The use_fft knob chooses between two implementations that return identical
  results up to numerical errors. For highest speed set use_fft to True on GPU,
  and False on TPU as jax.fft() is relatively slower compared to matrix
  multiplication on TPUs.
  TODO(stamas, kchoro): Improve efficiency further on TPU for small batch sizes
  (constructing the Toeplitz matrices is the bottleneck) and for very long
  sequences with >=8K tokens.
  """

  def __init__(self,
               use_fft = True,
               einsum_dot_general = lax.dot_general,
               einsum_precision = None):
    self._act_method = self._act_fft if use_fft else self._act_einsum
    self._einsum_dot_general = einsum_dot_general
    self._einsum_precision = einsum_precision

  def _act_fft(
      self,
      exp_first_rpe_array,
      exp_second_rpe_array,
      input_tensor,
  ):
    """Computes the action of the Toeplitz matrix using FFT."""
    # <exp_rpe_params> encodes the circulaw rows of the circulant embeddings
    # of the Toeplitz matrices corresponding to the RPE mechanism. It is of the
    # shape [H, 2L] (different RPE mechanisms for different heads).
    exp_rpe_params = jnp.concatenate(
        [
            exp_first_rpe_array,
            jnp.zeros(shape=(exp_first_rpe_array.shape[0], 1)),
            exp_second_rpe_array,
        ],
        axis=1,
    )
    # The method conducts fast Toeplitz matrix-matrix multiplication by
    # (see:  https://math.mit.edu/icg/resources/teaching/18.085-spring2015/
    # toeplitz.pdf):
    # (1) embedding (conceptually) Toeplitz matrix in the 2x larger circulant
    #     matrix,
    # (2) decomposing (conceptually) this larger circulant matrix C as:
    #     C = DFT * diag (DFT * c) * DFT^-1, where: DFT is the discrete Fourier
    #     transform matrix, c is the circulant row-vector defining C and DFT^-1
    #     is an inverse of DFT.
    # (3) left-multiplying <input_tensor> by DFT^-1 using Fast Fourier Transform
    #     FFT, computing diag (DFT * c) using FFT and finally: computing the
    #     Hadamard product with diag (DFT * c) and applying last time FFT.
    # (4) taking the part of the obtained tensor corresponding to the Toeplit
    #     submatrix of the circulant matrix C.
    #
    # The shape of the input and output tensor is [B, L, H, F], where: B - batch
    # dimension, L attention dimension, H - heads dimension and F - feature/
    # embeddings dimension.
    circ_vec_len = exp_rpe_params.shape[-1]
    diag_array = jnp.fft.fft(exp_rpe_params)
    inv_dft_trans = jnp.fft.ifft(input_tensor, n=circ_vec_len, axis=-3)
    had_product = jnp.einsum('...lhf,hl->...lhf',
                             inv_dft_trans, diag_array,
                             precision=self._einsum_precision,
                             _dot_general=self._einsum_dot_general)
    if len(had_product.shape) == 5:
      return jnp.real(
          jnp.fft.fft(had_product, n=circ_vec_len, axis=-3)[
              :, :, 0 : (exp_rpe_params.shape[-1] // 2), :, :
          ]
      )
    else:
      return jnp.real(
          jnp.fft.fft(had_product, n=circ_vec_len, axis=-3)[
              :, 0 : (exp_rpe_params.shape[-1] // 2), :, :
          ]
      )

  def _act_einsum(
      self,
      exp_first_rpe_array,
      exp_second_rpe_array,
      input_tensor,
  ):
    """Constructs the Toeplitz matrix explicitly and uses einsum."""

    # blakehechtman@'s recursive roll method from
    # https://github.com/jax-ml/jax/issues/1646#issuecomment-1139044324
    # modified to work with multiple heads (matrices) at once.
    #
    # This is the fastest on TPU of all the alternatives by far. It's slightly
    # slower on GPU than the best GPU friendly method based on reshaping.
    # However performance on GPU is less important as FFT is even faster there.
    #
    # Shape of x is [H, 2*L-1] on first call, returns shape [H, L, L]
    def toeplitz(x):
      if len(x.shape) == 2:
        x = jnp.expand_dims(x, axis=-1)  # shape [H, L, 1]
      # Keep appending rotated columns until we have enough.
      num_rows = x.shape[-2]
      num_cols = x.shape[-1]
      size_needed = num_rows // 2 + 1  # (==L)
      if num_cols >= size_needed:
        return x[:, :size_needed, :size_needed]
      r = jnp.roll(x, num_cols, axis=-2)
      return toeplitz(jnp.concatenate([x, r], axis=-1))

    rpe_matrices = toeplitz(
        jnp.concatenate([exp_first_rpe_array, exp_second_rpe_array], axis=1)
    )
    # Matrix multiplication, j is the length-index we sum over, h is head-index,
    # f is embedding-index. l-th column of the RPE matrix has the dist(.,l)
    # values used for computing l-th token.
    return jnp.einsum('...jhf,hjl->...lhf',
                      input_tensor, rpe_matrices,
                      precision=self._einsum_precision,
                      _dot_general=self._einsum_dot_general)

  def act(
      self, mask, input_tensor
  ):
    # The RPE masker is encoded with the two 2D arrays of shapes [H, L] and
    # [H, L - 1] respectively, where L stands for the length of the input
    # sequence and H for the number of heads. An ith row of the first array is
    # of the form: c^{i}_{1} = [b^{i}_{0,0},b^{i}_{0,1},...,b^{i}_{0,L-1}], and
    # the ith row of the second array is of the form: c^{i}_{2} =
    # [b^{i}_{L-1,0},...,b^{i}_{1,0}] where b^{i}_{i,j} encodes the relative
    # position distance between ith query and jth key in the ith head (the
    # b-entries that would be added to the corresponding logits entries in the
    # attention matrix in the brute-force masked attention mechanism).
    #
    # Note: We do not impose symmetry so the equality: b^{i}_{i,j} = b^{i}_{j,i}
    # does not necessarily need to hold.
    first_rpe_array, second_rpe_array = mask
    return self._act_method(
        jnp.exp(first_rpe_array), jnp.exp(second_rpe_array), input_tensor
    )


# ------------------------------------------------------------------------------
# Auxiliary functions for the RPE-masked Performer from:
# https://arxiv.org/abs/2302.01925.
# ------------------------------------------------------------------------------


#
# tau(x) = \sum_i w_i * exp(-(x-mu_i)**2 / sigma_i**2)
#
def compute_weighted_gaussians_from_flt_params(
    flt_params,
    points,
    num_ft_params_per_head,
    num_ft_rand_features,
):
  """Constructs the weighted sum of gaussians from the given flt params."""
  d = points.shape[-1]
  weights = flt_params[:, :num_ft_params_per_head]
  mus = flt_params[:, num_ft_params_per_head:((1 + d) * num_ft_params_per_head)]
  mus = jnp.reshape(mus, (flt_params.shape[0], num_ft_params_per_head, d))
  sqsigmas = flt_params[:, ((1 + d) * num_ft_params_per_head):]
  sqsigmas = jnp.exp(sqsigmas)
  h = flt_params.shape[0]
  b_points = jnp.broadcast_to(
      points, (num_ft_params_per_head, h, num_ft_rand_features, d))
  b_points = jnp.transpose(b_points, [2, 1, 0, 3])
  b_points -= mus
  b_points = -b_points**2
  b_points = jnp.sum(b_points, axis=-1)
  b_points /= jnp.expand_dims(sqsigmas, axis=0)
  b_points = jnp.exp(b_points)
  b_points *= jnp.expand_dims(weights, axis=0)
  b_points = jnp.sum(b_points, axis=-1)
  return jnp.transpose(b_points, [1, 0])


def create_random_points(
    d, nb_rows, nb_columns, seed
):
  return random.normal(
      key=random.key(seed), shape=(nb_rows, nb_columns, d)
  )


def create_point_densities(points):
  squared_points = points * points / 2.0
  point_squared_lengths = jnp.sum(squared_points, axis=-1)
  return (1.0 / jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-point_squared_lengths)


def create_flt_snippet(
    flt_params,
    coords,
    coeff,
    flt_num_blobs_per_head,
    flt_num_rand_features,
    einsum_dot_general = lax.dot_general,
    einsum_precision = None,
):
  """A procedure for creating the flt snippet.

  Args:
    flt_params: the learnable parameters defining FLT positional encoding.
    coords: tensor of shape [l, d] encoding Euclidean embeddings of the tokens.
    coeff: multiplieers used to define random featuremaps for FLTs.
    flt_num_blobs_per_head: number of Gaussian blobs per head used to encode
      the Fourier Transform (FT) of the function defining RPE masking in the
      FLT model.
    flt_num_rand_features: number of random features used to approximate
      the function defining RPE masking in the FLT model.
    einsum_dot_general: the function that performs dot product in the einsum.
    einsum_precision: the precision with which the einsum is performed.

  Returns:
    An FLT snippet.
  """
  h, _ = flt_params.shape
  l = coords.shape[-2]
  d = coords.shape[-1]
  points = create_random_points(d, h, flt_num_rand_features, 0)
  densities = create_point_densities(points)
  ft_matrix = compute_weighted_gaussians_from_flt_params(
      flt_params, points, flt_num_blobs_per_head, flt_num_rand_features
  )
  ratios = ft_matrix / densities
  result = jnp.broadcast_to(coords, (h, l, d))
  result = jnp.einsum('hld,hmd->hlm',
                      result, points,
                      precision=einsum_precision,
                      _dot_general=einsum_dot_general)
  result = jnp.exp(2.0 * jnp.pi * 1j * coeff * result)
  result = jnp.einsum('hlm,hm->lhm',
                      result, ratios,
                      precision=einsum_precision,
                      _dot_general=einsum_dot_general)
  return (1.0 / jnp.sqrt(flt_num_rand_features)) * result
