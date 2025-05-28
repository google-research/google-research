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

"""Fast clipping custom layer registry functions."""
from typing import Any

import tensorflow as tf
import tensorflow_models as tfm
from tensorflow_privacy.privacy.fast_gradient_clipping import type_aliases

from fast_gradient_clipping.src import einsum_utils


def naive_embedding_layer_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches=None,
):
  """Naive Embedding Layer Computation (works only for batch_size=1)."""
  if num_microbatches is not None:
    raise NotImplementedError("Microbatches are not supported.")
  del input_kwargs
  input_ids = tf.cast(*input_args, tf.int32)
  base_vars = layer_instance.trainable_variables[0]
  tape.watch(base_vars)
  outputs = tf.nn.embedding_lookup(base_vars, input_ids)

  def sqr_norm_fn(base_vars_grads):
    # ONLY WORKS FOR BATCH_SIZE = 1
    values = base_vars_grads.values
    indices = base_vars_grads.indices
    unique_indices, mapped_posns = tf.raw_ops.UniqueV2(  # Creates sparsity.
        x=indices, axis=[0]
    )
    num_uniques = unique_indices.shape[0]
    g = tf.sparse.segment_sum(
        values, tf.range(tf.shape(values)[0]), mapped_posns, num_uniques
    )
    # Creates a row-vector. Needs to be transposed below.
    u = tf.one_hot(mapped_posns, num_uniques, dtype=tf.float32)
    # Usual ghost-clipping rule
    ggt = tf.matmul(g, g, transpose_b=True)
    uut = tf.matmul(u, u, transpose_a=True)
    expanded_sqr_norm = tf.expand_dims(tf.reduce_sum(ggt * uut), axis=0)
    return expanded_sqr_norm

  return base_vars, outputs, sqr_norm_fn


def dense_layer_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches = None,
):
  """Registry function for `tf.keras.layers.Dense`.

  The logic for this computation is based on the following paper:
    https://arxiv.org/abs/1510.01799

  For the sake of efficiency, we fuse the variables and square grad norms
  for the kernel weights and bias vector together.

  Args:
    layer_instance: A `tf.keras.layers.Dense` instance.
    input_args: A `tuple` containing the first part of `layer_instance` input.
      Specifically, `layer_instance(*inputs_args, **input_kwargs)` should return
      a valid output.
    input_kwargs: A `tuple` containing the second part of `layer_instance`
      input. Specifically, `layer_instance(*inputs_args, **input_kwargs)` should
      return a valid output.
    tape: A `tf.GradientTape` instance that will be used to watch the output
      `base_vars`.
    num_microbatches: An optional numeric value or scalar `tf.Tensor` for
      indicating whether and how the losses are grouped into microbatches. If
      not None, num_microbatches must divide the batch size.

  Returns:
    A `tuple` `(base_vars, outputs, sqr_norm_fn)`. `base_vars` is the
    intermediate Tensor used in the chain-rule / "fast" clipping trick,
    `outputs` is the result of `layer_instance(*inputs)`, and `sqr_norm_fn` is
    a function that takes one input, a `tf.Tensor` that represents the output
    of the call `tape.gradient(summed_loss, base_vars)` where `tape` is a
    `tf.GradientTape` instance that records the dense layer computation and
    `summed_loss` is the sum of the per-example losses of the underlying model.
    This function then returns the per-example squared L2 gradient norms of the
    trainable variables in `layer_instance`. These squared norms should be a 1D
    `tf.Tensor` of length `batch_size`.
  """
  del input_kwargs  # Unused in dense layer calls.
  if len(input_args) != 1:
    raise ValueError("Only layer inputs of length 1 are permitted.")
  if num_microbatches is not None:
    raise NotImplementedError("Microbatching is not currently supported.")
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*input_args)
  tape.watch(base_vars)
  layer_instance.activation = orig_activation
  outputs = orig_activation(base_vars) if orig_activation else base_vars

  def sqr_norm_fn(grads):
    # `Dense` layers are special instances of `EinsumDense` layers
    return einsum_utils.compute_fast_einsum_squared_gradient_norm(
        "...b,bc->...c",
        input_args[0],
        grads,
        "c" if layer_instance.use_bias else None,
    )

  return base_vars, outputs, sqr_norm_fn


def einsum_layer_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches = None,
):
  """Registry function for `tf.keras.layers.EinsumDense`.

  For the technical details, see the documentation of
  `einsum_utils.compute_fast_einsum_gradient_norm()`.

  Args:
    layer_instance: A `tf.keras.layers.EinsumDense` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  del input_kwargs  # Unused in einsum layer calls.
  if num_microbatches is not None:
    raise NotImplementedError("Microbatching is not currently supported.")
  orig_activation = layer_instance.activation
  layer_instance.activation = None
  base_vars = layer_instance(*input_args)
  tape.watch(base_vars)
  layer_instance.activation = orig_activation
  outputs = orig_activation(base_vars) if orig_activation else base_vars

  def sqr_norm_fn(grads):
    return einsum_utils.compute_fast_einsum_squared_gradient_norm(
        layer_instance.equation,
        input_args[0],
        grads,
        layer_instance.bias_axes,
    )

  return base_vars, outputs, sqr_norm_fn


def nlp_on_device_embedding_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches = None,
):
  """Registry function for `tfm.nlp.layers.OnDeviceEmbedding`.

  Args:
    layer_instance: A `tfm.nlp.layers.OnDeviceEmbedding` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  # NOTE: Since the original code uses `.set_shape()`, we can assume that inputs
  # are not ragged.
  del input_kwargs
  if num_microbatches is not None:
    raise NotImplementedError("Microbatching is not currently supported.")
  inputs = tf.cast(input_args[0], dtype=tf.int64)
  scale = layer_instance._scale_factor  # pylint: disable=protected-access
  base_vars = layer_instance(inputs)
  if scale:
    base_vars /= scale
  tape.watch(base_vars)
  outputs = base_vars * scale if scale else base_vars

  def sqr_norm_fn(grads):
    # Gather the (row, input_id) pairs.
    nrows = tf.shape(inputs)[0]
    ncols = tf.reduce_prod(tf.shape(inputs)[1:])
    repeats = tf.repeat(ncols, nrows)
    row_indices = tf.reshape(tf.repeat(tf.range(nrows), repeats), [-1, 1])
    flattened_ids = tf.expand_dims(tf.reshape(inputs, [-1]), axis=-1)
    paired_indices = tf.concat(
        [tf.cast(row_indices, tf.int64), tf.cast(flattened_ids, tf.int64)],
        axis=1,
    )
    # Apply the adjoint operator (segment_sum).
    (unique_paired_indices, new_index_positions) = tf.raw_ops.UniqueV2(
        x=paired_indices, axis=[0]
    )
    unique_batch_ids = unique_paired_indices[:, 0]
    flattened_grads = tf.reshape(grads, [-1, layer_instance.embedding_width])
    summed_gradients = tf.math.unsorted_segment_sum(
        flattened_grads,
        new_index_positions,
        tf.shape(unique_paired_indices)[0],
    )
    # Compute the squared gradient norms at the per-example level.
    sqr_gradient_sum = tf.reduce_sum(tf.square(summed_gradients), axis=1)
    summed_data_range = tf.range(tf.shape(sqr_gradient_sum)[0])
    return tf.sparse.segment_sum(
        sqr_gradient_sum,
        summed_data_range,
        tf.sort(unique_batch_ids),
        num_segments=nrows,
    )  # fill in empty inputs

  return base_vars, outputs, sqr_norm_fn


def nlp_position_embedding_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches = None,
):
  """Registry function for `tfm.nlp.layers.PositionEmbedding`.

  Args:
    layer_instance: A `tf.keras.layers.EinsumDense` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  del input_kwargs
  if num_microbatches is not None:
    raise NotImplementedError("Microbatching is not currently supported.")
  inputs = input_args[0]
  base_vars = layer_instance(inputs)
  tape.watch(base_vars)

  def sqr_norm_fn(grads):
    broadcast_axes = list(range(len(grads.shape)))
    del broadcast_axes[layer_instance._seq_axis]  # pylint: disable=protected-access
    del broadcast_axes[-1], broadcast_axes[0]
    reduced_grads = tf.reduce_sum(grads, axis=broadcast_axes)
    reduction_axes = tf.range(1, len(reduced_grads.shape))
    return tf.reduce_sum(tf.square(reduced_grads), axis=reduction_axes)

  return base_vars, base_vars, sqr_norm_fn


def layer_normalization_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches = None,
):
  """Registry function for `tf.keras.layers.LayerNormalization`.

  This function computes actual per-example gradients and computes their
  norms directly, instead of employing a chain-rule trick. This is done using
  some slick reshaping calls.

  Args:
    layer_instance: A `tf.keras.layers.LayerNormalization` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  del input_kwargs  # Unused in layer normaliztion calls.
  if num_microbatches is not None:
    raise NotImplementedError("Microbatching is not currently supported.")

  # To make sure the watched variables (beta, gamma) generate per-example
  # gradients, we need to convert trainable variables from shape [S] to
  # [batch_size, S] via duplication to `tf.shape(inputs)` via broadcasting.
  inputs = input_args[0]
  base_vars = []
  batch_size = tf.shape(inputs)[0]

  def process_variable(var):
    expanded_var = tf.tile(
        tf.expand_dims(var, axis=0), [batch_size] + [1] * len(var.shape)
    )
    tape.watch(expanded_var)
    base_vars.append(expanded_var)
    broadcast_shape = [1] * len(inputs.shape)
    broadcast_shape[0] = batch_size
    for d in layer_instance.axis:
      broadcast_shape[d] = tf.shape(inputs)[d]
    final_var = tf.reshape(expanded_var, broadcast_shape)
    return final_var

  orig_gamma = layer_instance.gamma
  orig_beta = layer_instance.beta
  layer_instance.gamma = process_variable(orig_gamma)
  layer_instance.beta = process_variable(orig_beta)

  # Do the computation, ensure that the output conforms to the unexpanded
  # computation, and restore the state of the original instance.
  outputs = layer_instance.call(inputs)
  layer_instance.gamma = orig_gamma
  layer_instance.beta = orig_beta

  def sqr_norm_fn(grads):
    stacked_grads = tf.stack(grads, axis=-1)
    reduction_axes = tf.range(1, len(stacked_grads.shape))
    return tf.reduce_sum(tf.square(stacked_grads), axis=reduction_axes)

  return base_vars, outputs, sqr_norm_fn


def multi_head_attention_layer_computation(
    layer_instance,
    input_args,
    input_kwargs,
    tape,
    num_microbatches = None,
):
  """Registry function for `tf.keras.layers.MultiHeadAttention`.

  This function essentially applies the registry function for
  `tf.keras.layers.EinsumDense` three times. Some hints about the nature of
  the Einsum transforms are given below.

  -------------------
  ABOUT INPUT SHAPES
  -------------------
  For a given {query, key, value} input `I` of shape

    [Eq. A]  tf.shape(I) == [n, a[0],... , a[k-1], b]

  where `n` is the batch size, the corresponding Einsum equation for its
  `EinsumDense` transform is given by:

    {n a[0] ... a[k-1] b},{b c d}->{n a[1] ... a[k-1] c d}

  where `c` corresponds to the number of attention heads
  (`layer_instance.num_heads`) and `d` corresponds to the size per head
  (`layer_instance.key_dim` or `layer_instance.value_dim`).

  It is expected that the rank of the query, key, and value inputs are the same.

  ------------------
  ABOUT OUTPUT SHAPE
  ------------------
  Suppose the shape of the `query` input `Q` is given by [Eq. A] above with
  `I == Q`. Then, if `layer_instance.output_shape is None`, the output `O` of
  the layer satisfies `tf.shape(Q) == tf.shape(O)`. However, if we have
  `layer_instance.output_shape is not None`, then

    tf.shape(Q) == [n, a[0], ..., a[k-1], *layer_instance.output_shape]

  Args:
    layer_instance: A `tf.keras.layers.MultiHeadAttention` instance.
    input_args: See `dense_layer_computation()`.
    input_kwargs: See `dense_layer_computation()`.
    tape: See `dense_layer_computation()`.
    num_microbatches: See `dense_layer_computation()`.

  Returns:
    See `dense_layer_computation()`.
  """
  # ----------------------
  # PREPROCESS THE INPUTS.
  # ----------------------
  query = (
      input_kwargs.get("query")
      if input_kwargs.get("query") is not None
      else input_args[0]
  )
  value = (
      input_kwargs.get("value")
      if input_kwargs.get("value") is not None
      else input_args[1]
  )
  key = input_kwargs.get("key")
  attention_mask = input_kwargs.get("attention_mask")
  return_attention_scores = input_kwargs.get("return_attention_scores")
  training = input_kwargs.get("training")
  use_causal_mask = input_kwargs.get("use_causal_mask")
  attention_mask = layer_instance._compute_attention_mask(  # pylint: disable=protected-access
      query,
      value,
      key=key,
      attention_mask=attention_mask,
      use_causal_mask=use_causal_mask,
  )
  if not layer_instance._built_from_signature:  # pylint: disable=protected-access
    layer_instance._build_from_signature(query=query, value=value, key=key)  # pylint: disable=protected-access
  if key is None:
    key = value

  query_is_ragged = isinstance(query, tf.RaggedTensor)
  if query_is_ragged:
    query_lengths = query.nested_row_lengths()
    query = query.to_tensor()

  key_is_ragged = isinstance(key, tf.RaggedTensor)
  value_is_ragged = isinstance(value, tf.RaggedTensor)
  if key_is_ragged and value_is_ragged:
    bounding_shape = tf.math.maximum(
        key.bounding_shape(), value.bounding_shape()
    )
    key = key.to_tensor(shape=bounding_shape)
    value = value.to_tensor(shape=bounding_shape)
  elif key_is_ragged:
    key = key.to_tensor(shape=tf.shape(value))
  elif value_is_ragged:
    value = value.to_tensor(shape=tf.shape(key))
  # ------------------------------
  # APPLY THE FAST CLIPPING TRICK.
  # ------------------------------
  # trainable_op: W_q * QUERY
  query_base_vars, query, query_sqr_norm_fn = einsum_layer_computation(
      layer_instance._query_dense,  # pylint: disable=protected-access
      (query,),
      {},
      tape,
      num_microbatches,
  )
  # trainable_op: W_k * KEY
  key_base_vars, key, key_sqr_norm_fn = einsum_layer_computation(
      layer_instance._key_dense,  # pylint: disable=protected-access
      (key,),
      {},
      tape,
      num_microbatches,
  )
  # trainable_op: W_v * VALUE
  value_base_vars, value, value_sqr_norm_fn = einsum_layer_computation(
      layer_instance._value_dense,  # pylint: disable=protected-access
      (value,),
      {},
      tape,
      num_microbatches,
  )
  # op: TEMP = ATTENTION(W_q * QUERY, W_k * KEY, W_v * VALUE)
  temp_output, attention_scores = layer_instance._compute_attention(  # pylint: disable=protected-access
      query,
      key,
      value,
      attention_mask,
      training,
  )
  # trainable_op: W_o * OUTPUT
  (
      attention_output_base_vars,
      attention_output,
      attention_output_sqr_norm_fn,
  ) = einsum_layer_computation(
      layer_instance._output_dense,  # pylint: disable=protected-access
      (temp_output,),
      {},
      tape,
      num_microbatches,
  )
  # ------------------------
  # POSTPROCESS THE OUTPUTS.
  # ------------------------
  # Get registry output tensors ready.
  if query_is_ragged:
    attention_output = tf.RaggedTensor.from_tensor(
        attention_output, query_lengths
    )
  outputs = attention_output
  if return_attention_scores:
    outputs = (attention_output, attention_scores)
  base_vars = [
      query_base_vars,
      key_base_vars,
      value_base_vars,
      attention_output_base_vars,
  ]

  # The square norm function should just aggregate the squared norms
  # corresponding to each trainable op.
  def sqr_norm_fn(grad_list):
    if len(grad_list) != 4:
      raise ValueError(
          "Expected a container of 4 gradients for the `MultiheadAttention` "
          "square norm function's input. Instead, received a container of "
          "size "
          + str(len(grad_list))
      )
    combined_sqr_norms = tf.stack(
        [
            query_sqr_norm_fn(grad_list[0]),
            key_sqr_norm_fn(grad_list[1]),
            value_sqr_norm_fn(grad_list[2]),
            attention_output_sqr_norm_fn(grad_list[3]),
        ],
        axis=1,
    )
    return tf.reduce_sum(combined_sqr_norms, axis=1)

  return base_vars, outputs, sqr_norm_fn
