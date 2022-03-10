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

"""Sparse attention utils."""

import functools
import itertools
import math
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import


def is_xla_compiled():
  """Whether we are building graph that will be compiled by XLA.

  This checks whether the code is executing within an XLA context.

  If True, model authors should ensure the graph they build is compilable by
  XLA. Specifically, they should ensure that all ops have XLA implementations
  and that all shapes are statically known.

  Returns:
    bool, whether the current graph will be compiled for XLA.
  """
  ctxt = tf.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  x = control_flow_util.GetContainingXLAContext(ctxt) is not None
  return x


def get_channel_embeddings(io_depth,
                           targets,
                           hidden_size,
                           name="channel",
                           vocab_size=256):
  """Get separate embedding for each of the channels."""
  targets_split = tf.split(targets, io_depth, axis=3)
  rgb_embedding_var = tf.get_variable("rgb_target_emb_%s" % name,
                                      [vocab_size * io_depth, hidden_size])
  rgb_embedding_var = tf.identity(rgb_embedding_var)
  rgb_embedding_var *= float(hidden_size)**0.5
  channel_target_embs = []
  for i in range(io_depth):
    # Adding the channel offsets to get the right embedding since the
    # embedding tensor has shape 256 * io_depth, hidden_size
    target_ids = tf.squeeze(targets_split[i], axis=3) + i * vocab_size
    target_embs = tf.gather(rgb_embedding_var, target_ids)
    channel_target_embs.append(target_embs)

  return tf.concat(channel_target_embs, axis=-1)


def get_embeddings(targets, vocab_size, hidden_size, name="embeddings"):
  """Get embeddings for symbols in the targets."""
  with tf.variable_scope(name_or_scope=name):
    var = tf.get_variable("embedding", shape=[vocab_size, hidden_size])
    embed = tf.gather(var, targets)

  return embed


def right_shift_blockwise_nd(x, block_shape):
  """Right shift once in every block.

  Args:
    x: a [batch, d1, d2, ..., dn, depth] tensor
    block_shape: a tuple (q1, q2, ..., qn) representing the block shape

  Returns:
    a [batch, d1, d2, ..., dn, depth] tensor, right shifted.
  """
  blocked_x = break_into_blocks_nd(x, block_shape)
  blocked_x_shape = shape_list(blocked_x)
  blocked_x = tf.reshape(blocked_x,
                         [blocked_x_shape[0], -1, blocked_x_shape[-1]])
  padded_x = tf.pad(blocked_x, [[0, 0], [1, 0], [0, 0]])
  x = tf.slice(padded_x, [0, 0, 0],
               [-1, np.prod(blocked_x_shape[1:-1], dtype=np.int32), -1])
  x = tf.reshape(x, blocked_x_shape)
  return put_back_blocks_nd(x, block_shape)


def add_positional_embedding_nd(x, max_length, name=None):
  """Adds n-dimensional positional embedding.

  The embeddings add to all positional dimensions of the tensor.

  Args:
    x: Tensor with shape [batch, p1 ... pn, depth]. It has n positional
      dimensions, i.e., 1 for text, 2 for images, 3 for video, etc.
    max_length: int representing static maximum size of any dimension.
    name: str representing name of the embedding tf.Variable.

  Returns:
    Tensor of same shape as x.
  """
  with tf.name_scope("add_positional_embedding_nd"):
    x_shape = shape_list(x)
    num_dims = len(x_shape) - 2
    depth = x_shape[-1]
    base_shape = [1] * (num_dims + 1) + [depth]
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [1] * num_dims + [depth]
    for i in range(num_dims):
      shape = base_shape[:]
      start = base_start[:]
      size = base_size[:]
      shape[i + 1] = max_length
      size[i + 1] = x_shape[i + 1]
      var = tf.get_variable(
          name + "_%d" % i,
          shape,
          initializer=tf.random_normal_initializer(0, depth**-0.5))
      var = var * depth**0.5
      x += tf.slice(var, start, size)
    return x


def multihead_attention_nd_partial(hparams):
  """Returns partial multihead_attention_nd to reduce boilerplate."""
  multihead_fn = functools.partial(
      multihead_attention_nd,
      output_depth=hparams.hidden_size,
      query_shape=hparams.query_shape,
      memory_query_shape=hparams.memory_query_shape,
      memory_flange=hparams.memory_flange,
      sparsity_cluster_size=hparams.sparsity_cluster_size,
      sparsity_cluster_attention_window=hparams
      .sparsity_cluster_attention_window,
      sparsity_cluster_strided_num_heads=hparams
      .sparsity_cluster_strided_num_heads,
      sparsity_cluster_strided_relative=hparams
      .sparsity_cluster_strided_relative,
      sparsity_strided_num_heads=hparams.sparsity_strided_num_heads,
      sparsity_strided_relative=hparams.sparsity_strided_relative,
      mode=hparams.mode,
      cache_padding_bias=hparams.cache_padding_bias,
      max_relative_position=hparams.max_relative_position,
      dropout_rate=hparams.attention_dropout,
      ema=hparams.ema,
      beta=hparams.beta,
      decay=hparams.decay,
      hash_items=hparams.hash_items,
      use_tpu=hparams.use_tpu)
  return multihead_fn


def transformer_encoder_layers(inputs,
                               num_layers,
                               hparams,
                               losses,
                               name="transformer_encoder",
                               token_bias=None,
                               padding_bias=None):
  """Multi layer transformer encoder with default un-masked attention.

  Args:
    inputs: Input tensor to the attention layers.
    num_layers: Number of attention layers.
    hparams: Hparam object containing attention configurations.
    losses: Losses dict for training.
    name: Name of the layers.
    token_bias: Externally provided attention bias for self attention on inputs.
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).

  Returns:
    Output transformed by self-attention.
  """
  x = inputs
  if hparams.layer_prepostprocess_dropout:
    x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)
  key_depth = hparams.attention_key_channels or hparams.hidden_size
  value_depth = hparams.attention_value_channels or hparams.hidden_size

  # A placeholder for attention bias cache tensors to facilitate sharing across
  # attention types/layers.
  bias_cache = {}
  multihead_attention_fn = multihead_attention_nd_partial(hparams)
  local_heads, sparsity_cluster_heads = 0, 0
  for layer in range(num_layers):
    local_heads = hparams.local_num_heads
    sparsity_cluster_heads = hparams.sparsity_cluster_num_heads
    if layer < hparams.sparsity_skip_first:
      local_heads = hparams.local_num_heads + hparams.sparsity_cluster_num_heads
      sparsity_cluster_heads = 0
    with tf.variable_scope("%s_layer_%d" % (name, layer), reuse=tf.AUTO_REUSE):
      with tf.variable_scope("self_attention"):
        y = multihead_attention_fn(
            query_antecedent=layer_preprocess(x, hparams),
            memory_antecedent=None,
            total_key_depth=key_depth,
            total_value_depth=value_depth,
            masked=False,
            losses=losses,
            name="self_attention",
            bias_cache=bias_cache,
            local_relative=hparams.local_relative,
            local_num_heads=local_heads,
            sparsity_cluster_relative=hparams.sparsity_cluster_relative,
            sparsity_cluster_num_heads=sparsity_cluster_heads,
            is_recomputing=False,
            share_qk=False,  # No need to share qk for encoder self attn
            token_bias=token_bias,
            token_bias_wt_trainable=hparams.token_bias_wt_trainable,
            padding_bias=padding_bias)
        x = layer_postprocess(x, y, hparams)
        # feed-fwd layers + skip connections
        y = ffn_layer(layer_preprocess(x, hparams), hparams)
        x = layer_postprocess(x, y, hparams)
  return layer_preprocess(x, hparams)


def transformer_decoder_layers(inputs,
                               num_layers,
                               hparams,
                               losses,
                               encoder_output=None,
                               decode_step=None,
                               cache=None,
                               name="transformer_decoder",
                               decoding_stats=None,
                               token_bias_inputs=None,
                               token_bias_targets=None,
                               padding_bias=None):
  """Multi layer transformer encoder or decoder with default masked attention.

  Args:
    inputs: Input tensor to the attention layers.
    num_layers: Number of attention layers.
    hparams: Hparam object containing attention configurations.
    losses: Losses dict for training.
    encoder_output: Optional argument signifying encoder output.
    decode_step: Decode step for decoding.
    cache: Cache containing layer attention values for faster computation.
    name: Name of the layers.
    decoding_stats: Dictionary containing decoding stats.
    token_bias_inputs: Externally provided attention bias on encoder inputs.
    token_bias_targets: Externally provided attention bias on decoder targets.
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).

  Returns:
    Output transformed by self-attention.
  """
  x = inputs
  if hparams.layer_prepostprocess_dropout:
    x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)
  key_depth = hparams.attention_key_channels or hparams.hidden_size
  value_depth = hparams.attention_value_channels or hparams.hidden_size

  # A placeholder for attention bias cache tensors to facilitate sharing across
  # attention types/layers.
  bias_cache = {}
  multihead_attention_fn = multihead_attention_nd_partial(hparams)
  local_heads, sparsity_cluster_heads = 0, 0
  for layer in range(num_layers):
    local_heads = hparams.local_num_heads
    sparsity_cluster_heads = hparams.sparsity_cluster_num_heads
    if layer < hparams.sparsity_skip_first:
      local_heads = hparams.local_num_heads + hparams.sparsity_cluster_num_heads
      sparsity_cluster_heads = 0
    with tf.variable_scope("%s_layer_%d" % (name, layer), reuse=tf.AUTO_REUSE):
      layer_cache = None
      if decode_step is None and cache is not None:
        # Initialize layer cache.
        cache[layer] = {}
        layer_cache = cache[layer]
      if decode_step is not None:
        layer_cache = cache[layer]

      with tf.variable_scope("self_attention"):
        y = multihead_attention_fn(
            query_antecedent=layer_preprocess(x, hparams),
            memory_antecedent=None,
            total_key_depth=key_depth,
            total_value_depth=value_depth,
            masked=True,
            losses=losses,
            decode_step=decode_step,
            cache=layer_cache,
            name="self_attention",
            bias_cache=bias_cache,
            is_recomputing=False,
            local_relative=hparams.local_relative,
            local_num_heads=local_heads,
            sparsity_cluster_relative=hparams.sparsity_cluster_relative,
            sparsity_cluster_num_heads=sparsity_cluster_heads,
            decoding_stats=decoding_stats,
            share_qk=hparams.share_qk,
            token_bias=token_bias_targets,
            token_bias_wt_trainable=hparams.token_bias_wt_trainable,
            padding_bias=None)
        x = layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          y = multihead_attention_fn(
              query_antecedent=layer_preprocess(x, hparams),
              memory_antecedent=encoder_output,
              total_key_depth=key_depth,
              total_value_depth=value_depth,
              masked=False,
              losses=losses,
              decode_step=decode_step,
              cache=layer_cache,
              name="enc_dec_attention",
              bias_cache=bias_cache,
              is_recomputing=False,
              local_relative=False,
              local_num_heads=local_heads,
              sparsity_cluster_relative=False,
              sparsity_cluster_num_heads=sparsity_cluster_heads,
              decoding_stats=decoding_stats,
              share_qk=False,  # No need to share qk for encoder-decoder attn
              token_bias=token_bias_inputs,
              token_bias_wt_trainable=hparams.token_bias_wt_trainable,
              padding_bias=padding_bias)
          x = layer_postprocess(x, y, hparams)
        # feed-fwd layers + skip connections
        y = ffn_layer(layer_preprocess(x, hparams), hparams)
        x = layer_postprocess(x, y, hparams)
  if decode_step is not None:
    x = get_item_at_decode_step(x, decode_step, hparams.query_shape)
  return layer_preprocess(x, hparams)


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def geglu(inputs,
          filter_size,
          output_size,
          output_activation=None,
          dropout=0.0,
          dropout_broadcast_dims=None,
          name=None):
  """GEGLU activation as in https://arxiv.org/abs/2002.05202."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  layer_name = "%s_{}" % name if name else "{}"
  h = tf.layers.dense(
      inputs,
      filter_size,
      use_bias=False,
      activation=None,
      name=layer_name.format("weight1"))
  h = gelu(h)
  v = tf.layers.dense(
      inputs,
      filter_size,
      use_bias=False,
      activation=None,
      name=layer_name.format("weight2"))
  h *= v
  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = tf.layers.dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=False,
      name=layer_name.format("weight3"))
  return o


def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  layer_name = "%s_{}" % name if name else "{}"
  h = tf.layers.dense(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = tf.layers.dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      name=layer_name.format("conv2"))
  return o


def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias, layer_collection=None):
  """Layer norm raw computation."""

  # Save these before they get converted to tensors by the casting below
  params = (scale, bias)

  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(
      tf.squared_difference(x, mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

  output = norm_x * scale + bias

  if layer_collection:
    # Note that the first dimension of norm_x must be the batch size
    layer_collection.register_scale_and_shift(
        params, norm_x, output, approx="full")

  return output


def layer_norm(x,
               filters=None,
               epsilon=1e-6,
               name=None,
               reuse=None,
               layer_collection=None,
               scaling=True):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    if scaling:
      scale, bias = layer_norm_vars(filters)
    else:
      scale = tf.constant(1.0)
      bias = tf.constant(0.0)
    return layer_norm_compute(
        x, epsilon, scale, bias, layer_collection=layer_collection)


def ffn_layer(x, hparams):
  """ffn layer transformer."""
  with tf.variable_scope("ffn"):
    if hparams.ffn_layer == "none":
      return x
    elif hparams.ffn_layer == "geglu":
      return geglu(
          x,
          hparams.filter_size,
          hparams.hidden_size,
          dropout=hparams.relu_dropout)
    else:
      return dense_relu_dense(
          x,
          hparams.filter_size,
          hparams.hidden_size,
          dropout=hparams.relu_dropout)


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None,
                         layer_collection=None):
  """Apply a sequence of functions to the input or output of a layer.

  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout

  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))

  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    depth: an integer (size of last dimension of x).
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
    dropout_broadcast_dims:  an optional list of integers less than 3 specifying
      in which dimensions to broadcast the dropout decisions. saves memory.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.

  Returns:
    a Tensor
  """
  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x
    for c in sequence:
      if c == "a":
        x += previous_value
      elif c == "n":
        x = layer_norm(x, depth, epsilon, layer_collection=layer_collection)
      elif c == "d":
        x = dropout_with_broadcast_dims(
            x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
      else:
        raise ValueError("Unknown processing sequence.")
    return x


def comma_separated_string_to_integer_list(s):
  return [int(i) for i in s.split(",") if i]


def layer_preprocess(layer_input, hparams, layer_collection=None):
  """Apply layer preprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_preprocess_sequence
    layer_prepostprocess_dropout
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.

  Returns:
    a Tensor
  """
  assert "a" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  assert "z" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=hparams.layer_preprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_prepostprocess",
      layer_collection=layer_collection)


def layer_postprocess(layer_input, layer_output, hparams):
  """Apply layer postprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_postprocess_sequence
    layer_prepostprocess_dropout
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=hparams.layer_postprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_postprocess")


def multihead_attention_nd(query_antecedent,  # pylint: disable=dangerous-default-value
                           memory_antecedent,
                           total_key_depth,
                           total_value_depth,
                           output_depth,
                           query_shape,
                           memory_query_shape,
                           memory_flange,
                           local_num_heads,
                           local_relative=False,
                           sparsity_cluster_size=None,
                           sparsity_cluster_attention_window=None,
                           sparsity_cluster_num_heads=0,
                           sparsity_cluster_relative=False,
                           sparsity_cluster_strided_num_heads=0,
                           sparsity_cluster_strided_relative=False,
                           sparsity_strided_num_heads=0,
                           sparsity_strided_relative=False,
                           losses=None,
                           mode=tf.estimator.ModeKeys.EVAL,
                           masked=False,
                           cache=None,
                           decode_step=None,
                           name=None,
                           cache_padding_bias=False,
                           max_relative_position=None,
                           dropout_rate=0.,
                           bias_cache={},
                           ema=False,
                           beta=1e-4,
                           decay=0.99,
                           share_qk=False,
                           hash_items=False,
                           is_recomputing=False,
                           decoding_stats=None,
                           token_bias=None,
                           token_bias_wt_trainable=False,
                           padding_bias=None,
                           use_tpu=False):
  """n-d Multihead scaled-dot-product attention with in/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, d1, ..., dn, depth_q].
    memory_antecedent: a Tensor with shape [batch, d1, ..., dn, depth_m] or None
      for self attention.
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    query_shape: an tuple indicating the dimensions of each query block.
    memory_query_shape: query shape for memory antecedent (enc-dec).
    memory_flange: an integer indicating how much to look around a query block
      in each dimension
    local_num_heads: How many heads to use for local-nd attention. The sum of
      all heads should divide total_key_depth and total_value_depth.
    local_relative: whether to use relative for local heads,
    sparsity_cluster_size: Number of clusters.
    sparsity_cluster_attention_window: number of items within a cluster to
      attend to.
    sparsity_cluster_num_heads: how many heads to use for attention within
      cluster.
    sparsity_cluster_relative: whether to use relative for clustered attention
    sparsity_cluster_strided_num_heads: how many heads to use for attending to
      other clusters.
    sparsity_cluster_strided_relative: whether to use relative for strided
      clustering
    sparsity_strided_num_heads: how many heads to use for strided sparsity.
    sparsity_strided_relative: whether to use relative for strided heads.
    losses: a list of extra losses.
    mode: a tf.estimator.ModeKeys.
    masked: a boolean to specify whether to do masked or unmasked attention.
    cache: a dict like: {
      'q': [batch, num_heads, d1, ..., dn, depth_k // num_heads],
      'k': [batch, num_heads, d1, ..., dn, depth_k // num_heads],
      'v': [batch, num_heads, d1, ..., dn, depth_v // num_heads]} Caller should
        initially pass an empty dictionary and this method will update cache and
        caller should pass the same cache in consecutive calls. This works for
        both GPU and TPU inference. `memory_antecedent` should be None in this
        case, since auto-regressive decoding only applies to self attention.
    decode_step: integer to pass in decoding mode. `cache` and `decode_step`
      should both be set in decoding mode. Caller can also pass an empty `cache`
      without `decode_step`, for this method to initialize the cache for future
      calls with `decode_step` > 0.
    name: an optional string
    cache_padding_bias: If sequences are not variable length (e.g. images and
      videos) and the only source of padding is to be evenly divisible by blocks
      we can cache padding bias as well to save memory.
    max_relative_position: how much distance to consider for relative positions.
    dropout_rate: Rate of dropout.
    bias_cache: a dict containing attention bias cache.
    ema: a boolean to do ema updates.
    beta: multiplier for clustering loss.
    decay: decay factor for learning centroids.
    share_qk: Whether to share queries and keys.
    hash_items: Whether to hash items instead of clustering.
    is_recomputing: a boolean to represent whether this is a backward pass.
    decoding_stats: a dict to be used to return tensors to capture additional
      stats in decoding mode.
    token_bias: Externally provided attention bias over memory sequence (k / v).
    token_bias_wt_trainable: Whether or not token_bias_weight is trainable.
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).
    use_tpu: Whether to use TPU (default: False).

  Returns:
    A Tensor of shape [batch, d1, ..., dn, output_depth] or
      [batch, 1, ..., 1, output_depth] if decode_step is set.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  num_heads = (
      local_num_heads + sparsity_cluster_num_heads +
      sparsity_cluster_strided_num_heads + sparsity_strided_num_heads)

  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))

  # Validate that if we share keys and queries for clustering, memory is None
  if share_qk:
    assert memory_antecedent is None
  # Validate decoding input params are sensible.
  if decode_step is not None:
    assert "q" in cache and "k" in cache and "v" in cache

  with tf.variable_scope(
      name,
      default_name="multihead_attention_nd",
      values=[query_antecedent, memory_antecedent]):
    if decode_step is not None:
      latest_antecedent = get_item_at_decode_step(query_antecedent, decode_step,
                                                  query_shape)
      latest_q, latest_k, latest_v = compute_qkv(latest_antecedent,
                                                 memory_antecedent,
                                                 total_key_depth,
                                                 total_value_depth)
      latest_q = split_heads_nd(latest_q, num_heads)
      key_depth_per_head = total_key_depth // num_heads
      latest_q *= key_depth_per_head**-0.5
      latest_k = split_heads_nd(latest_k, num_heads)
      latest_v = split_heads_nd(latest_v, num_heads)
      # put latest q, k and v into their correct position in cache.
      q = cache["q"]
      k = cache["k"]
      v = cache["v"]
      q = put_item_in_decode_step(q, latest_q, decode_step, query_shape)
      if memory_antecedent is None:
        k = put_item_in_decode_step(k, latest_k, decode_step, query_shape)
        v = put_item_in_decode_step(v, latest_v, decode_step, query_shape)
      cache["q"] = q
      cache["k"] = k
      cache["v"] = v
    else:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth)
      q = split_heads_nd(q, num_heads)
      key_depth_per_head = total_key_depth // num_heads
      q *= key_depth_per_head**-0.5
      k = split_heads_nd(k, num_heads)
      v = split_heads_nd(v, num_heads)
      if cache is not None:
        cache["q"] = q
        cache["k"] = k
        cache["v"] = v

    x = attention_nd(
        q,
        k,
        v,
        query_shape=query_shape,
        memory_query_shape=memory_query_shape,
        memory_flange=memory_flange,
        memory_antecedent=memory_antecedent,
        local_num_heads=local_num_heads,
        local_relative=local_relative,
        sparsity_cluster_size=sparsity_cluster_size,
        sparsity_cluster_attention_window=sparsity_cluster_attention_window,
        sparsity_cluster_num_heads=sparsity_cluster_num_heads,
        sparsity_cluster_relative=sparsity_cluster_relative,
        sparsity_cluster_strided_num_heads=sparsity_cluster_strided_num_heads,
        sparsity_cluster_strided_relative=sparsity_cluster_strided_relative,
        sparsity_strided_num_heads=sparsity_strided_num_heads,
        sparsity_strided_relative=sparsity_strided_relative,
        masked=masked,
        losses=losses,
        mode=mode,
        decode_step=decode_step,
        cache_padding_bias=cache_padding_bias,
        max_relative_position=max_relative_position,
        dropout_rate=dropout_rate,
        bias_cache=bias_cache,
        ema=ema,
        beta=beta,
        decay=decay,
        share_qk=share_qk,
        hash_items=hash_items,
        is_recomputing=is_recomputing,
        decoding_stats=decoding_stats,
        token_bias=token_bias,
        token_bias_wt_trainable=token_bias_wt_trainable,
        padding_bias=padding_bias,
        use_tpu=use_tpu)

    x = combine_heads_nd(x)
    x = tf.layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    return x


def decode_step_to_index(decode_step, query_shape, tensor_shape):
  """Maps decode step to n-d index according to blocked raster scan order.

  Args:
    decode_step: an integer
    query_shape: a tuple (q1, q2, ..., qn) representing the query shape
    tensor_shape: a tuple (d1, d2, ..., dn) representing the tensor shape, minus
      the batch and depth dimensions.

  Returns:
    a tuple (i1, i2, ..., in) representing the index of the element at
    `decode_step` w.r.t. blocked raster scan order.
  """
  assert len(query_shape) == len(tensor_shape)
  blocks_per_dimension = [t // q for t, q in zip(tensor_shape, query_shape)]
  items_in_block = np.prod(query_shape, dtype=np.int32)
  step_block = decode_step // items_in_block
  step_within_block = decode_step % items_in_block

  block_index = []
  for q in blocks_per_dimension[::-1]:
    block_index.insert(0, step_block % q)
    step_block //= q

  within_block_index = []
  for q in query_shape[::-1]:
    within_block_index.insert(0, step_within_block % q)
    step_within_block //= q

  final_index = [
      w + b * q for w, b, q in zip(within_block_index, block_index, query_shape)
  ]
  return tuple(final_index)


def get_item_at_decode_step(x, decode_step, query_shape):
  """Extracts a single item from an n-d tensor at `decode_step` position.

  Args:
    x: a [batch, d1, d2, ..., dn, depth] tensor
    decode_step: an integer
    query_shape: a tuple (q1, q2, ..., qn) representing the query shape

  Returns:
    a [batch, 1, 1, ..., 1, depth] tensor that is a single element from `x` at
    `decode_step` w.r.t. blocked raster scan order.
  """
  x_shape = shape_list(x)
  index = decode_step_to_index(decode_step, query_shape, x_shape[1:-1])
  # TPU needs size to be non negative for the case when begins are not
  # compile-time constants.
  return tf.slice(x, [0] + list(index) + [0],
                  [x_shape[0]] + [1] * len(index) + [x_shape[-1]])


def put_item_in_decode_step(x, item, decode_step, query_shape):
  """Puts a single item into an n-d tensor at `decode_step` position.

  Args:
    x: a [batch, heads, d1, d2, ..., dn, depth] tensor
    item: a [batch, heads, 1, 1, ..., 1, depth] tensor
    decode_step: an integer
    query_shape: a tuple (q1, q2, ..., qn) representing the query shape

  Returns:
    a [batch, heads, d1, d2, ..., dn, depth] tensor with value at `decode_step`
    w.r.t. blocked raster scan order is updated to be `item`.
  """
  x_shape = shape_list(x)
  index = decode_step_to_index(decode_step, query_shape, x_shape[2:-1])
  # inplace_update only works on the first dimension, we need to flatten and
  # move batch to be the second dimension.
  flattened_x = tf.reshape(
      x, [-1, x_shape[1], np.prod(x_shape[2:-1]), x_shape[-1]])
  # transpose to [positions, batch, heads, depth]
  flattened_x = tf.transpose(flattened_x, [2, 0, 1, 3])

  flattened_index = 0
  factor = 1
  for d, idx in zip(x_shape[-2:1:-1], index[::-1]):
    flattened_index += idx * factor
    factor *= d

  item_shape = shape_list(item)
  item = tf.reshape(item, item_shape[:2] + item_shape[-1:])
  updated_x = inplace_ops.alias_inplace_update(
      flattened_x,
      flattened_index,
      item)
  # unflatten the results
  updated_x = tf.transpose(updated_x, [1, 2, 0, 3])
  return tf.reshape(updated_x, [-1, x_shape[1]] + x_shape[2:])


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
      to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables).

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads)
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads)
  return q, k, v


def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0):
  """Computes attention component (query, key or value).

  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)

  Returns:
    c : [batch, length, depth] tensor
  """
  if vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth**-0.5
    if "q" in name:
      initializer_stddev *= depth_per_head**-0.5
    var = tf.get_variable(
        name,
        [input_depth, vars_3d_num_heads, total_depth // vars_3d_num_heads],
        initializer=tf.random_normal_initializer(stddev=initializer_stddev))
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    return tf.layers.dense(antecedent, total_depth, use_bias=False, name=name)
  else:
    return tf.layers.conv1d(
        antecedent, total_depth, filter_width, padding=padding, name=name)


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def split_heads_nd(x, num_heads):
  """Split the depth dimension (last dimension) into multiple heads.

  Args:
    x: a [batch, d1, ..., dn, depth] tensor
    num_heads: an integer

  Returns:
    a [batch, num_heads, d1, ..., dn, depth // num_heads]
  """
  num_dimensions = len(shape_list(x)) - 2
  return tf.transpose(
      split_last_dimension(x, num_heads), [0, num_dimensions + 1] +
      list(range(1, num_dimensions + 1)) + [num_dimensions + 2])


def combine_heads_nd(x):
  """Inverse of split_heads_nd.

  Args:
    x: a [batch, num_heads, d1, ..., dn, depth // num_heads] tensor

  Returns:
    a [batch, d1, ...., dn, depth] tensor
  """
  num_dimensions = len(shape_list(x)) - 3
  return combine_last_two_dimensions(
      tf.transpose(x, [0] + list(range(2, num_dimensions + 2)) +
                   [1, num_dimensions + 2]))


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = shape_list(x)
  a, b = x_shape[-2:]  # pylint: disable=unbalanced-tuple-unpacking
  return tf.reshape(x, x_shape[:-2] + [a * b])


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])


def pad_to_multiple_nd(x, block_shape):
  """Making sure x is a multiple of shape.

  Args:
    x: a [batch, d1, d2, ..., dn, depth] tensor
    block_shape: a n-d list of integers representing block shape

  Returns:
    padded x where each dimension is a multiple of corresponding block length.
  """
  shape = shape_list(x)
  paddings = [-l % b for l, b in zip(shape[1:-1], block_shape)]
  return tf.pad(x, [[0, 0]] + [[0, p] for p in paddings] + [[0, 0]])


def break_into_blocks_nd(x, block_shape):
  """Break input tensor into blocks of `block_shape`.

  Args:
    x: a [batch, d1, d2, ..., dn, depth] tensor
    block_shape: a n-d list of integers representing block shape

  Returns:
    a [batch, d1//block1, ..., dn//blockn, block1 *... * blockn, depth] tensor
  """
  x_shape = shape_list(x)
  assert all([l % b == 0 for l, b in zip(x_shape[1:], block_shape)])
  blocks_per_dimension = [l // b for l, b in zip(x_shape[1:], block_shape)]
  # reshape to [-1, d1 // block1, block1, ..., dn // blockn, blockn, depth]
  reshape_to = list(
      itertools.chain.from_iterable(zip(blocks_per_dimension, block_shape)))
  x = tf.reshape(x, [-1] + reshape_to + x_shape[-1:])
  # transpose dimensions to bring the n-d blocks in consecutive dimensions.
  block_dimensions_index = [2 * (i + 1) for i in range(len(block_shape))]
  x = tf.transpose(x, [0] + [i - 1 for i in block_dimensions_index] +
                   block_dimensions_index + [2 * len(block_shape) + 1])
  return tf.reshape(x, [-1] + blocks_per_dimension +
                    [np.prod(block_shape, dtype=np.int32)] + x_shape[-1:])


def break_into_memory_blocks_nd(x, query_shape, memory_flange, masked=False):
  """Break a tensor into memory blocks around query blocks.

  This requires memory_flange to be divisible by query_shape in every dimension.

  Args:
    x: a [batch, d1, d2, ..., dn, depth] tensor
    query_shape: a n-d list of integers representing query shape
    memory_flange: an n-d list of integers representing memory flange.
    masked: a boolean for masked vs unmasked attention.

  Returns:
    a [batch, blocks_per_d1, ..., blocks_per_dn, b1 * ...* bn, depth] where bi
      is the memory block size in dimension i which is equal to q[i] + 2m[i] or
      q[i] + m[i] if masked attention and i = 1.
  """
  assert all([m % b == 0 for b, m in zip(query_shape, memory_flange)])

  original_x_shape = shape_list(x)
  # calculate the total number of query blocks in each dimension
  blocks_in_memory_flange = [m // b for b, m in zip(query_shape, memory_flange)]
  num_query_blocks = [
      l // q for l, q in zip(original_x_shape[1:-1], query_shape)
  ]
  # pad x to have enough items on the corners to form the  memory blocks.
  if masked:
    # Only pad the beginning of first dimension in masked mode.
    x = tf.pad(x, [[0, 0], [memory_flange[0], 0]] +
               [[p, p] for p in memory_flange[1:]] + [[0, 0]])
  else:
    x = tf.pad(x, [[0, 0]] + [[p, p] for p in memory_flange] + [[0, 0]])

  query_blocks = break_into_blocks_nd(x, query_shape)
  # stitch query blocks together to form memory blocks of the desired size.
  start_indices_per_dimension = []
  for dimension, blocks in enumerate(blocks_in_memory_flange):
    if masked and dimension == 0:
      # num blocks for first dimension in masked mode is blocks + 1
      size = blocks + 1
    else:
      size = 2 * blocks + 1
    start_indices_per_dimension.append(range(size))

  slices = []
  for start_indices in itertools.product(*start_indices_per_dimension):
    start = [0] + list(start_indices) + [0, 0]
    size = [-1] + num_query_blocks + [-1, -1]
    s = tf.slice(query_blocks, start, size)
    slices.append(s)
  # concat slices in their query block dimension to form the full memory blocks
  return tf.concat(slices, axis=-2)


def select_block_for_decode_step(blocked_x, decode_step, query_shape):
  """Selects one block from `x` that contains position `decode_step`.

  NOTE: This method only works for blocked inputs. It selects one block around
  `decode_step` position in blocked raster scan order.

  Args:
    blocked_x: a [batch, blocks_per_d1, ..., blocks_per_dn, b1 * ...* bn, depth]
      tensor
    decode_step: an integer
    query_shape: a tuple (q1, q2, ..., qn) representing query shape

  Returns:
     a [batch, [1] * n, b1 * ... * bn, depth] tensor
  """
  blocked_x_shape = shape_list(blocked_x)
  # calculate the shape of the normal x
  x_shape = [b * q for b, q in zip(blocked_x_shape[1:-2], query_shape)]
  # Get the position of `decode_step` element in the unblocked x.
  index = decode_step_to_index(decode_step, query_shape, x_shape)
  # Convert it to the blocked positions.
  blocked_index = [i // q for i, q in zip(index, query_shape)]
  # TPU needs size to be non negative for the case when begin is not
  # compile-time constants.
  return tf.slice(blocked_x, [0] + blocked_index + [0, 0],
                  [blocked_x_shape[0]] + [1] * len(blocked_index) +
                  blocked_x_shape[-2:])


def flatten_blocks_nd(x):
  """Flattens blocks of the input tensor.

  Args:
    x: a [batch, b1, ..., bn, items_in_block, depth] tensor

  Returns:
    a flattened tensor of shape [batch, b1 * ...* bm, items_in_block, depth]
  """
  x_shape = shape_list(x)
  num_blocks = np.prod(x_shape[1:-2], dtype=np.int32)
  return tf.reshape(x, [-1, num_blocks] + x_shape[-2:])


def unflatten_blocks_nd(x, blocks_per_dimension):
  """Converts a flattened tensor into a normal blocked tensor.

  Args:
    x: a [batch, d1 * ... dn, items_in_block, depth] tensor
    blocks_per_dimension: a n-d list of integers for number of blocks in each
      dimension.

  Returns:
    a [batch, d1, d2, ..., dn, items_in_block, depth] tensor
  """
  x_shape = shape_list(x)
  assert x_shape[1] == np.prod(blocks_per_dimension, dtype=np.int32)
  return tf.reshape(x, [-1] + list(blocks_per_dimension) + x_shape[-2:])


def causal_attention_bias_nd(query_shape,  # pylint: disable=dangerous-default-value
                             memory_flange,
                             decode_step=None,
                             bias_cache={}):
  """Creates causal attention bias for local nd attention.

  This assumes memory_flange is divisible by query_shape in every dimension.

  Args:
    query_shape: a n-d list of integers representing query shape
    memory_flange: a n-d list of integers representing memory flange
    decode_step: an integer
    bias_cache: attention bias cache

  Returns:
    a [1, 1, query_items, memory_items] tensor for masked attention bias or
      a [1, 1, 1, memory_items] tensor if decode_step is not None.
  """
  cache_key = "causal_attention_bias_{}_{}".format(query_shape, memory_flange)
  if cache_key in bias_cache and decode_step is None:
    return bias_cache[cache_key]
  assert all([m % q == 0 for q, m in zip(query_shape, memory_flange)])
  blocks_per_memory_flange = [
      m // q for q, m in zip(query_shape, memory_flange)
  ]
  # previous blocks will be half the number of all blocks if we select blocks
  # to the left and right of center block in every dimension.
  prev_blocks = np.prod([2 * b + 1 for b in blocks_per_memory_flange],
                        dtype=np.int32) // 2
  all_blocks = np.prod(
      [blocks_per_memory_flange[0] + 1] +
      [2 * b + 1 for b in blocks_per_memory_flange[1:]],
      dtype=np.int32)
  future_blocks = all_blocks - prev_blocks - 1
  # add unmasked biases for all prev blocks and a lower triangle for the center
  # block and all masked for future blocks.
  items_in_block = np.prod(query_shape, dtype=np.int32)
  items_in_query = items_in_block if decode_step is None else 1
  prev_blocks_attn = tf.zeros(
      [1, 1, items_in_query, prev_blocks * items_in_block])

  # add mask for the center block
  if decode_step is None:
    center_block_attn = attention_bias_lower_triangle(items_in_block,
                                                      bias_cache)
  else:
    step_in_block = decode_step % items_in_block
    cond = tf.reshape(
        tf.less_equal(tf.range(items_in_block, dtype=tf.int32), step_in_block),
        [1, 1, items_in_query, items_in_block])
    center_block_attn = tf.where(
        cond, tf.zeros([1, 1, items_in_query, items_in_block]),
        -1e9 * tf.ones([1, 1, items_in_query, items_in_block]))

  # add mask for all future blocks
  future_blocks_attn = -1e9 * tf.ones(
      [1, 1, items_in_query, future_blocks * items_in_block])
  bias = tf.concat([prev_blocks_attn, center_block_attn, future_blocks_attn],
                   axis=3)
  if decode_step is None:
    bias_cache[cache_key] = bias
  return bias


def put_back_blocks_nd(x, block_shape):
  """Restructure input tensor from blocks to normal ordering.

  Args:
    x: a [batch, b1, ..., bn, items_in_block, depth] tensor
    block_shape: a n-d list of integers representing block shape.

  Returns:
    a [batch, d1, ..., dn, depth] where blocks are put back to form the
      original tensor.
  """
  x_shape = shape_list(x)
  if isinstance(x_shape[-2], int):
    assert x_shape[-2] == np.prod(block_shape)
  x = tf.reshape(x, x_shape[:-2] + list(block_shape) + x_shape[-1:])
  block_dimension_index = [i + 1 for i in range(len(block_shape))]
  block_shape_index = [b + len(block_shape) for b in block_dimension_index]
  interleaved_dimensions = list(
      itertools.chain.from_iterable(
          zip(block_dimension_index, block_shape_index)))
  x = tf.transpose(x, [0] + interleaved_dimensions + [2 * len(block_shape) + 1])
  x_shape = shape_list(x)
  x = tf.reshape(x, [-1] + [
      x_shape[2 * i + 1] * x_shape[2 * i + 2] for i in range(len(block_shape))
  ] + x_shape[-1:])
  return x


def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.

  We have hacked symbol_modality to return all-zero embeddings for padding.

  Args:
    emb: a Tensor with shape [..., depth].

  Returns:
    a float Tensor with shape [...]. Each element is 1 if its corresponding
      embedding vector is all zero, and is 0 otherwise.
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.to_float(tf.equal(emb_sum, 0.))


def attention_nd(q,  # pylint: disable=dangerous-default-value
                 k,
                 v,
                 query_shape,
                 memory_query_shape,
                 memory_flange,
                 local_num_heads,
                 local_relative=False,
                 memory_antecedent=None,
                 sparsity_cluster_size=None,
                 sparsity_cluster_attention_window=None,
                 sparsity_cluster_num_heads=0,
                 sparsity_cluster_relative=False,
                 sparsity_cluster_strided_num_heads=0,
                 sparsity_cluster_strided_relative=False,
                 sparsity_strided_num_heads=0,
                 sparsity_strided_relative=False,
                 masked=True,
                 losses=None,
                 mode=tf.estimator.ModeKeys.EVAL,
                 decode_step=None,
                 name=None,
                 max_relative_position=None,
                 cache_padding_bias=True,
                 dropout_rate=0.,
                 bias_cache={},
                 ema=False,
                 beta=1e-4,
                 decay=0.99,
                 share_qk=False,
                 hash_items=False,
                 is_recomputing=False,
                 decoding_stats=None,
                 token_bias=None,
                 token_bias_wt_trainable=False,
                 padding_bias=None,
                 use_tpu=False):
  """Attention nd.

  Args:
    q: a [batch, heads, d1, d2, ..., dn, depth_k] tensor.
    k: a [batch, heads, d1, d2, ..., dn, depth_k] tensor.
    v: a [batch, heads, d1, d2, ..., dn, depth_v] tensor.
    query_shape: a tuple (q1, q2, ..., qn) indicating the shape of query blocks.
    memory_query_shape: query shape for memory antecedent (enc-dec).
    memory_flange: a tuple (m1, m2, ..., mn) indicating the number of extra
      positions in the attention memory. memory_shape=[q1 + m1, d2 + 2 * m2,
      ..., dn + 2 * mn]
    local_num_heads: How many heads to use for local attention
    local_relative: whether to use relative positions for local heads.
    memory_antecedent: Memory antecedent for attention.
    sparsity_cluster_size: Number of clusters for routing attention.
    sparsity_cluster_attention_window: how many positions to attend to within a
      cluster.
    sparsity_cluster_num_heads: how many heads to use for attention within
      cluster.
    sparsity_cluster_relative: whether to use relative positions for clustering.
    sparsity_cluster_strided_num_heads: how many heads to use for attending to
      items outside cluster.
    sparsity_cluster_strided_relative: whether to use relative for cluster
      strided
    sparsity_strided_num_heads: how many heads to use for strided attention.
    sparsity_strided_relative: whether to use relative for strided heads.
    masked: a boolean for masked/unmasked attention.
    losses: a list of extra losses.
    mode: a tf.estimator.ModeKeys.
    decode_step: an integer in fast decoding mode.
    name: an optional string
    max_relative_position: the max distance to consider for relative positions.
    cache_padding_bias: boolean to specify whether padding bias should be cached
      and reused. This should only be set for problems that do not have variable
      length sequences like images and videos.
    dropout_rate: Rate of dropout.
    bias_cache: attention bias cache.
    ema: a boolean to do ema updates.
    beta: multiplier for clustering loss.
    decay: decay factor for learning centroids.
    share_qk: Whether to share queries and keys.
    hash_items: Whether to hash items instead of clustering.
    is_recomputing: a boolean to represent whether this is a backward pass.
    decoding_stats: a dict to be used to return tensors to capture additional
      stats in decoding mode.
    token_bias: Externally provided attention bias over memory sequence (k / v).
    token_bias_wt_trainable: Whether or not token_bias_weight is trainable.
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).
    use_tpu: Whether to use TPU (default: False).

  Returns:
    a [batch, head, d1, d2, ..., dn, depth_v] tensor or
      [batch, head, 1, 1, ..., 1, depth_v] if decode_step is not None.
  """
  assert sparsity_cluster_strided_num_heads <= sparsity_cluster_num_heads
  assert mode is not None
  assert all([m % b == 0 for m, b in zip(memory_flange, query_shape)])
  num_heads = (
      local_num_heads + sparsity_cluster_num_heads +
      sparsity_strided_num_heads + sparsity_cluster_strided_num_heads)
  with tf.variable_scope(name, default_name="attention_nd", values=[q, k, v]):

    if decode_step is not None:
      q = tf.reshape(q, [-1] + shape_list(q)[2:])
      latest_q = get_item_at_decode_step(q, decode_step, query_shape)
      q = tf.reshape(q, [-1, num_heads] + shape_list(q)[1:])
      latest_q = tf.reshape(latest_q,
                            [-1, num_heads] + shape_list(latest_q)[1:])
      q_shape = shape_list(latest_q)
    else:
      q_shape = shape_list(q)
    k_shape = shape_list(k)
    v_shape = shape_list(v)
    remainder_num_heads = num_heads

    # split heads for different kinds of attention.
    outputs = []
    if sparsity_cluster_num_heads:
      remainder_num_heads -= sparsity_cluster_num_heads
      q_cluster, q = tf.split(
          q, [sparsity_cluster_num_heads, remainder_num_heads], axis=1)
      k_cluster, k = tf.split(
          k, [sparsity_cluster_num_heads, remainder_num_heads], axis=1)
      v_cluster, v = tf.split(
          v, [sparsity_cluster_num_heads, remainder_num_heads], axis=1)
      output_cluster, cluster_loss, cluster_attn_weights = (
          clustered_local_attention_helper(
              q=q_cluster,
              k=k_cluster,
              v=v_cluster,
              query_shape=query_shape,
              memory_antecedent=memory_antecedent,
              attention_window=sparsity_cluster_attention_window,
              sparsity_cluster_size=sparsity_cluster_size,
              masked=masked,
              decode_step=decode_step,
              name="cluster_attention",
              mode=mode,
              relative_attention=sparsity_cluster_relative,
              cache_padding_bias=cache_padding_bias,
              max_relative_position=max_relative_position,
              dropout_rate=dropout_rate,
              bias_cache=bias_cache,
              ema=ema,
              beta=beta,
              decay=decay,
              share_qk=share_qk,
              hash_items=hash_items,
              is_recomputing=is_recomputing,
              token_bias=token_bias,
              token_bias_wt_trainable=token_bias_wt_trainable,
              padding_bias=padding_bias,
              use_tpu=use_tpu))
      outputs.append(output_cluster)
      if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        losses.append(cluster_loss)
    if sparsity_cluster_strided_num_heads:
      remainder_num_heads -= sparsity_cluster_strided_num_heads
      q_cluster, q = tf.split(
          q, [sparsity_cluster_strided_num_heads, remainder_num_heads], axis=1)
      k_cluster, k = tf.split(
          k, [sparsity_cluster_strided_num_heads, remainder_num_heads], axis=1)
      v_cluster, v = tf.split(
          v, [sparsity_cluster_strided_num_heads, remainder_num_heads], axis=1)
      output_cluster, cluster_loss, cluster_strided_attn_weights = (
          clustered_local_attention_helper(
              q=q_cluster,
              k=k_cluster,
              v=v_cluster,
              query_shape=query_shape,
              attention_window=sparsity_cluster_attention_window,
              sparsity_cluster_size=sparsity_cluster_size,
              strided_attention=True,
              masked=masked,
              decode_step=decode_step,
              name="cluster_strided_attention",
              mode=mode,
              relative_attention=sparsity_cluster_strided_relative,
              max_relative_position=max_relative_position,
              cache_padding_bias=cache_padding_bias,
              dropout_rate=dropout_rate,
              bias_cache=bias_cache,
              ema=ema,
              is_recomputing=is_recomputing,
              token_bias=token_bias,
              token_bias_wt_trainable=token_bias_wt_trainable,
              padding_bias=padding_bias,
              use_tpu=use_tpu))
      outputs.append(output_cluster)
      if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        losses.append(cluster_loss)
    # Rest of attention types work on latest_q instead of the whole q.
    if decode_step is not None:
      start_head = (
          sparsity_cluster_strided_num_heads + sparsity_cluster_num_heads)
      if start_head:
        _, q = tf.split(latest_q, [start_head, remainder_num_heads], axis=1)
      else:
        q = latest_q
    if sparsity_strided_num_heads:
      remainder_num_heads -= sparsity_strided_num_heads
      q_strided, q = tf.split(
          q, [sparsity_strided_num_heads, remainder_num_heads], axis=1)
      k_strided, k = tf.split(
          k, [sparsity_strided_num_heads, remainder_num_heads], axis=1)
      v_strided, v = tf.split(
          v, [sparsity_strided_num_heads, remainder_num_heads], axis=1)
      # TODO(kalpeshk): Implement token_bias here?
      output_strided = strided_local_attention_helper(
          q=q_strided,
          k=k_strided,
          v=v_strided,
          query_shape=query_shape,
          masked=masked,
          decode_step=decode_step,
          name="strided_attention",
          relative_attention=sparsity_strided_relative,
          max_relative_position=max_relative_position,
          dropout_rate=dropout_rate,
          bias_cache=bias_cache)
      outputs.append(output_strided)
    if local_num_heads:
      # move heads to batch dimension. This is needed to reduce number of
      # dimensions as much as possible, since some ops support only up to 7
      # dimensions.
      q = tf.reshape(q, [-1] + q_shape[2:])
      k = tf.reshape(k, [-1] + k_shape[2:])
      v = tf.reshape(v, [-1] + v_shape[2:])

      # Set memory query shape if using local attn for enc-dec
      mem_query_shape = query_shape
      if memory_antecedent is not None:
        mem_query_shape = memory_query_shape
      # Pad query, key, value to ensure multiple of corresponding lengths.
      if decode_step is None:
        # don't pad query in fast decoding mode. We only need to calculate self
        # attention for one position.
        q = pad_to_multiple_nd(q, query_shape)
      k = pad_to_multiple_nd(k, mem_query_shape)
      v = pad_to_multiple_nd(v, mem_query_shape)

      # extract query and memory blocks
      if decode_step is None:
        q = break_into_blocks_nd(q, query_shape)
      else:
        # in fast decoding, q has 1 block with 1 item in it
        # q shape will be [batch] + [1] * n + [1, depth] which is equivalent of
        # [batch, b1, b2, ..., bn, items_in_block, depth] where there is 1 block
        # and 1 item in that block
        q = tf.reshape(q, [-1] + [1] * (len(q_shape) - 3) + [q_shape[-1]])
      k = break_into_memory_blocks_nd(
          k, mem_query_shape, memory_flange, masked=masked)
      v = break_into_memory_blocks_nd(
          v, mem_query_shape, memory_flange, masked=masked)
      blocks_per_dim = shape_list(q)[1:-2]
      # extract just one block of k and v in fast decoding mode.
      if decode_step is not None:
        k = select_block_for_decode_step(k, decode_step, mem_query_shape)
        v = select_block_for_decode_step(v, decode_step, mem_query_shape)

      # flatten q, k and v to [batch, num_blocks, items_in_block, depth]
      q = flatten_blocks_nd(q)
      k = flatten_blocks_nd(k)
      v = flatten_blocks_nd(v)

      # make attention bias for causal or unmasked attention.
      attn_bias = local_attention_bias_nd(
          query_shape=mem_query_shape,
          memory_flange=memory_flange,
          blocked_v=v,
          masked=masked,
          cache_padding_bias=cache_padding_bias,
          decode_step=decode_step,
          bias_cache=bias_cache)

      def break_bias_into_blocks(bias):
        # [b, s]
        bias = tf.expand_dims(bias, axis=1)
        bias = tf.tile(bias, [1, local_num_heads, 1])
        bias = tf.expand_dims(bias, axis=-1)
        bias = tf.reshape(bias, [-1] + shape_list(bias)[2:])
        bias = pad_to_multiple_nd(bias, mem_query_shape)
        bias = break_into_memory_blocks_nd(bias,
                                           mem_query_shape,
                                           memory_flange,
                                           masked=masked)
        if decode_step is not None:
          bias = select_block_for_decode_step(bias, decode_step,
                                              mem_query_shape)

        bias = flatten_blocks_nd(bias)
        bias = tf.squeeze(bias, axis=-1)
        return bias

      if padding_bias is not None:
        padding_bias = break_bias_into_blocks(padding_bias)
        padding_bias = tf.expand_dims(padding_bias * -1e9, axis=-2)
        attn_bias = tf.minimum(attn_bias, padding_bias)

      if token_bias is not None:
        token_bias = break_bias_into_blocks(token_bias)
        token_bias = tf.expand_dims(token_bias, axis=-2)
        token_bias_weight = tf.get_variable(name="token_bias_weight",
                                            initializer=1.0,
                                            trainable=token_bias_wt_trainable)
        attn_bias += token_bias_weight * token_bias

      # Calculate dot product attention
      output, local_attn_weights = dot_product_attention(
          q,
          k,
          v,
          attn_bias,
          dropout_rate=dropout_rate,
          name=name or "local_nd",
          relative_attention=local_relative,
          max_relative_position=max_relative_position,
          decode_step=decode_step,
          query_shape=query_shape)
      output = unflatten_blocks_nd(output, blocks_per_dim)
      output = tf.reshape(output, [q_shape[0], local_num_heads] +
                          shape_list(output)[1:])
      outputs.append(output)

    # Concat all the different types of attention results together
    output = tf.concat(outputs, axis=1)

    # Allow heads to talk to each other
    output_shape = shape_list(output)
    output = tf.reshape(output,
                        [output_shape[0], num_heads, -1, output_shape[-1]])
    combine_heads_nd(output)
    output = tf.layers.dense(output, output_shape[-1], use_bias=False)
    output = tf.reshape(output, output_shape)

    scope_name = tf.get_variable_scope().name
    # restructure the output from blocks ordering to the original ordering
    if decode_step is None:
      # In fast decoding, output only contains one element, this is not needed.
      output = tf.reshape(output, [-1] + shape_list(output)[2:])
      output = put_back_blocks_nd(output, query_shape)
      # bring back the heads dimension
      output = tf.reshape(output, q_shape[:2] + shape_list(output)[1:])
      # No padding is introduced in fast decoding, no need to do this.
      output_shape = shape_list(output)
      output = tf.slice(output, [0] * len(output_shape),
                        [-1, -1] + q_shape[2:-1] + [-1])
      if decoding_stats is not None:
        if local_num_heads:
          decoding_stats["%s/local_local_jsd" % scope_name] = tf.constant(0.0)
        if local_num_heads and sparsity_cluster_num_heads:
          decoding_stats["%s/local_cluster_jsd" % scope_name] = tf.constant(0.0)
        if local_num_heads and sparsity_cluster_strided_num_heads:
          decoding_stats["%s/local_strided_jsd" % scope_name] = tf.constant(0.0)
        if sparsity_cluster_num_heads:
          decoding_stats["%s/cluster_cluster_jsd" %
                         scope_name] = tf.constant(0.0)
        if sparsity_cluster_num_heads and sparsity_cluster_strided_num_heads:
          decoding_stats["%s/cluster_strided_jsd" %
                         scope_name] = tf.constant(0.0)
        if sparsity_cluster_strided_num_heads:
          decoding_stats["%s/strided_strided_jsd" %
                         scope_name] = tf.constant(0.0)

    if decode_step is not None and decoding_stats is not None:
      seq_length = np.prod(k_shape[2:-1])
      if local_num_heads:
        local_attn_weights = tf.reshape(local_attn_weights,
                                        [q_shape[0], local_num_heads, -1])
        # scatter the attention weights into [batch, heads, seq_length]
        block_len = shape_list(local_attn_weights)[-1]
        batch_idx = tf.reshape(tf.range(q_shape[0]), [q_shape[0], 1, 1, 1])
        batch_idx = tf.tile(batch_idx, [1, local_num_heads, block_len, 1])
        head_idx = tf.reshape(
            tf.range(local_num_heads), [1, local_num_heads, 1, 1])
        head_idx = tf.tile(head_idx, [q_shape[0], 1, block_len, 1])
        block_num = decode_step // seq_length
        pos_idx = tf.range(block_len) + (block_num * block_len)
        pos_idx = tf.reshape(pos_idx, [1, 1, block_len, 1])
        pos_idx = tf.tile(pos_idx, [q_shape[0], local_num_heads, 1, 1])
        idx = tf.concat([batch_idx, head_idx, pos_idx], axis=-1)
        local_attn_weights = tf.scatter_nd(
            idx, local_attn_weights, [q_shape[0], local_num_heads, seq_length])
      if sparsity_cluster_num_heads:
        cluster_attn_weights = tf.reshape(
            cluster_attn_weights,
            [q_shape[0], sparsity_cluster_num_heads, seq_length])
      if sparsity_cluster_strided_num_heads:
        cluster_strided_attn_weights = tf.reshape(
            cluster_strided_attn_weights,
            [q_shape[0], sparsity_cluster_strided_num_heads, seq_length])
      if local_num_heads:
        decoding_stats["%s/local_local_jsd" %
                       scope_name] += jensen_shannon_divergence(
                           local_attn_weights[:, 0], local_attn_weights[:, 1])
      if local_num_heads and sparsity_cluster_num_heads:
        decoding_stats["%s/local_cluster_jsd" % scope_name] += (
            jensen_shannon_divergence(local_attn_weights[:, 0],
                                      cluster_attn_weights[:, 0]))
      if local_num_heads and sparsity_cluster_strided_num_heads:
        decoding_stats["%s/local_strided_jsd" % scope_name] += (
            jensen_shannon_divergence(local_attn_weights[:, 0],
                                      cluster_strided_attn_weights[:, 0]))
      if sparsity_cluster_num_heads:
        decoding_stats["%s/cluster_cluster_jsd" % scope_name] += (
            jensen_shannon_divergence(cluster_attn_weights[:, 0],
                                      cluster_attn_weights[:, 1]))
      if sparsity_cluster_num_heads and sparsity_cluster_strided_num_heads:
        decoding_stats["%s/cluster_strided_jsd" % scope_name] += (
            jensen_shannon_divergence(cluster_attn_weights[:, 0],
                                      cluster_strided_attn_weights[:, 0]))
      if sparsity_cluster_strided_num_heads:
        decoding_stats["%s/strided_strided_jsd" % scope_name] += (
            jensen_shannon_divergence(cluster_strided_attn_weights[:, 0],
                                      cluster_strided_attn_weights[:, 1]))
    return output


def jensen_shannon_divergence(a, b):
  """Calculates JSD.

  Args:
    a: a [batch, seq_length] tensor representing a density function.
    b: a [batch, seq_length] tensor representing a density functon.

  Returns:
    the average JSD over batch as an scalar tensor.
  """
  a /= tf.reduce_sum(a, axis=-1, keepdims=True)
  b /= tf.reduce_sum(b, axis=-1, keepdims=True)
  m = (a + b) / 2
  jsd = kl_divergence(a, m) / 2 + kl_divergence(b, m) / 2
  return tf.reduce_mean(jsd)


def kl_divergence(a, b):
  eps = 1e-5
  return tf.reduce_sum(-a * tf.log(b / (a + eps) + eps), axis=-1)


def local_attention_bias_nd(query_shape,  # pylint: disable=dangerous-default-value
                            memory_flange,
                            blocked_v,
                            masked=False,
                            cache_padding_bias=True,
                            decode_step=None,
                            bias_cache={}):
  """create an attention bias for local n-d attention.

  This function creates/picks from cache an attention bias for local n-d
  attention type.

  Args:
    query_shape: a (q1, ..., qn) tuple
    memory_flange: a (m1, ..., mn) tuple
    blocked_v: a [batch, num_blocks, items_in_blocks, depth] tensor for v.
    masked: Whether to create masked/unmasked bias.
    cache_padding_bias: If sequences are not variable length (e.g. images and
      videos) and the only source of padding is to be evenly divisible by blocks
      we can cache padding bias as well to save memory.
    decode_step: the decode step in fast decoding mode or None.
    bias_cache: attention bias cache.

  Returns:
    the local attention bias tensor of shape
      [batch * heads, num_blocks, items_in_query, items_in_memory] or
      [1, num_blocks, items_in_query, items_in_meory] if cache padding bias is
      true.
  """
  cache_key = "local_attention_bias_{}_{}_{}_{}".format(query_shape,
                                                        memory_flange, masked,
                                                        cache_padding_bias)
  # do not use cache attention bias in fast decoding mode since each mask is
  # slightly different depending on decode step.
  if cache_key in bias_cache and decode_step is None:
    return bias_cache[cache_key]

  if cache_padding_bias:
    padding_attn_bias = tf.expand_dims(
        embedding_to_padding(blocked_v[:1, :, :, :]) * -1e9, axis=-2)
  else:
    padding_attn_bias = tf.expand_dims(
        embedding_to_padding(blocked_v) * -1e9, axis=-2)

  if masked:
    causal_attn_bias = causal_attention_bias_nd(
        query_shape,
        memory_flange,
        decode_step=decode_step,
        bias_cache=bias_cache)
    causal_attn_bias, padding_attn_bias = maybe_tile(causal_attn_bias,
                                                     padding_attn_bias)
    attn_bias = tf.minimum(causal_attn_bias, padding_attn_bias)
  else:
    attn_bias = padding_attn_bias

  if cache_padding_bias and decode_step is None:
    bias_cache[cache_key] = attn_bias

  return attn_bias


def maybe_tile(x, y):
  """Tile two tensors so they have the same shape except for batch and depth."""
  x_shape = shape_list(x)
  y_shape = shape_list(y)
  assert len(x_shape) == len(y_shape)
  x_tile = []
  y_tile = []
  for x_dim, y_dim in zip(x_shape[1:-1], y_shape[1:-1]):
    assert x_dim % y_dim == 0 or y_dim % x_dim == 0
    if x_dim == y_dim:
      x_tile.append(1)
      y_tile.append(1)
    elif x_dim > y_dim:
      x_tile.append(1)
      y_tile.append(x_dim // y_dim)
    else:
      x_tile.append(y_dim // x_dim)
      y_tile.append(1)

  return tf.tile(x, [1] + x_tile + [1]), tf.tile(y, [1] + y_tile + [1])


def strided_local_attention_helper(q,  # pylint: disable=dangerous-default-value
                                   k,
                                   v,
                                   query_shape,
                                   masked,
                                   decode_step,
                                   name,
                                   relative_attention=False,
                                   max_relative_position=None,
                                   dropout_rate=0.,
                                   bias_cache={}):
  """Strided local attention helper.

  Args:
    q: a [batch, heads, d1, ..., dn, depth_k] tensor or [batch, heads, 1, ...,
      1, depth_k] tensor in fast decoding mode.
    k: a [batch, heads, d1, ..., dn, depth_k] tensor.
    v: a [batch, heads, d1, ..., dn, depth_v] tensor.
    query_shape: a tuple of (q1, ..., qn) representing the query shape.
    masked: a boolean for masked/un masked attention.
    decode_step: decode step in fast decoding mode.
    name: variable scope name.
    relative_attention: whether to do relative attention.
    max_relative_position: the max distance to consider for relative positions.
    dropout_rate: Rate of dropout.
    bias_cache: attention bias cache.

  Returns:
    a [batch, heads, d1//q1, ..., dn//qn, items_in_block, depth_v] tensor where
    each positions attends to previous positions that are in the same relative
    position within their own query blocks. or [batch, heads, 1, ..., 1,
    depth_v] for fast decoding.
  """
  # This computation only applies to self attention, so assert q, k, v and
  # antecedent have the same dimensions.
  if decode_step is None:
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape()[:-1].assert_is_compatible_with(v.get_shape()[:-1])
  else:
    k.get_shape().assert_is_compatible_with(v.get_shape())
  with tf.variable_scope(
      name, default_name="strided_attention_nd", values=[q, k, v]):
    q_shape = shape_list(q)

    # rearrange q, k and v to blocked order and flatten them so
    # shape becomes [batch * heads, blocks, items_in_block, depth].
    k = tf.reshape(k, [-1] + shape_list(k)[2:])
    k = pad_to_multiple_nd(k, query_shape)
    k = break_into_blocks_nd(k, query_shape)
    k = flatten_blocks_nd(k)

    v = tf.reshape(v, [-1] + shape_list(v)[2:])
    v = pad_to_multiple_nd(v, query_shape)
    v = break_into_blocks_nd(v, query_shape)
    v = flatten_blocks_nd(v)

    # in fast decoding mode q will be [batch * heads, 1, depth]
    if decode_step is not None:
      q = tf.reshape(q, [-1, 1, q_shape[-1]])
    else:
      q = tf.reshape(q, [-1] + shape_list(q)[2:])
      q = pad_to_multiple_nd(q, query_shape)
      q = break_into_blocks_nd(q, query_shape)
      blocked_q_shape = shape_list(q)
      q = flatten_blocks_nd(q)

    # select the correct strides from k and v.
    if decode_step is not None:
      items_in_block = shape_list(k)[2]
      offset = decode_step % items_in_block
      block_num = decode_step // items_in_block
      # TPU needs size to be non negative for the case when begin is not
      # compile-time constants.
      k_shape = shape_list(k)
      k = tf.slice(k, [0, 0, offset, 0], k_shape[:2] + [1] + k_shape[-1:])
      v = tf.slice(v, [0, 0, offset, 0], k_shape[:2] + [1] + k_shape[-1:])
      k = tf.reshape(k, [shape_list(k)[0]] + [-1] + [shape_list(k)[-1]])
      v = tf.reshape(v, [shape_list(v)[0]] + [-1] + [shape_list(v)[-1]])
      cond = tf.less_equal(tf.range(shape_list(k)[1]), block_num)
      causal_attn_bias = tf.where(cond, tf.zeros_like(cond, dtype=tf.float32),
                                  tf.ones_like(cond, dtype=tf.float32) * -1e9)
      causal_attn_bias = tf.reshape(causal_attn_bias, [1, -1])
      padding_attn_bias = embedding_to_padding(v[0, :, :]) * -1e9
      if masked:
        attn_bias = tf.minimum(causal_attn_bias, padding_attn_bias)
      else:
        attn_bias = padding_attn_bias
    else:
      q = tf.transpose(q, [0, 2, 1, 3])
      k = tf.transpose(k, [0, 2, 1, 3])
      v = tf.transpose(v, [0, 2, 1, 3])
      causal_attn_bias = attention_bias_lower_triangle(
          shape_list(q)[2], bias_cache=bias_cache)
      padding_attn_bias = tf.expand_dims(
          embedding_to_padding(v[:1, :1, :, :]) * -1e9, axis=-1)
      causal_attn_bias, padding_attn_bias = maybe_tile(causal_attn_bias,
                                                       padding_attn_bias)
      if masked:
        attn_bias = tf.minimum(causal_attn_bias, padding_attn_bias)
      else:
        attn_bias = padding_attn_bias

    # [batch * heads, num_items_in_block, num_blocks, depth] or
    # [batch * heads, 1, depth] in fast decoding.
    output, _ = dot_product_attention(
        q,
        k,
        v,
        attn_bias,
        dropout_rate=dropout_rate,
        name=name or "strided_dot_product",
        relative_attention=relative_attention,
        max_relative_position=max_relative_position,
        decode_step=decode_step,
        query_shape=query_shape)

    if decode_step is None:
      output = tf.transpose(output, [0, 2, 1, 3])
      output = tf.reshape(output, [-1, q_shape[1]] + blocked_q_shape[1:])
    else:
      output = tf.reshape(output, q_shape)

    return output


def clustered_local_attention_helper(q,  # pylint: disable=dangerous-default-value
                                     k,
                                     v,
                                     query_shape,
                                     attention_window,
                                     sparsity_cluster_size,
                                     masked,
                                     decode_step,
                                     name,
                                     mode,
                                     memory_antecedent=None,
                                     strided_attention=False,
                                     relative_attention=False,
                                     max_relative_position=None,
                                     cache_padding_bias=False,
                                     dropout_rate=0.,
                                     bias_cache={},
                                     ema=False,
                                     beta=1e-4,
                                     decay=0.99,
                                     share_qk=False,
                                     hash_items=False,
                                     is_recomputing=False,
                                     skip_summaries=True,
                                     token_bias=None,
                                     token_bias_wt_trainable=False,
                                     padding_bias=None,
                                     use_tpu=False):
  """clustered local attention helper.

  Args:
    q: a [batch, heads, d1, ..., dn, depth_k] tensor.
    k: a [batch, heads, d1, ..., dn, depth_k] tensor.
    v: a [batch, heads, d1, ..., dn, depth_v] tensor.
    query_shape: a tuple of (q1, ..., qn) representing query shape.
    attention_window: how many positions to attend to within a cluster.
    sparsity_cluster_size: Number of clusters for routing attention.
    masked: a boolean for masked/unmasked attention.
    decode_step: decode step in fast decoding mode.
    name: variable scope name.
    mode: tf.estimator.ModeKeys.
    memory_antecedent: Memory antecedent for self attention.
    strided_attention: Whether to do strided attention in the cluster space.
    relative_attention: Whether to do relative attention.
    max_relative_position: the max distance to consider for relative positions.
    cache_padding_bias: If sequences are not variable length (e.g. images and
      videos) and the only source of padding is to be evenly divisible by blocks
      we can cache padding bias as well to save memory.
    dropout_rate: Rate of dropout.
    bias_cache: attention bias cache.
    ema: a boolean to do ema updates.
    beta: multiplier for clustering loss.
    decay: decay factor for learning centroids.
    share_qk: Whether to share queries and keys.
    hash_items: If True then use Locality Sensitive Hashing.
    is_recomputing: a boolean to represent whether this is a backward pass.
    skip_summaries: a boolean to represent whether to skip `tf.summary` ops.
    token_bias: Externally provided attention bias over memory sequence (k / v).
    token_bias_wt_trainable: Whether or not token_bias_weight is trainable.
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).
    use_tpu: Whether to use TPU (default: False).

  Returns:
    output: a [batch, heads, d1//q1, ..., dn//qn, items_in_block, depth_v]
      tensor with clustered attention. or [batch, heads, 1, ..., 1, depth_v]
      for fast decoding.
    loss: a scalar tensor of clustering loss.
    attention_weights: a [batch, heads, d1//q1, ..., dn//qn] tensor representing
      the attention weights for query item at `decode_step` or None if not in
      fast decoding mode.
  """
  # This computation only applies to self attention, so assert q, k, v and
  # antecedent have the same dimensions.
  if memory_antecedent is None:
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape()[:-1].assert_is_compatible_with(v.get_shape()[:-1])
  if share_qk:
    k = q
  with tf.variable_scope(
      name, default_name="clustered_attention_nd", values=[q, k, v]):
    q_shape = shape_list(q)
    v_shape = shape_list(v)
    num_heads = q_shape[1]
    batch = q_shape[0]

    # rearrange q, k and v  to blocked order and flatten them so
    # shape becomes [batch, heads, seq_length, depth].
    k = tf.reshape(k, [-1] + shape_list(k)[2:])
    k = pad_to_multiple_nd(k, query_shape)
    seq_length = np.prod(shape_list(k)[1:-1], dtype=np.int32)
    k = break_into_blocks_nd(k, query_shape)
    k = tf.reshape(k, [-1, num_heads, seq_length, shape_list(k)[-1]])

    v = tf.reshape(v, [-1] + shape_list(v)[2:])
    v = pad_to_multiple_nd(v, query_shape)
    v = break_into_blocks_nd(v, query_shape)
    v = tf.reshape(v, [-1, num_heads, seq_length, shape_list(v)[-1]])

    q = tf.reshape(q, [-1] + shape_list(q)[2:])
    q = pad_to_multiple_nd(q, query_shape)
    q = break_into_blocks_nd(q, query_shape)
    blocked_q_shape = [batch, num_heads] + shape_list(q)[1:]
    seq_q_length = np.prod(shape_list(q)[1:-1], dtype=np.int32)
    q = tf.reshape(q, [-1, num_heads, seq_q_length, q_shape[-1]])

    # Make sure keys and queries are normalized
    q = layer_norm(q, scaling=False)
    k = layer_norm(k, scaling=False)

    # Route information using queries and keys
    if hash_items:
      q_idx = hash_items_fn(q, sparsity_cluster_size,
                            shape_list(q)[-1],
                            decode_step, name)
      if share_qk:
        k_idx = q_idx
      else:
        k_idx = hash_items_fn(k, sparsity_cluster_size,
                              shape_list(k)[-1],
                              decode_step, name)
      clustering_loss = tf.constant(0.)
    else:
      # Keys and queries come from different sequences
      # Encoder-decoder attention, blank queries only
      q_cluster_dists, q_clustering_loss = cluster_items(
          q,
          sparsity_cluster_size,
          shape_list(q)[-1],
          mode,
          decode_step,
          name,
          ema,
          beta,
          decay,
          is_recomputing,
          skip_summaries,
          blank_future=masked,
          use_tpu=use_tpu)
      if share_qk:
        k_cluster_dists, k_clustering_loss = q_cluster_dists, q_clustering_loss
      else:
        k_cluster_dists, k_clustering_loss = cluster_items(
            k,
            sparsity_cluster_size,
            shape_list(k)[-1],
            mode,
            decode_step,
            name,
            ema,
            beta,
            decay,
            is_recomputing,
            skip_summaries,
            blank_future=masked,
            use_tpu=use_tpu)
      clustering_loss = q_clustering_loss + k_clustering_loss
    if decode_step is None and not hash_items and not skip_summaries:
      # Add a summary for cluster loss
      tf.summary.scalar("cluster_loss", clustering_loss)

    # gather cluster items.
    if hash_items:
      q, _, _, q_idx, _, _ = gather_hashed_attention_items(
          q=q,
          k=q,
          v=q,
          sparsity_cluster_size=sparsity_cluster_size,
          attention_window=attention_window,
          idx=q_idx,
          token_bias=token_bias,
          padding_bias=padding_bias if memory_antecedent is not None else None)
      _, k, v, k_idx, padding_bias, token_bias = gather_hashed_attention_items(
          q=k,
          k=k,
          v=v,
          sparsity_cluster_size=sparsity_cluster_size,
          attention_window=attention_window,
          idx=k_idx,
          token_bias=token_bias,
          padding_bias=padding_bias)
    else:
      q, _, _, q_idx, _, _ = gather_cluster_attention_items(
          q=q,
          k=q,
          v=q,
          attention_window=attention_window,
          cluster_dists=q_cluster_dists,
          strided_attention=strided_attention,
          token_bias=token_bias,
          padding_bias=padding_bias if memory_antecedent is not None else None)
      _, k, v, k_idx, padding_bias, token_bias = gather_cluster_attention_items(
          q=k,
          k=k,
          v=v,
          attention_window=attention_window,
          cluster_dists=k_cluster_dists,
          strided_attention=strided_attention,
          token_bias=token_bias,
          padding_bias=padding_bias)
    attn_bias = clustered_attention_bias_nd(
        attention_window=attention_window,
        clustered_v=v,
        masked=masked,
        cache_padding_bias=cache_padding_bias,
        bias_cache=bias_cache)

    if padding_bias is not None:
      padding_bias = tf.expand_dims(padding_bias * -1e9, axis=-2)
      attn_bias = tf.minimum(attn_bias, padding_bias)

    if token_bias is not None:
      token_bias = tf.expand_dims(token_bias, axis=-2)
      token_bias_weight = tf.get_variable(name="token_bias_weight",
                                          initializer=1.0,
                                          trainable=token_bias_wt_trainable)

      attn_bias += token_bias_weight * token_bias

    if relative_attention:
      q_shape = shape_list(q)
      k_shape = shape_list(k)
      v_shape = shape_list(v)
      q = tf.reshape(q, [q_shape[0], q_shape[1] * q_shape[2]] + q_shape[3:])
      k = tf.reshape(k, [k_shape[0], k_shape[1] * k_shape[2]] + k_shape[3:])
      v = tf.reshape(v, [v_shape[0], v_shape[1] * v_shape[2]] + v_shape[3:])
      bias_shape = shape_list(attn_bias)
      new_bias_shape = [bias_shape[0], bias_shape[1] * bias_shape[2]
                       ] + bias_shape[3:]
      attn_bias = tf.reshape(attn_bias, new_bias_shape)

    output, weights = dot_product_attention(
        q,
        k,
        v,
        attn_bias,
        dropout_rate=dropout_rate,
        name=name or "clustered_dot_product",
        relative_attention=relative_attention,
        max_relative_position=max_relative_position,
        decode_step=decode_step,
        query_shape=query_shape)
    if relative_attention:
      output = tf.reshape(output, q_shape[:-1] + [-1])
      weights = tf.reshape(weights, q_shape[:-1] + [-1])
    # scatter the results back into blocked raster scan order.
    output = scatter_cluster_items(output, q_idx, seq_q_length)

    if decode_step is not None:
      output = tf.slice(
          output, [0, 0, decode_step, 0],
          [batch, num_heads, 1, shape_list(output)[-1]])
      output = tf.reshape(output, [batch, num_heads] + [1] * len(query_shape) +
                          v_shape[-1:])
      # [batch, heads, num_clusters, attention_window, 1]
      weights = tf.transpose(weights, [0, 1, 2, 4, 3])
      # scatter the results to obtain [batch, heads, b1, ..., bn]
      weights = scatter_cluster_items(weights, q_idx, seq_length)
      weights = tf.slice(weights, [0, 0, 0, decode_step],
                         [batch, num_heads, seq_length, 1])
    else:
      output = tf.reshape(output, blocked_q_shape[:-1] + v_shape[-1:])

  return output, clustering_loss, weights if decode_step is not None else None


def clustered_attention_bias_nd(attention_window,  # pylint: disable=dangerous-default-value
                                clustered_v,
                                masked=True,
                                cache_padding_bias=False,
                                bias_cache={}):
  """create a cluster attention bias nd.

  Args:
    attention_window: an integer for the attention window.
    clustered_v: a [batch, heads, num_clusters, attention_window, depth] tensor.
    masked: a boolean for masked/un masked attention
    cache_padding_bias: If sequences are not variable length (e.g. images and
      videos) and the only source of padding is to be evenly divisible by blocks
      we can cache padding bias as well to save memory.
    bias_cache: attention bias cache.

  Returns:
    cluster attention bias of shape
      [batch, heads, num_clusters, attention_window, attention_window] or
      [1, heads, num_clusters, attention_window, attention_window] if cache
      padding bias is true.
  """
  cache_key = "clustered_attention_bias_{}_{}_{}".format(
      attention_window, masked, cache_padding_bias)
  if cache_key in bias_cache:
    return bias_cache[cache_key]
  if cache_padding_bias:
    padding_attn_bias = tf.expand_dims(
        embedding_to_padding(clustered_v[:1, :, :, :, :]) * -1e9, axis=-2)
  else:
    padding_attn_bias = tf.expand_dims(
        embedding_to_padding(clustered_v) * -1e9, axis=-2)
  if masked:
    causal_attn_bias = tf.expand_dims(
        attention_bias_lower_triangle(
            attention_window, bias_cache=bias_cache),
        axis=0)
    causal_attn_bias, padding_attn_bias = maybe_tile(causal_attn_bias,
                                                     padding_attn_bias)
    attn_bias = tf.minimum(causal_attn_bias, padding_attn_bias)
  else:
    attn_bias = padding_attn_bias

  if cache_padding_bias:
    bias_cache[cache_key] = attn_bias

  return attn_bias


def _generate_relative_positions_matrix(length_q,
                                        length_k,
                                        max_relative_position,
                                        query_shape,
                                        decode_step=None):
  """Generates matrix of relative positions between inputs."""
  if decode_step is None:
    if length_q == length_k:
      range_vec_q = range_vec_k = tf.range(length_q)
    else:
      range_vec_k = tf.range(length_k)
      range_vec_q = range_vec_k[-length_q:]
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
  else:
    block_len = np.prod(query_shape)
    positive_positions = block_len - decode_step % block_len
    distance_mat = tf.expand_dims(tf.range(-length_k, 0, 1),
                                  0) + positive_positions
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  # Shift values to be >= 0. Each integer still uniquely identifies a relative
  # position difference.
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat


def _generate_relative_positions_embeddings(length_q,
                                            length_k,
                                            depth,
                                            max_relative_position,
                                            name,
                                            query_shape,
                                            decode_step=None):
  """Generates tensor of size [1 if decode else length_q, length_k, depth]."""
  with tf.variable_scope(name):
    relative_positions_matrix = _generate_relative_positions_matrix(
        length_q, length_k, max_relative_position, query_shape, decode_step)
    vocab_size = max_relative_position * 2 + 1
    # Generates embedding for each relative position of dimension depth.
    embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)
    return embeddings


def _relative_attention_inner(x, y, z, transpose):
  """Relative position-aware dot-product attention inner calculation.

  This batches matrix multiply calculations to avoid unnecessary broadcasting.

  Args:
    x: Tensor with shape [batch_size, heads, length or 1, length or depth].
    y: Tensor with shape [batch_size, heads, length or 1, depth].
    z: Tensor with shape [length or 1, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
      last dimension of x is depth, not length.

  Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
  """
  # xy_matmul is [batch_size, heads, length or 1, length or depth]
  if transpose:
    xy_matmul = tf.einsum("bhxd,bhyd->bhxy", x, y)
    x_tz_matmul_r_t = tf.einsum("bhxd,xyd->bhxy", x, z)
  else:
    xy_matmul = tf.einsum("bhxd,bhdy->bhxy", x, y)
    x_tz_matmul_r_t = tf.einsum("bhxd,xdy->bhxy", x, z)
  return xy_matmul + x_tz_matmul_r_t


def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   query_shape,
                                   dropout_rate=0.0,
                                   name=None,
                                   decode_step=None):
  """Calculate relative position-aware dot-product self-attention.

  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.

  Args:
    q: a Tensor with shape [batch, heads, length, depth] or [batch, heads, 1,
      depth] in fast decoding mode.
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maximum distance between
      inputs that unique position embeddings should be learned for.
    query_shape: a tuple to represent the query shape.
    dropout_rate: a floating point number.
    name: an optional string.
    decode_step: the decode step in fast decoding mode.

  Returns:
    A Tensor of shape [batch, heads, length, depth].
    A Tensor fof shape [batch, heads, length, length] for attention weights.

  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))
  with tf.variable_scope(
      name, default_name="dot_product_attention_relative", values=[q, k, v]):

    # Use separate embeddings suitable for keys and values.
    depth = k.get_shape().as_list()[3]
    length_k = shape_list(k)[2]
    length_q = shape_list(q)[2]
    relations_keys = _generate_relative_positions_embeddings(
        length_q, length_k, depth, max_relative_position,
        "relative_positions_keys", query_shape, decode_step)
    relations_values = _generate_relative_positions_embeddings(
        length_q, length_k, depth, max_relative_position,
        "relative_positions_values", query_shape, decode_step)

    # Compute self attention considering the relative position embeddings.
    logits = _relative_attention_inner(q, k, relations_keys, True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if dropout_rate:
      weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return _relative_attention_inner(weights, v, relations_values,
                                     False), weights


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          query_shape,
                          dropout_rate=0.0,
                          name=None,
                          relative_attention=False,
                          max_relative_position=None,
                          decode_step=None):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must match
      with q.
    bias: bias Tensor (see attention_bias())
    query_shape: a tuple to represent the query shape.
    dropout_rate: a float.
    name: an optional string
    relative_attention: whether to do relative attention.
    max_relative_position: if relative attention is enabled, how much distance
      to use for relative positions.
    decode_step: the decode step in fast decoding mode.

  Returns:
    Tensor with shape [..., length_q, depth_v].
    Tensor with shape [..., length_q, length_kv] representing attention weights.
  """
  if relative_attention:
    assert max_relative_position
    return dot_product_attention_relative(
        q=q,
        k=k,
        v=v,
        bias=bias,
        max_relative_position=max_relative_position,
        dropout_rate=dropout_rate,
        name=name,
        decode_step=decode_step,
        query_shape=query_shape)
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = cast_like(bias, logits)
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # Drop out attention links for each head.
    if dropout_rate:
      weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    return tf.matmul(weights, v), weights


def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x


def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  This does not actually save any computation.

  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length, bias_cache={}):  # pylint: disable=dangerous-default-value
  """Create an bias tensor to be added to attention logits.

  Allows a query to attend to all positions up to and including its own.

  Args:
   length: a Scalar.
   bias_cache: attention bias cache.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  cache_key = "attention_bias_lower_triangle_{}".format(length)
  if cache_key in bias_cache:
    return bias_cache[cache_key]
  bias = attention_bias_local(length, -1, 0)
  bias_cache[cache_key] = bias
  return bias


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.matrix_band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.

  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x. The probability that
      each element is kept.
    broadcast_dims: an optional list of integers the dimensions along which to
      broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".

  Returns:
    Tensor of the same shape as x.
  """
  assert "noise_shape" not in kwargs
  if math.isclose(keep_prob, 1):
    return x
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    # Allow dimensions like "-1" as well.
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)
    ]
  return tf.nn.dropout(x, keep_prob, **kwargs)


def scatter_cluster_items(clustered_x, idx, seq_length):
  """Scatters items from clusters into their original positions.

  Args:
    clustered_x: a [batch, heads, num_clusters, attention_window, depth] tensor
      or [batch, heads, num_clusters, 1, depth] in fast decoding mode.
    idx: a [batch, heads, num_clusters, attention_window, 3] int tensor in which
      items in the last dimension are [batch_index, head_index, seq_len_index]
    seq_length: the sequence length.

  Returns:
    a [batch, heads, seq_length, depth] tensor where items in `clustered_x` are
      scattered to their positions or [batch, heads, 1, depth] in fast decoding.
  """
  x_shape = shape_list(clustered_x)
  batch = x_shape[0]
  heads = x_shape[1]
  res = tf.scatter_nd(
      idx, clustered_x, [batch, heads, seq_length, x_shape[-1]]) / (
          tf.scatter_nd(idx, tf.ones_like(clustered_x),
                        [batch, heads, seq_length, x_shape[-1]]) + 1e-2)
  return res


def gather_hashed_attention_items(q,
                                  k,
                                  v,
                                  idx,
                                  sparsity_cluster_size,
                                  attention_window,
                                  token_bias=None,
                                  padding_bias=None):
  """Gathers items that should attend to each other based on input hashing.

  Args:
    q: a [batch, heads, seq_length, depth_k] tensor or [batch, heads, 1,
      depth_k] in fast decoding mode.
    k: a [batch, heads, seq_length, depth_k] tensor.
    v: a [batch, heads, seq_length, depth_v] tensor.
    idx: Hash bucket ids.
    sparsity_cluster_size: Number of clusters for hashed attention.
    attention_window: How many positions to attend to in each cluster.
    token_bias: Externally provided attention bias over memory sequence (k / v).
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).

  Returns:
    q: a [batch, heads, num_clusters, attention_window, depth_k] or
      [batch, heads, num_clusters, 1, depth_k].
    k: a [batch, heads, num_clusters, attention_window, depth_k].
    v: a [batch, heads, num_clusters, attention_window, depth_v].
    idx: a [batch, heads, num_clusters, attention_window, 3] int tensor in
      which items in the last dimension is
      [batch_index, head_index, seq_length_index]. This is used for scattering
      the results back after dot product attention.
    padding_bias: Padding bias gathered according to indices.
    token_bias: token_bias gathered according to indices.
  """
  batch = shape_list(q)[0]
  heads = shape_list(q)[1]
  # [batch, num_heads, num_clusters, seq_length]
  idx = tf.one_hot(idx, depth=sparsity_cluster_size, axis=-1)
  idx = tf.transpose(idx, [0, 1, 3, 2])
  _, idx = tf.math.top_k(idx, k=attention_window)
  # idx = [batch, num_heads, seq_length, 1] (signifying idx)
  # ids correspond to decoding order, sort them to prevent peeking into future.
  idx = tf.sort(idx, axis=-1)
  idx = tf.expand_dims(idx, axis=-1)
  # to prepare gather indices we need to add batch index to idx.
  batch_idx = tf.reshape(tf.range(0, batch), [batch, 1, 1, 1, 1])
  # [batch, heads, num_clusters, attention_window, 1]
  batch_idx = tf.tile(batch_idx, [1, heads, sparsity_cluster_size,
                                  attention_window, 1])
  # we also need to add head index to idx.
  head_idx = tf.reshape(tf.range(0, heads), [1, heads, 1, 1, 1])
  head_idx = tf.tile(head_idx, [batch, 1, sparsity_cluster_size,
                                attention_window, 1])
  # [batch, heads, num_clusters, attention_window, 3]
  idx = tf.concat([batch_idx, head_idx, idx], axis=-1)
  k, v = tf.split(tf.gather_nd(tf.concat([k, v], -1), idx), 2, -1)

  def gather_idx_for_bias(bias):
    # Padding bias is of shape [batch, seq_length]
    bias = tf.expand_dims(bias, axis=1)
    bias = tf.tile(bias, [1, heads, 1])
    bias = tf.gather_nd(bias, idx)
    return bias

  if padding_bias is not None:
    padding_bias = gather_idx_for_bias(padding_bias)

  if token_bias is not None:
    token_bias = gather_idx_for_bias(token_bias)

  q = tf.gather_nd(q, idx)
  return q, k, v, idx, padding_bias, token_bias


def gather_cluster_attention_items(q,
                                   k,
                                   v,
                                   cluster_dists,
                                   attention_window,
                                   strided_attention=False,
                                   token_bias=None,
                                   padding_bias=None):
  """Gathers items that should attend to each other based on input clustering.

  Args:
    q: a [batch, heads, seq_length, depth_k] tensor or [batch, heads, 1,
      depth_k] in fast decoding mode.
    k: a [batch, heads, seq_length, depth_k] tensor.
    v: a [batch, heads, seq_length, depth_v] tensor.
    cluster_dists: a [batch, num_heads, seq_length, num_clusters] tensor
      representing the distance of each item from all clusters.
    attention_window: How many positions to attend to in each cluster.
    strided_attention: Whether to do strided attention in the cluster space.
    token_bias: Externally provided attention bias over memory sequence (k / v).
    padding_bias: Padding bias for seq2seq models (Shape: [b, s]).

  Returns:
    q: a [batch, heads, num_clusters, attention_window, depth_k] or
      [batch, heads, num_clusters, 1, depth_k].
    k: a [batch, heads, num_clusters, attention_window, depth_k].
    v: a [batch, heads, num_clusters, attention_window, depth_v].
    idx: a [batch, heads, num_clusters, attention_window, 3] int tensor in
      which items in the last dimension is
      [batch_index, head_index, seq_length_index]. This is used for scattering
      the results back after dot product attention.
    padding_bias: Padding bias gathered according to indices.
    token_bias: token_bias gathered according to indices.
  """
  shape = shape_list(cluster_dists)
  num_clusters = shape[-1]
  batch = shape_list(q)[0]
  heads = shape_list(q)[1]
  # [batch, num_heads, num_clusters, seq_length]
  cluster_dists = tf.transpose(cluster_dists, [0, 1, 3, 2])
  if strided_attention:
    # Simulate attending to the centroids by strided attention.
    seq_len = shape_list(cluster_dists)[-1]
    cluster_idx = tf.argsort(cluster_dists, axis=-1)
    stride = seq_len // attention_window
    idx = cluster_idx[:, :, :, ::stride]
    # we may need to trim down idx.
    if (seq_len % attention_window) != 0:
      idx = idx[:, :, :, :attention_window]
  else:
    _, idx = tf.math.top_k(-cluster_dists, k=attention_window)
  # ids correspond to decoding order, sort them to prevent peeking into future.
  idx = tf.sort(idx, axis=-1)
  idx = tf.expand_dims(idx, axis=-1)
  # to prepare gather indices we need to add batch index to idx.
  batch_idx = tf.reshape(tf.range(0, batch), [batch, 1, 1, 1, 1])
  # [batch, heads, num_clusters, attention_window, 1]
  batch_idx = tf.tile(batch_idx, [1, heads, num_clusters, attention_window, 1])
  # we also need to add head index to idx.
  head_idx = tf.reshape(tf.range(0, heads), [1, heads, 1, 1, 1])
  head_idx = tf.tile(head_idx, [batch, 1, num_clusters, attention_window, 1])
  # [batch, heads, num_clusters, attention_window, 3]
  idx = tf.concat([batch_idx, head_idx, idx], axis=-1)
  k, v = tf.split(tf.gather_nd(tf.concat([k, v], -1), idx), 2, -1)

  def gather_idx_for_bias(bias):
    # bias is of shape [batch, seq_length]
    bias = tf.expand_dims(bias, axis=1)
    bias = tf.tile(bias, [1, heads, 1])
    bias = tf.gather_nd(bias, idx)
    return bias

  if padding_bias is not None:
    padding_bias = gather_idx_for_bias(padding_bias)

  if token_bias is not None:
    token_bias = gather_idx_for_bias(token_bias)

  q = tf.gather_nd(q, idx)
  return q, k, v, idx, padding_bias, token_bias


def hash_items_fn(items, sparsity_cluster_size, codebook_depth,
                  decode_step, name):
  """Hash input items via random projections (LSH).

  Args:
    items: a [batch, heads, seq_length, depth] tensor
    sparsity_cluster_size: Number of clusters for LSH attention.
    codebook_depth: depth of the codebook entries.
    decode_step: decode step or None.
    name: variable scope name.

  Returns:
    idx: Membership index of each sequence item in hash bucket.
  """
  del decode_step
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    num_heads = shape_list(items)[1]
    num_bits = int(tf.log(sparsity_cluster_size) / tf.log(tf.constant(2)))
    projection_tensors = tf.get_variable(
        name="projection_tensors",
        shape=[num_heads, num_bits, codebook_depth],
        trainable=False,
        initializer=tf.initializers.orthogonal())
    # items have shape [bs, nh, seq_len, d]
    # projection_tensors have shape [nh, k, d]
    # inner product between the two has shape [bs, nh, seq_len, d]
    inner_product = tf.einsum("bnsd, nkd->bnsk", items, projection_tensors)
    signed_inner_product = tf.sign(inner_product)
    # So every sequence element gets a sign corresponding to which bucket it is
    binary_inner_product = (signed_inner_product + 1)//2
    idx = bit_to_int(binary_inner_product, num_bits=num_bits)
  return idx


def bit_to_int(x_bit, num_bits, base=2):
  """Turn x_bit representing numbers bitwise (lower-endian) to int tensor.

  Args:
    x_bit: Tensor containing numbers in a particular base to be converted to
      int.
    num_bits: Number of bits in the representation.
    base: Base of the representation.

  Returns:
    Integer representation of this number.
  """
  x_l = tf.to_int64(tf.reshape(x_bit, [-1, num_bits]))
  x_labels = [
      x_l[:, i] * tf.to_int64(base)**tf.to_int64(i) for i in range(num_bits)]
  res = sum(x_labels)
  return tf.to_int64(tf.reshape(res, x_bit.get_shape().as_list()[:-1]))


def cluster_items(items, sparsity_cluster_size, codebook_depth, mode,
                  decode_step, name, ema, beta, decay, is_recomputing,
                  skip_summaries, blank_future=False, use_tpu=False):
  """Cluster input items via a discrete bottleneck.

  Args:
    items: a [batch, heads, seq_length, depth] tensor
    sparsity_cluster_size: Number of clusters for routing attention.
    codebook_depth: depth of the codebook entries.
    mode: a tf.estimator.ModeKeys.
    decode_step: decode step or None.
    name: variable scope name.
    ema: a boolean to do ema updates or not.
    beta: multiplier for clustering loss.
    decay: decay factor for learning centroids.
    is_recomputing: a boolean to represent whether this is a backward pass.
    skip_summaries: a boolean to represent whether to skip `tf.summary` ops.
    blank_future: Whether to set future blank positions to infinity.
    use_tpu: Whether to use TPU (default: False).

  Returns:
    cluster_dist: a [batch, heads, seq_length, num_clusters] float tensor
      representing distance from all clusters.
    loss: Scalar Tensor. Sum of codebook and commitment losses
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    num_heads = shape_list(items)[1]
    seq_length = shape_list(items)[2]
    means = tf.get_variable(
        name="means",
        shape=[num_heads, sparsity_cluster_size, codebook_depth],
        trainable=True,
        dtype=tf.float32,
        initializer=lambda shape, dtype=None, partition_info=None,  # pylint: disable=g-long-lambda
                           verify_shape=None: layer_norm(
                               tf.random.normal(shape=shape,
                                                mean=0.0,
                                                stddev=1.0,
                                                dtype=dtype), scaling=False))
    ema_count, ema_means = None, None
    if ema:
      ema_count = tf.get_variable(
          name="ema_count",
          shape=[num_heads, sparsity_cluster_size],
          trainable=False,
          initializer=tf.constant_initializer(0))
      with tf.colocate_with(means):
        # In export mode, means becomes a Tensor that does not have
        # initialized_value defined.
        ema_means = tf.get_variable(
            name="ema_means",
            shape=None if isinstance(means, tf.Variable) else
            [num_heads, sparsity_cluster_size, codebook_depth],
            trainable=False,
            initializer=means.initialized_value() if isinstance(
                means, tf.Variable) else tf.constant_initializer(0))
    dist, loss = online_kmeans(
        inputs=items,
        sparsity_cluster_size=sparsity_cluster_size,
        beta=beta,
        means=means,
        ema=ema,
        decay=decay,
        ema_count=ema_count,
        ema_means=ema_means,
        mode=mode,
        is_recomputing=is_recomputing,
        skip_summaries=skip_summaries,
        use_tpu=use_tpu)
    # In decoding mode, set distances for blank positions to infinity.
    if decode_step is not None and blank_future:
      batch_size = shape_list(dist)[0]
      idx = tf.tile(
          tf.reshape(tf.range(seq_length), [1, 1, -1, 1]),
          [batch_size, num_heads, 1, shape_list(dist)[-1]])
      dist = tf.where(idx <= decode_step, dist, tf.ones_like(dist) * 1e9)
  return dist, loss


def online_kmeans(inputs,
                  sparsity_cluster_size,
                  mode=None,
                  beta=0.25,
                  ema=True,
                  means=None,
                  ema_count=None,
                  ema_means=None,
                  epsilon=1e-5,
                  decay=0.999,
                  is_recomputing=False,
                  skip_summaries=True,
                  use_tpu=False):
  """Clustering via online k-means.

  Args:
    inputs: Input to the bottleneck, a Tensor of shape [..., hidden_dim].
    sparsity_cluster_size: Number of clusters for routing attention.
    mode: tf.estimator.ModeKeys.
    beta: Scale factor for online k-means.
    ema: Whether to update embeddings using exponential moving averages.
    means: The embedding table. Used only if ema is True.
    ema_count: Table of counts for each embedding corresponding to how many
      examples in a batch it was the closest to. Used only if ema is True.
    ema_means: Exponentially averaged version of the embeddings. Used only if
      ema is True.
    epsilon: Small value to avoid dividing by zero in EMA update. Used only if
      ema is True.
    decay: Decay factor for the exponential moving average. Used only if ema is
      True.
    is_recomputing: a boolean to represent whether this is a backward pass.
    skip_summaries: a boolean to represent whether to skip `tf.summary` ops.
    use_tpu: Whether to use TPU (default: False).

  Returns:
    x_dist: Distance to the centroids for online k-means.
    extra_loss: Loss for training online k-means.
  """
  with tf.variable_scope("clustering", reuse=tf.AUTO_REUSE):
    # inputs [bs, n, s, h], means [n, k, h]
    input_shape = shape_list(inputs)
    num_heads = input_shape[1]
    x = inputs
    x_means_hot, x_dist, q_loss, e_loss = embedding_lookup(x, means=means)
    # Exclude pads from affecting centroids
    x_means_hot_pad_mask = 1 - embedding_to_padding(x)
    x_means_hot_pad_mask = tf.expand_dims(x_means_hot_pad_mask, axis=-1)
    x_means_hot = tf.multiply(x_means_hot, x_means_hot_pad_mask)
    extra_loss = 0
    # Update the EMA variables.
    if ema and mode == tf.estimator.ModeKeys.TRAIN and not is_recomputing:
      tf.logging.info("Using EMA with beta = {}".format(beta))
      # [bs, n, s, k], [n, k]
      count = tf.reduce_sum(
          tf.reshape(
              x_means_hot,
              shape=[-1, num_heads, sparsity_cluster_size]),
          axis=0)
      if use_tpu:
        count = tf.tpu.cross_replica_sum(count)
      updated_ema_count = moving_averages.assign_moving_average(
          ema_count,
          count,
          decay,
          zero_debias=False)
      # [bs, n, s, k], [bs, n, s, h]
      dw = tf.einsum("bnsk, bnsh -> nkh", x_means_hot, x)
      if use_tpu:
        dw = tf.tpu.cross_replica_sum(dw)
      updated_ema_means = moving_averages.assign_moving_average(
          ema_means, dw, decay, zero_debias=False)
      n = tf.reduce_sum(updated_ema_count, axis=-1, keep_dims=True)
      updated_ema_count = ((updated_ema_count + epsilon) /
                           (n + sparsity_cluster_size * epsilon) * n)
      updated_ema_means = updated_ema_means / tf.expand_dims(
          updated_ema_count, axis=-1)

      with tf.control_dependencies([e_loss]):
        update_means = tf.assign(means, updated_ema_means)
        with tf.control_dependencies([update_means]):
          extra_loss += beta * e_loss
    elif ema:
      extra_loss += beta * e_loss
    else:
      extra_loss += q_loss + beta * e_loss

    # Adjust shape of dist
    dist_shape = input_shape[:-1] + [sparsity_cluster_size]
    x_dist = tf.reshape(x_dist, dist_shape)

    # Add a tf summary for average cluster occupancy
    if mode != tf.estimator.ModeKeys.PREDICT and not skip_summaries:
      cluster_occupancy = tf.reduce_mean(
          tf.reduce_sum(x_means_hot, axis=2), axis=[0, 1])
      tf.summary.histogram("cluster_occupancy", cluster_occupancy)
      cluster_occupancy_min = tf.reduce_min(cluster_occupancy)
      cluster_occupancy_max = tf.reduce_max(cluster_occupancy)
      tf.summary.scalar("cluster_occupancy_min", cluster_occupancy_min)
      tf.summary.scalar("cluster_occupancy_max", cluster_occupancy_max)
  return x_dist, extra_loss


def embedding_lookup(x, means):
  """Compute nearest neighbors and loss for training the embeddings.

  Args:
    x: Continuous encodings of shape [batch_size, sequence_length, hidden_dim].
    means: Embedding table of shape [sparsity_cluster_size, hidden_dim].

  Returns:
    x_means_hot: The nearest neighbor in one hot form, with shape
      [batch_size, sequence_length, sparsity_cluster_size].
    x_dist: Distances to the centroids of shape [batch_size,
      sequence_length, sparsity_cluster_size].
    q_loss: Scalar Tensor representing codebook loss.
    e_loss: Scalar Tensor representing commitment loss.
  """
  x_means_hot, x_dist = nearest_neighbor(x, means)
  x_means = tf.einsum("bnsk, nkh -> bnsh", x_means_hot, means)
  q_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(x), x_means))
  e_loss = tf.reduce_mean(tf.squared_difference(x, tf.stop_gradient(x_means)))
  return x_means_hot, x_dist, q_loss, e_loss


def nearest_neighbor(x, means):
  """Find the nearest element in means to elements in x.

  Args:
    x: Continuous encodings of shape [batch_size, sequence_length, hidden_dim].
    means: Embedding table of shape [sparsity_cluster_size, hidden_dim].

  Returns:
    Tensor with nearest element in mean encoded in one-hot notation
    and distances.
  """
  sparsity_cluster_size = tf.shape(means)[1]
  scalar_prod = tf.einsum("bnsh, nkh -> bnsk", x, means)
  dist = - scalar_prod
  # computing cluster probabilities
  nearest_idx = tf.argmax(-dist, axis=-1)
  nearest_hot = tf.one_hot(nearest_idx, sparsity_cluster_size)
  dist = tf.reshape(dist, shape_list(nearest_hot))
  return nearest_hot, dist


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
      different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal
