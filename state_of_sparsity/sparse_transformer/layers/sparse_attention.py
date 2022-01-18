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

"""Sparse attention for the transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

import tensorflow.compat.v1 as tf

from state_of_sparsity.sparse_transformer.layers import common_sparse
from tensorflow.contrib.model_pruning.python import pruning  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import


def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0,
                                sparsity_technique=None,
                                threshold=3.0,
                                training=True,
                                clip_alpha=None,
                                initial_sparsity=None,
                                split_heads=False,
                                num_heads=None):
  """Computes attention compoenent (query, key or value).

  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)
    sparsity_technique: technique used for sparsifying weights.
    threshold: log alpha threshold used for evaluation with variational dropout.
    training: whether model is being trained or not.
    clip_alpha: alpha clipping threshold for variational dropout.
    initial_sparsity: initial sparsity level for lottery ticket &
      scratch experiments.
    split_heads: Whether to prune each head separately.
    num_heads: The number of heads in the attention module.

  Returns:
    c : [batch, length, depth] tensor
  """
  # We don't support 3d attention variables or filter_width > 1 with sparsity
  # techniques
  assert not sparsity_technique or (not vars_3d_num_heads and filter_width == 1)

  if vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth ** -0.5
    if "q" in name:
      initializer_stddev *= depth_per_head ** -0.5
    var = tf.get_variable(
        name, [input_depth,
               vars_3d_num_heads,
               total_depth // vars_3d_num_heads],
        initializer=tf.random_normal_initializer(stddev=initializer_stddev))
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    if sparsity_technique:
      if split_heads:
        # Prune each heads weights separately so that they are free
        # to have different weight magnitude distributions.
        if num_heads is None:
          raise ValueError("`num_heads` must be set for split head pruning.")
        if total_depth % num_heads != 0:
          raise ValueError("`total_depth` must be divisible by `num_heads`.")
        input_depth = antecedent.get_shape().as_list()[-1]
        depth_per_head = int(total_depth / num_heads)
        masked_head_weights = []
        for head_id in range(num_heads):
          head_name = name + "_shard_{}".format(head_id)
          with tf.variable_scope(head_name) as vs:
            head_weights = tf.get_variable(
                "kernel", [input_depth, depth_per_head])
            masked_head_weights.append(pruning.apply_mask(head_weights, vs))
        component_weights = tf.concat(masked_head_weights, axis=1)

        # compute the full component result
        return tf.tensordot(antecedent, component_weights, axes=1)
      else:
        return common_sparse.dense(
            antecedent,
            total_depth,
            use_bias=False,
            sparsity_technique=sparsity_technique,
            threshold=threshold,
            training=training,
            clip_alpha=clip_alpha,
            name=name,
            initial_sparsity=initial_sparsity)
    else:
      return common_layers.dense(
          antecedent, total_depth, use_bias=False, name=name)
  else:
    return common_layers.conv1d(
        antecedent, total_depth, filter_width, padding=padding, name=name)


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0,
                sparsity_technique=None,
                threshold=3.0,
                training=True,
                clip_alpha=None,
                initial_sparsity=None,
                split_heads=False,
                num_heads=None):
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
    vars_3d_num_heads: an optional (if we want to use 3d variables)
    sparsity_technique: technique used for sparsifying weights.
    threshold: log alpha threshold used for evaluation with variational dropout.
    training: whether model is being trained or not.
    clip_alpha: alpha clipping threshold for variational dropout.
    initial_sparsity: initial sparsity level for lottery ticket &
      scratch experiments.
    split_heads: Whether to prune each head separately.
    num_heads: The number of heads in the attention module.

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
      vars_3d_num_heads=vars_3d_num_heads,
      sparsity_technique=sparsity_technique,
      threshold=threshold,
      training=training,
      clip_alpha=clip_alpha,
      initial_sparsity=initial_sparsity,
      split_heads=split_heads,
      num_heads=num_heads)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads,
      sparsity_technique=sparsity_technique,
      threshold=threshold,
      training=training,
      clip_alpha=clip_alpha,
      initial_sparsity=initial_sparsity,
      split_heads=split_heads,
      num_heads=num_heads)
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads,
      sparsity_technique=sparsity_technique,
      threshold=threshold,
      training=training,
      clip_alpha=clip_alpha,
      initial_sparsity=initial_sparsity,
      split_heads=split_heads,
      num_heads=num_heads)
  return q, k, v


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        image_shapes=None,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        sparsity_technique=None,
                        threshold=3.0,
                        training=True,
                        clip_alpha=None,
                        initial_sparsity=None,
                        split_heads=False,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    sparsity_technique: technique used for sparsifying weights.
    threshold: log alpha threshold used for evaluation with variational dropout.
    training: whether model is being trained or not.
    clip_alpha: alpha clipping threshold for variational dropout.
    initial_sparsity: initial sparsity level for lottery ticket &
      scratch experiments.
    split_heads: Whether to prune each head separately.
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  if vars_3d:
    raise ValueError("3d attention variables not supported.")
  if attention_type != "dot_product":
    raise ValueError(
        "Sparse multihead attention only supports dot_product attention.")

  vars_3d_num_heads = 0
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):

    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            sparsity_technique=sparsity_technique,
                            threshold=threshold,
                            training=training,
                            clip_alpha=clip_alpha,
                            initial_sparsity=initial_sparsity,
                            split_heads=split_heads,
                            num_heads=num_heads)
    if cache is not None:
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q",
                                        vars_3d_num_heads=vars_3d_num_heads,
                                        sparsity_technique=sparsity_technique,
                                        threshold=threshold,
                                        training=training,
                                        clip_alpha=clip_alpha,
                                        initial_sparsity=initial_sparsity,
                                        split_heads=split_heads,
                                        num_heads=num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    q = common_attention.split_heads(q, num_heads)
    if cache is None:
      k = common_attention.split_heads(k, num_heads)
      v = common_attention.split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    # compute the attention
    x = common_attention.dot_product_attention(
        q, k, v, bias, dropout_rate, image_shapes,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=dropout_broadcast_dims)
    x = common_attention.combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if sparsity_technique:
      x = common_sparse.dense(
          x,
          output_depth,
          use_bias=False,
          sparsity_technique=sparsity_technique,
          threshold=threshold,
          training=training,
          clip_alpha=clip_alpha,
          name="output_transform",
          initial_sparsity=initial_sparsity)
    else:
      x = common_layers.dense(
          x,
          output_depth,
          use_bias=False,
          name="output_transform")
    return x
