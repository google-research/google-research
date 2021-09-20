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

"""T5 model components."""

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import attention
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import tensorflow.compat.v1 as tf


@gin.configurable
def causal_depthwise_conv(x, context, kernel_size=3):
  """Causal depthwise convolution."""

  def scale_var(shift_distance):
    return mtf.get_variable(
        context.mesh,
        "conv_%s" % shift_distance,
        mtf.Shape(context.model.ensemble_dims + x.shape.dims[-1:]),
        initializer=tf.constant_initializer(0.5 if shift_distance ==
                                            0 else 0.5 / kernel_size),
        dtype=context.variable_dtype)

  ret = x * scale_var(0)
  for shift_distance in range(1, kernel_size):
    x = mtf.shift(x, 1, context.length_dim, wrap=False)
    ret += x * scale_var(shift_distance)
  return ret


def primer_norm(x, dim, epsilon=1e-6, name="layer_prepostprocess"):
  """Primer normalization over dimension `dim`.

  Args:
    x: a mtf.Tensor whose shape contains `dim`.
    dim: a mtf.Dimension.
    epsilon: a floating point number.
    name: a string used for tf.variable_scope.

  Returns:
    a mtf.Tensor with same shape as x.
  """
  with tf.variable_scope(name + "/primer_norm"):
    scale = mtf.get_variable(
        x.mesh,
        "primer_norm_scale",
        mtf.Shape([dim]),
        initializer=tf.ones_initializer(),
        activation_dtype=x.dtype)
    bias = mtf.get_variable(
        x.mesh,
        "primer_norm_bias",
        mtf.Shape([dim]),
        initializer=tf.zeros_initializer(),
        activation_dtype=x.dtype)
    reduced_shape = x.shape - dim
    mean = mtf.reduce_mean(x, output_shape=reduced_shape)
    mean_centered_x = x - mean
    pseudo_variance = mtf.reduce_mean(
        x * mean_centered_x, output_shape=reduced_shape)
    norm_x = mean_centered_x * mtf.rsqrt(pseudo_variance + epsilon)
    return norm_x * scale + bias


@gin.configurable
def sublayer_prime_norm(x,
                        layer_stack,
                        context,
                        epsilon=1e-6,
                        name="primer_norm"):
  """Sublayer wrapper around Primer norm.

  Args:
    x: an input mtf.Tensor.
    layer_stack: a LayerStack.
    context: a Context.
    epsilon: a float.
    name: a string.

  Returns:
    a mtf.Tensor.
  """
  del layer_stack
  model_dim = context.model.model_dim
  with tf.variable_scope(name):
    return primer_norm(x, model_dim, epsilon)


@gin.configurable
class DoubleHeadsAttentionLayer(transformer.TransformerLayer):
  """Attention with twice as many heads for Evolved Transformer."""

  def __init__(self, base_num_heads, key_value_size, dropout_rate):
    self._self_attention = transformer_layers.SelfAttention(
        num_heads=int(2 * base_num_heads),
        key_value_size=int(key_value_size / 2),
        dropout_rate=dropout_rate)

  def call(self, context, x, losses=None):
    """Call the layer."""
    with tf.variable_scope("double_heads_attention"):
      return self._self_attention.call(context, x, losses)


class MDHAParams(attention.AttentionParams):
  """Multi-DConv-Head Attention parameters."""

  def __init__(self,
               create_k_weights,
               **kwargs):
    self._create_k_weights = create_k_weights
    super().__init__(**kwargs)

  def init_weights(self):
    """Initialize projection matrices."""
    if mtf.layers.unit_scaling_convention():
      init = tf.random_normal_initializer(stddev=1.0)
      q_init = init
      kv_init = init
      o_init = init
    else:
      stddev = self.query_input_dim.size ** -0.5
      if self.fold_scaling_into_initializer:
        stddev *= self.key_dim.size ** -0.5
      q_init = tf.random_normal_initializer(stddev=stddev)
      kv_init = tf.random_normal_initializer(
          stddev=self.memory_input_dim.size ** -0.5)
      o_init = tf.random_normal_initializer(
          stddev=mtf.Shape(self.query_heads_dims + [self.value_dim]).size**-0.5)

    if self.make_attention_vars:
      if not self.no_query:
        self.wq = mtf.get_variable(
            self.mesh,
            "q",
            self.q_shape,
            initializer=q_init,
            dtype=self.variable_dtype)
      if self.shared_kv:
        self.wkv = mtf.get_variable(
            self.mesh,
            "kv",
            self.k_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
      else:
        if self._create_k_weights:
          self.wk = mtf.get_variable(
              self.mesh,
              "k",
              self.k_shape,
              initializer=kv_init,
              dtype=self.variable_dtype)
        self.wv = mtf.get_variable(
            self.mesh,
            "v",
            self.v_shape,
            initializer=kv_init,
            dtype=self.variable_dtype)
    self.wo = mtf.get_variable(
        self.mesh,
        "o",
        self.o_shape,
        initializer=o_init,
        dtype=self.variable_dtype)

  def mdha_q(self, query_antecedent, context):
    """MDHA Q projection."""
    ret = mtf.layers.us_einsum([query_antecedent, self.wq],
                               reduced_dims=[self.query_input_dim])
    with tf.variable_scope("q_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.q_dims)
    if not self.fold_scaling_into_initializer:
      ret *= self.key_dim.size**-0.5
    return ret

  def mdha_k(self, memory_antecedent, context):
    """MDHA K projection."""
    ret = mtf.layers.us_einsum([memory_antecedent, self.wk],
                               reduced_dims=[self.memory_input_dim])
    with tf.variable_scope("k_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.k_dims)
    return ret

  def mdha_v(self, memory_antecedent, context):
    """MDHA V projection."""
    ret = mtf.layers.us_einsum([memory_antecedent, self.wv],
                               reduced_dims=[self.memory_input_dim])
    with tf.variable_scope("v_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim
    if self.combine_dims:
      ret = mtf.replace_dimensions(ret, ret.shape.dims[-1], self.v_dims)
    return ret

  def mdha_shared_qk(self, query_antecedent, context):
    """MDHA QK shared projection."""
    ret = mtf.layers.us_einsum([query_antecedent, self.wq],
                               reduced_dims=[self.query_input_dim])
    with tf.variable_scope("qk_dconv"):
      len_dim = context.length_dim
      context.length_dim = ret.shape.dims[-2]
      ret = causal_depthwise_conv(ret, context=context, kernel_size=3)
      context.length_dim = len_dim

    q = mtf.layers.dense(
        ret,
        ret.shape.dims[-1:],
        use_bias=False,
        activation=None,
        variable_dtype=context.variable_dtype,
        reduced_dims=ret.shape.dims[-1:],
        name="q_solo_project",
        expert_dims=context.model.ensemble_dims)

    k = ret

    if self.combine_dims:
      q = mtf.replace_dimensions(q, q.shape.dims[-1], self.q_dims)
      k = mtf.replace_dimensions(k, k.shape.dims[-1], self.k_dims)
    if not self.fold_scaling_into_initializer:
      q *= self.key_dim.size**-0.5

    return q, k


def mdha_params(context,
                kv_dim,
                num_heads,
                num_memory_heads=0,
                shared_kv=False,
                no_query=False,
                combine_dims=True,
                keep_query_heads_dims=False,
                fold_scaling_into_initializer=True,
                create_k_weights=True):
  """Multi-DConv Head Attention parameters."""
  if num_heads == 1:
    query_heads_dims = None
    memory_heads_dims = None
  elif num_memory_heads == 0:
    query_heads_dims = [mtf.Dimension("heads", num_heads)]
    memory_heads_dims = query_heads_dims
  elif num_memory_heads == 1:
    query_heads_dims = [mtf.Dimension("heads", num_heads)]
    memory_heads_dims = None
  else:
    if num_heads % num_memory_heads != 0:
      raise ValueError("num_memory_heads must divide num_heads")
    memory_heads_dims = [mtf.Dimension("heads", num_memory_heads)]
    query_heads_dims = memory_heads_dims + [
        mtf.Dimension("query_heads", num_heads // num_memory_heads)]
  return MDHAParams(
      create_k_weights=create_k_weights,
      mesh=context.mesh,
      query_input_dim=context.model.model_dim,
      memory_input_dim=context.model.model_dim,
      output_dim=context.model.model_dim,
      key_dim=kv_dim,
      value_dim=kv_dim,
      query_heads_dims=query_heads_dims,
      memory_heads_dims=memory_heads_dims,
      variable_dtype=context.variable_dtype,
      shared_kv=shared_kv,
      no_query=no_query,
      ensemble_dim=context.model.ensemble_dim,
      combine_dims=combine_dims,
      keep_query_heads_dims=keep_query_heads_dims,
      fold_scaling_into_initializer=fold_scaling_into_initializer)


@gin.configurable
class PrePostNormLayerStack(transformer.LayerStack):
  """Alternating pre and post normalization."""

  def call(self, context, x):
    """Call the layer stack."""
    x = self._call_sublayers(self._sublayers_initial, x, context, 0)
    context.layer_outputs.append(x)
    for lnum, layer in enumerate(self._layers):
      with tf.variable_scope(layer.name or ""):
        if self._recompute_grads:

          def fn(x, l=layer, c=context, lnum_arg=lnum):
            return self._layer_fn(x, l, c, lnum_arg)

          x = mtf.recompute_grad(fn, [x])
        else:
          x = self._layer_fn(x, layer, context, lnum)
      if lnum != len(self._layers) - 1:
        context.layer_outputs.append(x)
      context.layer_index += 1
    x = self._call_sublayers(self._sublayers_final, x, context, 0)
    x = transformer.sublayer_mask_padding(x, self, context)
    context.layer_outputs.append(x)
    return x

  # Pre and post norm.
  def _call_sublayers(self, sublayers, x, context, lnum):
    if lnum % 2 == 0:
      for s in sublayers:
        x = s(x, self, context)
    else:
      for s in [1, 2, 0, 3, 4]:
        x = sublayers[s](x, self, context)
    return x

  def _layer_fn(self, x, layer, context, lnum):
    context.current_layer = layer
    context.current_layer_input = x
    y = self._call_sublayers(self._sublayers_per_layer, x, context, lnum)
    if y.shape != x.shape:
      raise ValueError("Layer %s returned misshaped output x=%s y=%s" %
                       (layer.__class__.__name__, x, y))
    return y


@gin.configurable
class MDHA(transformer_layers.SelfAttention):
  """Multi-DConv-Head Attention."""

  def __init__(self,
               num_heads=8,
               num_memory_heads=0,
               key_value_size=128,
               shared_kv=False,
               dropout_rate=0.0,
               attention_kwargs=None,
               relative_attention_type=None,
               relative_attention_num_buckets=32,
               attention_func=None,
               combine_dims=True,
               keep_query_heads_dims=False,
               fold_scaling_into_initializer=True,
               z_loss_coeff=None,
               share_qk_rep=False):
    super().__init__(
        num_heads=num_heads,
        num_memory_heads=num_memory_heads,
        key_value_size=key_value_size,
        shared_kv=shared_kv,
        dropout_rate=dropout_rate,
        attention_kwargs=attention_kwargs,
        relative_attention_type=relative_attention_type,
        relative_attention_num_buckets=relative_attention_num_buckets,
        attention_func=attention_func,
        combine_dims=combine_dims,
        keep_query_heads_dims=keep_query_heads_dims,
        fold_scaling_into_initializer=fold_scaling_into_initializer,
        z_loss_coeff=z_loss_coeff)
    self.share_qk_rep = share_qk_rep

  def make_params(self, context):
    return mdha_params(
        context=context,
        kv_dim=self.kv_dim,
        num_heads=self.num_heads,
        num_memory_heads=self.num_memory_heads,
        shared_kv=self.shared_kv,
        combine_dims=self.combine_dims,
        keep_query_heads_dims=self.keep_query_heads_dims,
        fold_scaling_into_initializer=self.fold_scaling_into_initializer,
        create_k_weights=not self.share_qk_rep)

  @gin.configurable
  def call(self, context, x, losses=None):
    """Call the layer."""
    params = self.make_params(context)
    if self.share_qk_rep:
      q, k = params.mdha_shared_qk(x, context)
    else:
      q = params.mdha_q(x, context)
    memory_length = self.memory_length(context)
    if context.mode == "incremental":
      m = x
    else:
      if self.share_qk_rep:
        k = mtf.replace_dimensions(k, context.length_dim, memory_length)
      m = mtf.replace_dimensions(x, context.length_dim, memory_length)
    if self.shared_kv:
      kv = params.compute_kv(m)
    else:
      if not self.share_qk_rep:
        k = params.mdha_k(m, context)
      v = params.mdha_v(m, context)
    if context.mode == "incremental":
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      if self.shared_kv:
        old_kv = context.get_states(1)
        kv = old_kv * inv_one_hot + kv * one_hot
      else:
        old_k, old_v = context.get_states(2)
        k = old_k * inv_one_hot + k * one_hot
        v = old_v * inv_one_hot + v * one_hot
      memory_position = mtf.range(context.mesh, memory_length, tf.int32)
    else:
      memory_position = self.rename_length_to_memory_length(
          context.position, context)
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([kv] if self.shared_kv else [k, v])
    if self.shared_kv:
      k = kv
      v = kv
    o = self.attention_fn(
        q,
        k,
        v,
        context=context,
        memory_length_dim=memory_length,
        key_dim=self.kv_dim,
        value_dim=self.kv_dim,
        bias=self.compute_bias(context, memory_position, x,
                               params.query_heads_dims, q),
        **self.attention_kwargs_from_context(context))
    attention_output_shape = self.expected_attention_output_shape(x, params)
    attention_output = params.compute_output(
        o, output_shape=attention_output_shape)
    return self.layer_output_from_attention_output(context, attention_output,
                                                   losses)
