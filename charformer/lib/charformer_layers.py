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

"""Charformer layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import custom_attention
import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
from mesh_tensorflow.transformer.transformer import sublayer_legacy_dropout
from mesh_tensorflow.transformer.transformer import sublayer_legacy_final_rms_norm
from mesh_tensorflow.transformer.transformer import sublayer_legacy_rms_norm
from mesh_tensorflow.transformer.transformer import sublayer_mask_padding
from mesh_tensorflow.transformer.transformer import sublayer_residual
from mesh_tensorflow.transformer.transformer import sublayer_rms_norm
import tensorflow.compat.v1 as tf


@gin.configurable
class GradientSubwordLayerV2(transformer_layers.SelfAttention):
  """Gradient Subword layer for Charformers."""

  def __init__(self,
               radius=128,
               num_heads=8,
               num_memory_heads=0,
               key_value_size=128,
               shared_kv=False,
               dropout_rate=0.0,
               attention_kwargs=None,
               downsample_query=2,
               low_rank_features=32,
               project_kv=True,
               use_ffn=True,
               num_memory_slots=0,
               structured=False,
               pre_attention=False,
               local_gate=False,
               norm=False,
               pos_att=False,
               conv_type=None,
               query_func="linear",
               pool_func="max",
               local_attention=False,
               use_offsets=False,
               consider_chars_as_blocks=False,
               use_block_pos_embedding=False,
               canine_mode=False,
               filter_size=5,
               block_mixing_mode=None,
               rank_activation="softmax",
               gbst_pool="mean"
               ):
    super(GradientSubwordLayerV2, self).__init__(
        num_heads,
        num_memory_heads,
        key_value_size,
        shared_kv,
        dropout_rate,
        attention_kwargs)
    self.radius = radius
    self.downsample_query = downsample_query
    self.low_rank_features = low_rank_features
    self.project_kv = project_kv
    self.use_ffn = use_ffn
    if self.use_ffn:
      self.ffn = transformer_layers.DenseReluDense()
    self.num_memory_slots = num_memory_slots
    self.structured = structured
    self.pre_attention = pre_attention
    self.local_gate = local_gate
    self.norm = norm
    self.pos_att = pos_att
    self.conv_type = conv_type
    self.query_func = query_func
    self.pool_func = pool_func
    self.local_attention = local_attention
    self.use_offsets = use_offsets
    self.consider_chars_as_blocks = consider_chars_as_blocks
    self.use_block_pos_embedding = use_block_pos_embedding
    self.canine_mode = canine_mode
    self.filter_size = filter_size
    self.block_mixing_mode = block_mixing_mode
    self.rank_activation = rank_activation
    self.gbst_pool = gbst_pool

  def call_canine_encoder(self, context, x, losses=None):
    """Call Canine baseline encoder (Byte level T5 + LASC in paper)."""
    # local attention
    params = self.make_params(context)
    q = params.compute_q(x)
    if self.shared_kv:
      kv = params.compute_kv(x)
      k = kv
      v = kv
    else:
      k = params.compute_k(x)
      v = params.compute_v(x)
    # local attention
    output_shape = x.shape
    x = custom_attention.local_attention_1d(
        q,
        k,
        v,
        length_dim=context.length_dim,
        length_dim_num_splits=1,
        key_dim=self.kv_dim,
        value_dim=self.kv_dim,
        fully_autoregressive=False,
        radius=self.radius,
        sequence_id=context.sequence_id,
        write_priority=context.write_priority,
        read_priority=context.read_priority,
        context=context,
        attention_kwargs=self.attention_kwargs_from_context(context)
        )
    o = params.compute_output(x, output_shape=output_shape)
    # strided convolutions
    tmp_output = mtf.Dimension("tmp_dim", o.shape[-1].size)
    # downsample query args is reused here for "r"
    o = mtf.layers.conv1d(o, tmp_output, filter_size=self.filter_size,
                          stride=int(self.downsample_query))
    o = mtf.rename_dimension(o, "tmp_dim", "d_model")
    tf.logging.info(o)
    new_length_dim = o.shape.get_dim_by_name("length")
    context.length_dim = new_length_dim
    new_context_position = context.get_position()
    context.position = new_context_position
    context.sequence_id = mtf.slice(
        context.sequence_id,
        begin=0,
        size=new_length_dim.size,
        slice_dim_name=new_length_dim.name)
    return o, context

  def call(self, context, x, losses=None):
    """Call the layer."""

    if self.canine_mode:
      # This is the canine-like ByT5 + LASC baseline in paper.
      return self.call_canine_encoder(context, x, losses=losses)

    if self.conv_type:
      if self.conv_type == "conv1d":
        tf.logging.info("Using 1d conv")
        tmp_output = mtf.Dimension("tmp_dim", x.shape[-1].size)
        orig_dim = x.shape[-1]
        x = mtf.layers.conv1d(
            x, tmp_output, filter_size=self.filter_size, stride=1)
        x = mtf.rename_dimension(x, "tmp_dim", orig_dim.name)
        tf.logging.info(x)
    if self.norm:
      x = sublayer_rms_norm(x, None, context)
    o = x
    olength = o.shape.get_dim_by_name("length")
    o = custom_attention.gradient_based_subword_tokenization(
        o,
        olength,
        downsample=self.downsample_query,
        use_offsets=self.use_offsets,
        consider_chars_as_blocks=self.consider_chars_as_blocks,
        use_block_pos_embedding=self.use_block_pos_embedding,
        memory_embeddings=self.num_memory_slots,
        context=context,
        block_mixing_mode=self.block_mixing_mode,
        activation=self.rank_activation,
        downsample_function=self.gbst_pool)
    new_length_dim = o.shape.get_dim_by_name("length")
    context.length_dim = new_length_dim
    new_context_position = context.get_position()
    context.position = new_context_position
    context.sequence_id = mtf.slice(
        context.sequence_id,
        begin=0,
        size=new_length_dim.size,
        slice_dim_name=new_length_dim.name)
    if self.use_ffn:
      # not actually used in Charformer.
      tf.logging.info("Using FFN")
      o2 = self.ffn.call(context, o)
      o = o + o2
    if self.norm:
      o = sublayer_rms_norm(o, None, context)
    olength = o.shape.get_dim_by_name("length")
    return o, context

  def min_relative_position(self, context):
    return 1 - self.radius

  def max_relative_position(self, context):
    return None if context.model.fully_autoregressive else self.radius

  @property
  def window_dim(self):
    return mtf.Dimension("window", self.radius)


@gin.configurable
class CharformerLayerStack(transformer.TransformerLayer):
  """Custom layer stack for Charformer."""

  def __init__(self,
               layers,
               sublayers_initial=None,
               sublayers_per_layer=None,
               sublayers_final=None,
               dropout_rate=None,
               norm_epsilon=None,
               recompute_grads=False):
    """Create a Charformer LayerStack.

    This is an exact replica of Transformer Layerstack. However,
    the key difference now is that we update the context after
    calling this layerstack. This is not the best way to do things now,
    but having a standalone Charformer layerstack is bound to be handy
    for future extensions.

    `layers` is a list of TransformerLayer objects representing the
    building blocks of the transformer model, e.g.
    transformer_layers.SelfAttention.

    In addition, there are a bunch of other transformations which occur around
    the layer body, and at the beginning and the end of the layer stack.  We
    call these "sublayers".  They are configurable with the `sublayers_initial`,
    `sublayers_per_layer`, and `sublayers_final` arguments, each of which takes
    a list of sublayer functions.

    Each of the sublayer functions has signature:
      x, layer_stack, context -> y
    where x is the input tensor and y is the output tensor.

    The default sublayers specified in defaults.gin are:

      transformer.LayerStack.sublayers_initial = [
          @transformer.sublayer_dropout,
      ]
      transformer.LayerStack.sublayers_per_layer = [
          @transformer.sublayer_rms_norm,
          @transformer.sublayer_call_layer,
          @transformer.sublayer_dropout,
          @transformer.sublayer_residual,
      ]
      transformer.LayerStack.sublayers_final = [
          @transformer.sublayer_rms_norm,
          @transformer.sublayer_dropout,
      ]

    Refer to these as examples of how to write and call your own sublayer
    functions.

    `dropout_rate` and `norm_epsilon` should only be specified in a legacy mode,
    for compatibility with older checkpoints.

    Args:
      layers: a list of TransformerLayer
      sublayers_initial: an optional list of sublayer functions
      sublayers_per_layer: an optional list of sublayer functions
      sublayers_final: an optional list of sublayer functions
      dropout_rate: DEPRECATED - a floating-point number
      norm_epsilon: DEPRECATED - a floating-point number
      recompute_grads: a boolean
    """
    self._layers = layers
    self._recompute_grads = recompute_grads
    self._sublayers_initial = sublayers_initial
    self._sublayers_per_layer = sublayers_per_layer
    self._sublayers_final = sublayers_final
    if (dropout_rate is not None) != (norm_epsilon is not None):
      raise ValueError(
          "LayerStack.dropout_rate and LayerStack.norm_epsilon should either "
          "be both not None (legacy mode) or both None (normal mode)")
    if dropout_rate is not None:
      self._legacy_init(dropout_rate, norm_epsilon)

  def _legacy_init(self, dropout_rate, norm_epsilon):
    """Legacy initialization for use with old checkpoints.

    dropout_rate and norm_epsilon are specified in LayerStack.
    Custom sublayers are not specified.

    Args:
      dropout_rate: a float
      norm_epsilon: a float
    """
    self.dropout_rate = dropout_rate
    self.norm_epsilon = norm_epsilon
    if (self._sublayers_initial is not None or
        self._sublayers_per_layer is not None or
        self._sublayers_final is not None):
      tf.logging.warning("legacy mode - ignoring custom sublayers")
    self._sublayers_initial = [sublayer_legacy_dropout]
    self._sublayers_per_layer = [sublayer_legacy_rms_norm,
                                 sublayer_call_layer,
                                 sublayer_legacy_dropout,
                                 sublayer_residual]
    self._sublayers_final = [sublayer_legacy_final_rms_norm,
                             sublayer_legacy_dropout]
    self.context = None

  def call(self, context, x, run_layers_range=None):
    """Call the layer stack."""
    tf.logging.info("Calling Charformer layer stack")
    x = self._call_sublayers(self._sublayers_initial, x, context)
    context.layer_outputs.append(x)

    if run_layers_range:
      layers = self._layers[run_layers_range[0]:run_layers_range[1]]
    else:
      layers = self._layers

    for lnum, layer in enumerate(layers):
      tf.logging.info("Running=%d | %s", lnum, layer.__class__.__name__)
      tf.logging.info(layer)
      with tf.variable_scope(layer.name or ""):
        if self._recompute_grads:
          def fn(x, l=layer, c=context):
            return self._layer_fn(x, l, c)
          x = mtf.recompute_grad(fn, [x])
        else:
          x = self._layer_fn(x, layer, context)
      if lnum != len(self._layers) - 1:
        context.layer_outputs.append(x)
      context.layer_index += 1
    x = self._call_sublayers(self._sublayers_final, x, context)
    x = sublayer_mask_padding(x, self, context)
    context.layer_outputs.append(x)
    self.context = context
    return x

  def _call_sublayers(self, sublayers, x, context):
    for s in sublayers:
      x = s(x, self, context)
    return x

  def _layer_fn(self, x, layer, context):
    """Call the layer and its associated sublayers.

    Args:
      x: a Tensor
      layer: a Layer
      context: a Context
    Returns:
      a Tensor
    """
    context.current_layer = layer
    context.current_layer_input = x
    y = self._call_sublayers(self._sublayers_per_layer, x, context)
    if y.shape != x.shape:
      raise ValueError(
          "Layer %s returned misshaped output x=%s y=%s"
          % (layer.__class__.__name__, x, y))
    return y

  @property
  def num_layers(self):
    return len(self.layers)

  @property
  def layers(self):
    return self._layers


@gin.configurable
def sublayer_call_layer(x, layer_stack, context):
  x = sublayer_mask_padding(x, layer_stack, context)
  layer = context.current_layer
  with tf.variable_scope(layer.__class__.__name__):
    return layer.call(context, x)
