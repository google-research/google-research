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

# Lint as: python3
"""Core components of the colorization transfomer.

Consists of:

1. Grayscale Encoder.
2. Outer Decoder.
3. Inner Decoder.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils


def cond_with_context(inputs, cond_layer, context, cond_type, cond_act):
  cond_act_func = base_utils.act_to_func(cond_act)
  cond_out = cond_layer(context)
  if cond_type == 'shift':
    inputs += cond_out
  elif cond_type == 'affine':
    shift, scale = tf.split(cond_out, num_or_size_splits=2, axis=-1)
    inputs *= cond_act_func(scale)
    inputs += cond_act_func(shift)
  return inputs


def get_pos_embeddings(pos_embed, inputs_shape):
  embeddings = tf.zeros(shape=inputs_shape)
  return pos_embed(embeddings)


class GrayScaleEncoder(layers.Layer):
  """Encodes grayscale version of the image into a 2-D spatial context.

  Consists of a stack of row/column attention layers.
  """

  def __init__(self, config, **kwargs):
    super(GrayScaleEncoder, self).__init__(**kwargs)
    self.config = config
    self.dropout = config.get('dropout', 0.0)

  def build(self, input_shapes):
    self.embedding = layers.Dense(units=self.config.hidden_size)
    self.encoder = coltran_layers.FactorizedAttention(self.config)

  def call(self, inputs):
    if len(inputs.shape) == 4:
      if inputs.shape[-1] != 1:
        raise ValueError('Expected inputs is a grayscale image')
      grayscale = tf.squeeze(inputs, axis=-1)
    grayscale = tf.one_hot(grayscale, depth=256)
    h_gray = self.embedding(grayscale)
    return self.encoder(h_gray)


class OuterDecoder(layers.Layer):
  """Outer Decoder with optional conditioning.

  Contains the following sequence of operations:
    1. Positional Embeddings.
    2. (Unmasked Row + Masked Column) self attention * num_layers.
    3. Shift Down (to preserve causal ordering)

  The input is a tuple of 2 arguments (X, h) where h is the conditioning
  input. Transforms the input X into 2-D spatial context C (H, W, D)
  conditioned on h. Each location C[i, j] is a vector of size D that
  summarizes information from X[:i] and h.

  The conditional components can be activated by setting the corresponding
  conditional arguments to True.
    1. Conditional Layer Norm: config.cond_ln
    2. Conditional Self Attention: config.cond_att_k, config.cond_att_q,
                                   config.cond_att_v, config.cond_att_scale.
    3. Conditional MLP: config.cond_mlp
  """

  def __init__(self, config, **kwargs):
    super(OuterDecoder, self).__init__(**kwargs)
    self.config = config
    self.dropout = self.config.get('dropout', 0.0)
    self.skip = self.config.get('skip', True)

    # Conditional MLP
    self.cond_mlp = self.config.get('cond_mlp', 'affine')
    self.cond_mlp_act = self.config.get('cond_mlp_act', 'identity')

    # Conditional Layer Norm.
    self.cond_ln = self.config.get('cond_ln', True)
    self.cond_ln_act = self.config.get('cond_ln_act', 'identity')
    self.cond_ln_seq = self.config.get('cond_ln_seq', 'sc')
    self.cond_ln_sp_ave = self.config.get('cond_ln_sp_ave', 'learnable')
    self.cond_ln_init = self.config.get('cond_ln_init', 'glorot_uniform')

    # Conditional Self Attention.
    self.cond_att_act = self.config.get('cond_att_act', 'identity')
    self.cond_att_k = self.config.get('cond_att_k', True)
    self.cond_att_q = self.config.get('cond_att_q', True)
    self.cond_att_v = self.config.get('cond_att_v', True)
    self.cond_att_scale = self.config.get('cond_att_scale', True)
    self.cond_att_init = self.config.get('cond_att_init', 'glorot_uniform')
    self.cond_att = self.cond_att_v or self.cond_att_q or self.cond_att_k

  def build(self, input_shapes):
    embed_shape = input_shapes[0]
    height, width, num_filters = embed_shape[1:]
    hidden_size = self.config.hidden_size
    num_heads = self.config.num_heads
    ff_size = self.config.ff_size
    res = [height, width]

    self.pos_embed = coltran_layers.PositionEmbed(axes=[1, 2], max_lengths=res)

    self.residual_layers, self.layer_norms, self.cmlp_layers = [], [], []
    num_norms = self.config.num_outer_layers * 4
    if self.cond_ln:
      for _ in range(num_norms):
        curr_norm = coltran_layers.ConditionalLayerNorm(
            spatial_average=self.cond_ln_sp_ave,
            sequence=self.cond_ln_seq,
            out_init=self.cond_ln_init,
            out_act=self.cond_ln_act)
        self.layer_norms.append(curr_norm)
    else:
      self.layer_norms = [layers.LayerNormalization() for _ in range(num_norms)]

    for layer_ind in range(self.config.num_outer_layers):
      # unmasked row
      unmask_row = coltran_layers.SelfAttentionND(
          hidden_size=hidden_size, num_heads=num_heads,
          nd_block_size=[1, width], resolution=[height, width],
          cond_q=self.cond_att_q,
          cond_k=self.cond_att_k,
          cond_v=self.cond_att_v,
          cond_init=self.cond_att_init,
          cond_scale=self.cond_att_scale,
          cond_act=self.cond_att_act,
          name='unmask_row_att_%d' % layer_ind)

      ff_row = tf.keras.Sequential([
          layers.Dense(units=ff_size, activation='relu'),
          layers.Dense(units=num_filters)
      ], name='row_dense_%d' % layer_ind)

      # masked column,
      mask_col = coltran_layers.SelfAttentionND(
          hidden_size=hidden_size, num_heads=num_heads, mask='future',
          nd_block_size=[height, 1], resolution=[height, width],
          cond_q=self.cond_att_q,
          cond_k=self.cond_att_k,
          cond_v=self.cond_att_v,
          cond_act=self.cond_att_act,
          cond_init=self.cond_att_init,
          cond_scale=self.cond_att_scale,
          name='mask_col_att_%d' % layer_ind)

      ff_col = tf.keras.Sequential([
          layers.Dense(units=ff_size, activation='relu'),
          layers.Dense(units=num_filters)
      ], name='col_dense_%d' % layer_ind)

      self.residual_layers.append(unmask_row)
      self.residual_layers.append(ff_row)
      self.residual_layers.append(mask_col)
      self.residual_layers.append(ff_col)

      # Conditional MLP layers.
      if self.cond_mlp == 'shift':
        shift_r = layers.Dense(units=hidden_size, name='shift_r_%d' % layer_ind)
        shift_c = layers.Dense(units=hidden_size, name='shift_c_%d' % layer_ind)
        self.cmlp_layers.append(shift_r)
        self.cmlp_layers.append(shift_c)
      elif self.cond_mlp == 'affine':
        aff_r = layers.Dense(
            units=2*hidden_size, name='affine_r_%d' % layer_ind)
        aff_c = layers.Dense(
            units=2*hidden_size, name='affine_c_%d' % layer_ind)
        self.cmlp_layers.append(aff_r)
        self.cmlp_layers.append(aff_c)

    self.shift_down = coltran_layers.Shift(dimension=0, resolution=res)

  def call(self, inputs, training=True):
    embeddings, channel_context = inputs
    cond_layer_ind = 0

    output = self.pos_embed(embeddings)
    if self.skip:
      output += channel_context
    inputs = output

    for layer, norm in zip(self.residual_layers, self.layer_norms):
      if 'att' in layer.name and self.cond_att:
        output = layer((inputs, channel_context))
      else:
        output = layer(inputs)

      if 'dense' in layer.name:
        curr_cond_layer = self.cmlp_layers[cond_layer_ind]
        output = cond_with_context(output, curr_cond_layer, channel_context,
                                   self.cond_mlp, self.cond_mlp_act)
        cond_layer_ind += 1

      output = coltran_layers.residual_dropout(
          inputs, output, self.dropout, training)

      if self.cond_ln:
        inputs = norm((output, channel_context))
      else:
        inputs = norm(output)

    output = self.shift_down(inputs)
    return output


class InnerDecoder(layers.Layer):

  """Inner Decoder with optional conditioning.

  Contains the following sequence of operations:
    1. Adds positional Embeddings + context to the pixel embeddings.
    2. Shift right (to preserve causal order).
    2. (Masked Row) self attention * num_layers.

  The input is a tuple of 2 arguments (X, h_out, h) where h_out and h are the
  conditioning inputs from the grayscale image and the outer decoder
  respectively. Transforms the input X into 2-D spatial context C (H, W, D)
  conditioned on h. Each location C[i, j] is a vector of size D that
  summarizes information from X[:i], X[i, :j] and h.

  The conditional components can be activated by setting the corresponding
  conditional arguments to True.
    1. Conditional Layer Norm: config.cond_ln
    2. Conditional Self Attention: config.cond_att_k, config.cond_att_q,
                                   config.cond_att_v, config.cond_att_scale.
    3. Conditional MLP: config.cond_mlp
  """

  def __init__(self,
               config,
               **kwargs):
    super(InnerDecoder, self).__init__(**kwargs)
    self.config = config
    self.skip = self.config.get('skip', True)
    self.dropout = self.config.get('dropout', 0.0)

    self.cond_mlp = self.config.get('cond_mlp', 'affine')
    self.cond_mlp_act = self.config.get('cond_mlp_act', 'identity')

    self.cond_ln = self.config.get('cond_ln', True)
    self.cond_ln_act = self.config.get('cond_ln_act', 'identity')
    self.cond_ln_seq = self.config.get('cond_ln_seq', 'sc')
    self.cond_ln_sp_ave = self.config.get('cond_ln_sp_ave', 'learnable')
    self.cond_ln_init = self.config.get('cond_ln_init', 'glorot_uniform')

    self.cond_att_act = self.config.get('cond_att_act', 'identity')
    self.cond_att_k = self.config.get('cond_att_k', False)
    self.cond_att_q = self.config.get('cond_att_q', False)
    self.cond_att_v = self.config.get('cond_att_v', False)
    self.cond_att_scale = self.config.get('cond_att_scale', False)
    self.cond_att_init = self.config.get('cond_att_init', 'glorot_uniform')
    self.cond_att = self.cond_att_v or self.cond_att_q or self.cond_att_k

  def build(self, input_shapes):
    context_shape = input_shapes[1]
    height, width = context_shape[1:3]
    ff_size = self.config.ff_size
    hidden_size = self.config.hidden_size
    num_heads = self.config.num_heads
    res = [height, width]

    self.pos_embed = coltran_layers.PositionEmbed(axes=[1, 2], max_lengths=res)
    self.shift_right = coltran_layers.Shift(dimension=1, resolution=res)

    self.residual_layers, self.layer_norms, self.cmlp_layers = [], [], []
    num_norms = 2 * self.config.num_inner_layers
    if self.cond_ln:
      for _ in range(num_norms):
        curr_norm = coltran_layers.ConditionalLayerNorm(
            spatial_average=self.cond_ln_sp_ave,
            sequence=self.cond_ln_seq,
            out_init=self.cond_ln_init,
            out_act=self.cond_ln_act)
        self.layer_norms.append(curr_norm)
    else:
      self.layer_norms = [layers.LayerNormalization() for _ in range(num_norms)]

    for layer_ind in range(self.config.num_inner_layers):

      mask_row = coltran_layers.SelfAttentionND(
          hidden_size=hidden_size, num_heads=num_heads, mask='future',
          nd_block_size=[1, width], resolution=[height, width],
          cond_q=self.cond_att_q,
          cond_k=self.cond_att_k,
          cond_v=self.cond_att_v,
          cond_init=self.cond_att_init,
          cond_scale=self.cond_att_scale,
          cond_act=self.cond_att_act,
          name='mask_row_att_%d' % layer_ind)

      ff_block = tf.keras.Sequential([
          layers.Dense(units=ff_size, activation='relu'),
          layers.Dense(units=hidden_size)
      ], name='dense_%d' % layer_ind)

      self.residual_layers.append(mask_row)
      self.residual_layers.append(ff_block)

      if self.cond_mlp == 'shift':
        shift_c = layers.Dense(units=hidden_size, name='shift_c_%d' % layer_ind)
        self.cmlp_layers.append(shift_c)
      elif self.cond_mlp == 'affine':
        aff_c = layers.Dense(
            units=2*hidden_size, name='affine_c_%d' % layer_ind)
        self.cmlp_layers.append(aff_c)

  def call(self, inputs, row_ind=None, training=True):
    embeddings, upper_context, channel_context = inputs

    embeddings = self.shift_right(embeddings)
    if row_ind is None:
      embeddings = self.pos_embed(embeddings)
    # special case during sampling.
    else:
      input_shape = embeddings.shape.as_list()
      pos_embed = get_pos_embeddings(self.pos_embed, input_shape)
      pos_embed = pos_embed[:, row_ind: row_ind + 1]
      embeddings += pos_embed

    inputs = embeddings
    if self.skip:
      inputs += channel_context
      inputs += upper_context

    layer_zip = zip(self.residual_layers, self.layer_norms)
    all_context = tf.concat((channel_context, upper_context), -1)

    cond_layer_ind = 0
    for layer, norm in layer_zip:

      # Conditional Self-Attention.
      if 'att' in layer.name and self.cond_att:
        output = layer((inputs, all_context))
      else:
        output = layer(inputs)

      # Conditional MLP.
      if 'dense' in layer.name:
        curr_cond_layer = self.cmlp_layers[cond_layer_ind]
        output = cond_with_context(output, curr_cond_layer, all_context,
                                   self.cond_mlp, self.cond_mlp_act)
        cond_layer_ind += 1

      output = coltran_layers.residual_dropout(
          inputs, output, self.dropout, training)

      # providing all context here violates causal masking due to the spatial
      # averaging.
      # Conditional Layer norm.
      if self.cond_ln:
        inputs = norm((output, channel_context))
      else:
        inputs = norm(output)

    return inputs
