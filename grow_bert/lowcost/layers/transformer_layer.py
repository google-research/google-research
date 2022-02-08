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

"""Keras-based transformer block layer."""

import tensorflow as tf

from grow_bert.lowcost.layers import light_attention_layer


class TransformerLayer(tf.keras.layers.Layer):
  """Transformer layer.

  This layer implements the Transformer from "Attention Is All You Need".
  (https://arxiv.org/abs/1706.03762).
  """

  def __init__(self,
               num_attention_heads,
               intermediate_size,
               intermediate_activation,
               dropout_rate=0.0,
               attention_dropout_rate=0.0,
               output_range=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_lightatt=False,
               net2net_ratio=None,
               **kwargs):
    """Initialize.

    Arguments:
      num_attention_heads: Number of attention heads.
      intermediate_size: Size of the intermediate layer.
      intermediate_activation: Activation for the intermediate layer.
      dropout_rate: Dropout probability for the post-attention and output
        dropout.
      attention_dropout_rate: Dropout probability for within the attention
        layer.
      output_range: the sequence output range, [0, output_range) by slicing the
        target sequence. `None` means the target sequence is not sliced.
      kernel_initializer: Initializer for dense layer kernels.
      bias_initializer: Initializer for dense layer biases.
      kernel_regularizer: Regularizer for dense layer kernels.
      bias_regularizer: Regularizer for dense layer biases.
      activity_regularizer: Regularizer for dense layer activity.
      kernel_constraint: Constraint for dense layer kernels.
      bias_constraint: Constraint for dense layer kernels.
      use_lightatt: whether or not to use light attention.
      net2net_ratio: net2net ratio for the small fully connected matrices.
      **kwargs: **kwargs
    """
    super(TransformerLayer, self).__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._intermediate_size = intermediate_size
    self._intermediate_activation = intermediate_activation
    self._attention_dropout_rate = attention_dropout_rate
    self._dropout_rate = dropout_rate
    self._output_range = output_range
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.use_lightatt = use_lightatt
    self.net2net_ratio = net2net_ratio

  def build(self, input_shape):
    input_tensor = input_shape[0] if len(input_shape) == 2 else input_shape
    input_tensor_shape = tf.TensorShape(input_tensor)
    if len(input_tensor_shape.as_list()) != 3:
      raise ValueError("TransformerLayer expects a three-dimensional input of "
                       "shape [batch, sequence, width].")
    batch_size, sequence_length, hidden_size = input_tensor_shape

    if len(input_shape) == 2:
      mask_tensor_shape = tf.TensorShape(input_shape[1])
      expected_mask_tensor_shape = tf.TensorShape(
          [batch_size, sequence_length, sequence_length])
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          "The input size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (hidden_size, self._num_heads))
    self._attention_head_size = int(hidden_size // self._num_heads)
    common_kwargs = dict(
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)

    if self.use_lightatt:
      self._attention_layer = light_attention_layer.LightMultiHeadAttention(
          num_heads=self._num_heads,
          key_dim=self._attention_head_size,
          dropout=self._attention_dropout_rate,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activity_regularizer=self._activity_regularizer,
          kernel_constraint=self._kernel_constraint,
          bias_constraint=self._bias_constraint,
          name="self_attention")
    else:
      self._attention_layer = tf.keras.layers.MultiHeadAttention(
          num_heads=self._num_heads,
          key_dim=self._attention_head_size,
          dropout=self._attention_dropout_rate,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activity_regularizer=self._activity_regularizer,
          kernel_constraint=self._kernel_constraint,
          bias_constraint=self._bias_constraint,
          name="self_attention")
    self._attention_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32))
    if self.net2net_ratio is not None:
      small_interm_size = int(self._intermediate_size * self.net2net_ratio)
      self._intermediate_dense_small = tf.keras.layers.experimental.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, small_interm_size),
          bias_axes="d",
          kernel_initializer=self._kernel_initializer,
          name="intermediate_small",
          **common_kwargs)
      self._output_dense_small = tf.keras.layers.experimental.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, hidden_size),
          bias_axes="d",
          kernel_initializer=self._kernel_initializer,
          name="output_small")
    else:
      self._intermediate_dense = tf.keras.layers.experimental.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, self._intermediate_size),
          bias_axes="d",
          kernel_initializer=self._kernel_initializer,
          name="intermediate",
          **common_kwargs)
      self._output_dense = tf.keras.layers.experimental.EinsumDense(
          "abc,cd->abd",
          output_shape=(None, hidden_size),
          bias_axes="d",
          name="output",
          kernel_initializer=self._kernel_initializer,
          **common_kwargs)

    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == "mixed_bfloat16":
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._intermediate_activation, dtype=policy)

    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)

    super(TransformerLayer, self).build(input_shape)

  def get_config(self):
    config = {
        "num_attention_heads":
            self._num_heads,
        "intermediate_size":
            self._intermediate_size,
        "intermediate_activation":
            self._intermediate_activation,
        "dropout_rate":
            self._dropout_rate,
        "attention_dropout_rate":
            self._attention_dropout_rate,
        "output_range":
            self._output_range,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
        "activity_regularizer":
            tf.keras.regularizers.serialize(self._activity_regularizer),
        "kernel_constraint":
            tf.keras.constraints.serialize(self._kernel_constraint),
        "bias_constraint":
            tf.keras.constraints.serialize(self._bias_constraint)
    }
    base_config = super(TransformerLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, _target_tensor=None, shared_attention_scores=None):

    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
      input_tensor, attention_mask = inputs
    else:
      input_tensor, attention_mask = (inputs, None)

    if _target_tensor is not None:
      target_tensor = _target_tensor
    elif self._output_range:
      target_tensor = input_tensor[:, 0:self._output_range, :]
      attention_mask = attention_mask[:, 0:self._output_range, :]
    else:
      target_tensor = input_tensor

    if self.use_lightatt:
      attention_output, attention_scores = self._attention_layer(
          target_tensor,
          input_tensor,
          key=None,
          attention_mask=attention_mask,
          return_attention_scores=True,
          shared_attention_scores=shared_attention_scores)
    else:
      attention_output, attention_scores = self._attention_layer(
          target_tensor,
          input_tensor,
          key=None,
          attention_mask=attention_mask,
          return_attention_scores=True)

    attention_output = self._attention_dropout(attention_output)
    attention_output = self._attention_layer_norm(target_tensor +
                                                  attention_output)
    if self.net2net_ratio is not None:
      intermediate_output = self._intermediate_dense_small(attention_output)
    else:
      intermediate_output = self._intermediate_dense(attention_output)
    intermediate_output = self._intermediate_activation_layer(
        intermediate_output)

    if self.net2net_ratio is not None:
      layer_output = self._output_dense_small(intermediate_output)
    else:
      layer_output = self._output_dense(intermediate_output)
    layer_output = self._output_dropout(layer_output)
    # During mixed precision training, attention_output is from layer norm and
    # is always fp32 for now. Cast layer_output to fp32 for the subsequent
    # add.
    layer_output = tf.cast(layer_output, tf.float32)
    layer_output = self._output_layer_norm(layer_output + attention_output)

    return layer_output, attention_scores

  @property
  def attention_layer(self):
    return self._attention_layer
