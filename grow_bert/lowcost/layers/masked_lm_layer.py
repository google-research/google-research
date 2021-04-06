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

"""Masked language model network."""
import tensorflow as tf

from tensorflow_models.official.modeling import tf_utils


class MaskedLMLayer(tf.keras.layers.Layer):
  """Mask lm layer."""

  def __init__(self,
               embedding_table,
               activation=None,
               dropout_rate=0.1,
               initializer='glorot_uniform',
               output='logits',
               name='cls/predictions',
               **kwargs):
    super(MaskedLMLayer, self).__init__(name=name, **kwargs)
    self.embedding_table = embedding_table
    self.activation = activation
    self.initializer = tf.keras.initializers.get(initializer)

    if output not in ('predictions', 'logits'):
      raise ValueError(
          ('Unknown `output` value "%s". `output` can be either "logits" or '
           '"predictions"') % output)
    self._output_type = output

  def build(self, input_shape):
    self._vocab_size, hidden_size = self.embedding_table.shape
    self.dense = tf.keras.layers.Dense(
        hidden_size,
        activation=self.activation,
        kernel_initializer=self.initializer,
        name='transform/dense')
    self.layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name='transform/LayerNorm')
    self.bias = self.add_weight(
        'output_bias/bias',
        shape=(self._vocab_size,),
        initializer='zeros',
        trainable=True)

    super(MaskedLMLayer, self).build(input_shape)

  def call(self, target_embedding):
    lm_data = self.dense(target_embedding)
    lm_data = self.layer_norm(lm_data)
    lm_data = tf.matmul(lm_data, self.embedding_table, transpose_b=True)
    logits = tf.nn.bias_add(lm_data, self.bias)

    masked_positions_shape = tf_utils.get_shape_list(
        target_embedding, name='masked_positions_tensor')
    logits = tf.reshape(logits,
                        [-1, masked_positions_shape[1], self._vocab_size])
    if self._output_type == 'logits':
      return logits
    return tf.nn.log_softmax(logits)

  def get_config(self):
    raise NotImplementedError('MaskedLM cannot be directly serialized because '
                              'it has variable sharing logic.')
