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

"""Layers for contextual lenses."""

from typing import Any, Optional, Sequence
import gin
import tensorflow as tf


@gin.configurable
class MaskedGlobalMaxPooling1D(tf.keras.layers.GlobalMaxPooling1D):
  """Wraps GlobalMaxPooling1D with support for padding masks."""

  def __init__(self, pad_penalty = -1e9, **kwargs):
    super().__init__(**kwargs)
    self._pad_penalty = pad_penalty

  def call(self,
           inputs,
           mask = None):
    if mask is not None:
      pad_penalty = tf.cast(self._pad_penalty, inputs.dtype)
      mask = (mask[Ellipsis, tf.newaxis] if self.data_format == 'channels_last'
              else mask[:, tf.newaxis])
      inputs = tf.where(mask, inputs, pad_penalty)
    return super().call(inputs)


@gin.configurable
class GlobalAttentionPooling1D(tf.keras.layers.Layer):
  """Naive, attention-based weighted average pooling for sequential inputs."""

  def __init__(self,
               name = 'global_attention_pooling1d',
               normalize = True,
               dropout = 0.0,
               **kwargs):
    # TODO(fllinares): support 'channels_first' data_format, just in case.
    super().__init__(**kwargs)
    self.layer_norm = None
    if normalize:
      self.layer_norm = tf.keras.layers.LayerNormalization(
          epsilon=1e-12, name=f'{self.name}/layer_normalization')
    self.dropout = tf.keras.layers.Dropout(
        rate=dropout, name=f'{self.name}/dropout')
    self.dense = tf.keras.layers.Dense(units=1, name=f'{self.name}/dense')
    self.softmax = tf.keras.layers.Softmax()

  def call(self,
           inputs,
           mask = None,
           training = True):
    att_act = inputs if self.layer_norm is None else self.layer_norm(inputs)
    att_act = tf.squeeze(self.dense(att_act), axis=-1)
    att_act = self.dropout(att_act, training=training)
    att_weights = self.softmax(att_act, mask=mask)
    return tf.einsum('bld,bl->bd', inputs, att_weights)


@gin.configurable
class MLP(tf.keras.Sequential):
  """A generic multi-layer perceptron."""

  def __init__(self,
               output_size = gin.REQUIRED,
               output_activation = None,
               output_dropout = 0.0,
               pre_pooling_layer_norm = False,
               pre_pooling_hidden = (),
               pre_pooling_activation = 'relu',
               pre_pooling_dropout = 0.0,
               post_pooling_layer_norm = False,
               post_pooling_hidden = (),
               post_pooling_activation = 'relu',
               post_pooling_dropout = 0.0,
               pooling_cls=tf.keras.layers.GlobalMaxPooling1D,
               **kwargs):
    layers = []

    if pre_pooling_layer_norm:
      layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-12))

    for h in pre_pooling_hidden:
      layers.extend([tf.keras.layers.Dense(h),
                     tf.keras.layers.Activation(pre_pooling_activation)])
      if pre_pooling_dropout > 0.0:
        layers.append(tf.keras.layers.Dropout(pre_pooling_dropout))

    if pooling_cls is not None:
      layers.append(pooling_cls())

    if post_pooling_layer_norm:
      layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-12))

    for h in post_pooling_hidden:
      layers.extend([tf.keras.layers.Dense(h),
                     tf.keras.layers.Activation(post_pooling_activation)])
      if post_pooling_dropout > 0.0:
        layers.append(tf.keras.layers.Dropout(post_pooling_dropout))

    layers.extend([tf.keras.layers.Dense(output_size),
                   tf.keras.layers.Activation(output_activation)])
    if output_dropout > 0.0:
      layers.append(tf.keras.layers.Dropout(output_dropout))

    super().__init__(layers, **kwargs)


@gin.configurable
class NetSurfP2(tf.keras.Model):
  """NetSurfP-2.0 output head ala Klausen et al. 2009."""

  def __init__(self,
               output_size = gin.REQUIRED,
               output_layer_norm = False,
               cnn_layer_norm = False,
               cnn_filters = (),
               cnn_kernel_sizes = (),
               cnn_act=tf.keras.activations.relu,
               cnn_dropout = 0.0,
               concat_input = False,
               lstm_layer_norm = False,
               lstm_units = (),
               lstm_input_dropout = 0.0,
               lstm_recurrent_dropout = 0.0,
               name = 'NetSurfP2',
               **kwargs):
    super().__init__(name=name, **kwargs)

    self._cnn_layer_norm = None
    if cnn_layer_norm:
      self._cnn_layer_norm = tf.keras.layers.LayerNormalization(
          epsilon=1e-12, name=f'{self.name}/cnn_layer_norm')

    self._cnn_layers = []
    for i, (filters, kernel_size) in enumerate(
        zip(cnn_filters, cnn_kernel_sizes)):
      self._cnn_layers.append(tf.keras.layers.Conv1D(
          filters=filters,
          kernel_size=kernel_size,
          padding='same',
          name=f'{self.name}/cnn_layer_{i}'))
    self._cnn_act = cnn_act
    self._cnn_dropout = tf.keras.layers.Dropout(rate=cnn_dropout)

    self._concat_input = concat_input

    self._lstm_layer_norm = None
    if lstm_layer_norm:
      self._lstm_layer_norm = tf.keras.layers.LayerNormalization(
          epsilon=1e-12, name=f'{self.name}/lstm_layer_norm')

    self._lstm_layers = []
    for i, units in enumerate(lstm_units):
      lstm_layer = tf.keras.layers.LSTM(
          units=units,
          dropout=lstm_input_dropout,
          recurrent_dropout=lstm_recurrent_dropout,
          return_sequences=True,
          name=f'{self.name}/LSTM_layer_{i}')
      self._lstm_layers.append(tf.keras.layers.Bidirectional(
          lstm_layer, name=f'{self.name}/BiLSTM_layer_{i}'))

    self._output_layer_norm = None
    if output_layer_norm:
      self._output_layer_norm = tf.keras.layers.LayerNormalization(
          epsilon=1e-12, name=f'{self.name}/output_layer_norm')

    self._output_dense = tf.keras.layers.Dense(output_size)

  def call(self,
           embeddings,
           mask = None,
           training = True):
    # As described in the original article, the model is composed of:
    #  + Two CNN layers with 32 filters each and kernel sizes 129 and 257. No
    #    details are given about the activation function, present/absence of
    #    normalization (e.g. batch norm), use of dropout or regularization.
    #  + Concatenate input to CNN output.
    #  + Two BiLSTM layers with 1024 units each.
    masked = mask is not None
    float_mask = tf.cast(mask, embeddings.dtype) if masked else mask

    # CNN layers.
    x = embeddings
    if self._cnn_layer_norm is not None:
      x = self._cnn_layer_norm(x)
    for cnn_layer in self._cnn_layers:
      x = x * float_mask[Ellipsis, None] if masked else x
      x = cnn_layer(x, training=training)
      # TODO(fllinares): add BatchNormalization, but what about padding??
      x = self._cnn_act(x)
      x = self._cnn_dropout(x, training=training)

    # Concat original embeddings to output of CNN, optionally.
    if self._concat_input:
      x = tf.concat([embeddings, x], -1)

    # Bidirectional LSTM here.
    if self._lstm_layer_norm is not None:
      x = self._lstm_layer_norm(x)
    for lstm_layer in self._lstm_layers:
      x = lstm_layer(x, mask=mask, training=training)

    # Fully-connected output layer.
    if self._output_layer_norm is not None:
      x = self._output_layer_norm(x)
    x = x * float_mask[Ellipsis, None] if masked else embeddings
    return self._output_dense(x, training=training)
