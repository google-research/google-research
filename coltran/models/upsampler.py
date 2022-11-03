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

"""Color and spatial upsamplers of the Colorization Transformer.

1. Color Upsampler upsamples a coarsely colored image with 512 possible
colors into a low resolution RGB image.

2. Spatial Upsampler upsamples a 256x256 blurry low resolution image into
the final 256x256 high resolution output.

See Section 4.3 of https://openreview.net/pdf?id=5NA1PinlGFu for more details.
"""
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils


class ColorUpsampler(tf.keras.Model):
  """Color Upsampler."""

  def __init__(self, config, **kwargs):
    super(ColorUpsampler, self).__init__(**kwargs)
    self.config = config
    self.hidden_size = self.config.get('hidden_size', 512)

  def build(self, input_shape):
    self.bit_embedding = layers.Dense(units=self.hidden_size, use_bias=False)
    self.gray_embedding = layers.Dense(units=self.hidden_size, use_bias=False)
    self.input_dense = layers.Dense(units=self.hidden_size)
    self.encoder = coltran_layers.FactorizedAttention(self.config)
    self.final_dense = layers.Dense(units=256)

  def call(self, inputs, inputs_slice, channel_index=None, training=True):
    """Upsamples the coarsely colored input into a RGB image.

    Args:
      inputs: size (B, 64, 64, 3).
      inputs_slice: batch of randomly sliced channels, i.e (B, 64, 64, 1)
                    each element of the batch is either a R, G or B channel.
      channel_index: size (B,) Each element is (0, 1, or 2) denoting a
                     R, G or B channel.
      training: used only for dropout.
    Returns:
      logits: size (B, 64, 64, 3, 256) during training or
              size (B, 64, 64, 1, 256) during evaluation or sampling.
    """
    grayscale = tf.image.rgb_to_grayscale(inputs)
    # convert inputs to a coarse image.
    inputs_slice = base_utils.convert_bits(
        inputs_slice, n_bits_in=8, n_bits_out=3)

    logits = self.upsampler(inputs_slice, grayscale, training=training,
                            channel_index=channel_index)
    return logits, {}

  def upsampler(self, inputs, grayscale, channel_index=None, training=True):
    """Upsamples the coarse inputs to per-channel logits."""
    num_channels = inputs.shape[-1]
    logits = []

    # Embed grayscale image.
    grayscale = tf.one_hot(grayscale, depth=256)
    gray_embed = self.gray_embedding(grayscale)
    gray_embed = tf.squeeze(gray_embed, axis=-2)

    if channel_index is not None:
      channel_index = tf.reshape(channel_index, (-1, 1, 1))

    for channel_ind in range(num_channels):
      channel = inputs[Ellipsis, channel_ind]

      if channel_index is not None:
        # single random channel slice during training.
        # channel_index is the index of the random channel.
        # each channel has 8 possible symbols.
        channel += 8 * channel_index
      else:
        channel += 8 * channel_ind

      channel = tf.expand_dims(channel, axis=-1)
      channel = tf.one_hot(channel, depth=24)

      channel = self.bit_embedding(channel)
      channel = tf.squeeze(channel, axis=-2)

      channel = tf.concat((channel, gray_embed), axis=-1)
      channel = self.input_dense(channel)

      context = self.encoder(channel, training=training)
      channel_logits = self.final_dense(context)
      logits.append(channel_logits)
    logits = tf.stack(logits, axis=-2)
    return logits

  def sample(self, gray_cond, bit_cond, mode='argmax'):
    output = dict()
    bit_cond_viz = base_utils.convert_bits(bit_cond, n_bits_in=3, n_bits_out=8)
    output['bit_cond'] = tf.cast(bit_cond_viz, dtype=tf.uint8)

    logits = self.upsampler(bit_cond, gray_cond, training=False)

    if mode == 'argmax':
      samples = tf.argmax(logits, axis=-1)
    elif mode == 'sample':
      batch_size, height, width, channels = logits.shape[:-1]
      logits = tf.reshape(logits, (batch_size*height*width*channels, -1))
      samples = tf.random.categorical(logits, num_samples=1,
                                      dtype=tf.int32)[:, 0]
      samples = tf.reshape(samples, (batch_size, height, width, channels))

    samples = tf.cast(samples, dtype=tf.uint8)
    output[f'bit_up_{mode}'] = samples
    return output

  @property
  def metric_keys(self):
    return []

  def get_logits(self, inputs_dict, train_config, training):
    is_downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)

    # during training a random channel slice is upsampled noted by
    # suffix 'slice'.
    channel_index = inputs_dict['channel_index'] if training else None
    inputs_key = 'targets_%d' % downsample_res if is_downsample else 'targets'
    inputs = inputs_dict[inputs_key]
    if is_downsample and training:
      inputs_slice = inputs_dict['targets_slice_%d' % downsample_res]
    elif is_downsample:
      inputs_slice = inputs_dict['targets_%d' % downsample_res]
    elif training:
      inputs_slice = inputs_dict['targets_slice']
    else:
      inputs_slice = inputs_dict['targets']

    return self.call(
        inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)

  def loss(self, targets, logits, train_config, training, aux_output=None):
    is_downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    if is_downsample and training:
      labels = targets['targets_slice_%d' % downsample_res]
    elif is_downsample:
      labels = targets['targets_%d' % downsample_res]
    elif training:
      labels = targets['targets_slice']
    else:
      labels = targets['targets']

    height, width, num_channels = labels.shape[1:4]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    loss = loss / (height * width * num_channels)
    return loss, {}


class SpatialUpsampler(tf.keras.Model):
  """Spatial Upsampler."""

  def __init__(self, config, **kwargs):
    super(SpatialUpsampler, self).__init__(**kwargs)
    self.config = config
    self.num_symbols = 256
    self.hidden_size = self.config.get('hidden_size', 512)
    self.down_res = self.config.get('down_res', 32)
    self.down_method = self.config.get('down_method', 'area')

  def build(self, input_shape):
    self.channel_embedding = layers.Dense(
        units=self.hidden_size, use_bias=False)
    self.gray_embedding = layers.Dense(
        units=self.hidden_size, use_bias=False)
    self.input_dense = layers.Dense(units=self.hidden_size)
    self.encoder = coltran_layers.FactorizedAttention(self.config)
    self.final_dense = layers.Dense(units=self.num_symbols)

  def call(self, inputs, inputs_slice, channel_index=None, training=True):
    """Super resolves blurry high resolution inputs into per-pixel logits.

    Args:
      inputs: size (B, 256, 256, 3).
      inputs_slice: batch of randomly sliced channels, i.e (B, 256, 256, 1)
                    each element of the batch is either a R, G or B channel.
      channel_index: size (B,) Each element is (0, 1, or 2) denoting a
                     R, G or B channel.
      training: used only for dropout.
    Returns:
      logits: size (B, 256, 256, 3, 256) during training or
              size (B, 256, 256, 1, 256) during evaluation or sampling.
    """
    grayscale = tf.image.rgb_to_grayscale(inputs)
    logits = self.upsampler(inputs_slice, grayscale, training=training,
                            channel_index=channel_index)
    return logits, {}

  def upsampler(self, inputs, grayscale, channel_index=None, training=True):
    num_channels = inputs.shape[-1]
    logits = []

    grayscale = tf.one_hot(grayscale, depth=self.num_symbols)
    gray_embed = self.gray_embedding(grayscale)
    gray_embed = tf.squeeze(gray_embed, axis=-2)

    if channel_index is not None:
      channel_index = tf.reshape(channel_index, (-1, 1, 1))

    for channel_ind in range(num_channels):
      channel = inputs[Ellipsis, channel_ind]

      if channel_index is not None:
        # single random channel slice during training.
        # channel_index is the index of the random channel.
        channel += self.num_symbols * channel_index
      else:
        channel += self.num_symbols * channel_ind

      channel = tf.expand_dims(channel, axis=-1)
      channel = tf.one_hot(channel, depth=self.num_symbols*3)

      channel = self.channel_embedding(channel)
      channel = tf.squeeze(channel, axis=-2)

      channel = tf.concat((channel, gray_embed), axis=-1)
      channel = self.input_dense(channel)

      context = self.encoder(channel, training=training)
      channel_logits = self.final_dense(context)
      logits.append(channel_logits)
    logits = tf.stack(logits, axis=-2)
    return logits

  def sample(self, gray_cond, inputs, mode='argmax'):
    output = dict()
    output['low_res_cond'] = tf.cast(inputs, dtype=tf.uint8)
    logits = self.upsampler(inputs, gray_cond, training=False)

    if mode == 'argmax':
      samples = tf.argmax(logits, axis=-1)
    elif mode == 'sample':
      batch_size, height, width, channels = logits.shape[:-1]
      logits = tf.reshape(logits, (batch_size*height*width*channels, -1))
      samples = tf.random.categorical(logits, num_samples=1,
                                      dtype=tf.int32)[:, 0]
      samples = tf.reshape(samples, (batch_size, height, width, channels))

    samples = tf.cast(samples, dtype=tf.uint8)
    output[f'high_res_{mode}'] = samples
    return output

  @property
  def metric_keys(self):
    return []

  def loss(self, targets, logits, train_config, training, aux_output=None):
    if training:
      labels = targets['targets_slice']
    else:
      labels = targets['targets']

    height, width, num_channels = labels.shape[1:]
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    loss = loss / (height * width * num_channels)
    return loss, {}

  def get_logits(self, inputs_dict, train_config, training):
    downsample_res = train_config.get('downsample_res', 64)

    # Random channel slice during training denoted by suffix 'slice'.
    # up_back suffix denotes blurry high resolution input.
    inputs = inputs_dict['targets']
    if training:
      inputs_slice = inputs_dict['targets_slice_%d_up_back' % downsample_res]
      channel_index = inputs_dict['channel_index']
    else:
      inputs_slice = inputs_dict['targets_%d_up_back' % downsample_res]
      channel_index = None
    return self.call(
        inputs=inputs, inputs_slice=inputs_slice, channel_index=channel_index)
