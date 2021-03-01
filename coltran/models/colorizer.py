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
"""ColTran core.

Core autoregressive component of the colorization transformer based on
the AxialTransformer with conditional self-attention layers.

See Section 3 and Section 4.1 of https://openreview.net/pdf?id=5NA1PinlGFu
for more details.
"""
import tensorflow.compat.v2 as tf
from tensorflow.compat.v2.keras import layers
from coltran.models import core
from coltran.models import layers as coltran_layers
from coltran.utils import base_utils


class ColTranCore(tf.keras.Model):
  """Colorization Transformer."""

  def __init__(self, config, **kwargs):
    super(ColTranCore, self).__init__(**kwargs)
    self.config = config

    # 3 bits per channel, 8 colors per channel, a total of 512 colors.
    self.num_symbols_per_channel = 2**3
    self.num_symbols = self.num_symbols_per_channel**3
    self.gray_symbols, self.num_channels = 256, 1

    self.enc_cfg = config.encoder
    self.dec_cfg = config.decoder
    self.hidden_size = self.config.get('hidden_size',
                                       self.dec_cfg.hidden_size)

    # stage can be 'encoder_decoder' or 'decoder'
    # 1. decoder -> loss only due to autoregressive model.
    # 2. encoder_decoder -> loss due to both the autoregressive and parallel
    # model.
    # encoder_only and all
    self.stage = config.get('stage', 'decoder')
    self.is_parallel_loss = 'encoder' in self.stage
    stages = ['decoder', 'encoder_decoder']
    if self.stage not in stages:
      raise ValueError('Expected stage to be in %s, got %s' %
                       (str(stages), self.stage))

  @property
  def metric_keys(self):
    if self.stage == 'encoder_decoder':
      return ['encoder']
    return []

  def build(self, input_shape):
    # encoder graph
    self.encoder = core.GrayScaleEncoder(self.enc_cfg)
    if self.is_parallel_loss:
      self.parallel_dense = layers.Dense(
          units=self.num_symbols, name='parallel_logits', use_bias=False)

    # decoder graph: outer decoder -> inner decoder -> logits.
    self.pixel_embed_layer = layers.Dense(
        units=self.hidden_size, use_bias=False)
    self.outer_decoder = core.OuterDecoder(self.dec_cfg)
    self.inner_decoder = core.InnerDecoder(self.dec_cfg)
    self.final_dense = layers.Dense(
        units=self.num_symbols, name='auto_logits')
    self.final_norm = layers.LayerNormalization()

  def call(self, inputs, training=True):
    # encodes grayscale (H, W) into activations of shape (H, W, 512).
    gray = tf.image.rgb_to_grayscale(inputs)
    z = self.encoder(gray)

    if self.is_parallel_loss:
      enc_logits = self.parallel_dense(z)
      enc_logits = tf.expand_dims(enc_logits, axis=-2)

    dec_logits = self.decoder(inputs, z, training=training)
    if self.is_parallel_loss:
      return dec_logits, {'encoder_logits': enc_logits}
    return dec_logits, {}

  def decoder(self, inputs, z, training):
    """Decodes grayscale representation and masked colors into logits."""
    # (H, W, 512) preprocessing.
    # quantize to 3 bits.
    labels = base_utils.convert_bits(inputs, n_bits_in=8, n_bits_out=3)

    # bin each channel triplet -> (H, W, 3) with 8 possible symbols
    # (H, W, 512)
    labels = base_utils.labels_to_bins(labels, self.num_symbols_per_channel)

    # (H, W) with 512 symbols to (H, W, 512)
    labels = tf.one_hot(labels, depth=self.num_symbols)

    h_dec = self.pixel_embed_layer(labels)
    h_upper = self.outer_decoder((h_dec, z), training=training)
    h_inner = self.inner_decoder((h_dec, h_upper, z), training=training)

    activations = self.final_norm(h_inner)
    logits = self.final_dense(activations)
    return tf.expand_dims(logits, axis=-2)

  def image_loss(self, logits, labels):
    """Cross-entropy between the logits and labels."""
    height, width = labels.shape[1:3]
    logits = tf.squeeze(logits, axis=-2)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(loss, axis=0)
    loss = base_utils.nats_to_bits(tf.reduce_sum(loss))
    return loss / (height * width)

  def loss(self, targets, logits, train_config, training, aux_output=None):
    """Converts targets to coarse colors and computes log-likelihood."""
    downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    if downsample:
      labels = targets['targets_%d' % downsample_res]
    else:
      labels = targets['targets']

    if aux_output is None:
      aux_output = {}

    # quantize labels.
    labels = base_utils.convert_bits(labels, n_bits_in=8, n_bits_out=3)

    # bin each channel triplet.
    labels = base_utils.labels_to_bins(labels, self.num_symbols_per_channel)

    loss = self.image_loss(logits, labels)
    enc_logits = aux_output.get('encoder_logits')
    if enc_logits is None:
      return loss, {}

    enc_loss = self.image_loss(enc_logits, labels)
    return loss, {'encoder': enc_loss}

  def get_logits(self, inputs_dict, train_config, training):
    is_downsample = train_config.get('downsample', False)
    downsample_res = train_config.get('downsample_res', 64)
    if is_downsample:
      inputs = inputs_dict['targets_%d' % downsample_res]
    else:
      inputs = inputs_dict['targets']
    return self(inputs=inputs, training=training)

  def sample(self, gray_cond, mode='argmax'):
    output = {}

    z_gray = self.encoder(gray_cond, training=False)
    if self.is_parallel_loss:
      z_logits = self.parallel_dense(z_gray)
      parallel_image = tf.argmax(z_logits, axis=-1, output_type=tf.int32)
      parallel_image = self.post_process_image(parallel_image)

      output['parallel'] = parallel_image

    image, proba = self.autoregressive_sample(z_gray=z_gray, mode=mode)
    output['auto_%s' % mode] = image
    output['proba'] = proba
    return output

  def autoregressive_sample(self, z_gray, mode='sample'):
    """Generates pixel-by-pixel.

    1. The encoder is run once per-channel.
    2. The outer decoder is run once per-row.
    3. the inner decoder is run once per-pixel.

    The context from the encoder and outer decoder conditions the
    inner decoder. The inner decoder then generates a row, one pixel at a time.

    After generating all pixels in a row, the outer decoder is run to recompute
    context. This condtions the inner decoder, which then generates the next
    row, pixel-by-pixel.

    Args:
      z_gray: grayscale image.
      mode: sample or argmax.

    Returns:
      image: coarse image of shape (B, H, W)
      image_proba: probalities, shape (B, H, W, 512)
    """
    num_filters = self.config.hidden_size
    batch_size, height, width = z_gray.shape[:3]

    # channel_cache[i, j] stores the pixel embedding for row i and col j.
    canvas_shape = (batch_size, height, width, num_filters)
    channel_cache = coltran_layers.Cache(canvas_shape=(height, width))
    init_channel = tf.zeros(shape=canvas_shape)
    init_ind = tf.stack([0, 0])
    channel_cache(inputs=(init_channel, init_ind))

    # upper_context[row_ind] stores context from all previously generated rows.
    upper_context = tf.zeros(shape=canvas_shape)

    # row_cache[0, j] stores the pixel embedding for the column j of the row
    # under generation. After every row is generated, this is rewritten.
    row_cache = coltran_layers.Cache(canvas_shape=(1, width))
    init_row = tf.zeros(shape=(batch_size, 1, width, num_filters))
    row_cache(inputs=(init_row, init_ind))

    pixel_samples, pixel_probas = [], []

    for row in range(height):
      row_cond_channel = tf.expand_dims(z_gray[:, row], axis=1)
      row_cond_upper = tf.expand_dims(upper_context[:, row], axis=1)
      row_cache.reset()

      gen_row, proba_row = [], []
      for col in range(width):

        inner_input = (row_cache.cache, row_cond_upper, row_cond_channel)
        # computes output activations at col.
        activations = self.inner_decoder(inner_input, row_ind=row,
                                         training=False)

        pixel_sample, pixel_embed, pixel_proba = self.act_logit_sample_embed(
            activations, col, mode=mode)
        proba_row.append(pixel_proba)
        gen_row.append(pixel_sample)

        # row_cache[:, col] = pixel_embed
        row_cache(inputs=(pixel_embed, tf.stack([0, col])))

        # channel_cache[row, col] = pixel_embed
        channel_cache(inputs=(pixel_embed, tf.stack([row, col])))

      gen_row = tf.stack(gen_row, axis=-1)
      pixel_samples.append(gen_row)
      pixel_probas.append(tf.stack(proba_row, axis=1))

      # after a row is generated, recomputes the context for the next row.
      # upper_context[row] = self_attention(channel_cache[:row_index])
      upper_context = self.outer_decoder(
          inputs=(channel_cache.cache, z_gray), training=False)

    image = tf.stack(pixel_samples, axis=1)
    image = self.post_process_image(image)

    image_proba = tf.stack(pixel_probas, axis=1)
    return image, image_proba

  def act_logit_sample_embed(self, activations, col_ind, mode='sample'):
    """Converts activations[col_ind] to the output pixel.

    Activation -> Logit -> Sample -> Embedding.

    Args:
      activations: 5-D Tensor, shape=(batch_size, 1, width, hidden_size)
      col_ind: integer.
      mode: 'sample' or 'argmax'
    Returns:
      pixel_sample: 1-D Tensor, shape=(batch_size, 1, 1)
      pixel_embed: 4-D Tensor, shape=(batch_size, 1, 1, hidden_size)
      pixel_proba: 4-D Tensor, shape=(batch_size, 1, 512)
    """
    batch_size = activations.shape[0]
    pixel_activation = tf.expand_dims(activations[:, :, col_ind], axis=-2)
    pixel_logits = self.final_dense(self.final_norm(pixel_activation))
    pixel_logits = tf.squeeze(pixel_logits, axis=[1, 2])
    pixel_proba = tf.nn.softmax(pixel_logits, axis=-1)

    if mode == 'sample':
      pixel_sample = tf.random.categorical(
          pixel_logits, num_samples=1, dtype=tf.int32)
      pixel_sample = tf.squeeze(pixel_sample, axis=-1)
    elif mode == 'argmax':
      pixel_sample = tf.argmax(pixel_logits, axis=-1, output_type=tf.int32)

    pixel_sample_expand = tf.reshape(pixel_sample, [batch_size, 1, 1])
    pixel_one_hot = tf.one_hot(pixel_sample_expand, depth=self.num_symbols)
    pixel_embed = self.pixel_embed_layer(pixel_one_hot)
    return pixel_sample, pixel_embed, pixel_proba

  def post_process_image(self, image):
    """Post process image of size (H, W, 512) to a coarse RGB image."""
    image = base_utils.bins_to_labels(
        image, num_symbols_per_channel=self.num_symbols_per_channel)
    image = base_utils.convert_bits(image, n_bits_in=3, n_bits_out=8)
    image = tf.cast(image, dtype=tf.uint8)
    return image
