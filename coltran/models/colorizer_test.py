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
"""Tests for colorizer."""
from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
from coltran.models import colorizer


class ColTranCoreTest(tf.test.TestCase):

  def get_config(self, encoder_net='attention'):
    config = ConfigDict()
    config.image_bit_depth = 3
    config.encoder_1x1 = True
    config.resolution = [64, 64]
    config.batch_size = 2
    config.encoder_net = encoder_net
    config.hidden_size = 128
    config.stage = 'decoder'

    config.encoder = ConfigDict()
    config.encoder.dropout = 0.0
    config.encoder.ff_size = 128
    config.encoder.hidden_size = 128
    config.encoder.num_heads = 1
    config.encoder.num_encoder_layers = 1

    config.decoder = ConfigDict()
    config.decoder.ff_size = 128
    config.decoder.hidden_size = 128
    config.decoder.num_heads = 1
    config.decoder.num_outer_layers = 1
    config.decoder.num_inner_layers = 1
    config.decoder.resolution = [64, 64]
    config.decoder.dropout = 0.1
    config.decoder.cond_ln = True
    config.decoder.cond_q = True
    config.decoder.cond_k = True
    config.decoder.cond_v = True
    config.decoder.cond_q = True
    config.decoder.cond_scale = True
    config.decoder.cond_mlp = 'affine'
    return config

  def test_transformer_attention_encoder(self):
    config = self.get_config(encoder_net='attention')
    config.stage = 'encoder_decoder'
    transformer = colorizer.ColTranCore(config=config)
    images = tf.random.uniform(shape=(2, 2, 2, 3), minval=0,
                               maxval=256, dtype=tf.int32)
    logits = transformer(inputs=images, training=True)[0]
    self.assertEqual(logits.shape, (2, 2, 2, 1, 512))

    # batch-size=2
    gray = tf.image.rgb_to_grayscale(images)
    output = transformer.sample(gray, mode='argmax')
    output_np = output['auto_argmax'].numpy()
    proba_np = output['proba'].numpy()
    self.assertEqual(output_np.shape, (2, 2, 2, 3))
    self.assertEqual(proba_np.shape, (2, 2, 2, 512))
    # logging.info(output_np[0, ..., 0])

    # batch-size=1
    output_np_bs_1, proba_np_bs_1 = [], []
    for batch_ind in [0, 1]:
      curr_gray = tf.expand_dims(gray[batch_ind], axis=0)
      curr_out = transformer.sample(curr_gray, mode='argmax')
      curr_out_np = curr_out['auto_argmax'].numpy()
      curr_proba_np = curr_out['proba'].numpy()
      output_np_bs_1.append(curr_out_np)
      proba_np_bs_1.append(curr_proba_np)
    output_np_bs_1 = np.concatenate(output_np_bs_1, axis=0)
    proba_np_bs_1 = np.concatenate(proba_np_bs_1, axis=0)
    self.assertTrue(np.allclose(output_np, output_np_bs_1))
    self.assertTrue(np.allclose(proba_np, proba_np_bs_1))

  def test_transformer_encoder_decoder(self):
    config = self.get_config()
    config.stage = 'encoder_decoder'

    transformer = colorizer.ColTranCore(config=config)
    images = tf.random.uniform(shape=(1, 64, 64, 3), minval=0,
                               maxval=256, dtype=tf.int32)
    logits, enc_logits = transformer(inputs=images, training=True)
    enc_logits = enc_logits['encoder_logits']
    self.assertEqual(enc_logits.shape, (1, 64, 64, 1, 512))
    self.assertEqual(logits.shape, (1, 64, 64, 1, 512))


if __name__ == '__main__':
  tf.test.main()
