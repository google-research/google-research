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
"""Tests for core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging
from absl.testing import parameterized
from ml_collections import ConfigDict
import numpy as np
import tensorflow as tf
from coltran.models import core


def get_num_variables(model):
  var_shapes = [np.prod(variable.shape) for variable in model.variables]
  return np.sum(var_shapes)


cond_hparams = itertools.product(["shift", "affine"],
                                 [True, False],
                                 [True, False],
                                 [True, False])

new_hparams = []
for hparams in cond_hparams:
  new_args = "_".join([str(hparam) for hparam in hparams])
  new_hparams.append((new_args, *hparams))


class ColTranComponentsTest(tf.test.TestCase, parameterized.TestCase):

  def get_config(self):
    config = ConfigDict()
    config.hidden_size = 256
    config.ff_size = 256
    config.image_bit_depth = 5
    config.num_symbols = 32
    config.num_heads = 4
    config.resolution = [8, 8]
    config.num_outer_layers = 1
    config.num_inner_layers = 3
    config.num_encoder_layers = 1
    config.batch_size = 2
    config.skip = True

    config.cond_mlp = "affine_dense"
    config.cond_mlp_act = "identity"

    config.cond_ln = True
    config.cond_ln_act = "tanh"
    config.cond_ln_seq = "cs"
    config.cond_ln_sp_ave = "learnable"
    config.cond_ln_init = "glorot_uniform"

    config.cond_att_act = "identity"
    config.cond_att_scale = True
    config.cond_att_k = True
    config.cond_att_q = True
    config.cond_att_v = True
    return config

  def test_grayscale_encoder(self):
    config = self.get_config()
    inputs = tf.random.uniform(shape=(2, 32, 32, 3), minval=0, maxval=256,
                               dtype=tf.int32)
    gray = tf.image.rgb_to_grayscale(inputs)
    encoder = core.GrayScaleEncoder(config)
    output = encoder(gray)
    self.assertEqual(output.shape, (2, 32, 32, 256))

  @parameterized.named_parameters(*new_hparams)
  def test_inner_decoder(self, cond_mlp, cond_ln, cond_att_q, cond_att_scale):
    embeddings = tf.random.uniform(shape=(2, 8, 8, 256))
    channel_context = tf.random.uniform(shape=(2, 8, 8, 256))
    upper_context = tf.random.uniform(shape=(2, 8, 8, 256))
    config = self.get_config()
    config.cond_mlp = cond_mlp
    config.cond_ln = cond_ln
    config.cond_att_q = cond_att_q
    config.cond_att_scale = cond_att_scale

    model = core.InnerDecoder(config=config)
    output = model(inputs=(embeddings, upper_context, channel_context))
    num_vars = get_num_variables(model)
    logging.info(num_vars)
    self.assertEqual(output.shape, (2, 8, 8, 256))

  @parameterized.named_parameters(*new_hparams)
  def test_outer_decoder(self, cond_mlp, cond_ln, cond_att_q, cond_att_scale):
    embeddings = tf.random.uniform(shape=(2, 8, 8, 256))
    channel_context = tf.random.uniform(shape=(2, 8, 8, 256))
    config = self.get_config()
    config.cond_mlp = cond_mlp
    config.cond_ln = cond_ln
    config.cond_att_q = cond_att_q
    config.cond_att_scale = cond_att_scale

    model = core.OuterDecoder(config=config)
    num_vars = get_num_variables(model)
    logging.info(num_vars)
    upper_context = model(inputs=(embeddings, channel_context))
    upper_context_np = upper_context.numpy()

    # the first row slice should have zero context since both the present
    # and future are effectively masked.
    self.assertTrue(np.allclose(upper_context_np[:, 0], 0.0))
    self.assertEqual(upper_context_np.shape, (2, 8, 8, 256))


if __name__ == "__main__":
  tf.test.main()
