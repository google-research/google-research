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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from axial import transformers
from axial import utils

import numpy as np
import tensorflow.compat.v1 as tf


class Transformer2dTest(tf.test.TestCase):
  SEED = 31241
  # These are bad hyperparams to exaggerate differences due to sampling bugs
  MODEL_PARAMS = dict(
      num_embs=256,
      emb_dim=8,
      num_heads=2,
      hdim_factor=4,
      res_init_scale=1.0,
      emb_init_scale=1.0,
      logits_init_scale=1.0,
      num_outer_layers=2,
      num_inner_layers=2)

  def setUp(self):
    super(Transformer2dTest, self).setUp()
    tf.set_random_seed(self.SEED)
    np.random.seed(self.SEED)

  def _run_sampling_test(self, model, x_shape, cond_data):

    x_sym = tf.placeholder(tf.int32, x_shape)
    # sampling noise
    noise_sym = tf.placeholder(tf.float32, list(x_sym.shape) + [model.num_embs])
    # conditioning info
    cond = tf.constant(cond_data, dtype=tf.float32)
    # model graph
    logits_sym = model.compute_logits(x_sym, cond=cond, dropout=0.1)
    del logits_sym
    samples_slow_sym = model.sample_slow(
        noise=noise_sym, cond=cond, dropout=0.0)
    samples_sym = model.sample_fast(noise=noise_sym, cond=cond, dropout=0.0)
    samples_badcond_sym = model.sample_fast(
        noise=noise_sym, cond=None, dropout=0.0)
    for v in tf.trainable_variables():
      print(v.name)

    noise = utils.np_gumbel(
        np.random.RandomState(self.SEED),
        shape=noise_sym.shape,
        temperature=1.0)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print('sampling..')
      samples_a = sess.run(samples_slow_sym, {noise_sym: noise}).squeeze()
      samples_b = sess.run(samples_sym, {noise_sym: noise}).squeeze()
      samples_c = sess.run(samples_badcond_sym, {noise_sym: noise}).squeeze()
      print(samples_a)
      print(samples_b)
      self.assertAllClose(
          samples_a, samples_b, msg='fast/slow sampling do not agree')
      self.assertNotAllClose(
          samples_a, samples_c, msg='samples do not depend on conditioning')
      print('ok!')

  def test_transformer2d_sampling(self):
    """Unit test for fast and slow sampling."""
    bs = 3
    img_shape = [4, 5]
    model = transformers.Transformer2d(
        name='model',
        img_height=img_shape[0],
        img_width=img_shape[1],
        **self.MODEL_PARAMS)
    self._run_sampling_test(
        model,
        x_shape=[bs] + img_shape,
        cond_data=np.random.randn(*([bs] + img_shape + [model.emb_dim])))

  def test_multichannel_transformer2d_sampling(self):
    """Unit test for fast and slow sampling."""
    bs = 3
    img_shape = [4, 5, 7]
    model = transformers.MultiChannelTransformer2d(
        name='model',
        img_height=img_shape[0],
        img_width=img_shape[1],
        img_channels=img_shape[2],
        **self.MODEL_PARAMS)
    self._run_sampling_test(
        model,
        x_shape=[bs] + img_shape,
        cond_data=np.random.randn(*([bs] + img_shape[:2] + [model.emb_dim])))

  def test_transformer3d_sampling(self):
    """Unit test for fast and slow sampling."""
    bs = 3
    img_shape = [4, 5, 7]
    model = transformers.Transformer3d(
        name='model',
        img_height=img_shape[0],
        img_width=img_shape[1],
        img_channels=img_shape[2],
        num_exterior_layers=2,
        **self.MODEL_PARAMS)
    self._run_sampling_test(
        model,
        x_shape=[bs] + img_shape,
        cond_data=np.random.randn(*([bs] + img_shape[:2] + [model.emb_dim])))


if __name__ == '__main__':
  tf.test.main()
