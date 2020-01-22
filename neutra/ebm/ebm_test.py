# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# python3
"""Tests EBM."""

import numpy as np
import tensorflow.compat.v2 as tf

from neutra.ebm import train_ebm

tf.enable_v2_behavior()

N_CH = 3  # Number of channels
N_WH = 32  # Width of the images.


class EBMTest(tf.test.TestCase):

  def test_u_valid(self):
    """Tests that we can initialize U without error."""
    num_samples = 16
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    u = train_ebm.EbmConv(anchor_size=1)
    energy_x = u(x)
    # This should have a higher energy due to the quadratic prior.
    energy_x_far = u(x + tf.ones_like(x))
    self.assertTrue(np.all(energy_x < energy_x_far))

  def test_train_q_fwd_kl(self):
    """Verify that train_q_fwd_kl doesn't raise any exceptions."""
    num_samples = 16
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    q = train_ebm.MeanFieldGaussianQ()
    opt = tf.optimizers.Adam()
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=13)
    loss = train_ebm.train_q_fwd_kl(q, x, opt)
    self.assertTrue(np.all(np.isfinite(loss)))

  def test_train_q_rev_kl(self):
    """Verify that train_q_rev_kl doesn't raise any exceptions."""
    q = train_ebm.MeanFieldGaussianQ()
    u = lambda x: tf.reduce_sum(tf.square(x), axis=[1, 2, 3])
    opt = tf.optimizers.Adam()
    loss, entropy = train_ebm.train_q_rev_kl(q, u, opt)
    self.assertTrue(np.all(np.isfinite(loss)))
    self.assertTrue(np.all(np.isfinite(entropy)))

  def test_train_q_rev_kl_mle(self):
    """Verify that train_q_rev_kl_mle doesn't raise any exceptions."""
    num_samples = 16
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    q = train_ebm.MeanFieldGaussianQ()
    u = lambda x: tf.reduce_sum(tf.square(x), axis=[1, 2, 3])
    opt = tf.optimizers.Adam()
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    (loss, entropy, neg_e_q, mle_loss, grads_ebm_norm,
     grads_mle_norm) = train_ebm.train_q_rev_kl_mle(q, u, x, 1., opt)
    self.assertTrue(np.all(np.isfinite(loss)))
    self.assertTrue(np.all(np.isfinite(entropy)))
    self.assertTrue(np.all(np.isfinite(neg_e_q)))
    self.assertTrue(np.all(np.isfinite(mle_loss)))
    self.assertTrue(np.all(np.isfinite(grads_ebm_norm)))
    self.assertTrue(np.all(np.isfinite(grads_mle_norm)))

  def test_train_q_mle(self):
    """Verify that train_q_mle doesn't raise any exceptions."""
    num_samples = 16
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    q = train_ebm.MeanFieldGaussianQ()
    opt = tf.optimizers.Adam()
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=13)
    loss = train_ebm.train_q_mle(q, x, opt)
    self.assertTrue(np.all(np.isfinite(loss)))

  def test_train_p(self):
    """Verify that train_p doesn't raise any exceptions."""
    num_samples = 16
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    q = train_ebm.MeanFieldGaussianQ()
    u = train_ebm.EbmConv(anchor_size=1)
    opt = tf.optimizers.Adam()
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=13)
    (x_neg_q, x_neg_p, p_accept, step_size, pos_e, pos_e_updated, neg_e_q,
     neg_e_p, neg_e_p_updated) = train_ebm.train_p(q, u, x, 0.1, opt)
    self.assertTrue(np.all(np.isfinite(x_neg_q)))
    self.assertTrue(np.all(np.isfinite(x_neg_p)))
    self.assertTrue(np.all(np.isfinite(p_accept)))
    self.assertTrue(np.all(np.isfinite(step_size)))
    self.assertTrue(np.all(np.isfinite(pos_e)))
    self.assertTrue(np.all(np.isfinite(pos_e_updated)))
    self.assertTrue(np.all(np.isfinite(neg_e_q)))
    self.assertTrue(np.all(np.isfinite(neg_e_p)))
    self.assertTrue(np.all(np.isfinite(neg_e_p_updated)))

  def test_train_p_mh(self):
    """Verify that train_p doesn't raise any exceptions."""
    num_samples = 16
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=12)
    q = train_ebm.MeanFieldGaussianQ()
    u = train_ebm.EbmConv(anchor_size=1)
    opt = tf.optimizers.Adam()
    x = tf.random.normal(
        [num_samples, N_WH, N_WH, N_CH], seed=13)
    (x_neg_q, x_neg_p, p_accept, step_size, pos_e, pos_e_updated, neg_e_q,
     neg_e_p, neg_e_p_updated) = train_ebm.train_p_mh(q, u, x, 0.1, opt)
    self.assertTrue(np.all(np.isfinite(x_neg_q)))
    self.assertTrue(np.all(np.isfinite(x_neg_p)))
    self.assertTrue(np.all(np.isfinite(p_accept)))
    self.assertTrue(np.all(np.isfinite(step_size)))
    self.assertTrue(np.all(np.isfinite(pos_e)))
    self.assertTrue(np.all(np.isfinite(pos_e_updated)))
    self.assertTrue(np.all(np.isfinite(neg_e_q)))
    self.assertTrue(np.all(np.isfinite(neg_e_p)))
    self.assertTrue(np.all(np.isfinite(neg_e_p_updated)))


if __name__ == '__main__':
  tf.test.main()
