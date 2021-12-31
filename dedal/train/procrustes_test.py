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

"""Tests for procrustes."""

import tensorflow as tf
import tensorflow_probability as tfp

from dedal.train import losses


def embeddings_chain(batch_size, num_embs, dims, sigma):
  tri_int = tf.cast(num_embs * (num_embs + 1) / 2, dtype=tf.int32)
  lin_op = tfp.math.fill_triangular(tf.ones(tri_int))
  lin_op_repeat = tf.tile(lin_op[tf.newaxis, tf.newaxis, Ellipsis],
                          multiples=[batch_size, dims, 1, 1])
  initial_vec = tf.random.uniform((batch_size, 1, dims), minval=-1, maxval=1)
  noise = sigma * tf.random.normal((batch_size, num_embs - 1, dims))
  initial_and_noise = tf.transpose(tf.concat([initial_vec, noise], axis=1),
                                   (0, 2, 1))
  emb_chain = tf.matmul(lin_op_repeat, initial_and_noise[Ellipsis, tf.newaxis])
  return tf.squeeze(tf.transpose(emb_chain, (0, 2, 1, 3)))


class ProcrustesTest(tf.test.TestCase):

  def setUp(self):
    super(ProcrustesTest, self).setUp()
    tf.random.set_seed(0)

  def test_zeros(self):
    """Tests that two chains equal up to rotation have a zero loss."""
    loss = losses.ProcrustesLoss()
    batch_size = 8
    num_embs = 100
    dims = 17
    sigma = 0.01
    embs_chain = embeddings_chain(batch_size, num_embs, dims, sigma)

    m_random = tf.random.uniform((batch_size, dims, dims))
    _, u_r, _ = tf.linalg.svd(m_random)
    rotation_random = u_r

    embs_rotate = tf.transpose(tf.matmul(rotation_random,
                                         tf.transpose(embs_chain, (0, 2, 1))),
                               (0, 2, 1))

    self.assertAllClose(loss(embs_chain, embs_rotate), 0, atol=1e-5)

  def test_symmetric(self):
    """Tests that the loss is symmetric."""
    loss = losses.ProcrustesLoss()
    batch_size = 16
    num_embs = 50
    dims = 53
    sigma = 1e-4
    embs_chain_1 = embeddings_chain(batch_size, num_embs, dims, sigma)
    embs_chain_2 = embeddings_chain(batch_size, num_embs, dims, sigma)
    self.assertAllClose(loss(embs_chain_1, embs_chain_2),
                        loss(embs_chain_1, embs_chain_2), atol=1e-5)

  def scaling_norm_test(self):
    """Tests that the loss has some proper scaling properties."""
    loss = losses.ProcrustesLoss()
    batch_size = 8
    num_embs = 100
    dims = 17
    sigma = 0.01
    eps = 1e-3
    embs_true = embeddings_chain(batch_size, num_embs, dims, sigma)
    embs_pred = embs_true + eps * tf.random.normal(embs_true.shape)
    delta_norm_sq = tf.linalg.norm(embs_true, embs_pred) ** 2
    loss_value_sq = (loss(embs_pred, embs_true) ** 2) * batch_size
    self.assertGreater(delta_norm_sq, loss_value_sq)
    self.assertNear(loss_value_sq, (eps ** 2) * batch_size * num_embs * dims)


if __name__ == '__main__':
  tf.test.main()
