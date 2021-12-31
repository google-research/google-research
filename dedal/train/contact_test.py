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

"""Tests for contact-contact related losses."""

import tensorflow as tf
import tensorflow_probability as tfp

from dedal import pairs
from dedal.train import losses


def embeddings_chain(batch_size, num_embs, dims, sigma):
  """Creates a chain of embeddings, each element close to its two neighbours.

  Args:
    batch_size: an int, the size of the batch of chains.
    num_embs: an int, the number of desired embeddings in the chain.
    dims: an int, the ambiant dimension of the embeddings.
    sigma: a float, the variance of the noise between two neighbours.

  Returns:
    a tensor of shape [batch_size, num_embs, dims], a batch of chains, each of
    size num_embs, of embeddings of size dims, such that in each chain
      v_{k+1} = v_{k} + sigma * noise_{k}.
  """
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


def gauss_decrease(inputs, tau=1e-4):
  return tf.math.exp(- (2 * inputs/tau) ** 2)


def gauss_increase(inputs, tau=1e-4):
  return 1 - tf.math.exp(- (inputs/(2 * tau)) ** 2)


class ContactTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)

  def test_zeros(self):
    """Tests that two chains equal up to rotation have the same loss."""
    loss = losses.ContactLoss(no_contact_fun=gauss_decrease,
                              contact_fun=gauss_increase)
    batch_size = 8
    num_embs = 100
    dims = 17
    sigma = 0.01
    threshold = 1e-4
    embs_true = embeddings_chain(batch_size,
                                 num_embs,
                                 dims,
                                 sigma)

    m_random = tf.random.uniform((batch_size, dims, dims))
    _, u_r, _ = tf.linalg.svd(m_random)
    rotation_random = u_r

    embs_rotate = tf.transpose(tf.matmul(rotation_random,
                                         tf.transpose(embs_true, (0, 2, 1))),
                               (0, 2, 1))
    pairw_true = pairs.square_distances(embs_true, embs_true)
    pairw_rotate = pairs.square_distances(embs_rotate, embs_rotate)
    contact_true = tf.cast(pairw_true < threshold, dtype=tf.float32)
    loss_value = loss(contact_true, pairw_true)
    self.assertAllClose(loss_value, 0, atol=1e-4)
    self.assertEqual(loss_value, loss(contact_true, pairw_rotate))

  def test_from_embs(self):
    """Tests that the loss yields the same results from embeddings."""
    loss_embs = losses.ContactLoss(no_contact_fun=gauss_decrease,
                                   contact_fun=gauss_increase,
                                   from_embs=True)
    loss_mat = losses.ContactLoss(no_contact_fun=gauss_decrease,
                                  contact_fun=gauss_increase)
    batch_size = 16
    num_embs = 50
    dims = 53
    sigma = 1e-4
    threshold = 1e-4
    embs_chain_1 = embeddings_chain(batch_size, num_embs, dims, sigma)
    embs_chain_2 = embeddings_chain(batch_size, num_embs, dims, sigma)
    pairw_true = pairs.square_distances(embs_chain_1, embs_chain_1)
    pairw_pred = pairs.square_distances(embs_chain_2, embs_chain_2)
    contact_true = tf.cast(pairw_true < threshold, dtype=tf.float32)
    loss_value_embs = loss_embs(contact_true, embs_chain_2)
    loss_value_mat = loss_mat(contact_true, pairw_pred)
    self.assertAllClose(loss_value_embs,
                        loss_value_mat, atol=1e-5)

if __name__ == '__main__':
  tf.test.main()
