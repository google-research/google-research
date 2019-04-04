# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from norml import networks


class NetworksTest(tf.test.TestCase):

  def test_serialization_deserialization(self):
    np.random.seed(12345)
    with tf.Session() as sess:
      dim_in = np.random.randint(1, 100)
      dim_out = np.random.randint(1, 100)
      num_layers = np.random.randint(1, 10)
      layer_sizes = np.random.randint(1, 100, num_layers)
      gen = networks.FullyConnectedNetworkGenerator(dim_in, dim_out,
                                                    layer_sizes)
      weights = gen.construct_network_weights()
      init = tf.global_variables_initializer()
      sess.run(init)

      # Serialize weights to a 1-d numpy array
      serialized_weights = networks.serialize_weights(sess, weights)
      self.assertEqual(
          len(serialized_weights.shape),
          1,
          msg='Serialized weights should have dimension 1.')

      expected_length = 0
      for name in weights:
        expected_length += np.prod(weights[name].shape)
      self.assertEqual(
          expected_length, serialized_weights.shape[0],
          'Serialized weights should have the same length as original weights.')

      # Deserialize and compare weights
      deserialized_weights = networks.deserialize_weights(
          weights, serialized_weights)
      for name in weights:
        original_weight = weights[name].eval(session=sess)
        np.testing.assert_array_equal(
            original_weight, deserialized_weights[name],
            'Weights should be exactly the same before/after serialization')

  def test_fully_connected(self):
    np.random.seed(12345)
    with self.session() as sess:
      for _ in range(10):
        dim_in = np.random.randint(1, 100)
        dim_out = np.random.randint(1, 100)
        inp = tf.placeholder(tf.float32, shape=(None, dim_in), name='input')
        for num_layers in [1, 2, 10]:
          layer_sizes = np.random.randint(1, 100, num_layers)
          gen = networks.FullyConnectedNetworkGenerator(dim_in, dim_out,
                                                        layer_sizes)
          weights = gen.construct_network_weights()
          network = gen.construct_network(inp, weights)

          init = tf.global_variables_initializer()
          sess.run(init)
          test_out = sess.run(network, {inp: np.random.randn(100, dim_in)})
          self.assertFalse(np.isnan(test_out).any())

  def test_linear_network(self):
    np.random.seed(12345)
    with self.session() as sess:
      for _ in range(10):
        dim_in = np.random.randint(1, 100)
        dim_out = np.random.randint(1, 100)
        inp = tf.placeholder(tf.float32, shape=(None, dim_in), name='input')
        gen = networks.LinearNetworkGenerator(dim_in, dim_out)
        network = gen.construct_network(inp, gen.construct_network_weights())

        sess.run(tf.global_variables_initializer())
        test_inp = np.random.randn(100, dim_in)
        test_out = sess.run(network, {inp: test_inp})
        test_out_2 = sess.run(network, {inp: test_inp * 2})
        test_out_4 = sess.run(network, {inp: test_inp * 4})
        self.assertNotIn(np.nan, test_out)
        self.assertNotIn(np.nan, test_out_2)
        self.assertNotIn(np.nan, test_out_4)
        # Verify that the network is indeed linear.
        error = test_out_4 - test_out_2 - 2 * (test_out_2 - test_out)
        self.assertAlmostEqual(np.mean(np.abs(error)), 0)


if __name__ == '__main__':
  tf.test.main()
