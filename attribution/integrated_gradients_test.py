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

"""Tests for google_research.attribution.integrated_gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from attribution import integrated_gradients
from tensorflow.contrib import layers as contrib_layers


class AttributionTest(tf.test.TestCase):

  def testAddIntegratedGradientsOps(self):
    with tf.Graph().as_default() as graph:
      var1 = tf.compat.v1.get_variable(
          name='var1', initializer=[[[1., 2., 3.]]])
      input_tensor = tf.placeholder(shape=[None, None, 3], dtype=tf.float32)
      x = tf.multiply(input_tensor, [[[1.]]])
      var1_times_x = tf.multiply(var1, x)
      var2 = tf.compat.v1.get_variable(
          name='var2', initializer=[[4., 5.], [6., 7.], [4., 3.]])
      matmul = tf.einsum('ijk,kl->ijl', var1_times_x, var2)
      output_tensor = tf.reduce_sum(matmul, [1, 2], name='out')
      input_feed_dict = {input_tensor.name: [[[2., 3., 4.], [5., 6., 7.]]]}
      num_evals = tf.placeholder_with_default(
          tf.constant(20, name='num_evals'), shape=())
      attribution_hooks = integrated_gradients.AddIntegratedGradientsOps(
          graph=graph,
          attribution_tensors=[x],
          output_tensor=output_tensor,
          num_evals=num_evals,
          attribution_dims_map={x: [1]},
          zero_baseline_tensors=set([x]))

      with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(
            attribution_hooks['mean_grads'],
            attribution_hooks['create_combined_feed_dict'](input_feed_dict))
        self.assertTupleEqual(result[0].shape, (2, 1))
        self.assertAlmostEqual(result[0][0, 0], 180.)
        self.assertAlmostEqual(result[0][1, 0], 348.)

  def testAddBOWIntegratedGradientsOps(self):
    with tf.Graph().as_default() as graph:
      # pyformat: disable
      embedding_weights = tf.constant([[1., 3., 5.],
                                       [4., 6., 8.],
                                       [4., 5., 4.]])
      batch_size = tf.placeholder_with_default(tf.constant(1, tf.int64), [])
      sparse_ids = tf.SparseTensor(
          [[0, 0], [0, 1], [0, 2]], [2, 0, 2], [batch_size, 3])
      # pyformat: enable
      sparse_embedding = contrib_layers.safe_embedding_lookup_sparse(
          embedding_weights,
          sparse_ids,
          combiner='sum',
          partition_strategy='div')

      vector = tf.constant([1., 2., 4.])
      output_tensor = tf.reduce_sum(vector * sparse_embedding, axis=1)
      embedding_lookup = tf.nn.embedding_lookup(
          embedding_weights,
          tf.sparse_tensor_to_dense(sparse_ids),
          partition_strategy='div')
      bow_attribution_hooks = integrated_gradients.AddBOWIntegratedGradientsOps(
          graph, [embedding_lookup], [sparse_embedding], [], output_tensor)
      with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(bow_attribution_hooks['bow_attributions'])
        self.assertTupleEqual(result[0].shape, (3,))
        # Since the output is a sum of dot products, attributions are simply dot
        # products of the embedding with [1., 2., 4.].
        self.assertAlmostEqual(result[0][0], 30.)
        self.assertAlmostEqual(result[0][1], 27.)
        self.assertAlmostEqual(result[0][2], 30.)

  def testGetEmbeddingLookupList(self):
    with tf.Graph().as_default() as graph:
      # pyformat: disable
      embedding_weights = tf.constant([[1., 2., 3.],
                                       [4., 6., 8.],
                                       [5., 4., 3.]])
      batch_size = tf.placeholder_with_default(tf.constant(1, tf.int64), [])
      sparse_ids1 = tf.SparseTensor(
          [[0, 0], [0, 1]], [2, 0], [batch_size, 2])
      sparse_weights1 = tf.SparseTensor(
          [[0, 0], [0, 1]], [4., 3.], [batch_size, 2])
      sparse_ids2 = tf.SparseTensor(
          [[0, 0], [0, 1], [0, 2]], [1, 1, 2], [batch_size, 3])
      # pyformat: enable
      embedding_lookup_list = integrated_gradients.GetEmbeddingLookupList(
          ['feature1', 'feature2'],
          {'feature1': embedding_weights, 'feature2': embedding_weights},
          {'feature1': sparse_ids1, 'feature2': sparse_ids2},
          sparse_weights={'feature1': sparse_weights1, 'feature2': None},
          combiners={'feature1': 'sqrtn', 'feature2': 'mean'})
      with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(embedding_lookup_list)
        expected = [[[[4., 3.2, 2.4],
                      [0.6, 1.2, 1.8]]],
                    [[[1.33, 2., 2.66],
                      [1.33, 2., 2.66],
                      [1.66, 1.33, 1.]]]]
        self.assertTupleEqual(result[0].shape, (1, 2, 3))
        self.assertTupleEqual(result[1].shape, (1, 3, 3))
        for idx in range(len(expected)):
          for row in range(len(expected[idx][0])):
            for col in range(len(expected[idx][0][row])):
              self.assertAlmostEqual(result[idx][0, row, col],
                                     expected[idx][0][row][col],
                                     delta=0.1)


if __name__ == '__main__':
  tf.test.main()
