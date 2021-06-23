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

"""Tests model architecture functions."""

import tensorflow as tf

from poem.cv_mim import models


class ModelsTest(tf.test.TestCase):

  def test_simple_point_embedder_shapes(self):
    input_features = tf.zeros([4, 6], tf.float32)
    model = models.SimpleModel(
        output_shape=(4, 3),
        embedder=models.TYPE_EMBEDDER_POINT,
        hidden_dim=1024,
        num_residual_linear_blocks=2,
        num_layers_per_block=2)

    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 4, 3])
    self.assertAllEqual(outputs[1]['flatten'].shape, [4, 6])
    self.assertAllEqual(outputs[1]['fc0'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['res_fcs1'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['res_fcs2'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['embedder'].shape, [4, 4, 3])

  def test_simple_point_embedder_forward_pass(self):
    input_features = tf.constant([[1.0, 2.0, 3.0]])
    model = models.SimpleModel(
        output_shape=(4,),
        embedder=models.TYPE_EMBEDDER_GAUSSIAN,
        hidden_dim=2,
        num_residual_linear_blocks=3,
        num_layers_per_block=2,
        use_batch_norm=False,
        weight_initializer='ones')

    outputs = model(input_features)
    self.assertAllClose(outputs[0], [[1937.0, 1937.0, 1937.0, 1937.0]])

  def test_simple_gaussian_embedder_shapes(self):
    input_features = tf.zeros([4, 6], tf.float32)
    model = models.SimpleModel(
        output_shape=(4,),
        embedder=models.TYPE_EMBEDDER_GAUSSIAN,
        hidden_dim=1024,
        num_residual_linear_blocks=2,
        num_layers_per_block=2)

    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 4])
    self.assertAllEqual(outputs[1]['flatten'].shape, [4, 6])
    self.assertAllEqual(outputs[1]['fc0'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['res_fcs1'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['res_fcs2'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['embedder'].shape, [4, 4])

  def test_simple_gaussian_embedder(self):
    input_features = tf.ones([1, 6], tf.float32)
    model = models.SimpleModel(
        output_shape=(1,),
        embedder=models.TYPE_EMBEDDER_GAUSSIAN,
        hidden_dim=1024,
        num_residual_linear_blocks=2,
        num_layers_per_block=2,
        weight_initializer='ones')

    tf.random.set_seed(0)
    outputs_x = model(input_features, training=True)
    outputs_y = model(input_features, training=True)
    self.assertNotAllEqual(outputs_x[0], outputs_y[0])

    outputs_x = model(input_features, training=False)
    outputs_y = model(input_features, training=False)
    self.assertAllEqual(outputs_x[0], outputs_y[0])

  def test_semgcn_shapes(self):
    input_features = tf.zeros([4, 8, 2], tf.float32)
    model = models.GCN(
        output_dim=3,
        affinity_matrix=tf.ones(shape=(8, 8)),
        gconv_class=models.SemGraphConv,
        hidden_dim=128,
        num_residual_gconv_blocks=2,
        num_layers_per_block=2)

    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 8, 3])
    self.assertAllEqual(outputs[1]['gconv0'].shape, [4, 8, 128])
    self.assertAllEqual(outputs[1]['res_gconvs1'].shape, [4, 8, 128])
    self.assertAllEqual(outputs[1]['res_gconvs2'].shape, [4, 8, 128])
    self.assertAllEqual(outputs[1]['gconv3'].shape, [4, 8, 3])

  def test_semgcn_forward_pass(self):
    input_features = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    input_features = tf.reshape(input_features, [1, 3, 1])
    model = models.GCN(
        output_dim=1,
        affinity_matrix=tf.ones(shape=(3, 3)),
        gconv_class=models.SemGraphConv,
        hidden_dim=2,
        num_residual_gconv_blocks=2,
        num_layers_per_block=2,
        use_batch_norm=False,
        dropout_rate=0.0,
        kernel_initializer='ones',
        bias_initializer='zeros')

    outputs, _ = model(input_features)
    self.assertAllEqual(outputs.shape, [1, 3, 1])
    self.assertAllClose(outputs, tf.reshape([100.0, 100.0, 100.0], [1, 3, 1]))

  def test_likelihood_estimator_shapes(self):
    input_features = tf.zeros([4, 6], tf.float32)
    model = models.LikelihoodEstimator(output_dim=2)

    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 2])
    self.assertAllEqual(outputs[1].shape, [4, 2])


if __name__ == '__main__':
  tf.test.main()
