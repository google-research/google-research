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

"""Tests model architecture functions."""

import tensorflow.compat.v1 as tf

from poem.core import common
from poem.core import models
tf.disable_v2_behavior()


class ModelsTest(tf.test.TestCase):

  def test_simple_model_shapes(self):
    # Shape = [4, 2, 3].
    input_features = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                  [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                                  [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]])
    output_sizes = {'a': 8, 'b': [4, 3]}
    outputs, activations = models.simple_model(
        input_features,
        output_sizes,
        sequential_inputs=False,
        is_training=True,
        num_bottleneck_nodes=16)

    expected_global_variable_shapes = {
        'SimpleModel/InputFC/Linear/weight:0': ([3, 1024]),
        'SimpleModel/InputFC/Linear/bias:0': ([1024]),
        'SimpleModel/InputFC/BatchNorm/gamma:0': ([1024]),
        'SimpleModel/InputFC/BatchNorm/beta:0': ([1024]),
        'SimpleModel/InputFC/BatchNorm/moving_mean:0': ([1024]),
        'SimpleModel/InputFC/BatchNorm/moving_variance:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_0/Linear/weight:0': ([1024,
                                                                    1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_0/Linear/bias:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_0/BatchNorm/gamma:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_0/BatchNorm/beta:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_0/BatchNorm/moving_mean:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_0/BatchNorm/moving_variance:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_1/Linear/weight:0': ([1024,
                                                                    1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_1/Linear/bias:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_1/BatchNorm/gamma:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_1/BatchNorm/beta:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_1/BatchNorm/moving_mean:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_0/FC_1/BatchNorm/moving_variance:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_0/Linear/weight:0': ([1024,
                                                                    1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_0/Linear/bias:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_0/BatchNorm/gamma:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_0/BatchNorm/beta:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_0/BatchNorm/moving_mean:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_0/BatchNorm/moving_variance:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_1/Linear/weight:0': ([1024,
                                                                    1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_1/Linear/bias:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_1/BatchNorm/gamma:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_1/BatchNorm/beta:0': ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_1/BatchNorm/moving_mean:0':
            ([1024]),
        'SimpleModel/FullyConnectedBlock_1/FC_1/BatchNorm/moving_variance:0':
            ([1024]),
        'SimpleModel/BottleneckLogits/weight:0': ([1024, 16]),
        'SimpleModel/BottleneckLogits/bias:0': ([16]),
        'SimpleModel/OutputLogits/a/weight:0': ([16, 8]),
        'SimpleModel/OutputLogits/a/bias:0': ([8]),
        'SimpleModel/OutputLogits/b/weight:0': ([16, 12]),
        'SimpleModel/OutputLogits/b/bias:0': ([12]),
    }

    self.assertDictEqual(
        {var.name: var.shape.as_list() for var in tf.global_variables()},
        expected_global_variable_shapes)
    self.assertCountEqual(outputs.keys(), ['a', 'b'])
    self.assertAllEqual(outputs['a'].shape.as_list(), [4, 2, 8])
    self.assertAllEqual(outputs['b'].shape.as_list(), [4, 2, 4, 3])
    self.assertCountEqual(activations.keys(),
                          ['base_activations', 'bottleneck_activations'])
    self.assertAllEqual(activations['base_activations'].shape.as_list(),
                        [4, 2, 1024])
    self.assertAllEqual(activations['bottleneck_activations'].shape.as_list(),
                        [4, 2, 16])

  def test_simple_model_forward_pass(self):
    input_features = tf.constant([[1.0, 2.0, 3.0]])
    output_sizes = {'a': 4}
    outputs, activations = models.simple_model(
        input_features,
        output_sizes,
        sequential_inputs=False,
        is_training=True,
        num_hidden_nodes=2,
        weight_initializer=tf.initializers.ones(),
        bias_initializer=tf.initializers.zeros(),
        weight_max_norm=0.0,
        use_batch_norm=False,
        dropout_rate=0.0,
        num_fcs_per_block=2,
        num_fc_blocks=3)

    with self.session() as sess:
      sess.run(tf.initializers.global_variables())
      outputs_result, activations_result = sess.run([outputs, activations])

    self.assertCountEqual(outputs_result.keys(), ['a'])
    self.assertAllClose(outputs_result['a'], [[1500.0, 1500.0, 1500.0, 1500.0]])
    self.assertCountEqual(activations_result.keys(), ['base_activations'])
    self.assertAllClose(activations_result['base_activations'],
                        [[750.0, 750.0]])

  def test_get_simple_model(self):
    input_features = tf.constant([[1.0, 2.0, 3.0]])
    output_sizes = {'a': 4}
    model_fn = models.get_model(
        base_model_type=common.BASE_MODEL_TYPE_SIMPLE,
        is_training=True,
        num_hidden_nodes=2,
        weight_initializer=tf.initializers.ones(),
        bias_initializer=tf.initializers.zeros(),
        weight_max_norm=0.0,
        use_batch_norm=False,
        dropout_rate=0.0,
        num_fcs_per_block=2,
        num_fc_blocks=3)
    outputs, activations = model_fn(input_features, output_sizes)

    with self.session() as sess:
      sess.run(tf.initializers.global_variables())
      outputs_result, activations_result = sess.run([outputs, activations])

    self.assertCountEqual(outputs_result.keys(), ['a'])
    self.assertAllClose(outputs_result['a'], [[1500.0, 1500.0, 1500.0, 1500.0]])
    self.assertCountEqual(activations_result.keys(), ['base_activations'])
    self.assertAllClose(activations_result['base_activations'],
                        [[750.0, 750.0]])

  def test_get_simple_point_embedder(self):
    # Shape = [4, 2, 3].
    input_features = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                  [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                                  [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]])
    embedder_fn = models.get_embedder(
        base_model_type=common.BASE_MODEL_TYPE_SIMPLE,
        embedding_type=common.EMBEDDING_TYPE_POINT,
        num_embedding_components=3,
        embedding_size=16,
        is_training=True)
    outputs, activations = embedder_fn(input_features)

    self.assertCountEqual(outputs.keys(), [common.KEY_EMBEDDING_MEANS])
    self.assertAllEqual(outputs[common.KEY_EMBEDDING_MEANS].shape.as_list(),
                        [4, 2, 3, 16])
    self.assertCountEqual(activations.keys(), ['base_activations'])
    self.assertAllEqual(activations['base_activations'].shape.as_list(),
                        [4, 2, 1024])

  def test_get_simple_gaussian_embedder(self):
    # Shape = [4, 2, 3].
    input_features = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                  [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                                  [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]])
    embedder_fn = models.get_embedder(
        base_model_type=common.BASE_MODEL_TYPE_SIMPLE,
        embedding_type=common.EMBEDDING_TYPE_GAUSSIAN,
        num_embedding_components=3,
        embedding_size=16,
        num_embedding_samples=32,
        is_training=True,
        weight_max_norm=0.0)
    outputs, activations = embedder_fn(input_features)

    self.assertCountEqual(outputs.keys(), [
        common.KEY_EMBEDDING_MEANS,
        common.KEY_EMBEDDING_STDDEVS,
        common.KEY_EMBEDDING_SAMPLES,
    ])
    self.assertAllEqual(outputs[common.KEY_EMBEDDING_MEANS].shape.as_list(),
                        [4, 2, 3, 16])
    self.assertAllEqual(outputs[common.KEY_EMBEDDING_STDDEVS].shape.as_list(),
                        [4, 2, 3, 16])
    self.assertAllEqual(outputs[common.KEY_EMBEDDING_SAMPLES].shape.as_list(),
                        [4, 2, 3, 32, 16])
    self.assertCountEqual(activations.keys(), ['base_activations'])
    self.assertAllEqual(activations['base_activations'].shape.as_list(),
                        [4, 2, 1024])


if __name__ == '__main__':
  tf.test.main()
