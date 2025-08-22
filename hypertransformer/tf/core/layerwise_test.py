# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for `layerwise.py`."""

import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import common_ht
from hypertransformer.tf.core import layerwise
from hypertransformer.tf.core import layerwise_defs  # pylint:disable=unused-import


def make_layerwise_model_config():
  """Makes 'layerwise' model config."""
  return common_ht.LayerwiseModelConfig()


class LayerwiseTest(tf.test.TestCase):

  def test_number_of_trained_cnn_layers_param_should_give_trained_weights(self):
    """Tests the layerswise model with both generated and trained weights."""
    tf.reset_default_graph()
    model_config = make_layerwise_model_config()
    model_config.number_of_trained_cnn_layers = 1
    model = layerwise.build_model(
        model_config.cnn_model_name, model_config=model_config)
    images = tf.random.normal((100, 28, 28, 1))
    labels = tf.random.uniform((100,),
                               minval=0, maxval=10,
                               dtype=tf.dtypes.int32)
    weights = model.train(images, labels)
    self.assertIsNone(weights.weight_blocks[0])
    for weight_block in weights.weight_blocks[1:]:
      self.assertIsNotNone(weight_block)
    model.evaluate(images, weight_blocks=weights)
    self.assertIsInstance(model.layers[0].conv.weights[0], tf.Variable,
                          'First layer is trained directly and should be a'
                          'Variable.')
    self.assertIsInstance(model.layers[1].conv.weights[0], tf.Tensor,
                          'All other layers except for the first are computed'
                          'from a Transformer and should be Tensors.')

  def test_negative_number_of_trained_cnn_layers_param_trains_last_layers(self):
    """Tests the layerswise model with both generated and trained weights."""
    tf.reset_default_graph()
    model_config = make_layerwise_model_config()
    model_config.number_of_trained_cnn_layers = -1
    model = layerwise.build_model(
        model_config.cnn_model_name, model_config=model_config)
    images = tf.random.normal((100, 28, 28, 1))
    labels = tf.random.uniform((100,),
                               minval=0, maxval=10,
                               dtype=tf.dtypes.int32)
    weights = model.train(images, labels)
    self.assertIsNone(weights.weight_blocks[-2])
    for weight_block in weights.weight_blocks[:-2]:
      self.assertIsNotNone(weight_block)
    model.evaluate(images, weight_blocks=weights)
    self.assertIsInstance(model.layers[-2].conv.weights[0], tf.Variable,
                          'Last layer before the head is trained directly and '
                          'should be a Variable.')
    self.assertIsInstance(model.layers[0].conv.weights[0], tf.Tensor,
                          'All other layers except for the last are computed'
                          'from a Transformer and should be Tensors.')

  def test_layer_with_activation_after_bn_different_activation_before_bn(self):
    """Tests the option to use activation before or after batchnorm."""
    tf.reset_default_graph()
    model_config = make_layerwise_model_config()
    act_fn = tf.ones_like
    layer_act_after = layerwise.ConvLayer(
        name='test_layer_activation_after',
        model_config=model_config,
        act_fn=act_fn,
        act_after_bn=True)
    layer_act_before = layerwise.ConvLayer(
        name='test_layer_activation_before',
        model_config=model_config,
        act_fn=act_fn,
        act_after_bn=False)
    images = tf.random.normal((100, 28, 28, 3))
    out_after = layer_act_after(images)
    out_before = layer_act_before(images)

    sess = tf.InteractiveSession()
    sess.run(tf.initializers.global_variables())
    self.assertAllEqual(out_after, tf.ones_like(out_after),
                        'When evaluating layerwise.ConvLayer activation after'
                        'BatchNorm was not computed properly.')
    self.assertAllEqual(out_before, tf.zeros_like(out_before),
                        'When evaluating layerwise.ConvLayer activation before'
                        'BatchNorm was not computed properly.')

if __name__ == '__main__':
  tf.test.main()
