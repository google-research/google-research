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
"""Implementation for SuperCon projection head."""

import inspect

import tensorflow.compat.v1 as tf

from supcon import blocks


class ProjectionHead(tf.layers.Layer):
  """Supervised contrastive projection head.

  This layer implements a stack of dense layers.
  It also offers options for normalizing input and output activations.

  Attributes:
    feature_dims: An iterable indicating the number of units in each layer.
    activation: The activation to apply after each dense
      layer except the final one.
    normalize_output: Whether to normalize the output with euclidean norm.
    kernel_initializer: Initializer for weight matrices.
    bias_initializer: Intializer for biases. The final layer doesn't use a bias.
    use_batch_norm: Whether to include a Batch Normalization layer in between
      each dense layer and the activation function.
    use_batch_norm_beta: If use_batch_norm is True, whether the Batch
      Normalization layers should have a beta (bias) parameter.
    batch_norm_momentum: Momentum for the batchnorm moving average.
    use_global_batch_norm: Whether to use global batch norm, where statistics
      are aggregated across TPU cores, instead of local batch norm, where
      statistics are only computed on TPU core 0. This flag only has impact when
      running on TPU. Distributed GPU or CPU setups always use local batch norm.
    name: Name for the projection head.

  Input:
    Tensor of rank 2 and dtype which is a floating point dtype.
  Output:
    Tensor with the same dtype as input and shape
      (input_shape[0],feature_dims[-1])

  Raises:
    ValueError: If input rank is not 2.
  """

  def __init__(self,
               feature_dims=(2048, 128),
               activation=tf.nn.relu,
               normalize_output=True,
               kernel_initializer=tf.random_normal_initializer(stddev=.01),
               bias_initializer=tf.zeros_initializer(),
               use_batch_norm=False,
               batch_norm_momentum=blocks.BATCH_NORM_MOMENTUM,
               use_batch_norm_beta=False,
               use_global_batch_norm=True,
               name='ProjectionHead',
               **kwargs):
    super(ProjectionHead, self).__init__(name=name, **kwargs)

    self.normalize_output = normalize_output
    self.num_layers = len(feature_dims)

    for layer_idx, layer_dim in enumerate(feature_dims):
      is_last_layer = (layer_idx + 1) == len(feature_dims)
      # We can't just add all layers to a list, since keras.Layer uses
      # __setattr__ to monitor for sublayers that it needs to track, but it
      # doesn't handle lists of sublayers. We use setattr to enable using
      # dynamic variable naming given that the number of sublayers not
      # statically known.
      setattr(
          self, f'dense_{layer_idx}',
          tf.layers.Dense(
              layer_dim,
              activation=None,
              use_bias=not is_last_layer and not use_batch_norm,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer))

      if not is_last_layer:
        if use_batch_norm:
          # Cross-replica TPU BatchNorm.
          setattr(
              self, f'batchnorm_{layer_idx}',
              blocks.batch_norm(
                  use_trainable_beta=use_batch_norm_beta,
                  batch_norm_momentum=batch_norm_momentum,
                  use_global_batch_norm=use_global_batch_norm))

        setattr(self, f'activation_{layer_idx}',
                tf.keras.layers.Activation(activation))

  def call(self, inputs, training=None):
    if inputs.shape.rank != 2:
      raise ValueError(
          f'Input shape {inputs.shape} is expected to have rank 2, but does '
          'not.')
    x = inputs
    for layer_idx in range(self.num_layers):
      for layer_type in ['dense', 'batchnorm', 'activation']:
        if hasattr(self, f'{layer_type}_{layer_idx}'):
          layer = getattr(self, f'{layer_type}_{layer_idx}')
          kwargs = {}
          if 'training' in inspect.getfullargspec(layer.call):
            kwargs['training'] = training
          x = layer(x, **kwargs)
    if self.normalize_output:
      x = tf.nn.l2_normalize(x, axis=1)
    return x
