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

"""No padding Inception FCN neural network.

This is a variant of Inception v3 that removes all paddings. This change
allows the network to be trained and inference run with different patch size
(Fully Convolutional Network, FCN mode) while having the same inference results.
The network can be initialized for two different receptive fields: 911 and 129.
"""

import tensorflow.compat.v1 as tf

from nopad_inception_v3_fcn import inception_base_129
from nopad_inception_v3_fcn import inception_base_911
from nopad_inception_v3_fcn import network
from nopad_inception_v3_fcn import network_params
from nopad_inception_v3_fcn import scope_utils
from tensorflow.contrib import slim


def get_inception_base_and_downsample_factor(receptive_field_size):
  """Get the Inception base network and its downsample factor."""
  if receptive_field_size == 911:
    return inception_base_911.nopad_inception_v3_base_911, inception_base_911.MODEL_DOWNSAMPLE_FACTOR
  elif receptive_field_size == 129:
    return inception_base_129.nopad_inception_v3_base_129, inception_base_129.MODEL_DOWNSAMPLE_FACTOR
  else:
    raise ValueError(
        f'Receptive field size should be 911 or 129. {receptive_field_size} was provided.'
    )


class InceptionV3FCN(network.Network):
  """A no pad, fully convolutional InceptionV3 model."""

  def __init__(
      self,
      inception_params,
      conv_scope_params,
      num_classes = 2,
      is_training = True,
  ):
    """Creates a no pad, fully convolutional InceptionV3 model.

    Args:
      inception_params: parameters specific to the InceptionV3
      conv_scope_params: parameters used to configure the general convolution
        parameters used in the slim argument scope.
      num_classes: number of output classes from the model
      is_training: whether the network should be built for training or inference
    """
    super().__init__()
    self._num_classes = num_classes
    self._is_training = is_training
    self._network_base, self._downsample_factor = get_inception_base_and_downsample_factor(
        inception_params.receptive_field_size)
    self._prelogit_dropout_keep_prob = inception_params.prelogit_dropout_keep_prob
    self._depth_multiplier = inception_params.depth_multiplier
    self._min_depth = inception_params.min_depth
    self._inception_fcn_stride = inception_params.inception_fcn_stride
    self._conv_scope_params = conv_scope_params
    if self._depth_multiplier <= 0:
      raise ValueError('param depth_multiplier should be greater than zero.')
    self._logits_stride = int(
        self._inception_fcn_stride /
        self._downsample_factor) if self._inception_fcn_stride else 1

  def build(self, inputs):
    """Returns an InceptionV3FCN model with configurable conv2d normalization.

    Args:
      inputs: a map from input string names to tensors. Required:
        * IMAGES: a tensor of shape [batch, height, width, channels]

    Returns:
      A dictionary from network layer names to the corresponding layer
      activation Tensors. Includes:
        * PRE_LOGITS: activation layer preceding LOGITS
        * LOGITS: the pre-softmax activations, size [batch, num_classes]
        * PROBABILITIES: softmax probs, size [batch, num_classes]
    """
    images = self._get_tensor(inputs, self.IMAGES, expected_rank=4)
    with slim.arg_scope(
        scope_utils.get_conv_scope(self._conv_scope_params, self._is_training)):
      net, end_points = self._network_base(
          images,
          min_depth=self._min_depth,
          depth_multiplier=self._depth_multiplier)
      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        # 1 x 1 x 768
        net = slim.dropout(
            net,
            keep_prob=self._prelogit_dropout_keep_prob,
            is_training=self._is_training,
            scope='Dropout_1b')
        end_points[self.PRE_LOGITS] = net
        # 1 x 1 x num_classes
        logits = slim.conv2d(
            net,
            self._num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            stride=self._logits_stride,
            scope='Conv2d_1c_1x1')
      probabilities_tensor = tf.nn.softmax(logits)
      end_points[self.PROBABILITIES_TENSOR] = probabilities_tensor
      if self._logits_stride == 1:
        # Reshape to remove height and width
        end_points[self.LOGITS] = tf.squeeze(
            logits, [1, 2], name='SpatialSqueeze')
        end_points[self.PROBABILITIES] = tf.squeeze(
            probabilities_tensor, [1, 2], name='SpatialSqueeze')
      else:
        end_points[self.LOGITS] = logits
        end_points[self.PROBABILITIES] = probabilities_tensor
    return end_points


def get_inception_v3_fcn_network_fn(
    inception_params,
    conv_scope_params,
    num_classes = 2,
    is_training = True,
):
  """Returns a function that return logits and endpoints for slim uptraining."""

  net = InceptionV3FCN(inception_params, conv_scope_params, num_classes,
                       is_training)

  def network_fn(images):
    images_dict = {'Images': images}
    endpoints = net.build(images_dict)
    return endpoints['Logits'], endpoints

  return network_fn
