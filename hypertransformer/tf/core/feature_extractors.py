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

"""Feature extractors used to generate Layerwise models."""
import functools

from typing import Any, Optional

import tensorflow.compat.v1 as tf
import typing_extensions

from hypertransformer.tf.core import common_ht

Protocol = typing_extensions.Protocol


class FeatureExtractor(tf.Module):
  pass


class SimpleConvFeatureExtractor(FeatureExtractor):
  """Simple convolutional feature extractor."""

  def __init__(self,
               feature_layers,
               feature_dim,
               name,
               nonlinear_feature = False,
               kernel_size = 3,
               input_size = None,
               padding = 'valid'):
    super(SimpleConvFeatureExtractor, self).__init__(name=name)
    self.nonlinear_feature = nonlinear_feature
    self.convs = []
    def_stride = 2
    self.kernel_size = kernel_size
    if input_size < kernel_size:
      self.kernel_size = input_size
      def_stride = 1
    if feature_dim > 0:
      for idx, layer in enumerate(range(feature_layers)):
        stride = def_stride if idx < feature_layers - 1 else 1
        self.convs.append(
            tf.layers.Conv2D(
                filters=feature_dim,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(stride, stride),
                padding=padding,
                activation=None,
                name=f'layer_{layer + 1}'))

  def __call__(self, input_tensor):
    if not self.convs:
      return None
    with tf.variable_scope(None, default_name=self.name):
      tensor = input_tensor
      outputs = []
      for conv in self.convs:
        if int(tensor.shape[1]) < self.kernel_size:
          break
        tensor = conv(tensor)
        if not self.nonlinear_feature:
          feature = tensor
        # While the output is not employing nonlinearity, layer-to-layer
        # transformations use it.
        tensor = tf.nn.relu(tensor)
        if self.nonlinear_feature:
          feature = tensor
        outputs.append(feature)
      outputs = [tf.reduce_mean(tensor, axis=(1, 2)) for tensor in outputs]
      return tf.concat(outputs, axis=-1)


class SharedMultilayerFeatureExtractor(FeatureExtractor):
  """Simple shared convolutional feature extractor."""

  def __init__(self,
               feature_layers,
               feature_dim,
               name,
               kernel_size = 3,
               padding = 'valid',
               use_bn = False):
    super(SharedMultilayerFeatureExtractor, self).__init__(name=name)
    self.feature_dim = feature_dim
    self.convs = []
    self.bns = []
    assert feature_dim > 0
    for idx, layer in enumerate(range(feature_layers)):
      stride = 2 if idx < feature_layers - 1 else 1
      self.bns.append(tf.layers.BatchNormalization() if use_bn else None)
      self.convs.append(
          tf.layers.Conv2D(
              filters=feature_dim,
              kernel_size=(kernel_size, kernel_size),
              strides=(stride, stride),
              padding=padding,
              activation=tf.nn.relu,
              name=f'layer_{layer + 1}'))

  def __call__(self, input_tensor, training = True):
    with tf.variable_scope(None, default_name=self.name):
      tensor = input_tensor
      for conv, bn in zip(self.convs, self.bns):
        tensor = conv(tensor)
        tensor = bn(tensor) if bn is not None else tensor
      return tf.reduce_mean(tensor, axis=(-2, -3))


def fe_multi_layer(config, num_layers = 2,
                   use_bn = False):
  return SharedMultilayerFeatureExtractor(
      feature_layers=num_layers,
      feature_dim=config.shared_features_dim,
      name='shared_features',
      padding=config.shared_feature_extractor_padding,
      use_bn=use_bn)


class FeatureExtractorClass(Protocol):
  """Type declaring feature extractor builder class."""

  def __call__(self, *, name, **kwargs):
    Ellipsis


class PassthroughFeatureExtractor(FeatureExtractor):
  """Passthrough feature extractor."""

  def __init__(self,
               name,
               input_size = None,
               wrap_class = None):
    super(PassthroughFeatureExtractor, self).__init__(name=name)
    self.name = name
    if wrap_class is not None:
      self.wrap_feature_extractor = wrap_class(name=name)
    else:
      self.wrap_feature_extractor = None

  def __call__(self, input_tensor):
    output = tf.layers.Flatten()(input_tensor)
    if self.wrap_feature_extractor is not None:
      wrapped = self.wrap_feature_extractor(input_tensor)
      output = tf.concat([output, wrapped], axis=-1)
    return output


feature_extractors = {
    '2-layer': functools.partial(fe_multi_layer, num_layers=2),
    '3-layer': functools.partial(fe_multi_layer, num_layers=3),
    '4-layer': functools.partial(fe_multi_layer, num_layers=4),
    '2-layer-bn': functools.partial(fe_multi_layer, num_layers=2, use_bn=True),
    '3-layer-bn': functools.partial(fe_multi_layer, num_layers=3, use_bn=True),
}


def get_shared_feature_extractor(config):
  feature_extractor = config.shared_feature_extractor
  if feature_extractor in ['none', '']:
    return None
  if feature_extractor not in feature_extractors:
    raise ValueError(f'Unknown shared feature extractor "{feature_extractor}"')
  return feature_extractors[feature_extractor](config)
