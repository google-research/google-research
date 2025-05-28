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

"""`Layerwise` convolutional neural network models."""
import functools

from hypertransformer.tf.core import common_ht
from hypertransformer.tf.core import layerwise

ConvLayer = layerwise.ConvLayer
LogitsLayer = layerwise.LogitsLayer
FlattenLogitsLayer = layerwise.FlattenLogitsLayer


def maxpool_model(
    model_config,
    num_layers = 4,
    last_maxpool = 2):
  """Creates a larger 4-layer model (similar to the one in MAML paper)."""
  conv_args = {'model_config': model_config,
               'maxpool_size': 2,
               'padding': 'same'}
  layers = [
      ConvLayer(name=f'layer_{i + 1}', head_builder=LogitsLayer, **conv_args)
      for i in range(num_layers - 1)]
  # The last layer should not have a separate head (we already have a real head)
  conv_args['maxpool_size'] = last_maxpool
  layers += [
      ConvLayer(name=f'layer_{num_layers}', **conv_args),
      FlattenLogitsLayer(name=f'layer_{num_layers + 1}',
                         model_config=model_config),
  ]
  return layerwise.LayerwiseModel(layers=layers, model_config=model_config)


def avgpool_model(
    model_config,
    num_layers = 1):
  """Creates a basic 'layerwise' model."""
  layers = [
      ConvLayer(name=f'layer_{i + 1}', model_config=model_config,
                head_builder=LogitsLayer)
      for i in range(num_layers - 1)]
  # The last layer should not have a separate head (we already have a real head)
  layers += [
      ConvLayer(name=f'layer_{num_layers}', model_config=model_config),
      LogitsLayer(name=f'layer_{num_layers + 1}', model_config=model_config)
  ]
  return layerwise.LayerwiseModel(layers=layers, model_config=model_config)


def register():
  """Registers layerwise architectures."""
  layerwise.models['maxpool-4-layer'] = functools.partial(maxpool_model,
                                                          num_layers=4)
  layerwise.models['3-layer'] = functools.partial(avgpool_model, num_layers=3)
  layerwise.models['4-layer'] = functools.partial(avgpool_model, num_layers=4)


register()
