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

"""Contains functions used to build cnc and siamese net models."""
from __future__ import division

from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.regularizers import l2


def stack_layers(inputs, net_layers, kernel_initializer='glorot_uniform'):
  """Builds the architecture of the network by applying each layer specified in net_layers to inputs.

  Args:
    inputs: a dict containing input_types and input_placeholders for each key
      and value pair, respecively.
    net_layers:  a list of dicts containing all layers to be used in the
      network, where each dict describes one such layer. each dict requires the
      key 'type'. all other keys are dependent on the layer type.
    kernel_initializer: initialization configuration passed to keras (see keras
      initializers).

  Returns:
    outputs: a dict formatted in much the same way as inputs. it
      contains input_types and output_tensors for each key and value pair,
      respectively, where output_tensors are the outputs of the
      input_placeholders in inputs after each layer in net_layers is applied.
  """
  outputs = dict()

  for key in inputs:
    outputs[key] = inputs[key]

  for layer in net_layers:
    # check for l2_reg argument
    l2_reg = layer.get('l2_reg')
    if l2_reg:
      l2_reg = l2(layer['l2_reg'])

    # create the layer
    if layer['type'] in [
        'softplus', 'softsign', 'softmax', 'tanh', 'sigmoid', 'relu', 'selu'
    ]:
      l = layers.Dense(
          layer['size'],
          activation=layer['type'],
          kernel_initializer=kernel_initializer,
          kernel_regularizer=l2_reg,
          name=layer.get('name'))
    elif layer['type'] == 'None':
      l = layers.Dense(
          layer['size'],
          kernel_initializer=kernel_initializer,
          kernel_regularizer=l2_reg,
          name=layer.get('name'))
    elif layer['type'] == 'Conv2D':
      l = layers.Conv2D(
          layer['channels'],
          kernel_size=layer['kernel'],
          activation='relu',
          data_format='channels_last',
          kernel_regularizer=l2_reg,
          name=layer.get('name'))
    elif layer['type'] == 'BatchNormalization':
      l = layers.BatchNormalization(name=layer.get('name'))
    elif layer['type'] == 'MaxPooling2D':
      l = layers.MaxPooling2D(
          pool_size=layer['pool_size'],
          data_format='channels_first',
          name=layer.get('name'))
    elif layer['type'] == 'Dropout':
      l = layers.Dropout(layer['rate'], name=layer.get('name'))
    elif layer['type'] == 'Flatten':
      l = layers.Flatten(name=layer.get('name'))
    else:
      raise ValueError("Invalid layer type '{}'".format(layer['type']))

    # apply the layer to each input in inputs
    for k in outputs:
      outputs[k] = l(outputs[k])

  return outputs
