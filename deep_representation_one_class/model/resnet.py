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

# Lint as: python3
"""ResNet definitions."""

import functools
from deep_representation_one_class.model.resnet_util import basic_stack1
from deep_representation_one_class.model.resnet_util import bottleneck_stack1
from deep_representation_one_class.model.resnet_util import ResNet

__all__ = [
    'ResNet10',
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet101',
    'ResNet152',
]

BLOCK = {
    'ResNet10': [1, 1, 1, 1],
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
}

EXPANSION = {
    'ResNet10': 1,
    'ResNet18': 1,
    'ResNet34': 1,
    'ResNet50': 4,
    'ResNet101': 4,
    'ResNet152': 4,
}

STACK = {
    'ResNet10': basic_stack1,
    'ResNet18': basic_stack1,
    'ResNet34': basic_stack1,
    'ResNet50': bottleneck_stack1,
    'ResNet101': bottleneck_stack1,
    'ResNet152': bottleneck_stack1,
}


def ResNetV1(arch='ResNet18',
             width=1.0,
             head_dims=None,
             input_shape=None,
             num_class=1000,
             pooling='avg',
             normalization='bn',
             activation='relu'):
  """Instantiates the ResNet architecture."""

  def stack_fn(x, arch, width=1.0):
    block, stack, expansion = BLOCK[arch], STACK[arch], EXPANSION[arch]
    x = stack(
        x,
        int(64 * width),
        block[0],
        expansion=expansion,
        stride1=1,
        normalization=normalization,
        activation=activation,
        name='conv2')
    x = stack(
        x,
        int(128 * width),
        block[1],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv3')
    x = stack(
        x,
        int(256 * width),
        block[2],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv4')
    return stack(
        x,
        int(512 * width),
        block[3],
        expansion=expansion,
        normalization=normalization,
        activation=activation,
        name='conv5')

  return ResNet(
      stack_fn=functools.partial(stack_fn, arch=arch, width=width),
      preact=False,
      model_name='{}_width{:g}_{}_{}'.format(arch, width, normalization,
                                             activation),
      head_dims=head_dims,
      input_shape=input_shape,
      pooling=pooling,
      normalization=normalization,
      activation=activation,
      num_class=num_class)


def ResNet10(**kwargs):
  return ResNetV1(arch='ResNet10', **kwargs)


def ResNet18(**kwargs):
  return ResNetV1(arch='ResNet18', **kwargs)


def ResNet34(**kwargs):
  return ResNetV1(arch='ResNet34', **kwargs)


def ResNet50(**kwargs):
  return ResNetV1(arch='ResNet50', **kwargs)


def ResNet101(**kwargs):
  return ResNetV1(arch='ResNet101', **kwargs)


def ResNet152(**kwargs):
  return ResNetV1(arch='ResNet152', **kwargs)
