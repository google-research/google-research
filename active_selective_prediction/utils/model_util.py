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

"""Model utils."""

from typing import Tuple

from active_selective_prediction import models
import tensorflow as tf


def get_simple_mlp(
    input_shape,
    num_classes
):
  """Gets simple MLP."""
  model = models.custom_model.SimpleMLP(input_shape, num_classes)
  return model


def get_simple_convnet(
    input_shape,
    num_classes
):
  """Gets simple ConvNet."""
  model = models.custom_model.SimpleConvNet(input_shape, num_classes)
  return model


def get_cifar_resnet(
    input_shape,
    num_classes
):
  """Gets CifarResNet."""
  model = models.custom_model.CifarResNet(input_shape, num_classes)
  return model


def get_densenet121(
    input_shape,
    num_classes,
    weights = 'imagenet'
):
  """Gets DenseNet121."""
  model = models.custom_model.DenseNet(
      input_shape=input_shape,
      num_classes=num_classes,
      weights=weights,
      densenet_name='DenseNet121'
  )
  return model


def get_resnet50(
    input_shape,
    num_classes,
    weights = 'imagenet'
):
  """Gets ResNet50."""
  model = models.custom_model.ResNet(
      input_shape=input_shape,
      num_classes=num_classes,
      weights=weights,
      resnet_name='ResNet50'
  )
  return model


def get_roberta_mlp(
    input_shape,
    num_classes
):
  """Gets RoBerta MLP."""
  model = models.custom_model.RoBertaMLP(input_shape, num_classes)
  return model
