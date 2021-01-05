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

"""Utilities for getting the model object by name and configuration."""

from demogen.models.nin import Nin
from demogen.models.resnet import ResNet


def get_model(model_name, config):
  """Create a callable model function according to config.

  Args:
    model_name: The type of model to be created.
    config: A tf.contrib.training.HParams data structure
      for model hyperparameters.

  Returns:
    A callable model function built according to the config.
  """
  if model_name == 'nin':
    width = int(192 * config.wide)
    return Nin(
        width,
        dropout=config.dropout,
        batchnorm=config.batchnorm,
        decay_fac=config.decay_fac,
        num_classes=config.num_class,
        spatial_dropout=config.spatial_dropout
    )
  elif model_name == 'resnet':
    resnet_size = 32
    num_blocks = (resnet_size - 2) // 6
    return ResNet(
        bottleneck=False,
        num_filters=int(16 * config.wide),
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        pre_activation=True,
        weight_decay=config.weight_decay,
        norm_type=config.normalization,
        loss_filter_fn=lambda _: True,
        num_classes=config.num_class,
    )
  raise NotImplementedError('Model {} is not in dataset'.format(model_name))
