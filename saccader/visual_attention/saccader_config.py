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

"""Configuration for Saccader model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from saccader.feedforward import bagnet_config


class ConfigDict(object):
  pass


def get_config():
  """Returns the default configuration as instance of ConfigDict."""
  config = ConfigDict()
  # Optimization parameters (expected to be overridden by training script).
  config.mc_samples = 1
  config.reinforce_loss_wt = 1.0
  config.l2_loss_wt = 8e-5
  config.l2_loss_loc_wt = 8e-5
  config.xent_loss_wt = 1.0

  # Model parameters.
  config.num_times = 6
  config.attention_groups = 2
  config.attention_layers_per_group = 2

  config.soft_attention = False

  config.num_classes = -1  # Specify num_classes in main.
  # Representation network parameters.
  config.representation_config = bagnet_config.get_config()
  config.representation_config.blocks = [3, 4, 6, 3]
  config.representation_config.activation = tf.nn.relu
  config.representation_config.planes = [64, 128, 256, 512]

  # Saccader-77 (for ImageNet 224).
  config.representation_config.strides = [2, 2, 2, 1]
  config.representation_config.kernel3 = [2, 2, 2, 2]

  config.representation_config.final_bottleneck = True
  config.representation_config.batch_norm.enable = True
  return config
