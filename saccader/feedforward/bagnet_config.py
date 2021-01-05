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

"""Configuration for Bagnet network model.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class ConfigDict(object):
  pass


def get_config():
  """Returns the default configuration as instance of ConfigDict."""
  config = ConfigDict()
  # Optimization parameters (expected to be overridden by training script).
  config.l2_loss_wt = 8e-5

  config.num_classes = -1  # Specify num_classes in main.
  config.init_conv_channels = 64
  config.expansion = 4

  config.blocks = [3, 4, 6, 3]
  config.activation = tf.nn.relu
  config.planes = [64, 128, 256, 512]

  config.strides = [2, 2, 2, 1]
  config.kernel3 = [2, 2, 2, 2]

  config.final_bottleneck = True
  config.batch_norm = ConfigDict()
  config.batch_norm.enable = True
  config.batch_norm.momentum = 0.9
  config.batch_norm.epsilon = 1e-5
  return config
