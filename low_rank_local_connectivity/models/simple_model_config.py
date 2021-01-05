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

"""Configuration for simple model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ConfigDict(object):
  pass


def get_config():
  """Returns the default configuration as instance of ConfigDict."""
  config = ConfigDict()
  config.num_classes = -1  # Specify num_classes in main.
  config.kernel_size_list = [3, 3, 3]
  config.num_filters_list = [64, 64, 64]
  config.strides_list = [1, 1, 1]
  # layer_types options:
  # 'conv2d', 'low_rank_locally_connected2d',
  # 'locally_connected2d', 'wide_conv2d'
  config.layer_types = ['conv2d', 'conv2d', 'conv2d',]
  config.normalize_weights = 'softmax'
  config.input_dependent = False
  config.rank = 1
  config.kernel_initializer = 'he_uniform'
  config.combining_weights_initializer = 'conv_init'
  config.batch_norm = True
  config.num_channels = 0
  config.global_avg_pooling = True
  config.coord_conv = False
  config.share_row_combining_weights = True
  config.share_col_combining_weights = True

  return config
