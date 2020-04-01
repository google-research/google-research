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

"""Configuration for DRAM model.

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

  # Optimization parameters.
  config.mc_samples = 1
  config.reinforce_loss_wt = 0.1
  config.l2_loss_wt = 8e-5
  config.l2_loss_loc_wt = 0.0
  config.xent_loss_wt = 1.0
  config.sampling_stddev = 0.1

  # Model parameters.
  config.num_resolutions = 3
  # Limit the classification receptive field to the smallest RF.
  config.limit_classification_rf = True
  config.num_times = 6

  config.glimpse_shape = (77, 77)
  config.num_classes = -1  # Specify num_classes in main.
  # RNN Model parameters.
  config.rnn_dropout_rate = 0.
  config.num_units_rnn_layers = [
      1024,  # classification.
      1024,  # location.
  ]
  config.cell_type = "lstm"
  config.rnn_activation = tf.nn.tanh

  config.location_encoding = "absolute"  # Can be either absolute or relative.

  # Glimpse Model parameters.
  config.glimpse_model_config = ConfigDict()
  network_type = "resnet_v2_50"
  config.glimpse_model_config.network_type = network_type
  config.glimpse_model_config.output_dims = None
  config.glimpse_model_config.apply_stop_gradient = True
  if network_type == "wrn":
    config.glimpse_model_config.normalization_type = "batch"
    config.glimpse_model_config.residual_blocks_per_group = 6
    config.glimpse_model_config.number_groups = 3
    config.glimpse_model_config.init_conv_channels = 16
    config.glimpse_model_config.widening_factor = 10
    config.glimpse_model_config.activation = tf.nn.relu
    config.glimpse_model_config.num_units_fc_layers = None
    config.glimpse_model_config.regularizer = tf.nn.l2_loss
    config.glimpse_model_config.dropout_rate = 0
    config.glimpse_model_config.zero_pad = False
    config.glimpse_model_config.global_average_pool = True
  # Emission Model parameters.
  config.emission_model_config = ConfigDict()
  config.emission_model_config.num_units_fc_layers = None
  config.emission_model_config.activation = tf.nn.relu
  config.emission_model_config.regularizer = None

  # Classification Model parameters (additionally specify num_classes in main).
  config.classification_model_config = ConfigDict()
  config.classification_model_config.num_units_fc_layers = None
  config.classification_model_config.activation = tf.nn.relu
  config.classification_model_config.regularizer = tf.nn.l2_loss

  return config
