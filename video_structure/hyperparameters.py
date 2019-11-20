# coding=utf-8
# Copyright 2019 The Google Research Authors.
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
"""Hyperparameters of the structured video prediction models."""

import tensorflow as tf


class ConfigDict(dict):
  """A dictionary whose keys can be accessed as attributes."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

  def get(self, key, default=None):
    """Allows to specify defaults when accessing the config."""
    if key not in self:
      return default
    return self[key]


def get_config():
  """Default values for all hyperparameters."""

  cfg = ConfigDict()

  # Directories:
  cfg.dataset = 'debug'
  cfg.data_dir = 'video_structure/testdata'
  cfg.train_dir = ''
  cfg.test_dir = ''

  # Architecture:
  cfg.layers_per_scale = 2
  cfg.conv_layer_kwargs = _conv_layer_kwargs()
  cfg.dense_layer_kwargs = _dense_layer_kwargs()

  # Optimization:
  cfg.batch_size = 32
  cfg.steps_per_epoch = 100
  cfg.num_epochs = 100
  cfg.learning_rate = 0.001
  cfg.clipnorm = 10

  # Image sequence parameters:
  cfg.observed_steps = 8
  cfg.predicted_steps = 8

  # Keypoint encoding settings:
  cfg.num_keypoints = 64
  cfg.heatmap_width = 16
  cfg.heatmap_regularization = 5.0
  cfg.keypoint_width = 1.5
  cfg.num_encoder_filters = 32
  cfg.separation_loss_scale = 10.0
  cfg.separation_loss_sigma = 0.1

  # Dynamics:
  cfg.num_rnn_units = 512
  cfg.prior_net_dim = 128
  cfg.posterior_net_dim = 128
  cfg.latent_code_size = 16
  cfg.kl_loss_scale = 0.0
  cfg.kl_annealing_steps = 1000
  cfg.use_deterministic_belief = False
  cfg.scheduled_sampling_ramp_steps = (
      cfg.steps_per_epoch * int(cfg.num_epochs * 0.8))
  cfg.scheduled_sampling_p_true_start_obs = 1.0
  cfg.scheduled_sampling_p_true_end_obs = 0.1
  cfg.scheduled_sampling_p_true_start_pred = 1.0
  cfg.scheduled_sampling_p_true_end_pred = 0.5
  cfg.num_samples_for_bom = 10

  return cfg


def _conv_layer_kwargs():
  """Returns a configDict with default conv layer hyperparameters."""

  cfg = ConfigDict()

  cfg.kernel_size = 3
  cfg.padding = 'same'
  cfg.activation = tf.nn.leaky_relu
  cfg.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

  # He-uniform initialization is suggested by this paper:
  # https://arxiv.org/abs/1803.01719
  # The paper only considers ReLU units and it might be different for leaky
  # ReLU, but it is a better guess than Glorot.
  cfg.kernel_initializer = 'he_uniform'

  return cfg


def _dense_layer_kwargs():
  """Returns a configDict with default dense layer hyperparameters."""

  cfg = ConfigDict()
  cfg.activation = tf.nn.relu
  cfg.kernel_initializer = 'he_uniform'

  return cfg
