# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Define metric factory."""

import tensorflow as tf
from tensorflow_addons import optimizers as tfa_optimizers

from vatt.configs import experiment


def get_optimizer(learning_rate, config):
  """Returns the optimizer of choice given the configurations."""

  if isinstance(config, experiment.MomentumOptimizer):
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov
        )
  elif isinstance(config, experiment.MomentumWOptimizer):
    optimizer = tfa_optimizers.SGDW(
        weight_decay=config.weight_decay,
        learning_rate=learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov
        )
  elif isinstance(config, experiment.AdamOptimizer):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        epsilon=config.epsilon,
        )
  elif isinstance(config, experiment.AdamWOptimizer):
    optimizer = tfa_optimizers.AdamW(
        weight_decay=config.weight_decay,
        learning_rate=learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        epsilon=config.epsilon,
        )
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  return optimizer
