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

# python3
"""Adam with fixed hyper parameter configurations."""
import numpy as np
from task_set import registry
from task_set.optimizers import utils
import tensorflow.compat.v1 as tf


def make_lr_optimizer(lr):

  def fn(training_steps=10000):  # pylint: disable=unused-argument
    return tf.train.AdamOptimizer(lr)

  return fn


for _lr in np.logspace(-7, 1, 17):
  registry.optimizers_registry.register_fixed("adam_lr_%.2f" % np.log10(_lr))(
      make_lr_optimizer(_lr))


@registry.optimizers_registry.register_sampler("adam4p_wide_grid")
def sample_adam8p_wide_grid(seed):
  """Sample a random configuration from a wide grid for adam8p."""
  rng = np.random.RandomState(seed)
  cfg = {
      "learning_rate": utils.sample_log_float(rng, 1e-8, 1e1),
      "beta1": 1 - utils.sample_log_float(rng, 1e-4, 1e0),
      "beta2": 1 - utils.sample_log_float(rng, 1e-6, 1e0),
      "epsilon": utils.sample_log_float(rng, 1e-10, 1e3),
  }
  return cfg


def make_adam_optimizer(learning_rate, beta1, beta2, epsilon):
  return tf.train.AdamOptimizer(
      learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)


@registry.optimizers_registry.register_getter("adam4p_wide_grid")
def get_adam4p(cfg, training_steps=10000):  # pylint: disable=unused-argument
  return make_adam_optimizer(**cfg)


@registry.optimizers_registry.register_sampler("adam1p_wide_grid")
def sample_adam1p_wide_grid(seed):
  """Sample a random configuration from a wide grid for adam8p."""
  rng = np.random.RandomState(seed + 4123)
  cfg = {
      "learning_rate": utils.sample_log_float(rng, 1e-8, 1e1),
  }
  return cfg


@registry.optimizers_registry.register_getter("adam1p_wide_grid")
def get_adam1p(cfg, training_steps=10000):  # pylint: disable=unused-argument
  return tf.train.AdamOptimizer(cfg["learning_rate"])
