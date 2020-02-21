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
"""Adagrad with fixed hyper parameter configurations."""
import numpy as np
from task_set import registry
from task_set.optimizers import utils
import tensorflow.compat.v1 as tf


def make_lr_optimizer(lr):

  def fn():
    return tf.train.AdagradOptimizer(learning_rate=lr)

  return fn


for _lr in np.logspace(-7, 1, 17):
  registry.optimizers_registry.register_fixed("adagrad_lr_%.2f" %
                                              np.log10(_lr))(
                                                  make_lr_optimizer(_lr))


@registry.optimizers_registry.register_sampler("adagrad_wide_grid")
def sample_adagrad_wide_grid(seed):
  """Sample a random configuration from a wide grid for adagrad."""
  rng = np.random.RandomState(seed)
  cfg = {
      "learning_rate": utils.sample_log_float(rng, 1e-8, 1e1),
      "initial_accumulator_value": utils.sample_log_float(rng, 1e-10, 1e3),
  }
  return cfg


def make_adagrad_optimizer(learning_rate, initial_accumulator_value):
  return tf.train.AdagradOptimizer(
      learning_rate=learning_rate,
      initial_accumulator_value=initial_accumulator_value)


@registry.optimizers_registry.register_getter("adagrad_wide_grid")
def get_adagrad(cfg):
  return make_adagrad_optimizer(**cfg)
