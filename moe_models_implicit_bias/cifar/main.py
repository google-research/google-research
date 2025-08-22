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

"""Main file for experiment on replacing regular layer with MoE layer in a ResNet on CIFAR-10 dataset.
"""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
import ml_collections
import tensorflow as tf
from moe_models_implicit_bias.cifar import train

_WORKDIR = flags.DEFINE_string('workdir', '/cifar',
                               'Directory to store model data.')
_MODEL = flags.DEFINE_string('model', 'CNN', 'model')
_OPTIMIZER = flags.DEFINE_string('optim', 'adam', 'optimizer')
_LEARNING_RATE = flags.DEFINE_float('lr', 1e-4, 'learning rate')
_PROJ_DIM = flags.DEFINE_integer('proj_dim', 10,
                                 'dimension in which labels lie')
_SEED = flags.DEFINE_integer('seed', 0, 'random seed')


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()
  # As defined in the `models` module.
  config.model = _MODEL.value
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'cifar100'
  config.learning_rate = _LEARNING_RATE.value
  config.optim = _OPTIMIZER.value
  config.warmup_epochs = 0.0
  # config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 10000.0
  config.log_every_steps = 100  # CHANGE back to 100
  config.cache = False
  config.half_precision = False
  config.proj_dim = _PROJ_DIM.value
  config.seed = _SEED.value
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1
  return config


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       _WORKDIR.value, 'workdir')
  train.train_and_evaluate(get_config(), _WORKDIR.value)


if __name__ == '__main__':
  app.run(main)
