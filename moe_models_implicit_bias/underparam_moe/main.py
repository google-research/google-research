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

"""Main file for running the underparameterized MoE experiment.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
import ml_collections
import tensorflow as tf

from moe_models_implicit_bias.underparam_moe import train

_WORKDIR = flags.DEFINE_string('workdir',
                               '/semi_distributed/general_mixture/001',
                               'Directory to store model data.')
_DIM = flags.DEFINE_integer('dim', 6, 'dim')
_SAMPLES_RATIO = flags.DEFINE_float('samples_ratio', 2, 'samples_ratio')
_MARGIN = flags.DEFINE_float('margin', 0, 'margin')
_NOISE = flags.DEFINE_float('noise', 0.0, 'noise')
_NUM_EXPERTS = flags.DEFINE_integer('num_experts', 128, 'num_experts')
_BATCH_SIZE_RATIO = flags.DEFINE_float('batch_size_ratio', -1,
                                       'batch_size_ratio')
_BATCH_SIZE = flags.DEFINE_float('batch_size', -1, 'batch_size_ratio')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 3e-4, 'learning_rate')
_MODEL = flags.DEFINE_string('model', 'MLP', 'model')
_OPT = flags.DEFINE_string('opt', 'sgd', 'opt')
_DEPTH = flags.DEFINE_integer('depth', 2, 'depth')
_SEED = flags.DEFINE_integer('seed', 2, 'seed')
_WEIGHT_DECAY = flags.DEFINE_float('weight_decay', 0.0, 'weight_decay')
_WIDTHS_RATIO = flags.DEFINE_list('widths_ratio', None, 'widths_ratio')


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.dim = _DIM.value
  config.margin = _MARGIN.value

  config.num_experts = _NUM_EXPERTS.value

  config.num_samples = int(round(_SAMPLES_RATIO.value * _DIM.value))

  config.depth = _DEPTH.value
  config.widths = [
      int(round(float(w) * _DIM.value)) for w in _WIDTHS_RATIO.value
  ]

  # As defined in the `models` module.
  config.model = _MODEL.value
  config.opt = _OPT.value

  config.learning_rate = _LEARNING_RATE.value
  if _BATCH_SIZE_RATIO.value > 0:
    config.batch_size = int(round(_BATCH_SIZE_RATIO.value * config.num_samples))
  else:
    config.batch_size = _BATCH_SIZE.value

  config.num_train_steps = 250000
  config.log_every_steps = 1

  config.noise = _NOISE.value

  config.weight_decay = _WEIGHT_DECAY.value

  config.half_precision = False

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.steps_per_eval = max(2 * config.num_samples, 2000) // config.batch_size

  config.eval_every_steps = 10 * (config.num_samples // config.batch_size)

  config.seed = _SEED.value

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
