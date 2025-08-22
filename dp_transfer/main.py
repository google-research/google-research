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

"""Main file.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

from absl import app
from absl import flags
from clu import platform
from flax import linen as nn
import jax
from ml_collections import config_flags
import tensorflow as tf

from dp_transfer import train_fc
from dp_transfer import train_gd_clip_grad
from dp_transfer import train_linear_regression
from dp_transfer import train_newton

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    'configs/default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Enable wrapping of all module calls in a named_call for easier profiling:
  nn.enable_named_call()

  print('JAX process: %d / %d', jax.process_index(), jax.process_count())
  print('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  if FLAGS.config.solver == 'linear_regression':
    train_linear_regression.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.solver == 'newton':
    train_newton.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.solver == 'gd_clip_grad':
    train_gd_clip_grad.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.solver == 'fc':
    train_fc.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
