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

"""Main entry point for experiments."""

import functools
import json
import random

from absl import app
from absl import flags
from absl import logging

from clu import platform
import jax
from jax.config import config as jax_config
import ml_collections
from ml_collections.config_flags import config_flags
import numpy as onp
import tensorflow.compat.v2 as tf

from d3pm.images import utils


FLAGS = flags.FLAGS
flags.DEFINE_string('executable_name', None, 'executable name')
flags.DEFINE_string('experiment_dir', None, 'experiment output directory')
flags.DEFINE_string('work_unit_dir', None, 'work unit output directory')
flags.DEFINE_string('config_json', None, 'hyperparameter file to read in .json')
flags.DEFINE_string('extra_args_json_str', None, 'extra args to pass in')
config_flags.DEFINE_config_file('config', lock_config=False)


def main(executable_dict, argv):
  del argv


  work_unit = platform.work_unit()
  tf.enable_v2_behavior()
  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX host: %d / %d', jax.host_id(), jax.host_count())
  logging.info('JAX devices: %r', jax.devices())

  work_unit.set_task_status(
      f'host_id: {jax.host_id()}, host_count: {jax.host_count()}')

  # Read configuration
  if FLAGS.config_json:
    logging.info('Reading config from JSON: %s', FLAGS.config_json)
    with tf.io.gfile.GFile(FLAGS.config_json, 'r') as f:
      config = ml_collections.ConfigDict(json.loads(f.read()))
  else:
    config = FLAGS.config
  logging.info('config=%s',
               config.to_json_best_effort(indent=4, sort_keys=True))

  # Make output directories
  if FLAGS.experiment_dir:
    work_unit.create_artifact(platform.ArtifactType.DIRECTORY,
                              FLAGS.experiment_dir, 'experiment_dir')
  if FLAGS.work_unit_dir:
    work_unit.create_artifact(platform.ArtifactType.DIRECTORY,
                              FLAGS.work_unit_dir, 'work_unit_dir')
  logging.info('experiment_dir=%s work_unit_dir=%s', FLAGS.experiment_dir,
               FLAGS.work_unit_dir)

  # Seeding
  random.seed(config.seed * jax.host_count() + jax.host_id())
  onp.random.seed(config.seed * jax.host_count() + jax.host_id())
  rng = utils.RngGen(
      jax.random.fold_in(jax.random.PRNGKey(config.seed), jax.host_id()))

  # Run the main function
  logging.info('Running executable: %s', FLAGS.executable_name)

  extra_args = {}
  if FLAGS.extra_args_json_str:
    extra_args = json.loads(FLAGS.extra_args_json_str)
    logging.info('Extra args passed in: %r', extra_args)

  executable_dict[FLAGS.executable_name](
      config=config,
      experiment_dir=FLAGS.experiment_dir,
      work_unit_dir=FLAGS.work_unit_dir,
      rng=rng,
      **extra_args)

  utils.barrier()


def run(**executable_dict):
  # JAX uses a different flags library -- parse from absl so that the task
  # knows its task_id for pod training.
  jax_config.config_with_absl()
  app.run(functools.partial(main, executable_dict))
