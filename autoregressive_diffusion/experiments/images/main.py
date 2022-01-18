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

"""Entry point for the experiment, calls functions from train.

This entry point runs the training script located in train.py. It really does
not do so much more than that.
"""


from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections.config_flags import config_flags
import tensorflow as tf

from autoregressive_diffusion.experiments.images import train


FLAGS = flags.FLAGS
flags.DEFINE_string('executable_name', None, 'executable name')
flags.DEFINE_string('experiment_dir', None, 'experiment output directory')
flags.DEFINE_string('work_unit_dir', None, 'work unit output directory')
flags.DEFINE_string('config_json', None, 'hyperparameter file to read in .json')
flags.DEFINE_string('extra_args_json_str', None, 'extra args to pass in')
config_flags.DEFINE_config_file('config', lock_config=False)


def main(*_args, **_kwargs):
  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  info_string = f'JAX process: {jax.process_index()} / {jax.process_count()}'
  logging.info(info_string)

  info_string = f'JAX local devices: {jax.local_devices()}'
  logging.info(info_string)

  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.work_unit_dir, 'work_unit_dir')
  config = FLAGS.config
  train.train_and_evaluate(config, FLAGS.work_unit_dir)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
