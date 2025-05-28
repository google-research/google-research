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

r"""Trains a DEDAL model.

Example usage:

python3 -m dedal.main -- \
  --base_dir /tmp/example/1 \
  --gin_config data/uniref50.gin \
  --gin_config data/pfam34_alignment.gin \
  --gin_config data/pfam34_homology.gin \
  --gin_config model/dedal.gin \
  --gin_config task/task.gin \
  --gin_bindings UNIREF50_DATA_DIR=/path/to/masked_lm/data \
  --gin_bindings PFAM34_ALIGNMENT_DATA_DIR=/path/to/alignment/data \
  --gin_bindings PFAM34_HOMOLOGY_DATA_DIR=/path/to/homology/data \
  --gin_bindings MAIN_VOCAB_PATH=/path/to/main/vocab \
  --gin_bindings TOKEN_REPLACE_VOCAB_PATH=/path/to/token_replace/vocab \
  --gin_bindings ALIGNMENT_PATH_VOCAB_PATH=/path/to/alignment_path/vocab \
  --task train \
  --alsologtostderr
"""

import os.path

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf

from dedal.train import training_loop


flags.DEFINE_string(
    'base_dir', None,
    'Directory to save trained model in.')
flags.DEFINE_string(
    'reference_dir', None,
    'Directory where to read the reference model from (if exists).')
flags.DEFINE_boolean(
    'eval_in_train_job', True,
    'Whether to also run eval in a train job. Ignored for other tasks.')
flags.DEFINE_enum(
    'task', 'train', ['train', 'eval', 'downstream'],
    'Whether this is a train, eval or downstream task.')

flags.DEFINE_multi_string(
    'gin_config', [],
    'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Newline separated list of Gin parameter bindings.')
flags.DEFINE_string(
    'config_path', 'dedal/configs',
    'Where to find the gin configurations.')


FLAGS = flags.FLAGS


def main(unused_argv):
  filenames = [os.path.join(FLAGS.config_path, p) for p in FLAGS.gin_config]
  gin.parse_config_files_and_bindings(filenames, FLAGS.gin_bindings)
  logging.info('Gin Configuration:\n%s', gin.config_str())

  strategy_kwargs = {}

  # Worker preemption handling.
  keep_running = True
  preempted_count = 0
  while keep_running:
    strategy = training_loop.get_strategy(**strategy_kwargs)
    logging.info('Distribution strategy: %s', strategy)
    logging.info('Devices: %s', tf.config.list_physical_devices())
    logging.info('We have been preempted %d times so far.', preempted_count)

    kwargs = {'reference_workdir': FLAGS.reference_dir,
              'eval_in_train_job': FLAGS.eval_in_train_job}
    loop = training_loop.TrainingLoop(FLAGS.base_dir, strategy, **kwargs)

    try:
      loop.run(FLAGS.task)
      keep_running = False  # Finished training successfully.
    except tf.errors.UnavailableError as error:
      logging.warning('Job is likely being preempted: %s', error)
      logging.warning('Trying to recover...')
      preempted_count += 1


if __name__ == '__main__':
  app.run(main)
