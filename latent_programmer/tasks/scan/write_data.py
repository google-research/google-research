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

"""Write SCAN training tasks to TFRecord dataset."""

import os
import random

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from latent_programmer.tasks.scan import sample_random


gfile = tf.io.gfile

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_work_units', 1, 'Total number of work units.')
flags.DEFINE_integer('seed', None, 'Fixed random seed.')
flags.DEFINE_integer('num_tasks', 100000, 'Number of tasks to write.')
flags.DEFINE_string('save_dir', '/tmp/decomposition/scan',
                    'Directory to save results to.')
flags.DEFINE_boolean('output_separators', True,
                     'Whether to add separators between parts of the output.')
flags.DEFINE_enum('split', None, ['train', 'valid', 'test', 'finetune'],
                  'Which split of the dataset to generate.')
flags.DEFINE_enum('experiment', 'NONE',
                  [e.name for e in sample_random.ScanExperiment],
                  'Kind of experiment (see ScanExperiment for descriptions).')


def main(_):

  if FLAGS.seed is not None:
    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

  if not gfile.isdir(FLAGS.save_dir):
    gfile.makedirs(FLAGS.save_dir)

  worker_fname = os.path.join(
      FLAGS.save_dir,
      'program_tasks_{}.tf_records-00000-of-00001'.format(FLAGS.split))

  sample_random.write_examples(
      filename=worker_fname,
      num_tasks=FLAGS.num_tasks,
      experiment=FLAGS.experiment,
      split=FLAGS.split,
      output_separators=FLAGS.output_separators)


if __name__ == '__main__':
  app.run(main)
