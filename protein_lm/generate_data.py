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

# Lint as: python3
"""Process csv data to tfrecords."""

import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from protein_lm import data

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_dir', default='', help=('Directory to load CSVs from.'))
flags.DEFINE_string(
    'output_dir', default='', help=('Directory to output tfrecords to.'))


def main(argv):
  if not FLAGS.input_dir:
    raise ValueError('Must provide input directory.')
  if not FLAGS.output_dir:
    raise ValueError('Must provide output directory.')

  files = tf.gfile.Glob(os.path.join(FLAGS.input_dir, '*.csv'))
  tf.gfile.MakeDirs(FLAGS.output_dir)
  for i, file in enumerate(files):
    file = os.path.join(FLAGS.input_dir, file)
    print(file)
    data.csv_to_tfrecord(file, FLAGS.output_dir, idx=i, total=len(files))

if __name__ == '__main__':
  app.run(main)
