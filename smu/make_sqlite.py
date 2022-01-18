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

"""Generate the SQLite DB from TFRecord files."""

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from tensorflow.io import gfile
from smu import smu_sqlite

flags.DEFINE_string('input_tfrecord', None, 'Glob of tfrecord files to read')
flags.DEFINE_string('output_sqlite', None, 'Path of sqlite file to generate')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.get_absl_handler().use_absl_log_file()

  logging.info('Opening %s', FLAGS.output_sqlite)
  db = smu_sqlite.SMUSQLite(FLAGS.output_sqlite, 'c')

  dataset = tf.data.TFRecordDataset(gfile.glob(FLAGS.input_tfrecord))
  db.bulk_insert((raw.numpy() for raw in dataset), batch_size=10000)


if __name__ == '__main__':
  app.run(main)
