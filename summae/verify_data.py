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

"""Script that verifies data is set up properly for training and evaluation.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

from summae import util

FLAGS = flags.FLAGS


flags.DEFINE_string('data_dir', '', 'Where tfrecord data is.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  # TODO(peterjliu): Add checks for generated data
  shard0size = os.path.getsize(
      os.path.join(FLAGS.data_dir,
                   'rocstories_springwintertrain.all.0000.tfrecord'))
  assert shard0size == 2517241

  t = util.read_records(
      os.path.join(FLAGS.data_dir, 'rocstories_gt.test.tfrecord'))
  assert 500 == len(t)
  for x in t:
    s = tf.train.SequenceExample()
    s.ParseFromString(x)
    assert 3 == len(s.context.feature['summaries'].bytes_list.value)
  v = util.read_records(
      os.path.join(FLAGS.data_dir, 'rocstories_gt.valid.tfrecord'))
  assert 500 == len(v)
  print('Data looks good!')


if __name__ == '__main__':
  app.run(main)
