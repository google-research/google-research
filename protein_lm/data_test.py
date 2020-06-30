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

# Lint as: python3
"""Tests for data pipeline."""

import itertools
import os

from absl.testing import absltest
import tensorflow.compat.v1 as tf

from protein_lm import data

_TEST_DATA_DIR = './testdata'


tf.enable_eager_execution()


class DataTest(tf.test.TestCase):

  def setUp(self):
    self._tmpdir = self.create_tempdir().full_path
    super(DataTest, self).setUp()

  def test_preprocess_and_read(self):
    max_length = 30

    # Write 2 tfrecord shards
    csv_path = os.path.join(_TEST_DATA_DIR, 'trembl.csv')
    data.csv_to_tfrecord(csv_path=csv_path, outdir=self._tmpdir, idx=0, total=2)
    data.csv_to_tfrecord(csv_path=csv_path, outdir=self._tmpdir, idx=1, total=2)

    # Construct dataset
    train_files, test_files = data.get_train_test_files(self._tmpdir)
    train_ds, test_ds = data.load_dataset(
        train_files=train_files,
        test_files=test_files,
        batch_size=1,
        shuffle_buffer=1,
        max_train_length=max_length)

    # Load CSV manually
    seqs = []
    with tf.gfile.GFile(csv_path) as f:
      for line in f:
        print(line)
        seq = line.strip().split(',')[-1]
        enc = data.protein_domain.encode([seq], pad=False)[0][:max_length]
        seqs.append(enc)

    # Confirm we got the same sequences.
    for ds_x, target in itertools.zip_longest(iter(train_ds), seqs):
      ds_x = ds_x._numpy()[0]
      self.assertAllEqual(target, ds_x[:len(target)])

    for ds_x, target in itertools.zip_longest(iter(test_ds), seqs):
      ds_x = ds_x._numpy()[0]
      self.assertAllEqual(target, ds_x[:len(target)])


if __name__ == '__main__':
  absltest.main()
