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

"""Tests for count_duration_beam."""

import os
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import mock

import tensorflow as tf

from non_semantic_speech_benchmark.data_prep import count_duration_beam


TESTDIR = 'non_semantic_speech_benchmark/data_prep/testdata'


class CountDurationBeamTest(absltest.TestCase):

  @mock.patch.object(count_duration_beam, 'get_dataset_info_dict')
  @flagsaver.flagsaver
  def test_main_flow(self, mock_ds_dict):
    flags.FLAGS.output_file = os.path.join(
        absltest.get_default_test_tmpdir(), 'dummy_out.txt')
    dummy_fn = os.path.join(
        absltest.get_default_test_srcdir(), TESTDIR, 'test.tfrecord')
    mock_ds_dict.return_value = {
        'dummy': ([[dummy_fn]], 'tfrecord'),
    }

    # Run the beam pipeline, which writes to the output.
    count_duration_beam.main(None)

    # Do a check on the output file.
    ret = tf.io.gfile.glob(f'{flags.FLAGS.output_file}*')
    self.assertLen(ret, 1)
    out_file = ret[0]
    with tf.io.gfile.GFile(out_file) as f:
      lines = f.read().split('\n')[:-1]
    outs = [l.split(',') for l in lines]
    self.assertLen(outs, 1)
    savee_out = outs[0]
    self.assertEqual(savee_out[0], 'dummy')
    self.assertGreater(float(savee_out[1]), 0)
    self.assertEqual(int(savee_out[2]), 2)


if __name__ == '__main__':
  absltest.main()
