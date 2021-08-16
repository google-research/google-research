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

"""Tests for model_conversion_beam_main."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver

import mock

from non_semantic_speech_benchmark.export_model import model_conversion_beam_main


TESTDIR = 'non_semantic_speech_benchmark/export_model/testdata'


class ModelConversionBeamMainTest(absltest.TestCase):

  @mock.patch.object(model_conversion_beam_main.utils,
                     'convert_and_write_model')
  @flagsaver.flagsaver
  def test_full_flow(self, _):
    flags.FLAGS.xids = ['12321']
    flags.FLAGS.base_experiment_dir = os.path.join(
        absltest.get_default_test_srcdir(), TESTDIR)
    flags.FLAGS.output_dir = os.path.join(
        absltest.get_default_test_tmpdir(), 'dummy_out')

    # Frontend args.
    flags.FLAGS.frame_hop = 5
    flags.FLAGS.frame_width = 5
    flags.FLAGS.num_mel_bins = 80
    flags.FLAGS.n_required = 8000

    model_conversion_beam_main.main(None)


if __name__ == '__main__':
  absltest.main()
