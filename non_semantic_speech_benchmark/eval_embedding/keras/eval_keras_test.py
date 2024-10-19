# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for non_semantic_speech_benchmark.eval_embedding.keras.eval_keras."""

from absl import flags
from absl.testing import absltest
from non_semantic_speech_benchmark.eval_embedding.keras import eval_keras


class EvalKerasTest(absltest.TestCase):

  def test_full_flow(self):
    flags.FLAGS.label_name = 'test'
    flags.FLAGS.label_list = ['no', 'yes']
    flags.FLAGS.bucket_boundaries = [10, 20]
    flags.FLAGS.label_list = ['no', 'yes']
    flags.FLAGS.logdir = absltest.get_default_test_tmpdir()
    flags.FLAGS.eval_dir = absltest.get_default_test_tmpdir()
    flags.FLAGS.nc = 2
    flags.FLAGS.timeout = 5
    eval_keras.eval_and_report()


if __name__ == '__main__':
  absltest.main()
