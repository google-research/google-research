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

"""Tests for tflite_conversion."""

import os
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from non_semantic_speech_benchmark.export_model import tflite_conversion

TESTDIR = 'non_semantic_speech_benchmark/export_model/testdata'


class TfliteConversionTest(parameterized.TestCase):

  @parameterized.parameters(
      {'include_frontend': True},
      {'include_frontend': False},
  )
  @flagsaver.flagsaver
  def test_full_flow(self, include_frontend):
    flags.FLAGS.experiment_dir = os.path.join(
        absltest.get_default_test_srcdir(), TESTDIR)
    flags.FLAGS.checkpoint_number = '1000'
    flags.FLAGS.output_dir = absltest.get_default_test_tmpdir()
    flags.FLAGS.include_frontend = include_frontend

    tflite_conversion.main(None)

    tflite_model = os.path.join(flags.FLAGS.output_dir, 'model_1.tflite')
    self.assertTrue(tf.io.gfile.exists(tflite_model))

    # Check that input signature is as expected.
    with tf.io.gfile.GFile(tflite_model, 'rb') as model_file:
      model_content = model_file.read()
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    expected_input_shape = (1, 1) if include_frontend else (1, 96, 64, 1)
    np.testing.assert_array_equal(
        interpreter.get_input_details()[0]['shape'],
        expected_input_shape)


if __name__ == '__main__':
  absltest.main()
