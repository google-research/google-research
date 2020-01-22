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

"""Tests for learning.brain.research.deep_calibration.v2.criteo.run_train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from uq_benchmark_2019.criteo import models_lib
from uq_benchmark_2019.criteo import run_train


class RunTrainTest(parameterized.TestCase):

  @parameterized.named_parameters(*[(m, m) for m in models_lib.METHODS])
  def test_run_train(self, method):
    with tempfile.TemporaryDirectory() as model_dir:
      run_train.run(method, model_dir,
                    num_epochs=1, fake_data=True, fake_training=True)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
