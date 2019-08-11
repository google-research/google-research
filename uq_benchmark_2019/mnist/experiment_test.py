# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for mnist.experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from uq_benchmark_2019.mnist import experiment
from uq_benchmark_2019.mnist import models_lib

flags.DEFINE_bool('fake_data', True, 'Whether to use fake dataset data.')


class ExperimentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *[('%s_%s' % pair,) + pair for pair in itertools.product(
          models_lib.METHODS, models_lib.ARCHITECTURES)])
  def test_end_to_end_fake_data(self, method, arch):
    print('testing', method, arch)
    with tempfile.TemporaryDirectory() as output_dir:
      experiment.run(method, arch, output_dir, test_level=2)

  def test_end_to_end_real_data(self):
    if not flags.FLAGS.fake_data:
      with tempfile.TemporaryDirectory() as output_dir:
        experiment.run('vanilla', 'mlp', output_dir, test_level=1)

if __name__ == '__main__':
  absltest.main()
