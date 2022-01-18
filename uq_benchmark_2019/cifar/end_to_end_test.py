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

"""Tests for cifar.run_train and cifar.run_predict.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from uq_benchmark_2019.cifar import models_lib
from uq_benchmark_2019.cifar import run_predict
from uq_benchmark_2019.cifar import run_train

flags.DEFINE_bool('fake_data', True, 'Use dummy random data.')
flags.DEFINE_bool('fake_training', True, 'Train with trivial number of steps.')


class EndToEndTest(parameterized.TestCase):

  @parameterized.named_parameters(*[(m, m) for m in models_lib.METHODS])
  def test_end_to_end_train(self, method):
    with tempfile.TemporaryDirectory() as model_dir:
      run_train.run(method, model_dir,
                    fake_data=flags.FLAGS.fake_data,
                    fake_training=flags.FLAGS.fake_training)

      data_names = ['test']
      if not flags.FLAGS.fake_data:
        data_names.extend(['corrupt-static-gaussian_noise-2'])

      for data_name in data_names:
        run_predict.run(
            data_name, model_dir,
            predictions_per_example=8,
            max_examples=44,
            output_dir=model_dir+'/predictions/',
            fake_data=flags.FLAGS.fake_data)

if __name__ == '__main__':
  absltest.main()
