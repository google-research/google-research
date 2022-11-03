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

"""End-to-end test for ImageNet.

Tests for imagenet.resnet50_train, run_predict, run_temp_scaling, and
run_metrics. Real data doesn't work under blaze, so execute the test binary
directly.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow.compat.v2 as tf

from uq_benchmark_2019.imagenet import resnet50_train  # pylint: disable=line-too-long
from uq_benchmark_2019.imagenet import run_metrics
from uq_benchmark_2019.imagenet import run_predict
from uq_benchmark_2019.imagenet import run_temp_scaling

gfile = tf.io.gfile

flags.DEFINE_bool('fake_data', True, 'Use dummy random data.')
flags.DEFINE_bool('fake_training', True, 'Train with trivial number of steps.')

DATA_NAMES = ['train', 'test', 'corrupt-static-gaussian_noise-2', 'celeb_a']

METHODS = ['vanilla', 'll_dropout', 'll_svi', 'dropout']


class EndToEndTest(parameterized.TestCase):

  @parameterized.parameters(*[(d, m) for d in DATA_NAMES for m in METHODS])  # pylint: disable=g-complex-comprehension
  def test_end_to_end_train(self, data_name, method):
    with tempfile.TemporaryDirectory() as model_dir:
      metrics = ['sparse_categorical_crossentropy']
      if flags.FLAGS.fake_data and (data_name != 'test'):
        pass
      else:
        temp_model_dir = os.path.join(model_dir, data_name, method)
        resnet50_train.run(
            method, temp_model_dir, task_number=0, use_tpu=False, tpu=None,
            metrics=metrics, fake_data=flags.FLAGS.fake_data,
            fake_training=flags.FLAGS.fake_training)

        run_predict.run(
            data_name, temp_model_dir, batch_size=8, predictions_per_example=4,
            max_examples=44, output_dir=temp_model_dir,
            fake_data=flags.FLAGS.fake_data)

        tmpl = os.path.join(temp_model_dir, '*_small_*')
        glob_results = gfile.glob(tmpl)
        path = glob_results[0]
        if data_name == 'valid':
          run_temp_scaling(path)
        run_metrics.run(path, path, model_dir_ensemble=None,
                        use_temp_scaling=False)

if __name__ == '__main__':
  absltest.main()
