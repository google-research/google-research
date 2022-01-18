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

"""Tests for train."""

import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from multiple_user_representations import train

FLAGS = flags.FLAGS
_TESTDATA_DIR = 'third_party/google_research/google_research/multiple_user_representations/testdata'


@flagsaver.flagsaver
def run_experiment(config_name,
                   testdata_path,
                   num_user_representations,
                   use_density_smoothing=False):
  """Sets FLAGS and runs train.py.

  Args:
    config_name: Name of one of the configuration files in configs/.
    testdata_path: The relative path to testdata dir for the dataset. Should be
      one of the dirs in testdata/.
    num_user_representations: Number of user representations.
    use_density_smoothing: If True, uses iterative density smoothing.
  """

  dir_path = 'third_party/google_research/google_research/multiple_user_representations/configs'
  FLAGS.config_path = os.path.join(dir_path, config_name + '.yaml')
  FLAGS.seed = 1234
  FLAGS.epochs = 1
  results_dir = os.path.join(FLAGS.test_tmpdir, 'tests/results/')
  dataset_path = os.path.join(_TESTDATA_DIR, testdata_path)
  FLAGS.root_dir = os.path.commonprefix([results_dir, dataset_path])
  FLAGS.results_dir = os.path.relpath(results_dir, FLAGS.root_dir)
  FLAGS.dataset_path = os.path.relpath(dataset_path, FLAGS.root_dir)
  FLAGS.num_representations = num_user_representations
  FLAGS.metrics_k = [1, 3, 5]
  if use_density_smoothing:
    FLAGS.retrieval_model_type = 'density_smoothed_retrieval'
    FLAGS.delta = 0.005
  config = train.load_config()
  train.setup_and_train(config)


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(1, 3)
  def test_amazon_data_experiment(self, num_user_representations):
    """Tests the amazon data experiment."""

    testdata_path = 'test_amazon_category_data'
    run_experiment('amazon_review_experiment', testdata_path,
                   num_user_representations)


if __name__ == '__main__':
  tf.test.main()
