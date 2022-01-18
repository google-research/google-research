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

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

from cold_posterior_flax.cifar10 import train
from cold_posterior_flax.cifar10.configs import default


class TrainTest(tf.test.TestCase):
  """Test cases for CIFAR10."""

  def test_train_sgmcmc(self):
    config = default.get_config()
    config.algorithm = 'sgmcmc'
    config.optimizer = 'sym_euler'
    config.arch = 'wrn8_1'
    config.batch_size = 2
    config.num_epochs = 1
    # TODO(basv): include evaluation in testing (mock_data is preventing this).
    config.do_eval = False
    workdir = self.create_tempdir().full_path
    with tfds.testing.mock_data(num_examples=1):
      train.train_and_evaluate(config, workdir)
    logging.info('workdir content: %s', tf.io.gfile.listdir(workdir))


if __name__ == '__main__':
  tf.test.main()
