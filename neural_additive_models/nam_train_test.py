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

# Lint as: python3
"""Tests functionality of training NAM models."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

import nam_train

FLAGS = flags.FLAGS


class NAMTrainingTest(parameterized.TestCase):
  """Tests whether NAMs can be run without error."""

  @parameterized.named_parameters(
      ('classification', 'BreastCancer', False),
      ('regression', 'Housing', True),
  )
  @flagsaver.flagsaver
  def test_nam(self, dataset_name, regression):
    """Test whether the NAM training pipeline runs successfully or not."""
    FLAGS.training_epochs = 4
    FLAGS.save_checkpoint_every_n_epochs = 2
    FLAGS.early_stopping_epochs = 2
    FLAGS.dataset_name = dataset_name
    FLAGS.regression = regression
    FLAGS.num_basis_functions = 16
    logdir = os.path.join(self.create_tempdir().full_path, dataset_name)
    tf.gfile.MakeDirs(logdir)
    data_gen, _ = nam_train.create_test_train_fold(fold_num=1)
    nam_train.single_split_training(data_gen, logdir)


if __name__ == '__main__':
  absltest.main()
