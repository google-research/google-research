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

"""Tests for third_party.google_research.google_research.smug_saliency.models.train_mnist."""

import os
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from tensorflow.compat.v1 import gfile

from smug_saliency.mnist_models import train_mnist

FLAGS = flags.FLAGS


class TrainMnistTest():

  def test_train_mnist(self):
    # Create the random data and write it to the disk.
    test_subdirectory = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

    # Create the model parameters.
    model_path = os.path.join(test_subdirectory, 'temp_model')

    with flagsaver.flagsaver(
        model_path=model_path,
        save_period=1,
        num_dense_units='4,4',
        epochs=1,
        learning_rate=0.1,
        dropout=0.0,
        batch_size=32):
      train_mnist.main(argv=())

    # Verify that the trained model was saved.
    self.assertTrue(gfile.Exists(os.path.join(model_path, 'test_accuracy.txt')))
    self.assertLen(gfile.Glob(os.path.join(model_path, 'weights_epoch*')), 1)

if __name__ == '__main__':
  absltest.main()
