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

"""Tests for train_moco."""

import os
import tempfile

from absl.testing import absltest
import tensorflow_datasets as tfds

from flax_models.moco import moco_lib
from flax_models.moco import model_resnet


class TrainMocoTest(absltest.TestCase):

  def test_resnet50(self):
    emb_size = 128
    module = model_resnet.ResNet50.partial(num_outputs=emb_size)
    model_dir = tempfile.mkdtemp()
    data_dir = '/.tfds/metadata'
    if data_dir is not None and not os.path.isdir(data_dir):
      raise RuntimeError(
          'You must first download the TFDS metadata for this test: '
          'svn checkout https://github.com/tensorflow/datasets/trunk/tensorflow_datasets/testing/metadata .tfds/metadata -q'  # pylint: disable=line-too-long
      )
    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      metrics = moco_lib.train(
          module,
          model_dir=model_dir,
          batch_size=8,
          eval_batch_size=8,
          num_moco_epochs=1,
          num_clf_epochs=1,
          moco_learning_rate=1,
          clf_learning_rate=30,
          steps_per_epoch=1,
          steps_per_eval=1,
      )
    self.assertIn('loss', metrics)
    self.assertIn('error_rate', metrics)


if __name__ == '__main__':
  absltest.main()
