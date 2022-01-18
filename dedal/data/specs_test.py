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

"""Tests for the DatasetBuilder."""

import gin
import tensorflow as tf

from dedal.data import builder


GIN_CONFIG = """
import dedal.multi_task
import dedal.vocabulary
import dedal.data.nlp
import dedal.data.specs

vocabulary.get_default.vocab = %vocabulary.alternative
SEQUENCE_LENGTH = 512

DatasetBuilder.data_loader = @specs.FakePairsLoader()
specs.FakePairsLoader.max_len = %SEQUENCE_LENGTH
DatasetBuilder.batched_transformations = [
    @seq/Reshape(),
    @key/Reshape(),
    @CreateHomologyTargets(),
    @CreateBatchedWeights(),
    @PadNegativePairs(),
]
seq/Reshape.on = 'sequence'
seq/Reshape.shape = [-1, %SEQUENCE_LENGTH]
key/Reshape.on = 'fam_key'
key/Reshape.shape = [-1]
CreateHomologyTargets.on = 'fam_key'
CreateHomologyTargets.out = 'homology/targets'
CreateBatchedWeights.on = 'alignment/targets'
CreateBatchedWeights.out = 'alignment/weights'
PadNegativePairs.on = ('alignment/targets', 'alignment/weights')
DatasetBuilder.labels = @labels/multi_task.Backbone()
labels/multi_task.Backbone.alignments = [
   ('alignment/targets', 'alignment/weights'),
   'homology/targets',
]
"""


class FakePairsLoaderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()
    gin.parse_config(GIN_CONFIG)

  def test_transforms_from_gin(self):
    ds_builder = builder.DatasetBuilder()
    batch = 32
    ds = ds_builder.make('train', batch)
    inputs, y_true, weights, _ = next(iter(ds))
    self.assertIsInstance(inputs, tf.Tensor)
    self.assertEqual(inputs.dtype, tf.int32)
    self.assertIsInstance(y_true, dict)
    self.assertGreater(len(y_true), 0)
    self.assertIsInstance(weights, dict)
    self.assertGreater(len(weights), 0)
    self.assertEqual(y_true['alignments/0'].dtype, tf.int32)
    self.assertEqual(y_true['alignments/0'].shape, (2 * batch, 3, 1024))
    self.assertEqual(y_true['alignments/1'].dtype, tf.int32)
    self.assertEqual(y_true['alignments/1'].shape, (2 * batch, 1))
    self.assertEqual(weights['alignments/0'].dtype, tf.float32)
    self.assertEqual(weights['alignments/0'].shape, (2 * batch,))
    self.assertEqual(weights['alignments/1'].dtype, tf.float32)
    self.assertEqual(weights['alignments/1'].shape, (2 * batch,))


if __name__ == '__main__':
  tf.test.main()
