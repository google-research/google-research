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

"""Tests for non_semantic_speech_benchmark.eval_embedding.finetune.models."""

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from non_semantic_speech_benchmark.eval_embedding.finetune import models


class ModelTest(parameterized.TestCase):

  @parameterized.parameters(
      {'num_clusters': 0, 'alpha_init': 0},
      {'num_clusters': 4, 'alpha_init': 0},
      {'num_clusters': 0, 'alpha_init': 1.0},
  )
  def test_basic_model(self, num_clusters, alpha_init):
    m = models.get_keras_model(num_classes=5, input_length=16000,
                               num_clusters=num_clusters, alpha_init=alpha_init)
    o = m(tf.zeros([4, 16000], dtype=tf.float32))
    self.assertEqual(o.shape, (4, 5))

if __name__ == '__main__':
  absltest.main()
