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

"""Unit test for classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from saccader.visual_attention import classification_model
from saccader.visual_attention import dram_config


class ClassificationModelTest(tf.test.TestCase):

  def test_import(self):
    self.assertIsNotNone(classification_model)

  def test_build(self):
    config = dram_config.get_config()
    batch_size = 10
    input_dims = 64
    num_classes = 10
    config.classification_model_config.num_classes = 10
    state = tf.placeholder(shape=(batch_size, input_dims), dtype=tf.float32)

    model = classification_model.ClassificationNetwork(
        config.classification_model_config)

    logits, _ = model(state)
    init_op = model.init_op

    with self.test_session() as sess:
      sess.run(init_op)
      self.assertEqual(
          (batch_size, num_classes),
          sess.run(
              logits,
              feed_dict={
                  state: np.random.rand(batch_size, input_dims)
              }).shape)


if __name__ == "__main__":
  tf.test.main()
