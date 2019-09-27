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

"""Unit test for DRAM model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from saccader.visual_attention import dram
from saccader.visual_attention import dram_config


class DramTest(tf.test.TestCase, parameterized.TestCase):

  def test_import(self):
    self.assertIsNotNone(dram)

  @parameterized.named_parameters(("training mode", True),
                                  ("inference mode", False))
  def test_build(self, is_training):
    config = dram_config.get_config()
    num_times = 2
    image_shape = (28, 28, 1)
    num_classes = 10
    config.num_classes = num_classes
    config.num_units_rnn_layers = [10, 10,]
    config.num_times = num_times
    batch_size = 3
    images = tf.constant(
        np.random.rand(*((batch_size,) + image_shape)), dtype=tf.float32)
    model = dram.DRAMNetwork(config)

    logits_t = model(images, num_times=num_times, is_training=is_training)[0]
    init_op = model.init_op
    self.evaluate(init_op)
    self.assertEqual((batch_size, num_classes),
                     self.evaluate(logits_t[-1]).shape)

if __name__ == "__main__":
  tf.test.main()
