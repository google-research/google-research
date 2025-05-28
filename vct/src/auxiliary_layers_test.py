# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Tests for auxiliary_layers."""

import tensorflow as tf

from vct.src import auxiliary_layers


class AuxiliaryLayersTest(tf.test.TestCase):

  def test_make_embedding_layer(self):
    layer = auxiliary_layers.make_embedding_layer(num_channels=8, d_model=16)
    # Last dim is `num_channels`.
    otp = layer(tf.ones((15, 3, 8)))
    # Last dim is `d_model`.
    self.assertEqual(otp.shape, (15, 3, 16))

  def test_start_sym(self):
    start = auxiliary_layers.StartSym(8)
    otp = start(tf.ones((15, 5, 8)))
    self.assertEqual(otp.shape, (15, 5, 8))

  def test_learned_position(self):
    pos = auxiliary_layers.LearnedPosition("pos", 5, 8)
    otp = pos(tf.ones((15, 5, 8)))
    self.assertEqual(otp.shape, (15, 5, 8))
    with self.assertRaises(ValueError):
      otp = pos(tf.ones((15, 5, 9)))


if __name__ == "__main__":
  tf.test.main()
