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

"""Tests for vocabulary."""

import tensorflow as tf

from dedal import vocabulary


class VocabularyTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.voc = vocabulary.proteins

  def test_indices(self):
    self.assertLen(self.voc, 29)
    self.assertTrue(self.voc.get('*'), 28)
    self.assertEqual(self.voc.padding_code, 0)
    self.assertLess(self.voc.get('C'), self.voc.get('>'))
    self.assertLess(len(self.voc.get_specials(with_padding=False)),
                    len(self.voc.get_specials(with_padding=True)))

  def test_encode_decode(self):
    inputs = 'FGGGLMNPQPQPQQRV'
    encoded = self.voc.encode(inputs)
    self.assertLen(encoded, len(inputs))
    self.assertEqual(self.voc.decode(encoded), inputs)

  def test_mask(self):
    inputs = tf.constant([4, 5, 2, 26, 0, 0, 0], dtype=tf.int32)
    self.assertAllEqual(self.voc.padding_mask(inputs), inputs != 0)

  def test_serialize_deserialize(self):
    serialized = tf.keras.utils.serialize_keras_object(self.voc)
    self.assertIsInstance(serialized, dict)
    voc = tf.keras.utils.deserialize_keras_object(serialized)
    self.assertSequenceEqual(voc.tokens, self.voc.tokens)
    self.assertSequenceEqual(voc.specials, self.voc.specials)
    self.assertEqual(len(voc), len(self.voc))


if __name__ == '__main__':
  tf.test.main()
