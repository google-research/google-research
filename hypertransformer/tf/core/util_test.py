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

"""Tests for `util.py`."""
import tensorflow.compat.v1 as tf

from hypertransformer.tf.core import util


class UtilTest(tf.test.TestCase):

  def _encode_samples(self, batch_size, image_size,
                      io):
    """Creates empty images and labels and encodes for Transformer."""
    images = tf.zeros(shape=(batch_size, image_size, image_size, 1),
                      dtype=tf.float32)
    labels = tf.zeros(shape=(batch_size,), dtype=tf.int32)
    return io.encode_samples(images, labels)

  def _check_weights(self, encoded, num_weights,
                     image_size, io):
    """Decodes weights and checks their number and their shapes."""
    weights = io.decode_weights(encoded)
    self.assertLen(weights, num_weights)
    for weight in weights:
      self.assertEqual(weight.shape, (image_size ** 2,))

  def test_simple_transformer_io(self):
    """Tests `SimpleTransformerIO` Transformer adapter."""
    batch_size = 8
    embedding_dim = 8
    num_weights = 8
    image_size = 2
    io = util.SimpleTransformerIO(num_labels=4, num_weights=num_weights,
                                  embedding_dim=embedding_dim,
                                  weight_block_size=image_size ** 2)
    encoded = self._encode_samples(batch_size, image_size, io)
    self.assertEqual(encoded.shape,
                     (batch_size, embedding_dim + (image_size ** 2)))
    self._check_weights(encoded, num_weights=num_weights,
                        image_size=image_size, io=io)

  def test_joint_io(self):
    """Tests `JointTransformerIO` Transformer adapter."""
    batch_size = 8
    embedding_dim = 8
    num_weights = 8
    image_size = 2
    io = util.JointTransformerIO(num_labels=4, num_weights=num_weights,
                                 embedding_dim=embedding_dim,
                                 weight_block_size=image_size ** 2)
    encoded = self._encode_samples(batch_size, image_size, io)
    shape = (batch_size + num_weights, embedding_dim + (image_size ** 2))
    self.assertEqual(encoded.shape, shape)
    self._check_weights(encoded, num_weights=num_weights,
                        image_size=image_size, io=io)

if __name__ == '__main__':
  tf.test.main()
