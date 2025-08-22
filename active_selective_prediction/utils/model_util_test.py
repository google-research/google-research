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

"""Unit tests for `model_util.py`."""

import unittest

from active_selective_prediction.utils import model_util
import tensorflow as tf


class TestModelLoadingFunctions(unittest.TestCase):
  """Tests model loading functions."""

  def test_get_simple_mlp(self):
    """Tests get_simple_mlp function."""
    model = model_util.get_simple_mlp(input_shape=(2,), num_classes=2)
    batch_x = tf.ones((2, 2), dtype=tf.float32)
    batch_output = model(batch_x)
    self.assertEqual(batch_output.shape, (2, 2))

  def test_get_roberta_mlp(self):
    """Tests get_roberta_mlp function."""
    model = model_util.get_roberta_mlp(input_shape=(768,), num_classes=5)
    batch_x = tf.ones((2, 768), dtype=tf.float32)
    batch_output = model(batch_x)
    self.assertEqual(batch_output.shape, (2, 5))

  def test_get_simple_convnet(self):
    """Tests get_simple_convnet function."""
    model = model_util.get_simple_convnet(
        input_shape=(32, 32, 3), num_classes=2
    )
    batch_x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
    batch_output = model(batch_x)
    self.assertEqual(batch_output.shape, (2, 2))

  def test_get_cifar_resnet(self):
    """Tests get_cifar_resnet function."""
    model = model_util.get_cifar_resnet(
        input_shape=(32, 32, 3), num_classes=10
    )
    batch_x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
    batch_output = model(batch_x)
    self.assertEqual(batch_output.shape, (2, 10))

  def test_get_densenet121(self):
    """Tests get_densenet121 function."""
    model = model_util.get_densenet121(input_shape=(32, 32, 3), num_classes=2)
    batch_x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
    batch_output = model(batch_x)
    self.assertEqual(batch_output.shape, (2, 2))

  def test_get_resnet50(self):
    """Tests get_resnet50 function."""
    model = model_util.get_resnet50(input_shape=(32, 32, 3), num_classes=2)
    batch_x = tf.ones((2, 32, 32, 3), dtype=tf.float32)
    batch_output = model(batch_x)
    self.assertEqual(batch_output.shape, (2, 2))


if __name__ == '__main__':
  unittest.main()
