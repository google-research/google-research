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

"""Tests for parametric_attention."""

import numpy as np
import tensorflow as tf

from multiple_user_representations.models import parametric_attention


class ParametricAttentionTest(tf.test.TestCase):

  def test_parametric_attention_model_with_single_representation(self):

    model = parametric_attention.SimpleParametricAttention(
        output_dimension=2,
        input_embedding_dimension=2,
        vocab_size=10,
        num_representations=1,
        max_sequence_size=20)

    input_batch = tf.convert_to_tensor(
        np.random.randint(low=0, high=10, size=(10, 20)))
    output = model(input_batch)

    self.assertIsInstance(model, tf.keras.Model)
    self.assertSequenceEqual(output.numpy().shape, [10, 1, 2])

  def test_parametric_attention_model_with_multiple_representations(self):

    model = parametric_attention.SimpleParametricAttention(
        output_dimension=2,
        input_embedding_dimension=2,
        vocab_size=10,
        num_representations=3,
        max_sequence_size=20)

    input_batch = tf.convert_to_tensor(
        np.random.randint(low=0, high=10, size=(10, 20)))
    output = model(input_batch)

    self.assertIsInstance(model, tf.keras.Model)
    self.assertSequenceEqual(output.numpy().shape, [10, 3, 2])


if __name__ == '__main__':
  tf.test.main()
