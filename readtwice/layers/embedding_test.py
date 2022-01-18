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

"""Tests for embedding layers."""

from absl.testing import parameterized
import tensorflow as tf

from readtwice import layers as readtwice_layers


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_embedding_lookup_2d_ids_no_mask(self, use_one_hot_lookup):
    embedding_table = tf.constant([
        [1.0, -1.0],  #
        [1.1, -1.1],  #
        [1.2, -1.2],  #
        [1.3, -1.3],  #
        [1.4, -1.4],  #
    ])
    vocab_size, embedding_size = embedding_table.shape.as_list()

    input_ids = tf.constant([
        [3, 2, 1],  #
        [4, 0, 4],  #
    ])

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        use_one_hot_lookup=use_one_hot_lookup)
    layer.build(None)  # Shapes are unused so we pass None.
    layer.embedding_table = embedding_table

    expected = [
        [
            [1.3, -1.3],  #
            [1.2, -1.2],  #
            [1.1, -1.1],  #
        ],  #
        [
            [1.4, -1.4],  #
            [1.0, -1.0],  #
            [1.4, -1.4],  #
        ],  #
    ]
    result = layer(input_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected, result)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_embedding_lookup_2d_ids_with_mask(self, use_one_hot_lookup):
    embedding_table = tf.constant([
        [1.0, -1.0],  #
        [1.1, -1.1],  #
        [1.2, -1.2],  #
        [1.3, -1.3],  #
        [1.4, -1.4],  #
    ])
    vocab_size, embedding_size = embedding_table.shape.as_list()

    input_ids = tf.constant([
        [3, 2, 1],  #
        [4, -1, 5],  #
    ])
    input_mask = tf.constant([
        [1, 1, 0],  #
        [1, 0, 0],  #
    ])

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        use_one_hot_lookup=use_one_hot_lookup)
    layer.build(None)  # Shapes are unused so we pass None.
    layer.embedding_table = embedding_table

    expected = [
        [
            [1.3, -1.3],  #
            [1.2, -1.2],  #
            [0.0, 0.0],  #
        ],  #
        [
            [1.4, -1.4],  #
            [0.0, 0.0],  #
            [0.0, 0.0],  #
        ],  #
    ]
    result = layer(input_ids, input_mask=input_mask)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected, result)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_embedding_lookup_1d_ids(self, use_one_hot_lookup):
    embedding_table = tf.constant([
        [1.0, -1.0],  #
        [1.1, -1.1],  #
        [1.2, -1.2],  #
        [1.3, -1.3],  #
        [1.4, -1.4],  #
    ])
    vocab_size, embedding_size = embedding_table.shape.as_list()

    input_ids = tf.constant([1, 0, 0, 3])
    input_mask = tf.constant([1, 1, 0, 1])

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        use_one_hot_lookup=use_one_hot_lookup)
    layer.build(None)  # Shapes are unused so we pass None.
    layer.embedding_table = embedding_table

    expected = [
        [1.1, -1.1],  #
        [1.0, -1.0],  #
        [0.0, 0.0],  #
        [1.3, -1.3],  #
    ]
    result = layer(input_ids, input_mask=input_mask)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected, result)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_embedding_lookup_3d_ids(self, use_one_hot_lookup):
    embedding_table = tf.constant([
        [1.0, -1.0],  #
        [1.1, -1.1],  #
        [1.2, -1.2],  #
        [1.3, -1.3],  #
        [1.4, -1.4],  #
    ])
    vocab_size, embedding_size = embedding_table.shape.as_list()

    input_ids = tf.constant([[
        [3, 2, 1],  #
        [4, 0, 4],  #
    ]])

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        use_one_hot_lookup=use_one_hot_lookup)
    layer.build(None)  # Shapes are unused so we pass None.
    layer.embedding_table = embedding_table

    expected = [[
        [
            [1.3, -1.3],  #
            [1.2, -1.2],  #
            [1.1, -1.1],  #
        ],  #
        [
            [1.4, -1.4],  #
            [1.0, -1.0],  #
            [1.4, -1.4],  #
        ],  #
    ]]
    result = layer(input_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected, result)

  @parameterized.named_parameters(
      ('using_gather', False),
      ('using_one_hot', True),
  )
  def test_embedding_lookup_random_init_no_mask(self, use_one_hot_lookup):
    vocab_size = 5
    embedding_size = 2

    input_ids = tf.constant([1, 0, 0, 3])
    input_size = input_ids.shape.as_list()[0]

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        use_one_hot_lookup=use_one_hot_lookup)

    result = layer(input_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(result)
    self.assertAllEqual([input_size, embedding_size], result.shape)

  @parameterized.named_parameters(
      ('no_projection', 0),
      ('embedding_size_equals_projection_size', 3),
  )
  def test_embedding_lookup_no_projection(self, projection_size):
    # Create an embedding table with width = projection_size
    embedding_table = tf.constant([
        [1.0, -1.0, 0.5],  #
        [1.1, -1.1, -0.5],  #
        [1.2, -1.2, -0.2],  #
        [1.3, -1.3, 0.3],  #
        [1.4, -1.4, 0.4],  #
    ])
    vocab_size, embedding_size = embedding_table.shape.as_list()

    input_ids = tf.constant([
        [3, 2, 1],  #
        [4, 0, 4],  #
    ])

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        projection_size=projection_size,
        use_one_hot_lookup=True)

    layer.build(None)  # Shapes are unused so we pass None.
    layer.embedding_table = embedding_table

    expected = [
        [
            [1.3, -1.3, 0.3],  #
            [1.2, -1.2, -0.2],  #
            [1.1, -1.1, -0.5],  #
        ],  #
        [
            [1.4, -1.4, 0.4],  #
            [1.0, -1.0, 0.5],  #
            [1.4, -1.4, 0.4],  #
        ],  #
    ]
    result = layer(input_ids)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected, result)

  def test_embedding_lookup_with_projection(self):
    # Create an embedding table with width != projection_size
    embedding_table = tf.constant([
        [1.0, -1.0, 0.5],  #
        [1.1, -1.1, -0.4],  #
        [1.2, -1.2, -0.5],  #
        [1.3, -1.3, 0.8],  #
        [1.4, -1.4, 0.9],  #
    ])

    projection_size = 2  #  Different from the embedding_dimension.
    vocab_size, embedding_size = embedding_table.shape.as_list()
    input_ids = tf.constant([
        [3, 2, 1],  #
        [4, 0, 4],  #
    ])

    input_mask = tf.constant([
        [1, 0, 0],  #
        [0, 0, 1],  #
    ])

    layer = readtwice_layers.EmbeddingLookup(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        projection_size=projection_size,
        use_one_hot_lookup=True)

    layer.build(None)  # Shapes are unused so we pass None.
    layer.embedding_table = embedding_table

    # Dense layer to use for projection. Note that, we have a non-zero
    # bias initializer here to ensure that the bias term doesn't get through
    # to the masked_ids after projection.
    layer.embedding_projection = tf.keras.layers.Dense(
        units=projection_size,
        activation=None,
        use_bias=True,
        kernel_initializer='ones',
        bias_initializer='ones')

    expected = [
        [
            [1.8, 1.8],  # [1.3, -1.3, 0.8] * kernel_initializer + 1 (bias).
            [0., 0.],  #
            [0., 0.],  #
        ],  #
        [
            [0., 0.],  #
            [0., 0.],  #
            [1.9, 1.9],  # [1.4, -1.4, 0.9] * kernel_initializer + 1 (bias).
        ],  #
    ]
    result = layer(input_ids, input_mask)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(expected, result)


if __name__ == '__main__':
  tf.test.main()
