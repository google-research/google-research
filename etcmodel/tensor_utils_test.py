# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Tests for tensor utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from etcmodel import tensor_utils


class TensorUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_flatten_dims(self):
    tensor = tf.reshape(tf.range(2 * 3 * 4 * 5), [2, 3, 4, 5])

    result1 = tensor_utils.flatten_dims(tensor, last_dim=1)
    self.assertAllEqual([6, 4, 5], result1.shape)
    self.assertAllEqual(tf.range(2 * 3 * 4 * 5), tf.reshape(result1, [-1]))

    self.assertAllEqual([2, 3, 20],
                        tensor_utils.flatten_dims(tensor, first_dim=-2).shape)

    self.assertAllEqual([2, 12, 5],
                        tensor_utils.flatten_dims(
                            tensor, first_dim=1, last_dim=-2).shape)

    self.assertAllEqual([24, 5],
                        tensor_utils.flatten_dims(tensor, last_dim=-2).shape)

    self.assertAllEqual([2 * 3 * 4 * 5],
                        tensor_utils.flatten_dims(tensor).shape)

    self.assertAllEqual([2, 3, 4, 5],
                        tensor_utils.flatten_dims(
                            tensor, first_dim=1, last_dim=-3).shape)

    self.assertAllEqual([7], tensor_utils.flatten_dims(tf.ones([7])).shape)

    self.assertAllEqual([12], tensor_utils.flatten_dims(tf.ones([4, 3])).shape)

    with self.assertRaises(ValueError):
      tensor_utils.flatten_dims(tensor, first_dim=4)

    with self.assertRaises(ValueError):
      tensor_utils.flatten_dims(tensor, last_dim=-5)

    with self.assertRaises(ValueError):
      tensor_utils.flatten_dims(tensor, first_dim=2, last_dim=1)

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_gather_by_one_hot_1d_params(self, gather_fn):
    params = tf.constant([0.0, 0.1, 0.2, 0.3])

    self.assertAllClose(0.2, gather_fn(params, 2))

    self.assertAllClose([0.2, 0.0, 0.3], gather_fn(params, [2, 0, 3]))

    self.assertAllClose([[0.2, 0.0], [0.2, 0.3]],
                        gather_fn(params, [[2, 0], [2, 3]]))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_gather_by_one_hot_2d_params(self, gather_fn):
    params = tf.constant([
        [0.0, 0.0],  #
        [0.1, -0.1],  #
        [0.2, -0.2],  #
        [0.3, -0.3],  #
    ])

    self.assertAllClose([0.2, -0.2], gather_fn(params, 2))

    self.assertAllClose(
        [
            [0.2, -0.2],  #
            [0.0, 0.0],  #
            [0.3, -0.3],  #
        ],
        gather_fn(params, [2, 0, 3]))

    self.assertAllClose(
        [
            [[0.2, -0.2], [0.0, 0.0]],  #
            [[0.2, -0.2], [0.3, -0.3]],  #
        ],
        gather_fn(params, [[2, 0], [2, 3]]))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_gather_by_one_hot_3d_params(self, gather_fn):
    params = tf.constant([
        [[0.0, 0.0], [0.0, 0.0]],  #
        [[0.1, -0.1], [-0.1, 0.1]],  #
        [[0.2, -0.2], [-0.2, 0.2]],  #
        [[0.3, -0.3], [-0.3, 0.3]],  #
    ])

    self.assertAllClose([[0.2, -0.2], [-0.2, 0.2]], gather_fn(params, 2))

    self.assertAllClose(
        [
            [[0.2, -0.2], [-0.2, 0.2]],  #
            [[0.0, 0.0], [0.0, 0.0]],  #
            [[0.3, -0.3], [-0.3, 0.3]],  #
        ],
        gather_fn(params, [2, 0, 3]))

    self.assertAllClose(
        [
            [
                [[0.2, -0.2], [-0.2, 0.2]],  #
                [[0.0, 0.0], [0.0, 0.0]],  #
            ],  #
            [
                [[0.2, -0.2], [-0.2, 0.2]],  #
                [[0.3, -0.3], [-0.3, 0.3]],  #
            ]
        ],
        gather_fn(params, [[2, 0], [2, 3]]))

  def test_gather_by_one_hot_with_indices_out_of_range(self):
    params = tf.constant([
        [0.0, 0.0],  #
        [0.1, -0.1],  #
        [0.2, -0.2],  #
        [0.3, -0.3],  #
    ])

    self.assertAllClose(
        [
            [0.2, -0.2],  #
            [0.0, 0.0],  #
            [0.1, -0.1],  #
            [0.3, -0.3],  #
            [0.0, 0.0],  #
        ],
        tensor_utils.gather_by_one_hot(params, [2, -1, 1, 3, 4]))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.batch_gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_batch_gather_by_one_hot_1_batch_dim_plus_1d_params(self, gather_fn):
    params = tf.constant([
        [0.0, 0.1, 0.2, 0.3],  #
        [1.0, 1.1, 1.2, 1.3],  #
    ])

    # This case doesn't work for `tf.gather` so we don't use `gather_fn`.
    self.assertAllClose([0.2, 1.1],
                        tensor_utils.batch_gather_by_one_hot(
                            params, [2, 1], batch_dims=1))

    self.assertAllClose([[0.2, 0.0], [1.2, 1.3]],
                        gather_fn(params, [[2, 0], [2, 3]], batch_dims=1))

    self.assertAllClose(
        [
            [[0.2, 0], [0.2, 0.3]],  #
            [[1.1, 1.0], [1.2, 1.1]]
        ],
        gather_fn(
            params,
            [
                [[2, 0], [2, 3]],  #
                [[1, 0], [2, 1]]
            ],
            batch_dims=1))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.batch_gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_batch_gather_by_one_hot_1_batch_dim_plus_2d_params(self, gather_fn):
    params = tf.constant([
        [
            [0.0, 0.0],  #
            [0.1, -0.1],  #
            [0.2, -0.2],  #
            [0.3, -0.3],  #
        ],
        [
            [1.0, -1.0],  #
            [1.1, -1.1],  #
            [1.2, -1.2],  #
            [1.3, -1.3],  #
        ]
    ])

    # This case doesn't work for `tf.gather` so we don't use `gather_fn`.
    self.assertAllClose([[0.2, -0.2], [1.1, -1.1]],
                        tensor_utils.batch_gather_by_one_hot(
                            params, [2, 1], batch_dims=1))

    self.assertAllClose(
        [
            [[0.2, -0.2], [0.0, 0.0]],  #
            [[1.2, -1.2], [1.3, -1.3]]
        ],
        gather_fn(params, [[2, 0], [2, 3]], batch_dims=1))

    self.assertAllClose(
        [
            [
                [[0.2, -0.2], [0.0, 0.0]],  #
                [[0.2, -0.2], [0.3, -0.3]]
            ],
            [
                [[1.1, -1.1], [1.0, -1.0]],  #
                [[1.2, -1.2], [1.1, -1.1]]
            ]
        ],
        gather_fn(
            params,
            [
                [[2, 0], [2, 3]],  #
                [[1, 0], [2, 1]]
            ],
            batch_dims=1))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.batch_gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_batch_gather_by_one_hot_1_batch_dim_plus_3d_params(self, gather_fn):
    params = tf.constant([
        [
            [[0.0, 0.0], [0.0, 0.0]],  #
            [[0.1, -0.1], [-0.1, 0.1]],  #
            [[0.2, -0.2], [-0.2, 0.2]],  #
            [[0.3, -0.3], [-0.3, 0.3]],  #
        ],
        [
            [[1.0, 1.0], [1.0, 1.0]],  #
            [[1.1, -1.1], [-1.1, 1.1]],  #
            [[1.2, -1.2], [-1.2, 1.2]],  #
            [[1.3, -1.3], [-1.3, 1.3]],  #
        ]
    ])

    # This case doesn't work for `tf.gather` so we don't use `gather_fn`.
    self.assertAllClose(
        [
            [[0.2, -0.2], [-0.2, 0.2]],  #
            [[1.1, -1.1], [-1.1, 1.1]]
        ],
        tensor_utils.batch_gather_by_one_hot(params, [2, 1], batch_dims=1))

    self.assertAllClose(
        [
            [
                [[0.2, -0.2], [-0.2, 0.2]],  #
                [[0.0, 0.0], [0.0, 0.0]]
            ],  #
            [
                [[1.2, -1.2], [-1.2, 1.2]],  #
                [[1.3, -1.3], [-1.3, 1.3]]
            ]
        ],
        gather_fn(params, [[2, 0], [2, 3]], batch_dims=1))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.batch_gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_batch_gather_by_one_hot_2_batch_dims_plus_2d_params(self, gather_fn):
    params = tf.constant([
        [
            [
                [0.0, 0.0],  #
                [0.1, -0.1],  #
                [0.2, -0.2],  #
                [0.3, -0.3],  #
            ],
            [
                [1.0, -1.0],  #
                [1.1, -1.1],  #
                [1.2, -1.2],  #
                [1.3, -1.3],  #
            ]
        ],
        [
            [
                [2.0, -2.0],  #
                [2.1, -2.1],  #
                [2.2, -2.2],  #
                [2.3, -2.3],  #
            ],
            [
                [3.0, -3.0],  #
                [3.1, -3.1],  #
                [3.2, -3.2],  #
                [3.3, -3.3],  #
            ]
        ]
    ])

    # This case doesn't work for `tf.gather` so we don't use `gather_fn`.
    self.assertAllClose(
        [
            [[0.2, -0.2], [1.1, -1.1]],  #
            [[2.2, -2.2], [3.3, -3.3]]
        ],
        tensor_utils.batch_gather_by_one_hot(
            params, [[2, 1], [2, 3]], batch_dims=2))

    self.assertAllClose(
        [
            [
                [[0.2, -0.2], [0.0, 0.0]],  #
                [[1.2, -1.2], [1.3, -1.3]]
            ],
            [
                [[2.1, -2.1], [2.0, -2.0]],  #
                [[3.2, -3.2], [3.1, -3.1]]
            ]
        ],
        gather_fn(
            params,
            [
                [[2, 0], [2, 3]],  #
                [[1, 0], [2, 1]]
            ],
            batch_dims=2))

  @parameterized.named_parameters(
      ('using_one_hot', tensor_utils.batch_gather_by_one_hot),
      ('using_gather', tf.gather),
  )
  def test_batch_gather_by_one_hot_2_batch_dims_plus_3d_params(self, gather_fn):
    params = tf.constant([
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],  #
                [[0.1, -0.1], [-0.1, 0.1]],  #
                [[0.2, -0.2], [-0.2, 0.2]],  #
                [[0.3, -0.3], [-0.3, 0.3]],  #
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],  #
                [[1.1, -1.1], [-1.1, 1.1]],  #
                [[1.2, -1.2], [-1.2, 1.2]],  #
                [[1.3, -1.3], [-1.3, 1.3]],  #
            ]
        ],
        [
            [
                [[2.0, 2.0], [2.0, 2.0]],  #
                [[2.1, -2.1], [-2.1, 2.1]],  #
                [[2.2, -2.2], [-2.2, 2.2]],  #
                [[2.3, -2.3], [-2.3, 2.3]],  #
            ],
            [
                [[3.0, 3.0], [3.0, 3.0]],  #
                [[3.1, -3.1], [-3.1, 3.1]],  #
                [[3.2, -3.2], [-3.2, 3.2]],  #
                [[3.3, -3.3], [-3.3, 3.3]],  #
            ]
        ]
    ])

    # This case doesn't work for `tf.gather` so we don't use `gather_fn`.
    self.assertAllClose(
        [
            [
                [[0.2, -0.2], [-0.2, 0.2]],  #
                [[1.1, -1.1], [-1.1, 1.1]],  #
            ],
            [
                [[2.2, -2.2], [-2.2, 2.2]],  #
                [[3.3, -3.3], [-3.3, 3.3]],  #
            ]
        ],
        tensor_utils.batch_gather_by_one_hot(
            params, [[2, 1], [2, 3]], batch_dims=2))

    self.assertAllClose(
        [
            [
                [
                    [[0.2, -0.2], [-0.2, 0.2]],  #
                    [[0.0, 0.0], [0.0, 0.0]],  #
                ],
                [
                    [[1.2, -1.2], [-1.2, 1.2]],  #
                    [[1.3, -1.3], [-1.3, 1.3]],  #
                ],
            ],
            [
                [
                    [[2.1, -2.1], [-2.1, 2.1]],  #
                    [[2.0, 2.0], [2.0, 2.0]],  #
                ],
                [
                    [[3.2, -3.2], [-3.2, 3.2]],  #
                    [[3.1, -3.1], [-3.1, 3.1]],  #
                ],
            ]
        ],
        gather_fn(
            params,
            [
                [[2, 0], [2, 3]],  #
                [[1, 0], [2, 1]]
            ],
            batch_dims=2))

  def test_batch_gather_by_one_hot_with_default_batch_dim(self):
    params = tf.constant([
        [
            [0.0, 0.0],  #
            [0.1, -0.1],  #
            [0.2, -0.2],  #
            [0.3, -0.3],  #
        ],
        [
            [1.0, -1.0],  #
            [1.1, -1.1],  #
            [1.2, -1.2],  #
            [1.3, -1.3],  #
        ]
    ])

    # `batch_dim` should be inferred as 1.
    self.assertAllClose(
        [
            [[0.2, -0.2], [0.0, 0.0]],  #
            [[1.2, -1.2], [1.3, -1.3]]
        ],
        tensor_utils.batch_gather_by_one_hot(params, [[2, 0], [2, 3]]))

  def test_batch_gather_by_one_hot_with_indices_out_of_range(self):
    params = tf.constant([
        [
            [0.0, 0.0],  #
            [0.1, -0.1],  #
            [0.2, -0.2],  #
            [0.3, -0.3],  #
        ],
        [
            [1.0, -1.0],  #
            [1.1, -1.1],  #
            [1.2, -1.2],  #
            [1.3, -1.3],  #
        ]
    ])

    self.assertAllClose(
        [
            [
                [0.2, -0.2],  #
                [0.0, 0.0],  #
                [0.1, -0.1],  #
            ],
            [
                [1.0, -1.0],  #
                [1.3, -1.3],  #
                [0.0, 0.0],  #
            ]
        ],
        tensor_utils.batch_gather_by_one_hot(params, [[2, -1, 1], [0, 3, 4]]))

  def test_pad_to_multiple_1d(self):
    tensor = tf.range(3) + 1

    self.assertAllEqual([1, 2, 3],
                        tensor_utils.pad_to_multiple(tensor, factor=3, axis=0))

    self.assertAllEqual([1, 2, 3, 0, 0],
                        tensor_utils.pad_to_multiple(tensor, factor=5, axis=0))

    self.assertAllEqual([1, 2, 3, 0, 0, 0, 0],
                        tensor_utils.pad_to_multiple(tensor, factor=7, axis=0))

    self.assertAllEqual([1, 2, 3, 0],
                        tensor_utils.pad_to_multiple(tensor, factor=2, axis=0))

    self.assertAllEqual([1, 2, 3],
                        tensor_utils.pad_to_multiple(tensor, factor=1, axis=0))

  def test_pad_to_multiple_padding_mode(self):
    tensor = tf.range(3) + 1

    self.assertAllEqual([1, 2, 3, 2, 1],
                        tensor_utils.pad_to_multiple(
                            tensor, factor=5, axis=0, mode='REFLECT'))

    self.assertAllEqual([1, 2, 3, 3, 2],
                        tensor_utils.pad_to_multiple(
                            tensor, factor=5, axis=0, mode='SYMMETRIC'))

    self.assertAllEqual([1, 2, 3, -1, -1],
                        tensor_utils.pad_to_multiple(
                            tensor, factor=5, axis=0, constant_values=-1))

  def test_pad_to_multiple_2d(self):
    tensor = tf.ones([3, 5], dtype=tf.float32)

    self.assertAllEqual(
        [
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [0, 0, 0, 0, 0],  #
        ],
        tensor_utils.pad_to_multiple(tensor, factor=4, axis=0))

    self.assertAllEqual(
        [
            [1, 1, 1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 1, 1, 0, 0, 0],  #
        ],
        tensor_utils.pad_to_multiple(tensor, factor=4, axis=-1))

    self.assertAllEqual(tensor,
                        tensor_utils.pad_to_multiple(tensor, factor=5, axis=1))

  def test_pad_to_multiple_3d(self):
    tensor = tf.ones([2, 3, 5], dtype=tf.float32)

    self.assertAllEqual(
        [
            [
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
            ],  #
            [
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
            ],  #
            [
                [0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0],  #
            ]
        ],
        tensor_utils.pad_to_multiple(tensor, factor=3, axis=0))

    self.assertAllEqual(tensor,
                        tensor_utils.pad_to_multiple(tensor, factor=3, axis=-2))

    self.assertAllEqual(
        [
            [
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
            ],  #
            [
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 0],  #
            ]
        ],
        tensor_utils.pad_to_multiple(tensor, factor=3, axis=-1))

  def test_pad_to_multiple_static_shape(self):
    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # a static `batch_size` is unknown.
    tensor = tf.compat.v1.placeholder_with_default(
        np.ones(shape=[2, 5], dtype=np.int32), shape=[None, 5])

    result = tensor_utils.pad_to_multiple(tensor, factor=3, axis=-1)

    static_batch_size = tensor.shape.as_list()[0]
    self.assertAllEqual([static_batch_size, 6], result.shape.as_list())

  def test_split_into_blocks_1d(self):
    tensor = tf.range(6) + 1

    self.assertAllEqual([[1, 2], [3, 4], [5, 6]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=2, axis=0))

    self.assertAllEqual([[1, 2, 3], [4, 5, 6]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=3, axis=0))

    self.assertAllEqual([[1, 2, 3, 4], [5, 6, 0, 0]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=4, axis=0))

    self.assertAllEqual([[1, 2, 3, 4, 5], [6, 0, 0, 0, 0]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=5, axis=0))

    self.assertAllEqual([[1, 2, 3, 4, 5, 6]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=6, axis=0))

    self.assertAllEqual([[1, 2, 3, 4, 5, 6, 0]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=7, axis=0))

    self.assertAllEqual([[1, 2, 3, 4, 5, 6, -1, -1]],
                        tensor_utils.split_into_blocks(
                            tensor, block_len=8, axis=0, pad_value=-1))

  def test_split_into_blocks_3d(self):
    # shape: [2, 4, 2]
    tensor = tf.constant([
        [[1, -1], [2, -2], [3, -3], [4, -4]],  #
        [[11, 21], [12, 22], [13, 23], [14, 24]]
    ])

    self.assertAllEqual(
        [
            [
                [[1, -1], [2, -2]],  #
                [[3, -3], [4, -4]],  #
            ],
            [
                [[11, 21], [12, 22]],  #
                [[13, 23], [14, 24]],  #
            ]
        ],
        tensor_utils.split_into_blocks(tensor, block_len=2, axis=-2))

    self.assertAllEqual(
        [
            [
                [[1, -1], [2, -2], [3, -3]],  #
                [[4, -4], [0, 0], [0, 0]],  #
            ],
            [
                [[11, 21], [12, 22], [13, 23]],  #
                [[14, 24], [0, 0], [0, 0]],  #
            ]
        ],
        tensor_utils.split_into_blocks(tensor, block_len=3, axis=1))

    self.assertAllEqual(
        [
            [
                [[1, -1, 0]],  #
                [[2, -2, 0]],  #
                [[3, -3, 0]],  #
                [[4, -4, 0]],  #
            ],
            [
                [[11, 21, 0]],  #
                [[12, 22, 0]],  #
                [[13, 23, 0]],  #
                [[14, 24, 0]],  #
            ],
        ],
        tensor_utils.split_into_blocks(tensor, block_len=3, axis=-1))

  def test_split_into_blocks_static_shape(self):
    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # a static `batch_size` is unknown.
    tensor = tf.compat.v1.placeholder_with_default(
        np.ones(shape=[2, 5], dtype=np.int32), shape=[None, 5])

    result = tensor_utils.split_into_blocks(tensor, block_len=3, axis=-1)

    static_batch_size = tensor.shape.as_list()[0]
    self.assertAllEqual([static_batch_size, 2, 3], result.shape.as_list())

  def test_concat_3_blocks(self):
    # shape: [batch_size=2, num_blocks=3, block_len=2, hidden_size=2]
    blocked_seq = tf.constant([
        [
            [[1, -1], [2, -2]],  #
            [[3, -3], [4, -4]],  #
            [[5, -5], [6, -6]],  #
        ],  #
        [
            [[.1, -.1], [.2, -.2]],  #
            [[.3, -.3], [.4, -.4]],  #
            [[.5, -.5], [.6, -.6]],  #
        ],  #
    ])

    self.assertAllClose(
        [
            [
                [[0, 0], [0, 0], [1, -1], [2, -2], [3, -3], [4, -4]],  #
                [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5], [6, -6]],  #
                [[3, -3], [4, -4], [5, -5], [6, -6], [0, 0], [0, 0]],  #
            ],  #
            [
                [[0, 0], [0, 0], [.1, -.1], [.2, -.2], [.3, -.3], [.4, -.4]],  #
                [[.1, -.1], [.2, -.2], [.3, -.3], [.4, -.4], [.5, -.5],
                 [.6, -.6]],  #
                [[.3, -.3], [.4, -.4], [.5, -.5], [.6, -.6], [0, 0], [0, 0]],  #
            ],  #
        ],
        tensor_utils.concat_3_blocks(blocked_seq))

  def test_concat_3_blocks_with_extra_dim(self):
    # shape: [batch_size=1, num_blocks=3, block_len=2, num_heads=1,
    #         size_per_head=2]
    blocked_seq = tf.constant([[
        [[[1, -1]], [[2, -2]]],  #
        [[[3, -3]], [[4, -4]]],  #
        [[[5, -5]], [[6, -6]]],  #
    ]])

    self.assertAllClose(
        [[
            [[[0, 0]], [[0, 0]], [[1, -1]], [[2, -2]], [[3, -3]], [[4, -4]]],  #
            [[[1, -1]], [[2, -2]], [[3, -3]], [[4, -4]], [[5, -5]], [[6, -6]]],
            [[[3, -3]], [[4, -4]], [[5, -5]], [[6, -6]], [[0, 0]], [[0, 0]]],  #
        ]],
        tensor_utils.concat_3_blocks(blocked_seq))

  def test_shift_elements_right_1d(self):
    tensor = tf.constant([5, 4, 3, 2, 1])

    self.assertAllEqual([0, 5, 4, 3, 2],
                        tensor_utils.shift_elements_right(tensor))

    self.assertAllEqual([0, 0, 0, 5, 4],
                        tensor_utils.shift_elements_right(tensor, amount=3))

    self.assertAllEqual([3, 2, 1, 0, 0],
                        tensor_utils.shift_elements_right(tensor, amount=-2))

    self.assertAllEqual([3, 2, 1, -1, -1],
                        tensor_utils.shift_elements_right(
                            tensor, amount=-2, pad_value=-1))

    self.assertAllEqual([0, 0, 0, 0, 0],
                        tensor_utils.shift_elements_right(tensor, amount=10))

    self.assertAllEqual(tensor,
                        tensor_utils.shift_elements_right(tensor, amount=0))

    with self.assertRaises(ValueError):
      tensor_utils.shift_elements_right(tensor, axis=1)

  def test_shift_elements_right_2d(self):
    tensor = tf.constant([
        [1, 2, 3, 4],  #
        [5, 6, 7, 8],  #
        [9, 10, 11, 12],  #
    ])

    self.assertAllEqual(
        [
            [0, 1, 2, 3],  #
            [0, 5, 6, 7],  #
            [0, 9, 10, 11],  #
        ],
        tensor_utils.shift_elements_right(tensor))

    self.assertAllEqual(
        [
            [3, 4, -1, -1],  #
            [7, 8, -1, -1],  #
            [11, 12, -1, -1],  #
        ],
        tensor_utils.shift_elements_right(tensor, amount=-2, pad_value=-1))

    self.assertAllEqual(
        [
            [0, 0, 0, 0],  #
            [0, 0, 0, 0],  #
            [1, 2, 3, 4],  #
        ],
        tensor_utils.shift_elements_right(tensor, axis=-2, amount=2))

    self.assertAllEqual(tensor,
                        tensor_utils.shift_elements_right(tensor, amount=0))

  def test_shift_elements_right_3d(self):
    tensor = tf.constant([
        [
            [1, -1],  #
            [2, -2],  #
            [3, -3],  #
        ],  #
        [
            [4, -4],  #
            [5, -5],  #
            [6, -6],  #
        ],  #
    ])

    self.assertAllEqual(
        [
            [
                [-1, 0],  #
                [-2, 0],  #
                [-3, 0],  #
            ],  #
            [
                [-4, 0],  #
                [-5, 0],  #
                [-6, 0],  #
            ],  #
        ],
        tensor_utils.shift_elements_right(tensor, amount=-1))

    self.assertAllEqual(
        [
            [
                [-1, -1],  #
                [-1, -1],  #
                [-1, -1],  #
            ],  #
            [
                [1, -1],  #
                [2, -2],  #
                [3, -3],  #
            ],  #
        ],
        tensor_utils.shift_elements_right(tensor, axis=0, pad_value=-1))

    self.assertAllEqual(
        [
            [
                [0, 0],  #
                [0, 0],  #
                [1, -1],  #
            ],  #
            [
                [0, 0],  #
                [0, 0],  #
                [4, -4],  #
            ],  #
        ],
        tensor_utils.shift_elements_right(tensor, axis=1, amount=2))

    self.assertAllEqual(
        tensor, tensor_utils.shift_elements_right(tensor, axis=2, amount=0))

  def test_skew_elements_right_2d(self):
    tensor = tf.constant([
        [1, 2, 3],  #
        [4, 5, 6],  #
        [7, 8, 9],  #
    ])

    self.assertAllEqual(
        [
            [1, 2, 3, 0, 0],  #
            [0, 4, 5, 6, 0],  #
            [0, 0, 7, 8, 9],  #
        ],
        tensor_utils.skew_elements_right(tensor, -1))

    self.assertAllEqual(
        [
            [1, 2, 3, -2, -2],  #
            [-2, 4, 5, 6, -2],  #
            [-2, -2, 7, 8, 9],  #
        ],
        tensor_utils.skew_elements_right(tensor, 1, pad_value=-2))

    with self.assertRaises(ValueError):
      tensor_utils.skew_elements_right(tensor, 0)

    self.assertAllEqual(
        [
            [1, 2, 0, 0, 0, 0],  #
            [0, 3, 4, 0, 0, 0],  #
            [0, 0, 5, 6, 0, 0],  #
            [0, 0, 0, 7, 8, 0],  #
            [0, 0, 0, 0, 9, 10],  #
        ],
        tensor_utils.skew_elements_right(
            [
                [1, 2],  #
                [3, 4],  #
                [5, 6],  #
                [7, 8],  #
                [9, 10],  #
            ],
            1))

    self.assertAllEqual(
        [
            [1, 2, 3, 4, 0],  #
            [0, 5, 6, 7, 8],  #
        ],
        tensor_utils.skew_elements_right(
            [
                [1, 2, 3, 4],  #
                [5, 6, 7, 8],  #
            ],
            1))

    self.assertAllEqual(
        [
            [1, 2, 3],  #
        ],
        tensor_utils.skew_elements_right(
            [
                [1, 2, 3],  #
            ],
            1))

  def test_skew_elements_right_4d(self):
    # shape: [2, 3, 2, 2]
    tensor = tf.constant([
        [
            [[1, -1], [2, -2]],  #
            [[3, -3], [4, -4]],  #
            [[5, -5], [6, -6]],  #
        ],  #
        [
            [[.1, -.1], [.2, -.2]],  #
            [[.3, -.3], [.4, -.4]],  #
            [[.5, -.5], [.6, -.6]],  #
        ],  #
    ])

    self.assertAllClose(
        [
            [
                [[1, -1, 0], [0, 2, -2]],  #
                [[3, -3, 0], [0, 4, -4]],  #
                [[5, -5, 0], [0, 6, -6]],  #
            ],  #
            [
                [[.1, -.1, 0], [0, .2, -.2]],  #
                [[.3, -.3, 0], [0, .4, -.4]],  #
                [[.5, -.5, 0], [0, .6, -.6]],  #
            ],  #
        ],
        tensor_utils.skew_elements_right(tensor, -1))

    self.assertAllClose(
        [
            [
                [[1, -1], [2, -2], [0, 0], [0, 0]],  #
                [[0, 0], [3, -3], [4, -4], [0, 0]],  #
                [[0, 0], [0, 0], [5, -5], [6, -6]],  #
            ],  #
            [
                [[.1, -.1], [.2, -.2], [0, 0], [0, 0]],  #
                [[0, 0], [.3, -.3], [.4, -.4], [0, 0]],  #
                [[0, 0], [0, 0], [.5, -.5], [.6, -.6]],  #
            ],  #
        ],
        tensor_utils.skew_elements_right(tensor, -2))

    self.assertAllClose(
        [
            [
                [[1, -1], [2, -2]],  #
                [[3, -3], [4, -4]],  #
                [[5, -5], [6, -6]],  #
                [[0, 0], [0, 0]],  #
            ],  #
            [
                [[0, 0], [0, 0]],  #
                [[.1, -.1], [.2, -.2]],  #
                [[.3, -.3], [.4, -.4]],  #
                [[.5, -.5], [.6, -.6]],  #
            ],  #
        ],
        tensor_utils.skew_elements_right(tensor, 1))

  def test_unskew_elements_right_2d(self):
    tensor = tf.constant([
        [1, 2, 3, 0, 0],  #
        [0, 4, 5, 6, 0],  #
        [0, 0, 7, 8, 9],  #
    ])

    self.assertAllEqual(
        [
            [1, 2, 3],  #
            [4, 5, 6],  #
            [7, 8, 9],  #
        ],
        tensor_utils.unskew_elements_right(tensor, -1))

    with self.assertRaises(ValueError):
      tensor_utils.unskew_elements_right(tensor, 0)

    self.assertAllEqual(
        [
            [1, 2],  #
            [3, 4],  #
            [5, 6],  #
            [7, 8],  #
            [9, 10],  #
        ],
        tensor_utils.unskew_elements_right(
            [
                [1, 2, 0, 0, 0, 0],  #
                [0, 3, 4, 0, 0, 0],  #
                [0, 0, 5, 6, 0, 0],  #
                [0, 0, 0, 7, 8, 0],  #
                [0, 0, 0, 0, 9, 10],  #
            ],
            1))

    self.assertAllEqual(
        [
            [1, 2, 3, 4],  #
            [5, 6, 7, 8],  #
        ],
        tensor_utils.unskew_elements_right(
            [
                [1, 2, 3, 4, 0],  #
                [0, 5, 6, 7, 8],  #
            ],
            1))

    self.assertAllEqual(
        [
            [1, 2, 3],  #
        ],
        tensor_utils.unskew_elements_right(
            [
                [1, 2, 3],  #
            ],
            1))

    self.assertAllEqual(
        [
            [1],  #
            [2],  #
            [3],  #
        ],
        tensor_utils.unskew_elements_right(
            [
                [1, 0, 0],  #
                [0, 2, 0],  #
                [0, 0, 3],  #
            ],
            1))

    with self.assertRaises(ValueError):
      tensor_utils.unskew_elements_right(
          [
              [1, 2],  #
              [3, 4],  #
              [5, 6],  #
          ],
          1)

  def test_unskew_elements_right_4d(self):
    # shape: [2, 3, 2, 2]
    expected_tensor = tf.constant([
        [
            [[1, -1], [2, -2]],  #
            [[3, -3], [4, -4]],  #
            [[5, -5], [6, -6]],  #
        ],  #
        [
            [[.1, -.1], [.2, -.2]],  #
            [[.3, -.3], [.4, -.4]],  #
            [[.5, -.5], [.6, -.6]],  #
        ],  #
    ])

    self.assertAllClose(
        expected_tensor,
        tensor_utils.unskew_elements_right(
            [
                [
                    [[1, -1, 0], [0, 2, -2]],  #
                    [[3, -3, 0], [0, 4, -4]],  #
                    [[5, -5, 0], [0, 6, -6]],  #
                ],  #
                [
                    [[.1, -.1, 0], [0, .2, -.2]],  #
                    [[.3, -.3, 0], [0, .4, -.4]],  #
                    [[.5, -.5, 0], [0, .6, -.6]],  #
                ],  #
            ],
            -1))

    self.assertAllClose(
        expected_tensor,
        tensor_utils.unskew_elements_right(
            [
                [
                    [[1, -1], [2, -2], [0, 0], [0, 0]],  #
                    [[0, 0], [3, -3], [4, -4], [0, 0]],  #
                    [[0, 0], [0, 0], [5, -5], [6, -6]],  #
                ],  #
                [
                    [[.1, -.1], [.2, -.2], [0, 0], [0, 0]],  #
                    [[0, 0], [.3, -.3], [.4, -.4], [0, 0]],  #
                    [[0, 0], [0, 0], [.5, -.5], [.6, -.6]],  #
                ],  #
            ],
            -2))

    self.assertAllClose(
        expected_tensor,
        tensor_utils.unskew_elements_right(
            [
                [
                    [[1, -1], [2, -2]],  #
                    [[3, -3], [4, -4]],  #
                    [[5, -5], [6, -6]],  #
                    [[0, 0], [0, 0]],  #
                ],  #
                [
                    [[0, 0], [0, 0]],  #
                    [[.1, -.1], [.2, -.2]],  #
                    [[.3, -.3], [.4, -.4]],  #
                    [[.5, -.5], [.6, -.6]],  #
                ],  #
            ],
            1))

  def test_skew_elements_right_unskew_elements_right_round_trip(self):
    np.random.seed(1234)

    tensor = tf.constant(np.random.normal(size=[3, 5, 7, 11]))

    self.assertAllEqual(
        tensor,
        tensor_utils.unskew_elements_right(
            tensor_utils.skew_elements_right(tensor, 1), 1))

    self.assertAllEqual(
        tensor,
        tensor_utils.unskew_elements_right(
            tensor_utils.skew_elements_right(tensor, 2), 2))

    self.assertAllEqual(
        tensor,
        tensor_utils.unskew_elements_right(
            tensor_utils.skew_elements_right(tensor, -1), -1))

  @parameterized.parameters(
      dict(equation='ab,bc->ac', input_shapes=[[2, 3], [3, 4]]),
      dict(equation='...b,bc->...c', input_shapes=[[2, 3], [3, 4]]),
      dict(equation='ab,b...->a...', input_shapes=[[2, 3], [3, 4]]),
      dict(equation='ab,b...->a...', input_shapes=[[2, 3], [3]]),
      dict(equation='...ii->...i', input_shapes=[[2, 3, 4, 4]]),
      dict(equation='a...a', input_shapes=[[2, 3, 4, 2]]),
      dict(equation='A...A', input_shapes=[[2, 3, 2]]),
      dict(equation='...j->...', input_shapes=[[2, 3, 4, 5]]),
      dict(equation='...j->...', input_shapes=[[5]]),
      dict(equation='ki,...k->i...', input_shapes=[[3, 4], [2, 3]]),
      dict(equation='k...,jk', input_shapes=[[3, 4], [2, 3]]),
      dict(
          equation='...ij,...jk->...ik',
          input_shapes=[[2, 3, 4, 5], [2, 3, 5, 6]]),
      dict(equation='...ij,...jk->...ik', input_shapes=[[2, 3], [3, 4]]),
      dict(
          equation=' ... i j , ... j k  ->  ... i k ',
          input_shapes=[[2, 3], [3, 4]]),
  )
  def test_einsum_wrap_ellipses(self, equation, input_shapes):
    np.random.seed(1234)

    inputs = [
        tf.constant(np.random.normal(size=shape)) for shape in input_shapes
    ]
    self.assertAllClose(
        tf.einsum(equation, *inputs),
        tensor_utils.einsum_wrap_ellipses(equation, *inputs))

  @parameterized.parameters(
      dict(equation='...iii->...i', input_shapes=[[2, 2]]),
      dict(equation='...ij,...jk->...ik', input_shapes=[[2, 3], [2, 3, 4]]),
  )
  def test_einsum_wrap_ellipses_invalid_input(self, equation, input_shapes):
    np.random.seed(1234)

    inputs = [
        tf.constant(np.random.normal(size=shape)) for shape in input_shapes
    ]
    with self.assertRaises(ValueError):
      tensor_utils.einsum_wrap_ellipses(equation, *inputs)


if __name__ == '__main__':
  tf.test.main()
