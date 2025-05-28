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

"""Test io module."""

from absl.testing import parameterized
from google.protobuf import text_format
import tensorflow as tf

from graph_embedding.huge import io


def create_fake_graph_sampler_example():
  return text_format.Parse(
      """features: {
    feature: {
      key: "D"
      value: {
        int64_list: {
          value: [ 334 ]
        }
      }
    }
    feature: {
      key: "S"
      value: {
        int64_list: {
          value: [ 2375 ]
        }
      }
    }
    feature: {
      key: "W"
      value: {
        float_list: {
          value: [ 0.0, 0.0, 0.0, 2.0 ]
        }
      }
    }
  }""",
      tf.train.Example(),
  )


def create_fake_dataset(positive_batch_size = 8):
  parser = io.PositiveExampleParser(walk_length=4)
  serialized_example = create_fake_graph_sampler_example().SerializeToString()
  return (
      tf.data.Dataset.from_tensor_slices([serialized_example])
      .repeat()
      .batch(positive_batch_size, drop_remainder=True)
      .map(parser)
  )


class IoTest(tf.test.TestCase, parameterized.TestCase):

  def test_parser(self):
    walk_length = 4
    fake_example = create_fake_graph_sampler_example()
    parser = io.PositiveExampleParser(walk_length=walk_length)
    src, dst, w = parser(fake_example.SerializeToString())
    self.assertEqual(src, tf.constant(2375, shape=(), dtype=tf.int64))
    self.assertEqual(dst, tf.constant(334, shape=(), dtype=tf.int64))
    self.assertAllClose(
        w,
        tf.constant(
            [0.0, 0.0, 0.0, 2.0], shape=(walk_length,), dtype=tf.float32
        ),
    )

  def test_random_negatives(self):
    positive_batch_size = 8
    num_negs_per_pos = 4
    ds = create_fake_dataset(positive_batch_size=positive_batch_size)
    ds = io.add_uniform_random_negatives(
        ds, num_nodes=2708, num_negs_per_pos=num_negs_per_pos
    )
    it = iter(ds)
    src, dst, co = next(it)

    total_batch_size = positive_batch_size * (1 + num_negs_per_pos)
    self.assertAllEqual(
        src,
        tf.constant(
            [2375] * total_batch_size, shape=(total_batch_size,), dtype=tf.int64
        ),
    )
    self.assertAllEqual(
        dst[:positive_batch_size],
        tf.constant(
            [334] * positive_batch_size,
            shape=(positive_batch_size,),
            dtype=tf.int64,
        ),
    )
    self.assertEqual(dst.shape, (total_batch_size,))
    self.assertEqual(co.shape, (positive_batch_size, 4))

  @parameterized.parameters(
      {
          'weights': [
              1.0,
              1.0,
              1.0,
              2.0,
          ],
          'ees_norm': None,
          'expected': 4.0,
      },
      {
          'weights': [1.0, 2.0, 3.0, 4.0],
          'ees_norm': 5.0,
          'expected': 1.767767,
      },
  )
  def test_expected_edge_score(self, weights, ees_norm, expected):
    positive_batch_size = 8
    num_negs_per_pos = 4
    total_batch_size = positive_batch_size * (1 + num_negs_per_pos)
    ds = create_fake_dataset(positive_batch_size=positive_batch_size)
    ds = io.add_uniform_random_negatives(
        ds, num_nodes=2708, num_negs_per_pos=num_negs_per_pos
    )
    ds = io.add_expected_edge_score(
        ds, weights=weights, edge_score_norm=ees_norm
    )
    src, dst, ees = next(iter(ds))

    self.assertEqual(src.shape, (total_batch_size,))
    self.assertEqual(dst.shape, (total_batch_size,))
    self.assertAllClose(
        ees,
        tf.constant(
            [expected] * positive_batch_size,
            shape=(positive_batch_size,),
            dtype=tf.float32,
        ),
    )
    if ees_norm is not None:
      self.assertAllClose(
          tf.sqrt(tf.reduce_sum(ees * ees, axis=0)),
          tf.constant(
              ees_norm,
              shape=(),
              dtype=tf.float32,
          ),
      )


