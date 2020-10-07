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

"""Tests for multi_resolution_rec.modules."""

import tensorflow.compat.v1 as tf
from multi_resolution_rec import modules


class ModulesTest(tf.test.TestCase):

  def test_compute_head_weights_with_position_prior(self):
    # Test case: seq_length=4, num_heads=2 -> attn_size=2
    seq_len = 4
    num_heads = 2
    attn_size = 2

    weights = tf.constant(
        [[[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 3, 4]]] *
        num_heads)  # (2, 4, 4)
    masks = tf.constant(
        [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]] *
        num_heads)  # (2, 4, 4)
    paddings = tf.constant([[[-1] * seq_len] * seq_len] *
                           num_heads)  # (2, 4, 4)

    outputs = modules._compute_head_weights_with_position_prior(
        weights, masks, paddings, num_heads, attn_size)   # [2 x (1, 4, 4)]
    outputs = tf.concat(outputs, axis=0)  # (2, 4, 4)
    with self.test_session() as sess:
      expected_output = tf.constant([
          [
              [1, -1, -1, -1],  # First head weights for query 1
              [1, 2, -1, -1],
              [-1, 2, 3, -1],
              [-1, -1, 3, 4]  # First head weights for query 4
          ],
          [
              [-1, -1, -1, -1],  # Second head weights for query 1
              [-1, -1, -1, -1],
              [1, -1, -1, -1],
              [1, 2, -1, -1]  # Second head weights for query 4
          ]
      ])
      self.assertAllClose(sess.run(outputs), sess.run(expected_output))

  def test_compute_head_weights_with_time_prior(self):
    # Case 1: seq_len=4, num_heads=3, exp_base=3, overlapping_chunks=False
    # time_intervals = [[0,1] (1,3], (3,inf]]
    seq_len = 4
    num_heads = 3
    time_exp_base = 3
    overlapping_chunks = False

    weights = tf.constant([[[1, 0, 0, 0], [1, 2, 0, 0],
                            [1, 2, 3, 0], [1, 2, 3, 4]]] *
                          num_heads)  # (3, 4, 4)
    paddings = tf.constant([[[-1] * seq_len] * seq_len] *
                           num_heads)  # (3, 4, 4)
    time_deltas = tf.constant([[[0, -5, -6., -9.], [5, 0, -1, -4],
                                [6., 1, 0, -3], [9., 4, 3, 0]]])  # (1, 4, 4)

    outputs = modules._compute_head_weights_with_time_prior(
        weights, paddings, time_deltas, num_heads, time_exp_base,
        overlapping_chunks)
    outputs = tf.concat(outputs, axis=0)
    with self.test_session() as sess:
      expected_output = tf.constant([
          [
              [1, -1, -1, -1],  # First head weights for query 1
              [-1, 2, -1, -1],
              [-1, 2, 3, -1],
              [-1, -1, -1, 4]  # First head weights for query 4
          ],
          [
              [-1, -1, -1, -1],  # Second head weights for query 1
              [-1, -1, -1, -1],
              [-1, -1, -1, -1],
              [-1, -1, 3, -1]  # Second head weights for query 4
          ],
          [
              [-1, -1, -1, -1],  # Third head weights for query 1
              [1, -1, -1, -1],
              [1, -1, -1, -1],
              [1, 2, -1, -1]  # Third head weights for query 4
          ]
      ])
      self.assertAllClose(sess.run(outputs), sess.run(expected_output))

    # Case 2: With overlapping_chunks = True
    overlapping_chunks = True
    outputs = modules._compute_head_weights_with_time_prior(
        weights, paddings, time_deltas, num_heads, time_exp_base,
        overlapping_chunks)
    outputs = tf.concat(outputs, axis=0)
    with self.test_session() as sess:
      expected_output = tf.constant([
          [
              [1, -1, -1, -1],  # First chunk weights for query 1
              [-1, 2, -1, -1],
              [-1, 2, 3, -1],
              [-1, -1, -1, 4]  # First head weights for query 4
          ],
          [
              [1, -1, -1, -1],  # Second chunk weights for query 1
              [-1, 2, -1, -1],
              [-1, 2, 3, -1],
              [-1, -1, 3, 4]  # Second head weights for query 4
          ],
          [
              [1, -1, -1, -1],  # Third chunk weights for query 1
              [1, 2, -1, -1],
              [1, 2, 3, -1],
              [1, 2, 3, 4]  # Third head weights for query 4
          ]
      ])
      self.assertAllClose(sess.run(outputs), sess.run(expected_output))

  def test_compute_time_deltas(self):
    times = tf.constant([[0, 1, 6, 8]])
    times_delta = modules._compute_time_deltas(times)

    with self.test_session() as sess:
      expected_output = tf.constant([[
          [0, 0, -1, -6],
          [1, 1, 0, -5],
          [6, 6, 5, 0],
          [8, 8, 7, 2]
      ]])
      self.assertAllClose(sess.run(times_delta), sess.run(expected_output))

if __name__ == '__main__':
  tf.test.main()
