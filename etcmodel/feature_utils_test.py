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

"""Tests for feature utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from etcmodel import feature_utils


class TensorUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_relative_position_generator_init(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    self.assertEqual(3, relative_pos_gen.max_distance)
    self.assertEqual(False, relative_pos_gen.ignore_direction)
    self.assertEqual(7, relative_pos_gen.relative_vocab_size)
    self.assertEqual(6, relative_pos_gen.left_pad_value)
    self.assertEqual(3, relative_pos_gen.right_pad_value)

  def test_relative_position_generator_init_ignore_direction(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(
        max_distance=3, ignore_direction=True)

    self.assertEqual(3, relative_pos_gen.max_distance)
    self.assertEqual(True, relative_pos_gen.ignore_direction)
    self.assertEqual(4, relative_pos_gen.relative_vocab_size)
    self.assertEqual(3, relative_pos_gen.left_pad_value)
    self.assertEqual(3, relative_pos_gen.right_pad_value)

  def test_relative_position_generator_init_max_distance_0(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=0)

    self.assertEqual(0, relative_pos_gen.max_distance)
    self.assertEqual(False, relative_pos_gen.ignore_direction)
    self.assertEqual(1, relative_pos_gen.relative_vocab_size)
    self.assertEqual(0, relative_pos_gen.left_pad_value)
    self.assertEqual(0, relative_pos_gen.right_pad_value)

  def test_relative_position_generator_init_invalid_arguments(self):
    with self.assertRaises(ValueError):
      feature_utils.RelativePositionGenerator(max_distance=-1)

  def test_make_relative_att_ids_padding_case(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    expected = [[
        [0, 1, 2, 3, 3, 3],  #
        [4, 0, 1, 2, 3, 3],  #
        [5, 4, 0, 1, 2, 3],  #
        [6, 5, 4, 0, 1, 2],  #
        [6, 6, 5, 4, 0, 1],  #
        [6, 6, 6, 5, 4, 0],  #
    ]]
    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(6))

  def test_make_relative_att_ids_padding_case_ignore_direction(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(
        max_distance=3, ignore_direction=True)

    expected = [[
        [0, 1, 2, 3, 3, 3],  #
        [1, 0, 1, 2, 3, 3],  #
        [2, 1, 0, 1, 2, 3],  #
        [3, 2, 1, 0, 1, 2],  #
        [3, 3, 2, 1, 0, 1],  #
        [3, 3, 3, 2, 1, 0],  #
    ]]
    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(6))

  def test_make_relative_att_ids_trimming_case(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=9)

    expected = [[
        [0, 1, 2, 3, 4],  #
        [10, 0, 1, 2, 3],  #
        [11, 10, 0, 1, 2],  #
        [12, 11, 10, 0, 1],  #
        [13, 12, 11, 10, 0],  #
    ]]
    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(5))

  def test_make_relative_att_ids_no_pad_or_trim_case(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=4)

    expected = [[
        [0, 1, 2, 3, 4],  #
        [5, 0, 1, 2, 3],  #
        [6, 5, 0, 1, 2],  #
        [7, 6, 5, 0, 1],  #
        [8, 7, 6, 5, 0],  #
    ]]
    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(5))

  def test_make_relative_att_ids_max_distance_0(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=0)

    expected = [[
        [0, 0, 0, 0],  #
        [0, 0, 0, 0],  #
        [0, 0, 0, 0],  #
        [0, 0, 0, 0],  #
    ]]
    self.assertAllEqual(expected, relative_pos_gen.make_relative_att_ids(4))

  def test_make_relative_att_ids_batch_size_2(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    expected = [
        [
            [0, 1, 2, 3, 3],  #
            [4, 0, 1, 2, 3],  #
            [5, 4, 0, 1, 2],  #
            [6, 5, 4, 0, 1],  #
            [6, 6, 5, 4, 0],  #
        ],
        [
            [0, 1, 2, 3, 3],  #
            [4, 0, 1, 2, 3],  #
            [5, 4, 0, 1, 2],  #
            [6, 5, 4, 0, 1],  #
            [6, 6, 5, 4, 0],  #
        ]
    ]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_relative_att_ids(seq_len=5, batch_size=2))

  def test_make_relative_att_ids_batch_size_2_tensor(self):
    dummy_batch = tf.ones([2, 5])

    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    expected = [
        [
            [0, 1, 2, 3, 3],  #
            [4, 0, 1, 2, 3],  #
            [5, 4, 0, 1, 2],  #
            [6, 5, 4, 0, 1],  #
            [6, 6, 5, 4, 0],  #
        ],
        [
            [0, 1, 2, 3, 3],  #
            [4, 0, 1, 2, 3],  #
            [5, 4, 0, 1, 2],  #
            [6, 5, 4, 0, 1],  #
            [6, 6, 5, 4, 0],  #
        ]
    ]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_relative_att_ids(
            seq_len=5, batch_size=tf.shape(dummy_batch)[0]))

  def test_overwrite_relative_att_ids_outside_segments(self):

    # batch_size = 2, seq_len = 5, max_distance = 3
    rel_att_ids = [
        [
            [0, 1, 2, 3, 3],  #
            [4, 0, 1, 2, 3],  #
            [5, 4, 0, 1, 2],  #
            [6, 5, 4, 0, 1],  #
            [6, 6, 5, 4, 0],  #
        ],
        [
            [0, 1, 2, 3, 3],  #
            [4, 0, 1, 2, 3],  #
            [5, 4, 0, 1, 2],  #
            [6, 5, 4, 0, 1],  #
            [6, 6, 5, 4, 0],  #
        ]
    ]

    segment_ids = [[10, 10, 20, 30, 30], [10, 20, 20, 10, 10]]
    overwrite_value = 100

    expected_rel_att_ids = [
        [
            [0, 1, 100, 100, 100],  #
            [4, 0, 100, 100, 100],  #
            [100, 100, 0, 100, 100],  #
            [100, 100, 100, 0, 1],  #
            [100, 100, 100, 4, 0],  #
        ],
        [
            [0, 100, 100, 3, 3],  #
            [100, 0, 1, 100, 100],  #
            [100, 4, 0, 100, 100],  #
            [6, 100, 100, 0, 1],  #
            [6, 100, 100, 4, 0],  #
        ]
    ]

    self.assertAllEqual(
        expected_rel_att_ids,
        feature_utils.overwrite_relative_att_ids_outside_segments(
            rel_att_ids=rel_att_ids,
            segment_ids=segment_ids,
            overwrite_value=overwrite_value))

  def test_make_relative_att_ids_invalid_arguments(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    with self.assertRaises(ValueError):
      relative_pos_gen.make_relative_att_ids(0)

    with self.assertRaises(ValueError):
      relative_pos_gen.make_relative_att_ids(seq_len=5, batch_size=0)

  def test_make_local_relative_att_ids_padding_case(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    expected = [[
        [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],  #
        [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],  #
        [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],  #
        [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],  #
    ]]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(seq_len=4, local_radius=5))

  def test_make_local_relative_att_ids_padding_case_ignore_direction(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(
        max_distance=3, ignore_direction=True)

    expected = [[
        [3, 3, 3, 2, 1, 0, 1, 2, 3, 3, 3],  #
        [3, 3, 3, 2, 1, 0, 1, 2, 3, 3, 3],  #
        [3, 3, 3, 2, 1, 0, 1, 2, 3, 3, 3],  #
        [3, 3, 3, 2, 1, 0, 1, 2, 3, 3, 3],  #
    ]]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(seq_len=4, local_radius=5))

  def test_make_local_relative_att_ids_trimming_case(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=9)

    expected = [[
        [13, 12, 11, 10, 0, 1, 2, 3, 4],  #
        [13, 12, 11, 10, 0, 1, 2, 3, 4],  #
        [13, 12, 11, 10, 0, 1, 2, 3, 4],  #
    ]]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(seq_len=3, local_radius=4))

  def test_make_local_relative_att_ids_no_pad_or_trim_case(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=4)

    expected = [[
        [8, 7, 6, 5, 0, 1, 2, 3, 4],  #
        [8, 7, 6, 5, 0, 1, 2, 3, 4],  #
        [8, 7, 6, 5, 0, 1, 2, 3, 4],  #
    ]]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(seq_len=3, local_radius=4))

  def test_make_local_relative_att_ids_max_distance_0(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=0)

    expected = [[
        [0, 0, 0, 0, 0],  #
        [0, 0, 0, 0, 0],  #
    ]]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(seq_len=2, local_radius=2))

  def test_make_local_relative_att_ids_batch_size_2(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    expected = [
        [
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
        ],
        [
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
        ],
    ]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(
            seq_len=3, local_radius=4, batch_size=2))

  def test_make_local_relative_att_ids_batch_size_2_tensor(self):
    dummy_batch = tf.ones([2, 5])

    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    expected = [
        [
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
        ],
        [
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
            [6, 6, 5, 4, 0, 1, 2, 3, 3],  #
        ],
    ]
    self.assertAllEqual(
        expected,
        relative_pos_gen.make_local_relative_att_ids(
            seq_len=3, local_radius=4, batch_size=tf.shape(dummy_batch)[0]))

  def test_make_local_relative_att_ids_invalid_arguments(self):
    relative_pos_gen = feature_utils.RelativePositionGenerator(max_distance=3)

    with self.assertRaises(ValueError):
      relative_pos_gen.make_local_relative_att_ids(seq_len=0, local_radius=3)

    with self.assertRaises(ValueError):
      relative_pos_gen.make_local_relative_att_ids(seq_len=5, local_radius=0)

    with self.assertRaises(ValueError):
      relative_pos_gen.make_local_relative_att_ids(
          seq_len=5, local_radius=3, batch_size=0)

  def test_make_att_mask_from_input_mask(self):
    input_mask = [
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    expected = [
        [
            [1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 0, 0, 0],  #
            [1, 1, 1, 0, 0, 0],  #
            [0, 0, 0, 1, 1, 1],  #
            [0, 0, 0, 1, 1, 1],  #
            [0, 0, 0, 1, 1, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [0, 0, 0, 0, 0, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
        ],  #
    ]
    self.assertAllEqual(expected,
                        feature_utils.make_att_mask_from_input_mask(input_mask))

  def test_make_segmented_att_mask(self):
    segment_ids = [
        [0, 0, 1, 1, 0, 0],
        [2, 2, 2, 2, 2, 2],
        [0, 0, 3, 0, 3, 0],
        [0, 5, 4, 3, 2, 1],
    ]

    expected = [
        [
            [1, 1, 0, 0, 1, 1],  #
            [1, 1, 0, 0, 1, 1],  #
            [0, 0, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 0, 0],  #
            [1, 1, 0, 0, 1, 1],  #
            [1, 1, 0, 0, 1, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
        ],  #
        [
            [1, 1, 0, 1, 0, 1],  #
            [1, 1, 0, 1, 0, 1],  #
            [0, 0, 1, 0, 1, 0],  #
            [1, 1, 0, 1, 0, 1],  #
            [0, 0, 1, 0, 1, 0],  #
            [1, 1, 0, 1, 0, 1],  #
        ],  #
        [
            [1, 0, 0, 0, 0, 0],  #
            [0, 1, 0, 0, 0, 0],  #
            [0, 0, 1, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0],  #
            [0, 0, 0, 0, 1, 0],  #
            [0, 0, 0, 0, 0, 1],  #
        ],  #
    ]
    self.assertAllEqual(expected,
                        feature_utils.make_segmented_att_mask(segment_ids))

  def test_make_att_mask_from_breakpoints(self):
    att_breakpoints = [
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ]

    expected = [
        [
            [1, 1, 0, 0, 0, 0],  #
            [1, 1, 0, 0, 0, 0],  #
            [0, 0, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 0, 0],  #
            [0, 0, 0, 0, 1, 1],  #
            [0, 0, 0, 0, 1, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
        ],  #
        [
            [1, 0, 0, 0, 0, 0],  #
            [0, 1, 1, 1, 1, 1],  #
            [0, 1, 1, 1, 1, 1],  #
            [0, 1, 1, 1, 1, 1],  #
            [0, 1, 1, 1, 1, 1],  #
            [0, 1, 1, 1, 1, 1],  #
        ],  #
        [
            [1, 0, 0, 0, 0, 0],  #
            [0, 1, 0, 0, 0, 0],  #
            [0, 0, 1, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0],  #
            [0, 0, 0, 0, 1, 0],  #
            [0, 0, 0, 0, 0, 1],  #
        ],  #
    ]
    self.assertAllEqual(
        expected, feature_utils.make_att_mask_from_breakpoints(att_breakpoints))

  def test_make_att_mask_from_breakpoints_use_starting_breakpoints(self):
    att_breakpoints = [
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ]

    expected = [
        [
            [1, 1, 0, 0, 0, 0],  #
            [1, 1, 0, 0, 0, 0],  #
            [0, 0, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 0, 0],  #
            [0, 0, 0, 0, 1, 1],  #
            [0, 0, 0, 0, 1, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1],  #
        ],  #
        [
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0],  #
            [0, 0, 0, 0, 0, 1],  #
        ],  #
        [
            [1, 0, 0, 0, 0, 0],  #
            [0, 1, 0, 0, 0, 0],  #
            [0, 0, 1, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0],  #
            [0, 0, 0, 0, 1, 0],  #
            [0, 0, 0, 0, 0, 1],  #
        ],  #
    ]
    self.assertAllEqual(
        expected,
        feature_utils.make_att_mask_from_breakpoints(
            att_breakpoints, use_starting_breakpoints=True))

  def test_make_local_segmented_att_mask(self):
    segment_ids = [
        [0, 0, 1, 0, 1, 0, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2],
        [4, 3, 3, 3, 4, 1, 1, 1],
        [0, 6, 5, 4, 3, 2, 1, 0],
    ]

    expected = [
        [
            [0, 0, 1, 1, 0],  #
            [0, 1, 1, 0, 1],  #
            [0, 0, 1, 0, 1],  #
            [1, 0, 1, 0, 1],  #
            [1, 0, 1, 0, 1],  #
            [1, 0, 1, 0, 0],  #
            [1, 0, 1, 1, 0],  #
            [0, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
        ],  #
    ]
    self.assertAllEqual(
        expected,
        feature_utils.make_local_segmented_att_mask(
            segment_ids, local_radius=2))

  def test_make_local_segmented_att_mask_uneven_blocking_case(self):
    segment_ids = [
        [0, 0, 1, 0, 1, 0, 1, 1, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    expected = [
        [
            [0, 0, 0, 1, 1, 0, 1],  #
            [0, 0, 1, 1, 0, 1, 0],  #
            [0, 0, 0, 1, 0, 1, 0],  #
            [1, 1, 0, 1, 0, 1, 0],  #
            [0, 1, 0, 1, 0, 1, 1],  #
            [0, 1, 0, 1, 0, 0, 0],  #
            [0, 1, 0, 1, 1, 0, 0],  #
            [1, 0, 1, 1, 0, 0, 0],  #
            [0, 0, 0, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 0, 0, 0],  #
        ],  #
        [
            [0, 0, 0, 1, 1, 1, 1],  #
            [0, 0, 1, 1, 1, 1, 1],  #
            [0, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 0, 0],  #
            [1, 1, 1, 1, 0, 0, 0],  #
        ],  #
    ]
    self.assertAllEqual(
        expected,
        feature_utils.make_local_segmented_att_mask(
            segment_ids, local_radius=3))

  def test_make_local_segmented_att_mask_single_block_case(self):
    segment_ids = [
        [0, 1],
        [0, 0],
    ]

    expected = [
        [
            [0, 0, 0, 1, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0, 0],  #
        ],  #
        [
            [0, 0, 0, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 0, 0, 0],  #
        ],  #
    ]
    self.assertAllEqual(
        expected,
        feature_utils.make_local_segmented_att_mask(
            segment_ids, local_radius=3))

  def test_make_local_segmented_att_mask_static_shape(self):
    # This test is only relevant for TF v1 session mode.  If the batch size
    # is statically unknown (None), we want to make sure all shapes in the
    # output other than batch size are still statically known.

    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # the static `batch_size` is unknown.
    segment_ids = tf.compat.v1.placeholder_with_default(
        np.zeros([1, 8]), shape=[None, 8])

    local_radius = 2
    result = feature_utils.make_local_segmented_att_mask(
        segment_ids, local_radius=local_radius)

    self.assertAllEqual([8, 2 * local_radius + 1], result.shape.as_list()[1:])

  def test_make_local_att_mask_from_breakpoints(self):
    att_breakpoints = [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    expected = [
        [
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
        ],  #
    ]
    self.assertAllEqual(
        expected,
        feature_utils.make_local_att_mask_from_breakpoints(
            att_breakpoints, local_radius=2))

  def test_make_local_att_mask_from_breakpoints_use_starting_breakpoints(self):
    att_breakpoints = [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    expected = [
        [
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 1],  #
            [1, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 1, 1],  #
            [0, 1, 1, 1, 0],  #
            [1, 1, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
        ],  #
        [
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
            [0, 0, 1, 0, 0],  #
        ],  #
    ]
    self.assertAllEqual(
        expected,
        feature_utils.make_local_att_mask_from_breakpoints(
            att_breakpoints, local_radius=2, use_starting_breakpoints=True))


if __name__ == '__main__':
  tf.test.main()
