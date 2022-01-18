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

"""Tests for alignment."""

import tensorflow as tf

from dedal import alignment


class AlignmentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(1)

    # Ground-truth alignment of
    #   XX-XXXXXX
    #   YYYY--YYY.
    self.len_x = 8
    self.len_y = 7
    self.alignments = tf.convert_to_tensor(
        [[[1, 2, 2, 3, 4, 5, 6, 7, 8],
          [1, 2, 3, 4, 4, 4, 5, 6, 7],
          [0, 1, 4, 2, 6, 8, 3, 1, 1]]], tf.int32)

    # Mock sequences whose predicted alignment matches `self.alignments` by
    # design.
    seq1 = tf.convert_to_tensor([1, 1, 2, 4, 4, 5, 5, 5], tf.int32)
    seq2 = tf.convert_to_tensor([1, 1, 3, 2, 5, 5, 5, 0], tf.int32)
    mask1, mask2 = seq1 != 0, seq2 != 0
    paired_masks = tf.logical_and(mask1[:, None], mask2[None, :])

    sim_mat = tf.where(seq1[:, None] == seq2[None, :], 20.0, -30.0)
    sim_mat = tf.where(paired_masks, sim_mat, -1e9)
    sim_mat = tf.expand_dims(sim_mat, 0)
    gap_open = tf.convert_to_tensor([12.0], tf.float32)
    gap_extend = tf.convert_to_tensor([1.0], tf.float32)
    self.sw_params = (sim_mat, gap_open, gap_extend)

  def test_alignments_to_paths(self):
    exp_out = tf.convert_to_tensor(
        [[[1., 0., 0., 0., 0., 0., 0.],
          [0., 2., 5., 0., 0., 0., 0.],
          [0., 0., 0., 3., 0., 0., 0.],
          [0., 0., 0., 7., 0., 0., 0.],
          [0., 0., 0., 9., 0., 0., 0.],
          [0., 0., 0., 0., 4., 0., 0.],
          [0., 0., 0., 0., 0., 2., 0.],
          [0., 0., 0., 0., 0., 0., 2.]]], tf.float32)

    paths = alignment.alignments_to_paths(
        self.alignments, self.len_x, self.len_y)
    sq_paths = alignment.path_label_squeeze(paths)
    self.assertAllEqual(sq_paths, exp_out)

    # Invariance to padding.
    padded_alignments = tf.concat(
        [self.alignments, tf.zeros([1, 3, 3], tf.int32)], 2)
    paths = alignment.alignments_to_paths(
        padded_alignments, self.len_x, self.len_y)
    sq_paths = alignment.path_label_squeeze(paths)
    self.assertAllEqual(sq_paths, exp_out)

    # Pads correctly via length arguments.
    paths = alignment.alignments_to_paths(
        self.alignments, self.len_x + 3, self.len_y + 3)
    sq_paths = alignment.path_label_squeeze(paths)
    self.assertAllEqual(sq_paths[:, :self.len_x, :self.len_y], exp_out)
    self.assertAllEqual(sq_paths[:, self.len_x:, :],
                        tf.zeros([1, 3, self.len_y + 3], tf.float32))
    self.assertAllEqual(sq_paths[Ellipsis, self.len_y:],
                        tf.zeros([1, self.len_x + 3, 3], tf.float32))

    # Deals with empty ground-truth alignments.
    paths = alignment.alignments_to_paths(
        tf.zeros_like(self.alignments), self.len_x, self.len_y)
    sq_paths = alignment.path_label_squeeze(paths)
    self.assertAllEqual(sq_paths, tf.zeros_like(exp_out))

  def test_alignments_to_state_indices(self):
    exp_out = tf.convert_to_tensor([[0, 0, 0],
                                    [0, 1, 1],
                                    [0, 2, 3],
                                    [0, 5, 4],
                                    [0, 6, 5],
                                    [0, 7, 6]], tf.int32)
    match_indices = alignment.alignments_to_state_indices(
        self.alignments, 'match')
    self.assertAllEqual(match_indices, exp_out)

    # Invariance to padding.
    padded_alignments = tf.concat(
        [self.alignments, tf.zeros([1, 3, 3], tf.int32)], 2)
    match_indices = alignment.alignments_to_state_indices(
        padded_alignments, 'match')

    # Deals with empty ground-truth alignments.
    exp_out = tf.zeros([0, 3], tf.int32)
    match_indices = alignment.alignments_to_state_indices(
        tf.zeros_like(self.alignments), 'match')
    self.assertAllEqual(match_indices, exp_out)

    # Gaps.
    self.assertAllEqual(match_indices, exp_out)
    exp_out = tf.convert_to_tensor([[0, 1, 2],
                                    [0, 3, 3]], tf.int32)
    gap_open_indices = alignment.alignments_to_state_indices(
        self.alignments, 'gap_open')
    self.assertAllEqual(gap_open_indices, exp_out)

    exp_out = tf.convert_to_tensor([[0, 4, 3]], tf.int32)
    gap_extend_indices = alignment.alignments_to_state_indices(
        self.alignments, 'gap_extend')
    self.assertAllEqual(gap_extend_indices, exp_out)

  def test_sw_score(self):
    # Test sw_score from sparse representation.
    sw_score = alignment.sw_score(self.sw_params, self.alignments)
    self.assertAllEqual(sw_score, [95.0])
    # Test sw_score from dense representation.
    paths = alignment.alignments_to_paths(
        self.alignments, self.len_x, self.len_y + 1)  # Testing padding too.
    sw_score = alignment.sw_score(self.sw_params, paths)
    self.assertAllEqual(sw_score, [95.0])

    # Test empty alignments / paths
    sw_score = alignment.sw_score(
        self.sw_params, tf.zeros_like(self.alignments))
    self.assertAllEqual(sw_score, [0.0])
    sw_score = alignment.sw_score(
        self.sw_params, tf.zeros_like(paths))
    self.assertAllEqual(sw_score, [0.0])

if __name__ == '__main__':
  tf.test.main()
