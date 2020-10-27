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

"""Tests for multihop_utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel.models import multihop_utils


class InputUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def _compare_side_inputs(self,
                           side_inputs,
                           use_hard_g2l_mask=False,
                           use_hard_l2g_mask=False):
    # Shape (1, 15, 9). 9 because of 2 * radius + 1
    self.assertAllEqual(
        [
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 0],  #
                [0, 0, 0, 1, 1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1, 1, 0, 0, 0],  #
                [0, 1, 1, 1, 1, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 1, 1, 1, 1, 0],  #
                [0, 0, 0, 1, 1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1, 1, 0, 0, 0],  #
                [0, 1, 1, 1, 1, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 1, 1, 1, 1, 0],  #
                [0, 0, 0, 1, 1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1, 1, 0, 0, 0],  #
                [0, 1, 1, 1, 1, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 1, 1, 1, 0, 0],  #
                [0, 0, 0, 1, 1, 1, 0, 0, 0],  #
                [0, 0, 1, 1, 1, 0, 0, 0, 0],  #
            ],
        ],
        side_inputs.l2l_att_mask)

    # Shape: (1, 14, 14)
    self.assertAllEqual(
        [
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #
            ],
        ],
        side_inputs.g2g_att_mask)

    expected_l2g_mask = np.array([
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #
        ],
    ])

    # Shape: (1, 14, 15)
    expected_g2l_mask = np.transpose(expected_l2g_mask, [0, 2, 1])

    # Shape: (1, 15, 14)
    expected_hard_l2g_mask = np.array([
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
        ],
    ])

    # Shape: (1, 14, 15)
    expected_hard_g2l_mask = np.array([
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  #
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  #
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
        ],
    ])

    if use_hard_l2g_mask:
      self.assertAllEqual(expected_hard_l2g_mask, side_inputs.l2g_att_mask)
    else:
      self.assertAllEqual(expected_l2g_mask, side_inputs.l2g_att_mask)

    if use_hard_g2l_mask:
      self.assertAllEqual(expected_hard_g2l_mask, side_inputs.g2l_att_mask)
    else:
      self.assertAllEqual(expected_g2l_mask, side_inputs.g2l_att_mask)

    # Shape: (1, 15, 9). 9 because of 2 * radius + 1
    self.assertAllEqual(
        [
            [
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
                [4, 4, 4, 3, 0, 1, 2, 2, 2],  #
            ],
        ],
        side_inputs.l2l_relative_att_ids)

    # Relative position overwrite value = 7 as the vocab_size for relative
    # position generator =. 2 * max_relative_pos_distance + 1
    # and overwrite value for g2g_rel_att_ids = vocab_size + 2
    # Shape: (1, 14, 14)
    self.assertAllEqual(
        [
            [
                [0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
                [7, 0, 1, 2, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
                [7, 3, 0, 1, 2, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
                [7, 4, 3, 0, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
                [7, 4, 4, 3, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7],  #
                [7, 7, 7, 7, 7, 0, 1, 2, 2, 7, 7, 7, 7, 7],  #
                [7, 7, 7, 7, 7, 3, 0, 1, 2, 7, 7, 7, 7, 7],  #
                [7, 7, 7, 7, 7, 4, 3, 0, 1, 7, 7, 7, 7, 7],  #
                [7, 7, 7, 7, 7, 4, 4, 3, 0, 7, 7, 7, 7, 7],  #
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 1, 2, 2, 7],  #
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 0, 1, 2, 7],  #
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 3, 0, 1, 7],  #
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 4, 3, 0, 7],  #
                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0],  #
            ],
        ],
        side_inputs.g2g_relative_att_ids)

    # Shape: (1, 15, 14)
    # 2 new ids (vocab_size, vocab_size + 1) here to account for collisions
    # in the fused attention.
    expected_l2g_relative_att_ids = np.array([
        [
            [5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 6, 5, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 6, 5, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 6, 5, 5, 6, 5, 6, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 5, 5, 5, 6, 5, 5, 6, 5, 5, 5, 5, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 6, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  #
        ],
    ])

    # Shape: (1, 14, 15)
    expected_g2l_relative_att_ids = np.transpose(expected_l2g_relative_att_ids,
                                                 [0, 2, 1])

    self.assertAllEqual(expected_l2g_relative_att_ids,
                        side_inputs.l2g_relative_att_ids)
    self.assertAllEqual(expected_g2l_relative_att_ids,
                        side_inputs.g2l_relative_att_ids)

  @parameterized.named_parameters(
      ('disable_hard_g2l_disable_hard_l2g', False, False),
      ('enable_hard_g2l_disable_hard_l2g', True, False),
      ('disable_hard_g2l_enable_hard_l2g', False, True),
      ('enable_hard_g2l_hard_l2g', True, True),
  )
  def test_make_global_local_transformer_side_inputs(self, use_hard_g2l_mask,
                                                     use_hard_l2g_mask):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Example input:
    # Q_ID is token corresponding to question. P* represents paragraph tokens
    # and S* represents sentence level tokens.
    #
    # A total of 14 global tokens as follows:
    #. 0.    1.   2.   3.   4.   5. 6  7  8   9.  10 11 12    13
    # [CLS] Q_ID Q_ID Q_ID Q_ID  P1 T1 S1 S2  P2  T2 S1 S2  --padding--
    #
    # Long Input:
    # T*, W* represent (Title, Words) WordPieces in HotpotQA context.
    # For example, (T1, W1) correspond to (Title1, Title1Words) and belong
    # to the same sentence in the long input. Hence, there is only one
    # corresponding global token for both of them.
    # Q* represent the question WordPieces.
    #
    # S1, S2 are sentences (each with one WordPiece) of T1 and S3, S4 are
    # sentences (each with one WordPiece) of T2
    # Q1 Q2 Q3 Q4 T1 W1 S1 S2 T2 W2 S3 S4
    #
    # A total of 15 long tokens.
    #
    # Padding with 0s
    long_paragraph_breakpoints = tf.convert_to_tensor(
        [[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])

    # Note the difference here - padding with -1s instead.
    long_sentence_ids = tf.convert_to_tensor(
        [[1, 2, 3, 4, 6, 6, 7, 8, 10, 10, 11, 12, -1, -1, -1]])

    # Note the difference here - padding with -1s instead.
    long_paragraph_ids = tf.convert_to_tensor(
        [[-1, -1, -1, -1, 5, 5, 5, 5, 9, 9, 9, 9, -1, -1, -1]])

    # Let's say we want to link 0-th, 4-th, 5-th, 6-th long tokens to the 1st
    # 2nd global tokens.
    l2g_linked_ids = tf.convert_to_tensor(
        [[1, -1, -1, -1, 1, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1]])

    # Padding with 0s
    global_paragraph_breakpoints = tf.convert_to_tensor(
        [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]])

    # Let's say we want the first and the third global tokens to attend to
    # everything in the long (except padding) even if `hard_g2l` is enabled.
    # Note that this tensor will be used / applicable only when
    # `use_hard_g2l_mask` is enabled.
    ignore_hard_g2l_mask = tf.convert_to_tensor(
        [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Let's say we want the first long token to attend to everything in the
    # global (except padding) even if `hard_l2g` is enabled. Note that this
    # tensor will be used / applicable only when `use_hard_l2g_mask` is
    # enabled.
    ignore_hard_l2g_mask = tf.convert_to_tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    side_inputs = (
        multihop_utils.make_global_local_transformer_side_inputs(
            long_paragraph_breakpoints=long_paragraph_breakpoints,
            long_paragraph_ids=long_paragraph_ids,
            long_sentence_ids=long_sentence_ids,
            global_paragraph_breakpoints=global_paragraph_breakpoints,
            local_radius=4,
            relative_pos_max_distance=2,
            use_hard_g2l_mask=use_hard_g2l_mask,
            ignore_hard_g2l_mask=ignore_hard_g2l_mask,
            use_hard_l2g_mask=use_hard_l2g_mask,
            ignore_hard_l2g_mask=ignore_hard_l2g_mask,
            l2g_linked_ids=l2g_linked_ids))

    self._compare_side_inputs(
        side_inputs,
        use_hard_g2l_mask=use_hard_g2l_mask,
        use_hard_l2g_mask=use_hard_l2g_mask)


if __name__ == '__main__':
  tf.test.main()
