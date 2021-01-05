# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests for input_utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from etcmodel.models import input_utils


class InputUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def _compare_side_inputs(self,
                           side_inputs,
                           use_hard_g2l_mask=False,
                           use_hard_l2g_mask=False):
    self.assertAllEqual(
        [
            [
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
            ],  #
            [
                [0, 0, 1, 0, 0],  #
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
            ],  #
        ],
        side_inputs.l2l_att_mask)

    self.assertAllEqual(
        [
            [
                [1, 1, 1, 0],  #
                [1, 1, 1, 0],  #
                [1, 1, 1, 0],  #
                [0, 0, 0, 1],  #
            ],  #
            [
                [1, 0, 0, 0],  #
                [0, 1, 0, 0],  #
                [0, 0, 1, 0],  #
                [0, 0, 0, 1],  #
            ],  #
        ],
        side_inputs.g2g_att_mask)

    expected_l2g_att_mask = np.transpose(
        np.array([
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  #
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  #
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],  #
            ],  #
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  #
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            ],  #
        ]),
        [0, 2, 1])

    expected_hard_l2g_att_mask = np.transpose(
        np.array([
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],  #
            ],  #
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  #
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            ],  #
        ]),
        [0, 2, 1])

    expected_hard_g2l_att_mask = np.transpose(expected_hard_l2g_att_mask,
                                              [0, 2, 1])
    expected_g2l_att_mask = np.transpose(expected_l2g_att_mask, [0, 2, 1])

    self.assertAllEqual(
        expected_hard_g2l_att_mask if use_hard_g2l_mask else
        expected_g2l_att_mask, side_inputs.g2l_att_mask)

    self.assertAllEqual(
        expected_hard_l2g_att_mask if use_hard_l2g_mask else
        expected_l2g_att_mask, side_inputs.l2g_att_mask)

    self.assertAllEqual(
        [
            [
                [2, 2, 0, 1, 1],  #
            ] * 10,  #
        ] * 2,
        side_inputs.l2l_relative_att_ids)

    self.assertAllEqual(
        [
            [
                [0, 1, 1, 1],  #
                [2, 0, 1, 1],  #
                [2, 2, 0, 1],  #
                [2, 2, 2, 0],  #
            ],  #
        ] * 2,
        side_inputs.g2g_relative_att_ids)

    expected_g2l_relative_att_ids = np.array([
        [
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # Last 3 `1`s are `att_mask`ed out.
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],  #
        ],  #
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  #
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #
        ],  #
    ])
    # We add an offset of 3 for the 3 relative position ids used in
    # `l2l_relative_att_ids` and `g2g_relative_att_ids`.
    expected_g2l_relative_att_ids += 3

    self.assertAllEqual(
        np.transpose(expected_g2l_relative_att_ids, [0, 2, 1]),
        side_inputs.l2g_relative_att_ids)

    self.assertAllEqual(expected_g2l_relative_att_ids,
                        side_inputs.g2l_relative_att_ids)

  def test_global_local_transformer_side_inputs_to_dict(self):
    side_input_dict = input_utils.GlobalLocalTransformerSideInputs(
        l2l_att_mask=tf.ones([2, 16, 3]),
        g2g_att_mask=tf.ones([2, 4, 4]),
        l2g_att_mask=tf.ones([2, 16, 4]),
        g2l_att_mask=tf.ones([2, 4, 16]),
        l2l_relative_att_ids=None,
        g2g_relative_att_ids=None,
        l2g_relative_att_ids=None,
        g2l_relative_att_ids=None).to_dict()

    expected_keys = {
        'l2l_att_mask',
        'g2g_att_mask',
        'l2g_att_mask',
        'g2l_att_mask',
        'l2l_relative_att_ids',
        'g2g_relative_att_ids',
        'l2g_relative_att_ids',
        'g2l_relative_att_ids',
    }

    self.assertEqual(expected_keys, set(side_input_dict.keys()))

  def test_global_local_transformer_side_inputs_to_dict_exclude_nones(self):
    side_input_dict = input_utils.GlobalLocalTransformerSideInputs(
        l2l_att_mask=tf.ones([2, 16, 3]),
        g2g_att_mask=tf.ones([2, 4, 4]),
        l2g_att_mask=tf.ones([2, 16, 4]),
        g2l_att_mask=tf.ones([2, 4, 16]),
        l2l_relative_att_ids=None,
        g2g_relative_att_ids=None,
        l2g_relative_att_ids=None,
        g2l_relative_att_ids=None).to_dict(exclude_none_values=True)

    expected_keys = {
        'l2l_att_mask', 'g2g_att_mask', 'l2g_att_mask', 'g2l_att_mask'
    }

    self.assertEqual(expected_keys, set(side_input_dict.keys()))

  @parameterized.named_parameters(
      ('disable_hard_g2l_disable_hard_l2g', False, False),
      ('enable_hard_g2l_disable_hard_l2g', True, False),
      ('disable_hard_g2l_enable_hard_l2g', False, True),
      ('enable_hard_g2l_hard_l2g', True, True),
  )
  def test_make_global_local_transformer_side_inputs(self, use_hard_g2l_mask,
                                                     use_hard_l2g_mask):
    long_breakpoints = [
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  #
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #
    ]
    global_breakpoints = [
        [0, 0, 1, 1],  #
        [1, 1, 1, 0],  #
    ]
    sentence_ids = [
        [0, 1, 1, 2, 3, 3, 3, 0, 0, 0],  #
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 3],  #
    ]

    side_inputs = input_utils.make_global_local_transformer_side_inputs(
        long_breakpoints=long_breakpoints,
        global_breakpoints=global_breakpoints,
        sentence_ids=sentence_ids,
        local_radius=2,
        relative_pos_max_distance=1,
        use_hard_g2l_mask=use_hard_g2l_mask,
        use_hard_l2g_mask=use_hard_l2g_mask)

    self._compare_side_inputs(
        side_inputs,
        use_hard_g2l_mask=use_hard_g2l_mask,
        use_hard_l2g_mask=use_hard_l2g_mask)

  @parameterized.named_parameters(
      ('disable_hard_g2l_disable_hard_l2g', False, False),
      ('enable_hard_g2l_disable_hard_l2g', True, False),
      ('disable_hard_g2l_enable_hard_l2g', False, True),
      ('enable_hard_g2l_hard_l2g', True, True),
  )
  def test_make_global_local_transformer_side_inputs_from_example_ids(
      self, use_hard_g2l_mask, use_hard_l2g_mask):
    long_example_ids = [
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],  #
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 2],  #
    ]
    global_example_ids = [
        [0, 0, 0, 1],  #
        [0, 1, 2, 3],  #
    ]
    sentence_ids = [
        [0, 1, 1, 2, 3, 3, 3, 0, 0, 0],  #
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 3],  #
    ]

    side_inputs = (
        input_utils.make_global_local_transformer_side_inputs_from_example_ids(
            long_example_ids=long_example_ids,
            global_example_ids=global_example_ids,
            sentence_ids=sentence_ids,
            local_radius=2,
            relative_pos_max_distance=1,
            use_hard_g2l_mask=use_hard_g2l_mask,
            use_hard_l2g_mask=use_hard_l2g_mask))

    self._compare_side_inputs(
        side_inputs,
        use_hard_g2l_mask=use_hard_g2l_mask,
        use_hard_l2g_mask=use_hard_l2g_mask)

  def test_make_fixed_block_side_inputs(self):
    input_mask = [
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  #
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
    ]

    side_inputs, global_token_ids = input_utils.make_fixed_block_side_inputs(
        input_mask=input_mask,
        num_tokens_per_block=3,
        local_radius=2,
        relative_pos_max_distance=2)

    self.assertAllEqual(
        [
            [
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
            ],
            [
                [0, 0, 1, 1, 1],  #
                [0, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 0],  #
                [1, 1, 1, 0, 0],  #
            ],
        ],
        side_inputs.l2l_att_mask)

    self.assertAllEqual(
        [
            [
                [1, 1, 0, 0],  #
                [1, 1, 0, 0],  #
                [0, 0, 1, 1],  #
                [0, 0, 1, 1],  #
            ],
            [
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
                [1, 1, 1, 1],  #
            ]
        ],
        side_inputs.g2g_att_mask)

    self.assertAllEqual(
        [
            [
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  #
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  #
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  #
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  #
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #
            ]
        ],
        side_inputs.g2l_att_mask)

    self.assertAllEqual(
        tf.transpose(side_inputs.g2l_att_mask, [0, 2, 1]),
        side_inputs.l2g_att_mask)

    self.assertAllEqual(
        [
            [
                [4, 3, 0, 1, 2],  #
            ] * 10,  #
        ] * 2,
        side_inputs.l2l_relative_att_ids)

    self.assertAllEqual(
        [
            [
                [0, 1, 2, 2],  #
                [3, 0, 1, 2],  #
                [4, 3, 0, 1],  #
                [4, 4, 3, 0],  #
            ],  #
        ] * 2,
        side_inputs.g2g_relative_att_ids)

    self.assertAllEqual(
        [
            [
                [6, 6, 6, 5, 5, 5, 5, 5, 5, 5],  #
                [5, 5, 5, 6, 6, 6, 5, 5, 5, 5],  #
                [5, 5, 5, 5, 5, 5, 6, 6, 6, 5],  #
                [5, 5, 5, 5, 5, 5, 5, 5, 5, 6],  #
            ],  #
        ] * 2,
        side_inputs.g2l_relative_att_ids)

    self.assertAllEqual(
        tf.transpose(side_inputs.g2l_relative_att_ids, [0, 2, 1]),
        side_inputs.l2g_relative_att_ids)

    self.assertAllEqual(
        [
            [1, 1, 0, 0],  #
            [1, 1, 1, 1],  #
        ],
        global_token_ids)


if __name__ == '__main__':
  tf.test.main()
