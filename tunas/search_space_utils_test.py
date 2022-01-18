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

"""Tests for search_space_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import mobile_search_space_v3
from tunas import schema
from tunas import search_space_utils


class SearchSpaceUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_normalize_strides(self):
    self.assertEqual(search_space_utils.normalize_strides(1), (1, 1))
    self.assertEqual(search_space_utils.normalize_strides((2, 2)), (2, 2))
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'neither an integer nor a pair of integers'):
      search_space_utils.normalize_strides((1, 2, 3))

  def test_scale_filters(self):
    self.assertEqual(search_space_utils.scale_filters(32, 1/4, 8), 8)
    self.assertEqual(search_space_utils.scale_filters(64, 1/3, 8), 24)
    self.assertEqual(search_space_utils.scale_filters(64, 1.4, 32), 96)
    self.assertEqual(search_space_utils.scale_filters(64, 3, 32), 192)
    self.assertEqual(search_space_utils.scale_filters(68, 1.0, 8), 72)
    self.assertEqual(search_space_utils.scale_filters(68, 1.2, 8), 80)
    self.assertEqual(search_space_utils.scale_filters(76, 1.0, 8), 80)
    self.assertEqual(search_space_utils.scale_filters(76, 1.2, 8), 88)

  def test_make_divisible(self):
    # If value < divisor then make_divisor(value, divisor) == divisor
    self.assertEqual(search_space_utils.make_divisible(7, 8), 8)
    self.assertEqual(search_space_utils.make_divisible(92, 100), 100)

    # If value % divisor == 0 then make_divisible(value, divisor) == value
    self.assertEqual(search_space_utils.make_divisible(8, 8), 8)
    self.assertEqual(search_space_utils.make_divisible(56, 8), 56)
    self.assertEqual(search_space_utils.make_divisible(72, 8), 72)

    # Otherwise, try to round to the nearest multiple of `base`.
    self.assertEqual(search_space_utils.make_divisible(1023, 8), 1024)
    self.assertEqual(search_space_utils.make_divisible(1025, 8), 1024)

    # We shouldn't round down by more than 10%. So this function outputs 24
    # rather than 16.
    self.assertEqual(search_space_utils.make_divisible(18, 8), 24)

  def test_tf_make_divisible(self):
    # If value < divisor then make_divisor(value, divisor) == divisor
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(7, 8)), 8)
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(92, 100)), 100)

    # If value % divisor == 0 then make_divisible(value, divisor) == value
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(8, 8)), 8)
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(56, 8)), 56)
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(72, 8)), 72)

    # Otherwise, try to round to the nearest multiple of `base`.
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(1023, 8)), 1024)
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(1025, 8)), 1024)

    # We shouldn't round down by more than 10%. So this function outputs 24
    # rather than 16.
    self.assertEqual(
        self.evaluate(search_space_utils.tf_make_divisible(18, 8)), 24)

    # Test behavior with Tensor inputs of different dtypes.
    self.assertEqual(
        self.evaluate(
            search_space_utils.tf_make_divisible(
                tf.constant(56, tf.int32), 8)),
        56)
    self.assertEqual(
        self.evaluate(
            search_space_utils.tf_make_divisible(
                tf.constant(56, tf.float32), 8)),
        56)

  def test_tf_scale_filters_values(self):
    output1 = search_space_utils.tf_scale_filters(32, 1/4, 8)
    self.assertEqual(self.evaluate(output1), 8)

    output2 = search_space_utils.tf_scale_filters(64, 1/3, 8)
    self.assertEqual(self.evaluate(output2), 24)

    output3 = search_space_utils.tf_scale_filters(64, 1.4, 32)
    self.assertEqual(self.evaluate(output3), 96)

    output4 = search_space_utils.tf_scale_filters(64, 3, 32)
    self.assertEqual(self.evaluate(output4), 192)

    output5 = search_space_utils.tf_scale_filters(68, 1.0, 8)
    self.assertEqual(self.evaluate(output5), 72)

    output6 = search_space_utils.tf_scale_filters(68, 1.2, 8)
    self.assertEqual(self.evaluate(output6), 80)

    output7 = search_space_utils.tf_scale_filters(76, 1.0, 8)
    self.assertEqual(self.evaluate(output7), 80)

    output8 = search_space_utils.tf_scale_filters(76, 1.2, 8)
    self.assertEqual(self.evaluate(output8), 88)

  def test_tf_scale_filters_types_and_shapes(self):
    output1 = search_space_utils.tf_scale_filters(32, 1/4, 8)
    self.assertEqual(output1.shape, [])
    self.assertEqual(output1.dtype, tf.int32)

    output2 = search_space_utils.tf_scale_filters(
        tf.constant(64, tf.int32),
        tf.constant(1/3, tf.float32),
        tf.constant(8, tf.int32))
    self.assertEqual(output2.shape, [])
    self.assertEqual(output2.dtype, tf.int32)

  def test_prune_simple_model_spec_no_tags(self):
    pruned_spec = search_space_utils.prune_model_spec(
        model_spec=[
            schema.OneOf(['a', 'b', 'c'], basic_specs.OP_TAG),
            schema.OneOf(['z', 'w'], basic_specs.FILTERS_TAG),
        ],
        genotype=[2, 1],
        prune_filters_by_value=True)

    self.assertEqual(
        pruned_spec,
        [
            schema.OneOf(['c'], basic_specs.OP_TAG),
            schema.OneOf(['w'], basic_specs.FILTERS_TAG),
        ])

  def test_prune_simple_model_spec_validation_no_tags(self):
    model_spec = [
        schema.OneOf(['a', 'b', 'c'], basic_specs.OP_TAG),
        schema.OneOf(['z', 'w'], basic_specs.FILTERS_TAG),
    ]

    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Genotype contains 1 oneofs but model_spec contains 2'):
      search_space_utils.prune_model_spec(
          model_spec=model_spec,
          genotype=[0],
          prune_filters_by_value=True)

  def test_prune_model_spec_with_path_dropout_training(self):
    model_spec = {
        'op1': schema.OneOf(
            [
                mobile_search_space_v3.ConvSpec(kernel_size=2, strides=2),
                basic_specs.ZeroSpec(),
            ], basic_specs.OP_TAG),
        'op2': schema.OneOf(
            [
                mobile_search_space_v3.ConvSpec(kernel_size=3, strides=4),
            ], basic_specs.OP_TAG),
        'filter': schema.OneOf([32], basic_specs.FILTERS_TAG),
    }

    model_spec = search_space_utils.prune_model_spec(
        model_spec,
        {basic_specs.OP_TAG: [0, 0]},
        path_dropout_rate=0.2,
        training=True)

    self.assertCountEqual(model_spec.keys(), ['op1', 'op2', 'filter'])
    self.assertEqual(model_spec['op1'].mask.shape, tf.TensorShape([1]))
    self.assertIsNone(model_spec['op2'].mask)
    self.assertIsNone(model_spec['filter'].mask)

    self.assertEqual(
        model_spec['op1'].choices,
        [mobile_search_space_v3.ConvSpec(kernel_size=2, strides=2)])
    self.assertEqual(
        model_spec['op2'].choices,
        [mobile_search_space_v3.ConvSpec(kernel_size=3, strides=4)])
    self.assertEqual(
        model_spec['filter'].choices,
        [32])

    self.assertEqual(model_spec['op1'].tag, basic_specs.OP_TAG)
    self.assertEqual(model_spec['op2'].tag, basic_specs.OP_TAG)
    self.assertEqual(model_spec['filter'].tag, basic_specs.FILTERS_TAG)

    op_mask_sum = 0
    for _ in range(100):
      # The value should either be 0 or 1 / (1 - path_dropout_rate) = 1.25
      op_mask_value = self.evaluate(model_spec['op1'].mask)
      self.assertTrue(
          abs(op_mask_value - 0) < 1e-6 or abs(op_mask_value - 1.25) < 1e-6,
          msg='Unexpected op_mask_value: {}'.format(op_mask_value))
      op_mask_sum += op_mask_value[0]

    # The probability of this test failing by random chance is roughly 0.002%.
    # Our random number generators are deterministically seeded, so the test
    # shouldn't be flakey.
    self.assertGreaterEqual(op_mask_sum, 75)
    self.assertLessEqual(op_mask_sum, 113)

  def test_prune_model_spec_with_path_dropout_rate_tensor(self):
    model_spec = {
        'op1': schema.OneOf(
            [
                mobile_search_space_v3.ConvSpec(kernel_size=2, strides=2),
                basic_specs.ZeroSpec(),
            ], basic_specs.OP_TAG),
        'op2': schema.OneOf(
            [
                mobile_search_space_v3.ConvSpec(kernel_size=3, strides=4),
            ], basic_specs.OP_TAG),
        'filter': schema.OneOf([32], basic_specs.FILTERS_TAG),
    }

    model_spec = search_space_utils.prune_model_spec(
        model_spec,
        {basic_specs.OP_TAG: [0, 0]},
        path_dropout_rate=tf.constant(2.0)  / tf.constant(10.0),
        training=True)

    self.assertCountEqual(model_spec.keys(), ['op1', 'op2', 'filter'])
    self.assertEqual(model_spec['op1'].mask.shape, tf.TensorShape([1]))
    self.assertIsNone(model_spec['op2'].mask)
    self.assertIsNone(model_spec['filter'].mask)

    # The value should either be 0 or 1 / (1 - path_dropout_rate) = 1.25
    op_mask_value = self.evaluate(model_spec['op1'].mask)
    self.assertTrue(
        abs(op_mask_value - 0) < 1e-6 or abs(op_mask_value - 1.25) < 1e-6,
        msg='Unexpected op_mask_value: {}'.format(op_mask_value))

  def test_prune_model_spec_with_path_dropout_eval(self):
    model_spec = {
        'op1': schema.OneOf(
            [
                mobile_search_space_v3.ConvSpec(kernel_size=2, strides=2),
                basic_specs.ZeroSpec(),
            ], basic_specs.OP_TAG),
        'op2': schema.OneOf(
            [
                mobile_search_space_v3.ConvSpec(kernel_size=3, strides=4),
            ], basic_specs.OP_TAG),
        'filter': schema.OneOf([32], basic_specs.FILTERS_TAG),
    }
    model_spec = search_space_utils.prune_model_spec(
        model_spec,
        {basic_specs.OP_TAG: [0, 0]},
        path_dropout_rate=0.2,
        training=False)

    self.assertCountEqual(model_spec.keys(), ['op1', 'op2', 'filter'])
    # Even though path_dropout_rate=0.2, the controller should not populate
    # the mask for op1 because we called prune_model_spec() with training=False.
    # In other words, path_dropout_rate should only affect the behavior during
    # training, not during evaluation.
    self.assertIsNone(model_spec['op1'].mask)
    self.assertIsNone(model_spec['op2'].mask)
    self.assertIsNone(model_spec['filter'].mask)

    self.assertEqual(model_spec['op1'].tag, basic_specs.OP_TAG)
    self.assertEqual(model_spec['op2'].tag, basic_specs.OP_TAG)
    self.assertEqual(model_spec['filter'].tag, basic_specs.FILTERS_TAG)

    self.assertEqual(
        model_spec['op1'].choices,
        [mobile_search_space_v3.ConvSpec(kernel_size=2, strides=2)])
    self.assertEqual(
        model_spec['op2'].choices,
        [mobile_search_space_v3.ConvSpec(kernel_size=3, strides=4)])
    self.assertEqual(
        model_spec['filter'].choices,
        [32])

  def test_prune_model_spec_prune_filters_by_value(self):
    model_spec = {
        'op': schema.OneOf(['a', 'b', 'c'], basic_specs.OP_TAG),
        'filters': schema.OneOf([128, 256, 512], basic_specs.FILTERS_TAG),
    }
    genotype = {
        basic_specs.OP_TAG: [1],
        basic_specs.FILTERS_TAG: [512],  # Use values instead of indices
    }
    pruned_spec = search_space_utils.prune_model_spec(
        model_spec, genotype, prune_filters_by_value=True)

    expected_spec = {
        'op': schema.OneOf(['b'], basic_specs.OP_TAG),
        'filters': schema.OneOf([512], basic_specs.FILTERS_TAG),
    }
    self.assertEqual(pruned_spec, expected_spec)

  def test_prune_model_spec_prune_filters_by_value_with_invalid_value(self):
    model_spec = {
        'op': schema.OneOf(['a', 'b', 'c'], basic_specs.OP_TAG),
        'filters': schema.OneOf([128, 256, 512], basic_specs.FILTERS_TAG),
    }
    genotype = {
        basic_specs.OP_TAG: [1],
        basic_specs.FILTERS_TAG: [1024],  # Use values instead of indices
    }
    with self.assertRaises(ValueError):
      search_space_utils.prune_model_spec(
          model_spec, genotype, prune_filters_by_value=True)

  def test_scale_conv_tower_spec_filter_multipliers(self):
    model_spec = basic_specs.ConvTowerSpec(
        blocks=[
            basic_specs.block(
                layers=[
                    schema.OneOf(
                        [
                            basic_specs.FilterMultiplier(3.0),
                            basic_specs.FilterMultiplier(6.0)
                        ], basic_specs.FILTERS_TAG)
                ],
                filters=48)
        ],
        filters_base=8)
    scaled_spec = search_space_utils.scale_conv_tower_spec(
        model_spec,
        multipliers=(0.5, 1, 2),
        base=8)

    # FilterMultiplier objects should not be affected by the scaling function.
    self.assertEqual(
        scaled_spec.blocks[0].layers[0].choices,
        [
            basic_specs.FilterMultiplier(3.0),
            basic_specs.FilterMultiplier(6.0),
        ])

    # However, absolute filter sizes should still be scaled.
    self.assertEqual(
        scaled_spec.blocks[0].filters.choices,
        [24, 48, 96])

  def test_tf_argmax_or_zero(self):
    # Pruned OneOf structure without mask
    oneof = schema.OneOf([1], 'foo')
    self.assertEqual(
        self.evaluate(search_space_utils.tf_argmax_or_zero(oneof)), 0)

    # Unpruned OneOf structure without mask
    with self.assertRaisesWithPredicateMatch(
        ValueError, 'Expect pruned structure'):
      oneof = schema.OneOf([1, 2], 'foo')
      search_space_utils.tf_argmax_or_zero(oneof)

    # Unpruned OneOf structure with mask
    oneof = schema.OneOf([1, 2], 'foo', tf.constant([0.0, 1.0]))
    self.assertEqual(
        self.evaluate(search_space_utils.tf_argmax_or_zero(oneof)), 1)

  def test_tf_indices_with_masks(self):
    model_spec = [
        schema.OneOf([1], 'foo', mask=tf.constant([1])),
        schema.OneOf([2, 3], 'bar', mask=tf.constant([0, 1])),
        schema.OneOf([4, 5, 6], 'baz', mask=tf.constant([0, 0, 1])),
    ]
    indices = search_space_utils.tf_indices(model_spec)
    self.assertAllEqual(self.evaluate(indices), [0, 1, 2])

  def test_parse_list(self):
    self.assertEqual([], search_space_utils.parse_list('', str))
    self.assertEqual([], search_space_utils.parse_list('   \t', str))
    self.assertEqual(['hello'], search_space_utils.parse_list('hello', str))
    self.assertEqual(['he', 'lo'], search_space_utils.parse_list('he:lo', str))
    self.assertEqual([42], search_space_utils.parse_list('42', int))
    self.assertEqual([4, 2], search_space_utils.parse_list('4:2', int))
    self.assertEqual([1, 2, 3], search_space_utils.parse_list('1:2:3', int))
    self.assertAllClose(
        [1.5, 2.5], search_space_utils.parse_list('1.5:2.5', float))

  def test_reward_for_single_cost_model_mnas(self):
    # Test for MNAS (soft) reward function
    estimated_cost = 1
    rl_cost_model_target = 0.5
    rl_cost_model_exponent = -0.07
    accuracy = 0.8

    rl_stats = search_space_utils.reward_for_single_cost_model(
        quality=accuracy,
        rl_reward_function='mnas',
        estimated_cost=estimated_cost,
        rl_cost_model_target=rl_cost_model_target,
        rl_cost_model_exponent=rl_cost_model_exponent)
    expected_rl_cost_ratio = estimated_cost / rl_cost_model_target
    self.assertAllClose(expected_rl_cost_ratio,
                        rl_stats['rl_cost_ratio'])
    expected_rl_cost_adjustment = pow(expected_rl_cost_ratio,
                                      rl_cost_model_exponent)
    self.assertAllClose(expected_rl_cost_adjustment,
                        self.evaluate(rl_stats['rl_cost_adjustment']))
    expected_rl_reward = accuracy * expected_rl_cost_adjustment
    self.assertAllClose(expected_rl_reward,
                        self.evaluate(rl_stats['rl_reward']))

  def test_reward_for_single_cost_model_abs(self):
    # Test for ABS reward function.
    estimated_cost = 1
    rl_cost_model_target = 0.5
    rl_cost_model_exponent = -0.07
    accuracy = 0.8

    rl_stats = search_space_utils.reward_for_single_cost_model(
        accuracy,
        rl_reward_function='abs',
        estimated_cost=estimated_cost,
        rl_cost_model_target=rl_cost_model_target,
        rl_cost_model_exponent=rl_cost_model_exponent)
    expected_rl_cost_ratio = estimated_cost / rl_cost_model_target
    self.assertAllClose(expected_rl_cost_ratio,
                        rl_stats['rl_cost_ratio'])
    expected_rl_cost_adjustment = (
        rl_cost_model_exponent * abs(expected_rl_cost_ratio - 1))
    self.assertAllClose(expected_rl_cost_adjustment,
                        self.evaluate(rl_stats['rl_cost_adjustment']))
    expected_rl_reward = accuracy + expected_rl_cost_adjustment
    self.assertAllClose(expected_rl_reward,
                        self.evaluate(rl_stats['rl_reward']))

  def test_reward_for_single_cost_model_mnas_hard(self):
    # Test for MNAS_HARD reward function.
    estimated_cost = 1
    rl_cost_model_target = 0.5
    rl_cost_model_exponent = -0.07
    accuracy = 0.8

    rl_stats = search_space_utils.reward_for_single_cost_model(
        accuracy,
        rl_reward_function='mnas_hard',
        estimated_cost=estimated_cost,
        rl_cost_model_target=rl_cost_model_target,
        rl_cost_model_exponent=rl_cost_model_exponent)
    expected_rl_cost_ratio = estimated_cost / rl_cost_model_target
    self.assertAllClose(expected_rl_cost_ratio,
                        rl_stats['rl_cost_ratio'])
    expected_rl_cost_adjustment = min(
        pow(expected_rl_cost_ratio, rl_cost_model_exponent), 1)
    self.assertAllClose(expected_rl_cost_adjustment,
                        self.evaluate(rl_stats['rl_cost_adjustment']))
    expected_rl_reward = accuracy * expected_rl_cost_adjustment
    self.assertAllClose(expected_rl_reward,
                        self.evaluate(rl_stats['rl_reward']))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
