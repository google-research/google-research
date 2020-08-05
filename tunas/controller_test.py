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

"""Tests for controller.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow.compat.v1 as tf

from tunas import basic_specs
from tunas import controller
from tunas import schema


class ControllerTest(tf.test.TestCase):

  def assertOneHot(self, array):
    self.assertLen(array.shape, 1)

    argmax = np.argmax(array)
    self.assertEqual(array[argmax], 1)

    array_copy = np.copy(array)
    array_copy[argmax] = 0
    self.assertAllEqual(array_copy, [0]*len(array_copy))

  def test_independent_sample_basic(self):
    structure = {
        'filters': schema.OneOf([48], basic_specs.FILTERS_TAG),
        'opA': schema.OneOf(['foo', 'bar', 'baz'], basic_specs.OP_TAG),
        'opB': schema.OneOf(['blah', 'yatta'], basic_specs.OP_TAG),
        'other': schema.OneOf(['W', 'X', 'Y', 'Z'], 'some_other_tag'),
    }

    rl_structure, dist_info = controller.independent_sample(structure)
    self.assertItemsEqual(structure.keys(), rl_structure.keys())
    self.assertEqual(
        {k: v.choices for (k, v) in structure.items()},
        {k: v.choices for (k, v) in rl_structure.items()})
    self.assertEqual(
        {k: v.tag for (k, v) in structure.items()},
        {k: v.tag for (k, v) in rl_structure.items()})

    self.assertEqual(rl_structure['opA'].mask.shape, [3])
    self.assertEqual(rl_structure['opB'].mask.shape, [2])
    self.assertEqual(rl_structure['filters'].mask.shape, [1])
    self.assertEqual(rl_structure['other'].mask.shape, [4])

    self.evaluate(tf.global_variables_initializer())
    self.assertEqual(dist_info['entropy'].shape, [])
    self.assertEqual(dist_info['entropy'].dtype, tf.float32)

    # Initially, all the logits are zero, so the entropy of a distribution with
    # N possible choices is -log(N). We sum up the entropies of four different
    # distributions, for opA, opB, filters, and other.
    self.assertAlmostEqual(
        self.evaluate(dist_info['entropy']),
        math.log(1) + math.log(2) + math.log(3) + math.log(4))

    self.assertEqual(dist_info['sample_log_prob'].shape, [])
    self.assertEqual(dist_info['sample_log_prob'].dtype, tf.float32)
    self.assertAlmostEqual(
        self.evaluate(dist_info['sample_log_prob']),
        math.log(1) + math.log(1/2) + math.log(1/3) + math.log(1/4))

    # The controller will visit the elements of 'structure' in sorted order
    # (based on their keys). So op_indices_0 will correspond to opA, and
    # op_indices_1 will correspond to opB. All variables are initialized to 0.
    self.assertItemsEqual(
        dist_info['logits_by_tag'].keys(),
        ['op_indices_0',
         'op_indices_1',
         'filters_indices_0',
         'some_other_tag_0'])

    self.assertEqual(dist_info['logits_by_tag']['filters_indices_0'].shape, [1])
    self.assertEqual(dist_info['logits_by_tag']['op_indices_0'].shape, [3])
    self.assertEqual(dist_info['logits_by_tag']['op_indices_1'].shape, [2])
    self.assertEqual(dist_info['logits_by_tag']['some_other_tag_0'].shape, [4])

    # Repeat, but with logits grouped by path instead of tag.
    self.assertItemsEqual(
        dist_info['logits_by_path'],
        ['filters', 'opA', 'opB', 'other'])
    self.assertEqual(dist_info['logits_by_path']['filters'].shape, [1])
    self.assertEqual(dist_info['logits_by_path']['opA'].shape, [3])
    self.assertEqual(dist_info['logits_by_path']['opB'].shape, [2])
    self.assertEqual(dist_info['logits_by_path']['other'].shape, [4])

  def test_independent_sample_increase_ops_probability_1(self):
    structure = schema.OneOf(['foo', 'bar', 'baz'], basic_specs.OP_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_ops_probability=1.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(rl_structure.mask), [1/3, 1/3, 1/3])
    self.assertEqual(self.evaluate(dist_info['sample_log_prob']), 0)

  def test_independent_sample_increase_ops_probability_0(self):
    structure = schema.OneOf(['foo', 'bar', 'baz'], basic_specs.OP_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_ops_probability=0.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertOneHot(self.evaluate(rl_structure.mask))
    self.assertAlmostEqual(
        self.evaluate(dist_info['sample_log_prob']),
        math.log(1/3))

  def test_independent_sample_increase_ops_does_not_affect_filters(self):
    structure = schema.OneOf([4, 8, 12], basic_specs.FILTERS_TAG)

    rl_structure, dist_info = controller.independent_sample(
        structure, increase_ops_probability=1.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertOneHot(self.evaluate(rl_structure.mask))
    self.assertAlmostEqual(
        self.evaluate(dist_info['sample_log_prob']),
        math.log(1/3))

  def test_independent_sample_increase_filters_probability_1(self):
    # Make sure that increase_filters does the right thing when the choices do
    # not appear in sorted order.
    structure = schema.OneOf([4, 12, 8], basic_specs.FILTERS_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_filters_probability=1.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(rl_structure.mask), [0, 1, 0])
    self.assertEqual(self.evaluate(dist_info['sample_log_prob']), 0)

  def test_independent_sample_increase_filters_probability_1_big_space(self):
    # Use a large enough number of choices that we're unlikely to select the
    # right one by random chance.
    structure = schema.OneOf(list(range(100)), basic_specs.FILTERS_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_filters_probability=1.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertAllClose(self.evaluate(rl_structure.mask), [0]*99 + [1])
    self.assertEqual(self.evaluate(dist_info['sample_log_prob']), 0)

  def test_independent_sample_increase_filters_probability_0(self):
    structure = schema.OneOf([4, 12, 8], basic_specs.FILTERS_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_filters_probability=0.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertOneHot(self.evaluate(rl_structure.mask))
    self.assertAlmostEqual(
        self.evaluate(dist_info['sample_log_prob']),
        math.log(1/3))

  def test_independent_sample_increase_ops_does_not_affect_ops(self):
    structure = schema.OneOf([42, 64], basic_specs.OP_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_filters_probability=1.0)

    self.evaluate(tf.global_variables_initializer())
    self.assertOneHot(self.evaluate(rl_structure.mask))
    self.assertAlmostEqual(
        self.evaluate(dist_info['sample_log_prob']),
        math.log(1/2))

  def test_independent_sample_hierarchical(self):
    structure = schema.OneOf(
        [
            schema.OneOf(['a', 'b', 'c'], basic_specs.OP_TAG),
            schema.OneOf(['d', 'e', 'f', 'g'], basic_specs.OP_TAG),
        ], basic_specs.OP_TAG)
    rl_structure, dist_info = controller.independent_sample(
        structure, increase_ops_probability=0, increase_filters_probability=0,
        hierarchical=True)

    tensors = {
        'outer_mask': rl_structure.mask,
        'entropy': dist_info['entropy'],
        'sample_log_prob': dist_info['sample_log_prob'],
    }

    self.evaluate(tf.global_variables_initializer())
    for _ in range(10):
      values = self.evaluate(tensors)
      if np.all(values['outer_mask'] == np.array([1, 0])):
        self.assertAlmostEqual(values['entropy'], math.log(2) + math.log(3))
        self.assertAlmostEqual(
            values['sample_log_prob'], math.log(1/2) + math.log(1/3))
      elif np.all(values['outer_mask'] == np.array([0, 1])):
        self.assertAlmostEqual(values['entropy'], math.log(2) + math.log(4))
        self.assertAlmostEqual(
            values['sample_log_prob'], math.log(1/2) + math.log(1/4))
      else:
        self.fail('Unexpected outer_mask: %s', values['outer_mask'])

  def test_independent_sample_not_hierarchical(self):
    structure = schema.OneOf(
        [
            schema.OneOf(['a', 'b', 'c'], basic_specs.OP_TAG),
            schema.OneOf(['d', 'e', 'f', 'g'], basic_specs.OP_TAG),
        ], basic_specs.OP_TAG)
    unused_rl_structure, dist_info = controller.independent_sample(
        structure, increase_ops_probability=0, increase_filters_probability=0,
        hierarchical=False)

    tensors = {
        'entropy': dist_info['entropy'],
        'sample_log_prob': dist_info['sample_log_prob'],
    }

    self.evaluate(tf.global_variables_initializer())
    for _ in range(10):
      values = self.evaluate(tensors)
      self.assertAlmostEqual(
          values['entropy'], math.log(2) + math.log(3) + math.log(4))
      self.assertAlmostEqual(
          values['sample_log_prob'],
          math.log(1/2) + math.log(1/3) + math.log(1/4))

  def test_independent_sample_temperature(self):
    structure = schema.OneOf(['foo', 'bar', 'baz'], basic_specs.OP_TAG)
    temperature = tf.placeholder_with_default(
        tf.constant(5.0, tf.float32), shape=(), name='temperature')
    rl_structure, dist_info = controller.independent_sample(
        structure, temperature=temperature)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())

      # Samples should be valid even when the temperature is set to a value
      # other than 1.
      self.assertOneHot(sess.run(rl_structure.mask))

      # Before training, the sample log-probability and entropy shouldn't be
      # affected by the temperature, since the probabilities are initialized
      # to a uniform distribution.
      self.assertAlmostEqual(
          sess.run(dist_info['sample_log_prob']), math.log(1/3))
      self.assertAlmostEqual(sess.run(dist_info['entropy']), math.log(3))

      # The gradients should be multiplied by (1 / temperature).
      # The OneOf has three possible choices. The gradient for the selected one
      # will be positive, while the gradients for the other two will be
      # negative. Since the selected choice can change between steps, we compare
      # the max, which should always give us gradients w.r.t. the selected one.
      trainable_vars = tf.trainable_variables()
      self.assertLen(trainable_vars, 1)

      grad_tensors = tf.gradients(dist_info['sample_log_prob'], trainable_vars)
      grad1 = np.max(sess.run(grad_tensors[0], {temperature: 1.0}))
      grad5 = np.max(sess.run(grad_tensors[0], {temperature: 5.0}))
      self.assertAlmostEqual(grad1 / 5, grad5)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
