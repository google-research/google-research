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

"""Tests for evolutionary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deap import gp
from deap import tools
import numpy as np
import tensorflow.compat.v1 as tf

from neural_guided_symbolic_regression.models import evolutionary


class EvolutionaryTest(parameterized.TestCase):

  def setUp(self):
    super(EvolutionaryTest, self).setUp()
    self.pset = evolutionary.get_univariate_one_constant_primitives_set()

  @parameterized.parameters(
      ('( 1 + x )', True),
      (gp.Terminal('x', symbolic=True, ret=None), True),
      (gp.Primitive('c', args=(), ret=None), True),
      (gp.Primitive('add', args=(int, int), ret=int), False),
      )
  def test_is_terminal(self, node, expected):
    self.assertEqual(evolutionary.is_terminal(node), expected)

  @parameterized.parameters(
      (gp.Primitive('add', args=(int, int), ret=int),
       gp.Terminal('x', symbolic=True, ret=None),
       gp.Terminal('y', symbolic=True, ret=None),
       '( x + y )'),
      (gp.Primitive('add', args=(int, int), ret=int),
       gp.Primitive('add', args=(int, int), ret=int),
       gp.Terminal('y', symbolic=True, ret=None),
       None),
      (gp.Primitive('add', args=(int, int), ret=int),
       gp.Terminal('x', symbolic=True, ret=None),
       gp.Primitive('add', args=(int, int), ret=int),
       None),
      )
  def test_combine_nodes(self, node0, node1, node2, expected):
    self.assertEqual(evolutionary.combine_nodes(node0, node1, node2), expected)

  def test_primitive_sequence_to_expression_string(self):
    #       add
    #       / \
    #      x  mul
    #         / \
    #       sub  y
    #       / \
    #      a  b
    primitive_sequence = [
        gp.Primitive('add', args=(int, int), ret=int),
        gp.Terminal('x', symbolic=True, ret=None),
        gp.Primitive('mul', args=(int, int), ret=int),
        gp.Primitive('sub', args=(int, int), ret=int),
        # Whether symbolic is True or False does not matter.
        gp.Terminal(1, symbolic=True, ret=None),
        gp.Terminal(2, symbolic=False, ret=None),
        gp.Terminal('y', symbolic=True, ret=None),
    ]
    self.assertEqual(
        evolutionary.primitive_sequence_to_expression_string(
            primitive_sequence),
        '( x + ( ( 1 - 2 ) * y ) )')

  def test_primitive_sequence_to_expression_string_constant(self):
    primitive_sequence = [gp.Terminal('ARG0', symbolic=True, ret=None)]
    self.assertEqual(
        evolutionary.primitive_sequence_to_expression_string(
            primitive_sequence),
        'x')

  def test_primitive_sequence_to_expression_string_wrong_length(self):
    primitive_sequence = [
        gp.Primitive('add', args=(int, int), ret=int),
        gp.Terminal('x', symbolic=True, ret=None),
    ]
    with self.assertRaisesRegex(
        ValueError, r'The length of sequence should be 1 \+ 2 \* n, but got 2'):
      evolutionary.primitive_sequence_to_expression_string(primitive_sequence)

  def test_evolutionary_algorithm_with_num_evals_limit(self):
    evolutionary.set_creator()
    toolbox = evolutionary.get_toolbox(pset=self.pset, max_height=50)
    toolbox.register(
        'evaluate',
        evolutionary.evaluate_individual,
        input_values=np.array([1., 2., 3.]),
        output_values=np.array([2., 3., 4.]),
        toolbox=toolbox)
    population = toolbox.population(n=10)
    halloffame = tools.HallOfFame(1)

    evolutionary.evolutionary_algorithm_with_num_evals_limit(
        population=population,
        toolbox=toolbox,
        cxpb=0.5,
        mutpb=0.1,
        num_evals_limit=500,
        halloffame=halloffame)

    func = toolbox.compile(expr=halloffame[0])
    np.testing.assert_allclose(func(np.array([5., 6., 7.])), [6., 7., 8.])

  @parameterized.parameters(
      (None, None, None, False),
      (0, 1, None, True),
      (0, 1, 50., True),
      )
  def test_search_expression(
      self,
      leading_at_0,
      leading_at_inf,
      hard_penalty_default_value,
      include_leading_powers):
    # Test search several expressions.
    evolutionary.set_creator()

    evolutionary.search_expression(
        input_values=np.array([1., 2., 3.]),
        output_values=np.array([2., 3., 4.]),
        pset=self.pset,
        max_height=50,
        population_size=10,
        cxpb=0.5,
        mutpb=0.1,
        num_evals_limit=30,
        leading_at_0=leading_at_0,
        leading_at_inf=leading_at_inf,
        hard_penalty_default_value=hard_penalty_default_value,
        include_leading_powers=include_leading_powers,
        default_value=50.)

    evolutionary.search_expression(
        input_values=np.array([1., 2., 3.]),
        output_values=np.array([1., 4., 9.]),
        pset=self.pset,
        max_height=50,
        population_size=10,
        cxpb=0.5,
        mutpb=0.1,
        num_evals_limit=30,
        leading_at_0=leading_at_0,
        leading_at_inf=leading_at_inf,
        hard_penalty_default_value=hard_penalty_default_value,
        include_leading_powers=include_leading_powers,
        default_value=50.)


if __name__ == '__main__':
  tf.test.main()
