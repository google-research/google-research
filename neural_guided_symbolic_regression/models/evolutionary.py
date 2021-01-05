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

"""Find expression by evolutionary algorithm.

This library includes code for symbolic regression with evolutionary algorithm.

Most code in this library are adapted from DEAP's documentation:

Symbolic Regression Problem: Introduction to GP
http://deap.gel.ulaval.ca/doc/dev/examples/gp_symbreg.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

import numpy as np

from neural_guided_symbolic_regression.models import metrics
from neural_guided_symbolic_regression.utils import evaluators


_NODE_TO_SYMBOL = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'divide_with_zero_divisor': '/',
    'constant1': '1',
    'ARG0': 'x',
}


def is_terminal(node):
  """Whether a node in primitive tree is terminal.

  Args:
    node: String, deap.gp.Primitive or deap.gp.Terminal.

  Returns:
    Boolean.
  """
  return isinstance(node, str) or node.arity == 0


def combine_nodes(node0, node1, node2):
  r"""Combine three nodes if they are operator, terminal, terminal.

  For example, + 1 x is the preorder of
      +
    /  \
   1   x
  The combined expressin is ( 1 + x ).

  If three nodes are not in this pattern, returns None.

  Args:
    node0: String, deap.gp.Primitive or deap.gp.Terminal.
    node1: String, deap.gp.Primitive or deap.gp.Terminal.
    node2: String, deap.gp.Primitive or deap.gp.Terminal.

  Returns:
    String.
  """
  def _get_name(node):
    if isinstance(node, (gp.Primitive, gp.Terminal)):
      return node.name
    else:
      return node
  if (not is_terminal(node0)) and is_terminal(node1) and is_terminal(node2):
    return '( %s %s %s )' % tuple(
        _NODE_TO_SYMBOL.get(node_name, node_name)
        for node_name in [_get_name(node1), _get_name(node0), _get_name(node2)])
  else:
    return None


def primitive_sequence_to_expression_string(sequence):
  """Converts primitive sequence to expression string.

  Args:
    sequence: List of deap.gp.Primitive.

  Returns:
    String.
  """
  # The sequence is a preorder traversal of a tree. So the length of the
  # sequence should be 1 + 2 * n, n is 0, 1, 2, ...
  if (len(sequence) - 1) % 2:
    raise ValueError(
        'The length of sequence should be 1 + 2 * n, but got %d'
        % len(sequence))

  if len(sequence) == 1:
    return _NODE_TO_SYMBOL.get(sequence[0].name, sequence[0].name)

  i = 0
  while len(sequence) > 1:
    if i >= 2:
      combined_element = combine_nodes(
          sequence[i - 2], sequence[i - 1], sequence[i])
      if combined_element is None:
        i += 1
      else:
        sequence = sequence[:i - 2] + [combined_element] + sequence[i + 1:]
        # sequence[i - 2], sequence[i - 1], sequence[i] are combined. The new
        # combined element is the (i - 2)-th element in the sequence.
        i -= 2
    else:
      i += 1
  return sequence[0]


def get_univariate_one_constant_primitives_set():
  """Gets primitives set.

  The operators, argument and constant defined in this primitives set is used
  as building blocks of symbolic expressions.

  This primitives set includes +, -, *, /, x, 1.

  Returns:
    deap.gp.PrimitiveSet.
  """
  pset = gp.PrimitiveSet('MAIN', 1)
  pset.addPrimitive(operator.add, 2)
  pset.addPrimitive(operator.sub, 2)
  pset.addPrimitive(operator.mul, 2)
  pset.addPrimitive(evaluators.divide_with_zero_divisor, 2)
  pset.addTerminal(1, 'constant1')
  pset.renameArguments(ARG0='x')
  return pset


def set_creator():
  """Sets creator."""
  creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
  creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)


def get_toolbox(pset, max_height):
  """Gets toolbox.

  Args:
    pset: deap.gp.PrimitiveSet.
    max_height: Integer, the max value of the height of tree.

  Returns:
    deap.base.Toolbox.
  """
  toolbox = base.Toolbox()
  toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
  toolbox.register(
      'individual', tools.initIterate, creator.Individual, toolbox.expr)
  toolbox.register('population', tools.initRepeat, list, toolbox.individual)
  toolbox.register('compile', gp.compile, pset=pset)
  toolbox.register('select', tools.selTournament, tournsize=3)
  toolbox.register('mate', gp.cxOnePoint)
  toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
  toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

  toolbox.decorate(
      'mate',
      gp.staticLimit(key=operator.attrgetter('height'), max_value=max_height))
  toolbox.decorate(
      'mutate',
      gp.staticLimit(key=operator.attrgetter('height'), max_value=max_height))
  return toolbox


def evolutionary_algorithm_with_num_evals_limit(
    population,
    toolbox,
    cxpb,
    mutpb,
    num_evals_limit,
    halloffame):
  """Runs evolutionary algorithm with limit of number of evaluations.

  The main logic of this function is from deap.algorithms.eaSimple(). The major
  difference is that the number of iteration is controlled by num_evals_limit
  instead of number of generations.

  Args:
    population: List of individuals.
    toolbox: deap.base.Toolbox, it contains the evolution operators.
    cxpb: Float, the probability of mating two individuals.
    mutpb: Float, the probability of mutating an individual.
    num_evals_limit: Integer, the limit of the number of evaluations.
    halloffame: deap.tools.HallOfFame, it records the best individuals.

  Returns:
    The final population.
  """
  # Evaluate the individuals with an invalid fitness (new individuals which
  # have not been evaluated).
  num_evals = 0

  invalid_ind = [ind for ind in population if not ind.fitness.valid]
  for ind, fit in zip(invalid_ind, toolbox.map(toolbox.evaluate, invalid_ind)):
    ind.fitness.values = fit

  halloffame.update(population)

  num_evals += len(invalid_ind)

  # Begin the generational process.
  while num_evals < num_evals_limit:
    # Select the next generation individuals.
    offspring = toolbox.select(population, len(population))

    # Vary the pool of individuals.
    offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

    # Evaluate the individuals with an invalid fitness.
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

    # Update the hall of fame with the generated individuals.
    halloffame.update(offspring)

    # Replace the current population by the offspring.
    population[:] = offspring

    num_evals += len(invalid_ind)

  return population


def evaluate_individual(
    individual,
    input_values,
    output_values,
    toolbox,
    leading_at_0=None,
    leading_at_inf=None,
    hard_penalty_default_value=None,
    include_leading_powers=False,
    default_value=50.):
  """Evaluates individual on input_values.

  NOTE(leeley): deap's evaluate function must be a generator.

  Args:
    individual: creator.Individual.
    input_values: Numpy array with shape [num_input_values]. List of input
        values to univariate function.
    output_values: Numpy array with shape [num_output_values]. List of output
        values from the univariate function.
    toolbox: deap.base.Toolbox, it contains the evolution operators.
    leading_at_0: Float, desired leading power at 0.
    leading_at_inf: Float, desired leading power at inf.
    hard_penalty_default_value: Float, the default value for hard penalty.
        Default None, the individual will be evaluated by soft penalty instead
        of hard penalty.
    include_leading_powers: Boolean, whether to include leading powers in
        evaluation.
    default_value: Float, default value if leading power error is nan.

  Returns:
    (Float,)
  """
  # Transform the tree expression in a callable function.
  func = toolbox.compile(expr=individual)
  ind_values = np.asarray([func(x) for x in input_values])
  input_values_rmse = np.sqrt(np.mean((output_values - ind_values) ** 2))
  if not include_leading_powers:
    if np.isfinite(input_values_rmse):
      return (input_values_rmse,)
    else:
      return (default_value,)

  true_leading_at_0, true_leading_at_inf = (
      metrics.evaluate_leading_powers_at_0_inf(
          expression_string=primitive_sequence_to_expression_string(individual),
          symbol='x'))
  leading_power_error = (
      abs(true_leading_at_0 - leading_at_0)
      + abs(true_leading_at_inf - leading_at_inf))

  if hard_penalty_default_value is None:
    # Soft penalty.
    if np.isfinite(leading_power_error):
      return (input_values_rmse + leading_power_error,)
    else:
      return (default_value,)
  else:
    # Hard penalty.
    if (np.isfinite(leading_power_error)
        and np.isclose(leading_power_error, 0)):
      return (input_values_rmse,)
    else:
      return (hard_penalty_default_value,)


def search_expression(
    input_values,
    output_values,
    pset,
    max_height=50,
    population_size=10,
    cxpb=0.5,
    mutpb=0.1,
    num_evals_limit=500,
    leading_at_0=None,
    leading_at_inf=None,
    hard_penalty_default_value=None,
    include_leading_powers=False,
    default_value=50.):
  """Searches expression using evolutionary algorithm.

  Args:
    input_values: Numpy array with shape [num_input_values]. List of input
        values to univariate function.
    output_values: Numpy array with shape [num_output_values]. List of output
        values from the univariate function.
    pset: deap.gp.PrimitiveSet.
    max_height: Integer, the max value of the height of tree.
    population_size: Integer, the size of population.
    cxpb: Float, the probability of mating two individuals.
    mutpb: Float, the probability of mutating an individual.
    num_evals_limit: Integer, the limit of the number of evaluations.
    leading_at_0: Float, desired leading power at 0.
    leading_at_inf: Float, desired leading power at inf.
    hard_penalty_default_value: Float, the default value for hard penalty.
        Default None, the individual will be evaluated by soft penalty instead
        of hard penalty.
    include_leading_powers: Boolean, whether to include leading powers in
        evaluation.
    default_value: Float, default value if leading power error is nan.

  Returns:
    individual: creator.Individual, the best individual in population.
    toolbox: deap.base.Toolbox, it contains the evolution operators.
  """
  toolbox = get_toolbox(pset, max_height)
  toolbox.register(
      'evaluate',
      evaluate_individual,
      input_values=input_values,
      output_values=output_values,
      toolbox=toolbox,
      leading_at_0=leading_at_0,
      leading_at_inf=leading_at_inf,
      hard_penalty_default_value=hard_penalty_default_value,
      include_leading_powers=include_leading_powers,
      default_value=default_value)
  population = toolbox.population(n=population_size)
  halloffame = tools.HallOfFame(1)

  evolutionary_algorithm_with_num_evals_limit(
      population=population,
      toolbox=toolbox,
      cxpb=cxpb,
      mutpb=mutpb,
      num_evals_limit=num_evals_limit,
      halloffame=halloffame)
  return halloffame[0], toolbox
