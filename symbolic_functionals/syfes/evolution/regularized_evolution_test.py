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

"""Tests for evolution.regularized_evolution."""

import random
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from symbolic_functionals.syfes.evolution import regularized_evolution


class IndividualTest(absltest.TestCase):

  def test_str(self):
    self.assertEqual(
        str(regularized_evolution.Individual('foo', 1.5)),
        'gene: foo\nfitness: 1.5')

  def test_eq_true(self):
    self.assertEqual(
        # Two individuals are equal even when the fitness values are different.
        regularized_evolution.Individual('foo', 1.5),
        regularized_evolution.Individual('foo', 99.5))

  def test_eq_false(self):
    self.assertNotEqual(
        # Two individuals are not equal if their gene are different.
        regularized_evolution.Individual('foo', 1.5),
        regularized_evolution.Individual('bar', 1.5))

  def test_serialize_gene(self):
    self.assertEqual(
        regularized_evolution.Individual(3.14, 1.5).serialize_gene(), '3.14')


class RegularizedEvolutionTest(parameterized.TestCase):

  def test_get_parent(self):
    population = regularized_evolution.Population(
        population_size=5, tournament_size=2, mutation_probability=1.)
    with mock.patch.object(
        population, '_sample_tournament',
        return_value=[
            (123, regularized_evolution.Individual('foo', 1.5)),
            (456, regularized_evolution.Individual('bar', -1.5))]):
      self.assertEqual(population.get_parent(), 'bar')

  def test_get_parent_mutation_probability(self):
    population = regularized_evolution.Population(
        population_size=5, tournament_size=2, mutation_probability=0.9)
    with mock.patch.object(random, 'uniform', side_effect=[0.95, 0.8]):
      with mock.patch.object(
          population, '_sample_tournament',
          side_effect=[
              [
                  # Best individual from the first call is ignored since the
                  # mutation is not allowed.
                  (123, regularized_evolution.Individual('foo', 1.5)),
                  (456, regularized_evolution.Individual('bar', -1.5))],
              [
                  # Best individual from the second call is used.
                  (123, regularized_evolution.Individual('koo', 0.5)),
                  (456, regularized_evolution.Individual('par', -0.5))],
          ]):
        self.assertEqual(population.get_parent(), 'par')

  def test_add_to_population(self):
    population = regularized_evolution.Population(
        population_size=3, tournament_size=2)
    self.assertEmpty(population)
    population.add_to_population('a', 1.)
    self.assertLen(population, 1)
    population.add_to_population('b', 4.)
    self.assertLen(population, 2)
    population.add_to_population('c', 2.)
    self.assertLen(population, 3)
    # Exceed the population_size. The oldest individual 'a' will be removed.
    population.add_to_population('d', -1.)
    # assert False, list(population._individuals.values())
    self.assertCountEqual(
        population._individuals.values(),
        [
            regularized_evolution.Individual('b', 4.),
            regularized_evolution.Individual('c', 2.),
            regularized_evolution.Individual('d', -1.),
        ])

  def test_add_to_population_continue_search(self):
    population = regularized_evolution.Population(
        population_size=3, tournament_size=2, max_mutations=2)
    self.assertTrue(population.add_to_population('a', 1.))
    self.assertFalse(population.add_to_population('b', 1.))
    self.assertFalse(population.add_to_population('c', 1.))
    self.assertFalse(population.add_to_population('d', 1.))

  def test_history_writer(self):
    mock_writer = mock.MagicMock()
    mock_writer.write = mock.MagicMock()
    population = regularized_evolution.Population(
        population_size=1, tournament_size=2, history_writer=mock_writer)

    with mock.patch.object(time, 'time', return_value=123.456):
      population.add_to_population('foo', 1., train_error=99.)

    record = mock_writer.write.call_args[0][0]
    self.assertCountEqual(
        record.keys(), ['received_time', 'fitness', 'gene', 'train_error'])
    self.assertEqual(record['received_time'], '123.456')
    self.assertAlmostEqual(record['fitness'], 1.)
    self.assertEqual(record['gene'], 'foo')
    self.assertAlmostEqual(record['train_error'], 99.)


if __name__ == '__main__':
  absltest.main()
