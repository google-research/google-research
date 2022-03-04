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

"""A distributed implementation of regularized evolution.

E. Real, A. Aggarwal, Y. Huang, and Q. V. Le 2018
Regularized Evolution for Image Classifier Architecture Search
https://arxiv.org/abs/1802.01548
"""

import queue
import random
import time
import uuid

from absl import logging


class Individual:
  """Individual in the population.

  The solution of the problem is wrapped by this class.

  Attributes:
    gene: The information of the individual.
    fitness: Float, the fitness value of this individual.
  """

  def __init__(self, gene, fitness):
    """Initializer.

    Args:
      gene: The information of the individual.
      fitness: Float, the fitness value of this individual.
    """
    self.gene = gene
    self.fitness = fitness

  def __str__(self):
    """Readable version of the individual."""
    return f'gene: {self.gene}\nfitness: {self.fitness}'

  def __eq__(self, other):
    return isinstance(other, Individual) and self.gene == other.gene

  def serialize_gene(self):
    """Serializes the gene.

    Subclass can override this method.

    Returns:
      String, the serialization of the gene.
    """
    return f'{self.gene}'


class Population:
  """Population for task server."""

  def __init__(
      self,
      population_size,
      tournament_size,
      mutation_probability=0.9,
      max_mutations=-1,
      history_writer=None,
      other_config=None):
    """Initializes the population.

    Args:
      population_size: Integer, the number of individuals to keep in the
          population.
      tournament_size: Integer, the number of individuals that should
          participate in each tournament.
      mutation_probability: Float [0, 1], the probability of mutation.
      max_mutations: Integer, the max number of mutations. If exceeds, stop the
          search. -1 if there is no max mutations.
      history_writer: datatables.Writer, the history of individuals will be
          written to this datatable writer.
      other_config: ConfigDict or other parameter container for additional
          customized setting.
    """
    self._population_size = population_size
    self._tournament_size = tournament_size
    self._mutation_probability = mutation_probability
    self._history_writer = history_writer
    self._max_mutations = max_mutations
    self._history_counter = 0
    # Queue containing `individual_id` for each individual to solve.
    self._queue = queue.Queue()
    # Dictionary with `individual_id` as key and Individual as value.
    self._individuals = {}
    self._start_time = time.time()
    self._cfg = other_config
    self.create_initial_population()

  def create_initial_population(self):
    """Customized initialization of the population.

    The subclass overrides this method.
    """
    pass

  def _sample_tournament(self):
    """Samples tournament_size unique individuals from the population.

    Returns:
      List of Individual instances.
    """
    return random.sample(
        list(self._individuals.items()),
        k=min(len(self), self._tournament_size))

  def get_parent(self):
    """Gets parent for mutation.

    Obtain the best individual in the tournament.

    Returns:
      gene: The information of the individual.
    """
    probability = random.uniform(0, 1)
    while probability > self._mutation_probability:
      # Keep adding parents back to the population until mutation is allowed.
      individual = min(
          self._sample_tournament(), key=lambda x: x[1].fitness)[1]
      self.add_to_population(individual.gene, individual.fitness)
      probability = random.uniform(0, 1)

    individual = min(
        self._sample_tournament(), key=lambda x: x[1].fitness)[1]
    return individual.gene

  def add_to_population(self, gene, fitness, **kwargs):
    """Adds an individual to the population.

    Args:
      gene: The information of the individual.
      fitness: Float, the fitness value of this individual.
      **kwargs: Other information to record in history.

    Returns:
      Boolean. Whether to continue the search.
    """
    individual_id = str(uuid.uuid4())
    individual = Individual(gene=gene, fitness=fitness)
    self._individuals[individual_id] = individual
    self._queue.put_nowait(individual_id)
    if self._history_writer is not None:
      self._history_writer.write({
          'received_time': str(time.time()),
          'fitness': individual.fitness,
          'gene': individual.serialize_gene(),
          **kwargs,
      })
      # Flush and wait every 5 minutes.
      if time.time() - self._start_time > 300:
        logging.info('Flush the writer and wait the data server complete.')
        self._history_writer.wait()
        self._start_time = time.time()

    logging.info('Put %s into population with id %s', individual, individual_id)
    if len(self) > self._population_size:
      # pop the oldest individual.
      pop_individual_id = self._queue.get_nowait()
      pop_individual = self._individuals.pop(pop_individual_id)
      logging.info(
          'Remove %s from population with id %s',
          pop_individual, pop_individual_id)

    self._history_counter += 1
    if self._max_mutations > 0 and self._history_counter >= self._max_mutations:
      if self._history_writer is not None:
        self._history_writer.wait()
      return False
    else:
      return True

  def __len__(self):
    return len(self._individuals)
