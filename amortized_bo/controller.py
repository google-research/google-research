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

# Lint as: python3
"""A basic controller for in-silico optimization experiments."""

import abc

from absl import logging
import gin

from amortized_bo import data
from amortized_bo import utils


class BaseController(abc.ABC):
  """Controller base class."""

  def __init__(self,
               problem,
               solver,
               initial_population=None,
               callbacks=None,
               output_dir=None,
               verbose=False,
               seed=None):
    """Creates an instance of this class.

    Args:
      problem: An instance of a `BaseProblem`.
      solver: An instance of a `BaseSolver`.
      initial_population: An instance of a `Population` that is used to
        initialize the optimization.
      callbacks: A list of callables that take a population as input argument
        and that are called at the end of each round.
      output_dir: An output directory for storing data.
      verbose: Whether to log messages to the console.
      seed: The global seed of random number generators. If `None`, no seed is
        set explicitly.
    """
    if seed is not None:
      utils.set_seed(seed)

    self._callbacks = utils.to_list(callbacks)
    self._output_dir = output_dir
    self._problem = problem
    self._solver = solver
    self._population = (
        data.Population()
        if initial_population is None else initial_population.copy())
    self._verbose = verbose

    self._log('Initial population size: %d', len(self._population))
    if self._population:
      for callback in self._callbacks:
        callback(initial_population)

  def _log(self, *args, **kwargs):
    if self._verbose:
      logging.info(*args, **kwargs)

  @property
  def problem(self):
    """Returns the problem."""
    return self._problem

  @property
  def solver(self):
    """Returns the solver."""
    return self._solver

  @property
  def population(self):
    """Returns the population."""
    return self._population

  @abc.abstractmethod
  def step(self, batch_size):
    """Runs one optimization round with `batch_size` samples."""

  def run(self, num_rounds, batch_size):
    """Runs the controller for multiple optimization rounds.

    Args:
      num_rounds: The number of optimization rounds.
      batch_size: The number of samples per round. A single int to use the same
        batch size for each round a list of int of length `num_rounds` to use
        different batch sizes for different rounds.

    Returns:
      The resulting population.
    """
    if isinstance(batch_size, int):
      batch_sizes = [batch_size] * num_rounds
    else:
      batch_sizes = batch_size
      if len(batch_sizes) != num_rounds:
        raise ValueError(
            f'Got num_rounds={num_rounds} but got {len(batch_sizes)} '
            'batch_sizes {batch_sizes}')

    for batch_index, batch_size in enumerate(batch_sizes):
      self._log('Round %d', batch_index)
      self.step(batch_size)
      # TODO(christofa): Move into step after refactoring callbacks.
      for callback in self._callbacks:
        callback(self.population, force_write=batch_index == num_rounds - 1)

    return self.population


@gin.configurable
class BasicController(BaseController):
  """A basic controller for in-silico optimization experiments."""

  def __init__(self, problem, solver, log_population=False, **kwargs):
    """Creates an instance of this class.

    Args:
      problem: An instance of a `BaseProblem`.
      solver: An instance of a `BaseSolver` or a class reference.
      log_population: Whether to log the population to the console.
      **kwargs: Named arguments passed to the base class.
    """
    solver = utils.get_instance(solver, domain=problem.domain)
    super().__init__(problem, solver, **kwargs)
    self._log_population = log_population

  def step(self, batch_size):
    """Runs one optimization round with `batch_size` samples."""
    self._log('Requesting %d samples from solver %s', batch_size,
              self.solver.name)
    samples = self.solver.propose(batch_size, self.population.copy())
    if not isinstance(samples[0], data.Sample):
      samples = [data.Sample(structure=structure) for structure in samples]

    self._log('Evaluating problem %s on %d structures', self.problem.name,
              len(samples))
    rewards = self.problem([sample.structure for sample in samples])

    batch_index = self.population.current_batch_index + 1
    samples = [
        sample.copy(reward=reward, batch_index=batch_index, new_key=False)
        for sample, reward in zip(samples, rewards)
    ]

    self.population.add_samples(samples)
    if self._log_population:
      utils.log_pandas(self.population.get_last_batch().to_frame())


def run(problem, solver, num_rounds, batch_size, **kwargs):
  """Helper function to run an optimization experiment with the BasicController.

  Args:
    problem: An instance of a `BaseProblem`.
    solver: An instance of a `BaseSolver` or a class reference.
    num_rounds: The number of optimization rounds.
    batch_size: The number of samples per round. A single int to use the same
      batch size for each round a list of int of length `num_rounds` to use
      different batch sizes for different rounds.
    **kwargs: Named arguments passed to the constructor of the
      `BasicController`.

  Returns:
    The population at the end of the experiment.
  """
  controller = BasicController(problem, solver, **kwargs)
  controller.run(num_rounds, batch_size)
  return controller.population
