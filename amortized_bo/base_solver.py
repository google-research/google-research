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

# Lint as: python3
"""Solver base class."""

import abc

from absl import logging

from amortized_bo import utils


class BaseSolver(abc.ABC):
  """Solver base class."""

  def __init__(self,
               domain,
               random_state=None,
               name=None,
               log_level=logging.INFO,
               **kwargs):
    """Creates an instance of this class.

    Args:
      domain: An instance of a `Domain`.
      random_state: An instance of or integer seed to build a
        `np.random.RandomState`.
      name: The name of the solver. If `None`, will use the class name.
      log_level: The logging level of the solver-specific logger.
        -2=ERROR, -1=WARN,  0=INFO, 1=DEBUG.
      **kwargs: Named arguments stored in `self.cfg`.
    """
    self._domain = domain
    self._name = name or self.__class__.__name__
    self._random_state = utils.get_random_state(random_state)
    self._log = utils.get_logger(self._name, level=log_level)

    cfg = utils.Config(self._config())
    cfg.update(kwargs)
    self.cfg = cfg

  def _config(self):
    return {}

  @property
  def domain(self):
    """Return the optimization domain."""
    return self._domain

  @property
  def name(self):
    """Returns the solver name."""
    return self._name

  def __str__(self):
    return self._name

  @abc.abstractmethod
  def propose(self,
              num_samples,
              population=None,
              pending_samples=None,
              counter=0):
    """Proposes num_samples from `self.domain`.

    Args:
      num_samples: The number of samples to return.
      population: A `Population` of samples or None if the population is empty.
      pending_samples: A list of structures without reward that were already
        proposed.
      counter: The number of times `propose` has been called with the same
        `population`. Can be used by solver to avoid repeated computations on
        the same `population`, e.g. updating a model.

    Returns:
      `num_samples` structures from the domain.
    """
