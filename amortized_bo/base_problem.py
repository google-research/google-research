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

"""Defines the BaseProblem. Problems are expected to inherit from this class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from amortized_bo import data as data_utils
from amortized_bo import utils


@six.add_metaclass(abc.ABCMeta)
class BaseProblem(object):
  """Base problem specification."""

  def __init__(self, batch_size=None, num_rounds=None, name=None):
    """Base specification for a Problem.

    Args:
      batch_size: Size of batch that the problem should be queried with. If
        None, any batch size is allowed
      num_rounds: Number of times the problem is allowed to query the problem
        oracle.  If None, any number of queries is allowed
      name: The name of the problem.
    """
    self._batch_size = batch_size
    self._num_rounds = num_rounds
    self._name = name or self.__class__.__name__
    # TODO(christofa): Make domain a required argument of this class.
    if not hasattr(self, '_domain'):
      self._domain = None

  @property
  def name(self):
    return self._name

  def __str__(self):
    return '%s (batch_size=%s num_rounds=%s)' % (
        self._name, self._batch_size, self._num_rounds)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def num_rounds(self):
    return self._num_rounds

  @property
  def domain(self):
    """Get the domain specifying allowed solution space for to the problem."""
    return self._domain

  def get_seed_structures(self):
    """Returns an iterable of seed structures from the problem domain.

    These seed structures can be used to seed the optimization or to build
    an initial dataset.

    Returns:
      An iterable of seed structures from the problem domain, which can be
      empty if the problem does not provide seed structures.
    """
    return []

  def build_labeled_dataset(self, num_samples=100, seed=0):
    """Builds a problem-specific initial dataset.

    By default, samples sequences randomly and evaluates the problem on those
    sequences. Can be overwritten by problem-specific dataset generators.

    Args:
      num_samples: The number of samples to return.
      seed: An optional integer seed or np.random.RandomState to be used
        for the random number generator.

    Returns:
      A tf.data.Dataset with data_utils.DatasetSamples.
    """
    structures = self.domain.sample_uniformly(num_samples, seed=seed)
    rewards = self(structures)
    return utils.dataset_from_tensors(
        data_utils.DatasetSample(structure=structures, reward=rewards))

  def compute_metrics(self, population, fast_only=False):
    """Compute problem-specific metrics for a population.

    Args:
      population: A `data.Population`.
      fast_only: Whether to only compute metrics that are fast to compute.

    Returns:
      A dict mapping metric names to metric values.
    """
    del population, fast_only
    return dict()

  @abc.abstractmethod
  def __call__(self, inputs):
    """Given a batch of sequences, return a batch of rewards.

    Args:
      inputs: A [batch x length] array of points to evaluate.

    Returns:
      A [batch] size array of rewards.
    """
    raise NotImplementedError
