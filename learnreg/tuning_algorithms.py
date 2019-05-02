# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""The TuneReg algorithm for tuning regularization hyperparameters.

See section 3.1 of paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import six
from typing import Generator, Iterable, List, Mapping, Sequence, Text, Tuple

from learnreg import learn_lin_reg

DataPoint = learn_lin_reg.DataPoint
ParameterValue = float
ParameterMapping = Mapping[Text, ParameterValue]

TunerCoroutine = Generator[ParameterMapping, DataPoint, None]


class TuningAlgorithm(object):
  """Base class for algorithms for tuning regularization hyperparameters."""

  def get_initial_hparam_dicts(self):
    # type: (...) -> Iterable[ParameterMapping]:
    raise NotImplementedError()

  def update(
      self,
      pairs  # type: List[Tuple[ParameterMapping, DataPoint]]
  ):
    # type: (...) -> Iterable[ParameterMapping]:
    raise NotImplementedError()

  def run(self,
          get_data_point  # type: Callable[[ParameterMapping], DataPoint]
         ):
    # type: (...) -> Iterator[Tuple[ParameterMapping, DataPoint]]
    """Runs the algorithm, and yields (hparam_dict, data_point tuples)."""
    dicts = self.get_initial_hparam_dicts()
    while dicts:
      pairs = []
      for d in dicts:
        point = get_data_point(d)
        pair = (d, point)
        yield pair
        pairs.append(pair)
      dicts = self.update(pairs)


class CoroutineTuningAlgorithm(TuningAlgorithm):
  """Base class for TuningAlgorithms written as a coroutine.

  The coroutine yields ParameterMappings, and receives back the corresponding
  DataPoints.
  """

  def __init__(self,
               coroutine  # type: TunerCoroutine
              ):
    self.coroutine = coroutine

  def get_initial_hparam_dicts(self):
    # type: (...) -> Iterable[ParameterMapping]
    hparams = next(self.coroutine)
    return [hparams]

  def update(
      self,
      pairs  # type: List[Tuple[ParameterMapping, DataPoint]]
  ):
    # type: (...) -> Iterable[ParameterMapping]
    try:
      assert len(pairs) == 1
      _, point = pairs[0]
      hparams = self.coroutine.send(point)
      return [hparams]
    except StopIteration:
      return []


class TuneReg(CoroutineTuningAlgorithm):
  """The TuneReg algorithm defined in section 3 of paper."""

  def __init__(self,
               hparam_names,           # type: Sequence[Text]
               initial_hparams,        # type: Iterable[ParameterMapping]
               sample_hparam_mapping,  # type: Callable[[], ParameterMapping]
               eps=1e-6):
    """Initializer.

    Args:
      hparam_names: sequence of hyperparameter names
      initial_hparams: iterable of initial hyperparameter settings to try
        first, before solving LP
      sample_hparam_mapping: a callable that returns a randoml-sampled
        ParameterMapping
      eps: if a hyperparameter vector obtained by solving the LP is within
        distance eps of a previously-tried hyperparameter vector, we revert
        to random sampling
    """
    self.hparam_names = hparam_names
    self.initial_hparams = initial_hparams
    self.sample_hparams = sample_hparam_mapping
    self.eps = eps

    CoroutineTuningAlgorithm.__init__(self, self._coroutine())

  def _coroutine(self):
    # type: (...) -> TunerCoroutine
    """Function that returns generator coroutine that optimizes hyperparams."""

    def key_for_hparams(hparams):
      return tuple(sorted(six.iteritems(hparams)))

    hparams_to_point = {}
    for hparams in self.initial_hparams:
      point = yield hparams
      key = key_for_hparams(hparams)
      hparams_to_point[key] = point

    def dist(hparams0, hparams1):
      assert set(six.iterkeys(hparams0)) == set(six.iterkeys(hparams1))
      squared_dist = sum((hparams0[k] - hparams1[k])**2 for k in hparams0)
      return squared_dist**0.5

    while True:
      points = list(six.itervalues(hparams_to_point))
      _, coeffs = learn_lin_reg.learn_linear_regularizer(points)
      hparams = dict(zip(self.hparam_names, coeffs))

      # If the hparams we get from solving the LP are very close to ones
      # we've already tried, revert to random sampling instead.
      min_dist = min(dist(hparams, dict(prev_hparams))
                     for prev_hparams in hparams_to_point)
      if min_dist < self.eps:
        hparams = self.sample_hparams()

      point = yield hparams
      key = key_for_hparams(hparams)
      hparams_to_point[key] = point


def sample_hparams(ranges,      # type: Mapping[Text, Tuple[float, float]]
                   rand=random  # type: random.Random
                  ):
  # type: (...) -> ParameterMapping
  """Returns a randomly-sampled ParameterMapping."""
  return {
      name: min_val + rand.random() * (max_val - min_val)
      for name, (min_val, max_val) in six.iteritems(ranges)
  }

