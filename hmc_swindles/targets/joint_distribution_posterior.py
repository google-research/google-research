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

"""The `JointDistributionPosterior` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional, Text, Tuple

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

__all__ = [
    'JointDistributionPosterior',
]


def _pick_unconditioned(values,
                        conditioning):
  return tuple(v for v, cv in zip(values, conditioning) if cv is None)


class JointDistributionPosterior(tfd.Distribution):
  """A posterior over a subset of variables of a `JointDistribution`.

  This distribution has an un-normalized `log_prob` and it cannot generate
  samples. Currently only tuple and list valued `JointDistribution`s are
  accepted.
  """

  def __init__(self,
               distribution,
               conditioning,  # pylint: disable=g-bare-generic
               validate_args = False,
               name = None):
    """Constructs a `JointDistributionPosterior`.

    Args:
      distribution: Base `JointDistribution`.
      conditioning: A tuple of the same length as the event tuple of
        `distribution`, with `None` elements representing the unconditioned
        values and `Tensor` elements representing the conditioned ones.
      validate_args: Python `bool`.  Whether to validate input with asserts. If
        `validate_args` is `False`, and the inputs are invalid, correct behavior
        is not guaranteed.
      name: Python `str` name prefixed to Ops created by this class. Default:
        `distribution.name + 'Posterior'`.
    """
    parameters = dict(locals())
    name = name or distribution.name + 'Posterior'
    with tf.name_scope(name) as name:
      self._distribution = distribution
      self._conditioning = conditioning

      super(JointDistributionPosterior, self).__init__(
          dtype=_pick_unconditioned(self._distribution.dtype,
                                    self._conditioning),
          reparameterization_type=_pick_unconditioned(
              self._distribution.reparameterization_type, self._conditioning),
          validate_args=validate_args,
          allow_nan_stats=self._distribution.allow_nan_stats,
          parameters=parameters,
          name=name)

  @property
  def distribution(self):
    return self._distribution

  def batch_shape_tensor(self):
    return _pick_unconditioned(self._distribution.batch_shape_tensor(),
                               self._conditioning)

  @property
  def batch_shape(self):
    return _pick_unconditioned(self._distribution.batch_shape,
                               self._conditioning)

  def event_shape_tensor(self):
    return _pick_unconditioned(self._distribution.event_shape_tensor(),
                               self._conditioning)

  @property
  def event_shape(self):
    return _pick_unconditioned(self._distribution.event_shape,
                               self._conditioning)

  def _log_prob(self, value):
    inner_value = list(self._conditioning)
    value_idx = 0
    for inner_idx, one_inner_value in enumerate(inner_value):
      if one_inner_value is None:
        inner_value[inner_idx] = value[value_idx]
        value_idx += 1
    return self._distribution.log_prob(inner_value)
