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

# python3
"""Defines the target spec class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Any, Callable, Dict, List, NamedTuple
from typing import Optional, Text, Tuple, Type, TypeVar, Union

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'TargetDensity',
    'Expectation',
]


class Expectation(
    NamedTuple('Expectation', [
        ('fn', Callable[[Any], tf.Tensor]),
        ('human_name', Text),
        ('ground_truth_mean', Optional[np.ndarray]),
        ('ground_truth_mean_standard_error', Optional[np.ndarray]),
        ('ground_truth_standard_deviation', Optional[np.ndarray]),
    ])):
  """Describes an expectation with respect to a specific target's samples.

  Specifically, `E_{x~p}[f(x)]` for a target `p` and function `f`.  The target
  `p` is implicit, in that the `Expectation` appears in the `expectations` field
  of that `TargetDensity`.  The `f` is given as `fn` so that candidate
  samples may be passed through it.  The `fn` may close over the
  parameters of `p`, and the `ground_truth_mean` will presumably depend on `p`
  implicitly via some sampling process.

  If the `ground_truth_mean` is estimated by sampling, then
  `ground_truth_standard_deviation` and `ground_truth_mean_standard_error` are
  related using the standard formula:
  ```none
  SEM = SD / sqrt(N)
  ```
  where `N` is the number of samples. `ground_truth_standard_deviation`
  describes the distribution of `f(x)`, while `ground_truth_mean_standard_error`
  desribes how accurately we know `ground_truth_mean`.

  Attributes:
    fn: Function that takes samples from the target and returns
      a `Tensor`. The returned `Tensor` must retain the leading non-event
      dimensions.
    human_name: Human readable name, suitable for a table in a paper.
    ground_truth_mean: Ground truth value of this expectation. Can be `None` if
      not available.
    ground_truth_mean_standard_error: Standard error of the ground truth. Can be
      `None` if not available.
    ground_truth_standard_deviation: Standard deviation of samples transformed
      by `fn`. Can be `None` if not available.

  #### Examples

  An identity `fn` for a vector-valued target would look like:

  ```python
  fn = lambda x: x
  ```

  If the target is over a tuple-shaped space, then you could opt to return a
  particular element:

  ```python
  fn = lambda x: x[0]
  ```

  or concatenate all the elements (assuming they're vectors):

  ```python
  fn = lambda x: tf.concatenate(x, axis=-1)
  ```
  """
  __slots__ = ()

  def __str__(self):
    return self.human_name

  def __call__(self, arg):
    return self.fn(arg)


def expectation(
    fn,
    human_name,
    ground_truth_mean = None,
    ground_truth_mean_standard_error = None,
    ground_truth_standard_deviation = None,
):
  """Construct an `Expectation` with reasonable defaults."""
  if ground_truth_mean is not None and ground_truth_mean_standard_error is None:
    # TODO(axch): Make sure pytype complains about passing `float` here, or
    # adjust the type to accept it.
    ground_truth_mean = np.array(ground_truth_mean)
    eps = np.finfo(ground_truth_mean.dtype).eps
    ground_truth_mean_standard_error = eps * np.ones_like(ground_truth_mean)
  return Expectation(
      fn=fn,
      human_name=human_name,
      ground_truth_mean=ground_truth_mean,
      ground_truth_mean_standard_error=ground_truth_mean_standard_error,
      ground_truth_standard_deviation=ground_truth_standard_deviation,
  )


T = TypeVar('T', bound='TargetDensity')


class TargetDensity(
    NamedTuple('TargetDensity', [
        ('unnormalized_log_prob', Callable[[Any], tf.Tensor]),
        ('event_shape', tf.TensorShape),
        ('dtype', 'tf.Dtype'),
        ('constraining_bijectors', Union[tfb.Bijector, Tuple[tfb.Bijector],
                                         List[tfb.Bijector]]),
        ('expectations', 'collections.OrderedDict[Text, Expectation]'),
        ('distribution', Optional[tfd.Distribution]),
        ('code_name', Text),
        ('human_name', Text),
    ])):
  """Describes a target density.

  See `logistic_regression.py` for an example.

  Attributes:
    unnormalized_log_prob: A Python Callable computing the target density.
    event_shape: The event shape of the target, as a `tf.TensorShape`.
    dtype: The dtype in which the target expects to be computed.
    constraining_bijectors: Bijectors to constrain parameters from R^N to the
      `distribution`'s support.
    expectations: Ordered dictionary of code names to `Expectation`.
      Transformations of the `distribution`s samples.
    distribution: A `Distribution` representing the target, if available (e.g.,
      analytically).
    code_name: Code name of this target distribution.
    human_name: Human readable name, suitable for a table in a paper.
  """
  __slots__ = ()

  def __call__(self, arg):
    return self.unnormalized_log_prob(arg)

  def event_shape_tensor(self):
    """Returns the shape of this density's sample space, as a Tensor or nest.

    Returns:
      event_shape_tensor: The event shape tensor or a nest thereof.
    """
    return tf.constant(self.event_shape)

  def name(self):
    return self.code_name

  def __str__(self):
    return self.human_name

  def __hasattr__(self, name):
    return name in self.expectations

  def __getattr__(self, name):
    if name in self.expectations:
      return self.expectations[name]
    else:
      raise AttributeError('%r object has no attribute %r' %
                           (self.__class__.__name__, name))

  @classmethod
  def from_distribution(
      cls,
      distribution,
      constraining_bijectors,
      expectations,
      code_name,
      human_name):
    """Constructs a TargetDensity from a Distribution and annotations."""
    if isinstance(expectations, collections.OrderedDict):
      expectations_ord = expectations
    else:
      expectations_ord = collections.OrderedDict(
          sorted(expectations.items()))
    return TargetDensity(
        distribution.log_prob,
        distribution.event_shape,
        distribution.dtype,
        constraining_bijectors,
        expectations_ord,
        distribution,
        code_name,
        human_name)
