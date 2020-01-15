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

# Lint as: python2, python3
"""Library of functions for model calibration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.optimize
import scipy.special
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from uq_benchmark_2019 import uq_utils


def find_scaling_temperature(labels, logits, temp_range=(1e-5, 1e5)):
  """Find max likelihood scaling temperature using binary search.

  Args:
    labels: Integer labels (shape=[num_samples]).
    logits: Floating point softmax inputs (shape=[num_samples, num_classes]).
    temp_range: 2-tuple range of temperatures to consider.
  Returns:
    Floating point temperature value.
  """
  if not tf.executing_eagerly():
    raise NotImplementedError(
        'find_scaling_temperature() not implemented for graph-mode TF')
  if len(labels.shape) != 1:
    raise ValueError('Invalid labels shape=%s' % str(labels.shape))
  if len(logits.shape) not in (1, 2):
    raise ValueError('Invalid logits shape=%s' % str(logits.shape))
  if len(labels.shape) != 1 or len(labels) != len(logits):
    raise ValueError('Incompatible shapes for logits (%s) vs labels (%s).' %
                     (logits.shape, labels.shape))

  @tf.function(autograph=False)
  def grad_fn(temperature):
    """Returns gradient of log-likelihood WRT a logits-scaling temperature."""
    temperature *= tf.ones([])
    if len(logits.shape) == 1:
      dist = tfp.distributions.Bernoulli(logits=logits / temperature)
    elif len(logits.shape) == 2:
      dist = tfp.distributions.Categorical(logits=logits / temperature)
    nll = -dist.log_prob(labels)
    nll = tf.reduce_sum(nll, axis=0)
    grad, = tf.gradients(nll, [temperature])
    return grad

  tmin, tmax = temp_range
  return scipy.optimize.bisect(lambda t: grad_fn(t).numpy(), tmin, tmax)


def apply_temperature_scaling(temperature, probs):
  """Apply temperature scaling to an array of probabilities.

  Args:
    temperature: Floating point temperature.
    probs: Array of probabilities with probabilities over axis=-1.
  Returns:
    Temperature-scaled probabilities; same shape as input probs.
  """
  logits_t = uq_utils.np_inverse_softmax(probs).T / temperature
  return scipy.special.softmax(logits_t.T, axis=-1)
