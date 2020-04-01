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

"""Shared utilities for agents."""
import math
import tensorflow.compat.v1 as tf


def variable_summaries(var, name):
  """Attach summaries to a Tensor."""
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean/' + name, mean)
  stddev = tf.math.reduce_std(var)
  tf.summary.scalar('sttdev/' + name, stddev)
  tf.summary.scalar('max/' + name, tf.reduce_max(var))
  tf.summary.scalar('min/' + name, tf.reduce_min(var))
  tf.summary.histogram(name, var)


def compute_eval_gamma_interval(gamma_max, hyp_exponent, number_of_gammas):
  r"""This method computes the eval_gammas intervals.

  These are the non-exponentiated gammas.  We produce this eval_gammas
  sequence in a way that when the largest eval_gamma is exponentiated, the
  largest gamma being learned via Bellman updates is approximately equal
  to self.gamma_max.

  We can see that if we use a power-method for choosing the gamma interval,
  the base b must satisfy the relation,

        (1. - b ^ N) ^ k = \gamma_max

  where N is the number of gammas, k is the hyperbolic exponent.  The base b
  to use is then solved as

        b = exp[ ln(1. - \gamma_max ^ (1 \ k)) / N]

  Args:
    gamma_max:  The maximum gamma to use in Bellman equations.
    hyp_exponent:  The k-coefficient for hyperbolic discounting.
    number_of_gammas:  Number of gammas to simultaneously model.

  Returns:
    eval_gammas:  List of gammas selected for the x-axis of the Riemann
      rectangles.
  """
  if number_of_gammas > 1:
    b = math.exp(
        math.log(1. - math.pow(gamma_max, 1. / hyp_exponent)) /
        number_of_gammas)
    eval_gammas = [1. - math.pow(b, i) for i in range(number_of_gammas)]

  # If using only a single head, overwrite and set gammas to be the only
  # gamma being used.
  else:
    tf.logging.info('Running baseline model with single head.')
    eval_gammas = [gamma_max]

  return eval_gammas


def integrate_q_values(q_values, integral_estimate, eval_gammas,
                       number_of_gammas, gammas):
  """Estimate the integral for the hyperbolic discount q-values.

  This does a Riemann sum to approximate the integral.  This builds a lower or
  an upper estimate of the integral via a set of rectangles.

  Args:
    q_values:  List of q-values.
    integral_estimate: Type of Riemann integral estimate. One of 'lower',
      'upper'.
    eval_gammas:  List of gammas selected for the x-axis of the Riemann
      rectangles.
    number_of_gammas:  Number of gammas to simultaneously model.
    gammas:  List of exponentiated gammas that are actually used for Bellman
      updates.

  Returns:
    integral:  Scalar estimate of the integral.
  """
  integral = 0.

  if integral_estimate == 'lower':
    tf.logging.info('\nintegral_estimate == lower')
    gamma_plus_one = eval_gammas + [1.]

    # The weights are derived by which interval of gammas we wish to consider
    # i.e. the self.eval_gammas.  These are the non-exponentiated gammas.
    #
    # Example
    # If we're evaluating at \gamma = 0.5 and our k = 0.1
    # then we still evaluate the integral at \gamma = 0.5, but we are now
    # estimating the Q-values associated with \gamma ^ k = 0.5 ^ 0.1 = 0.933
    weights = [gamma_plus_one[i + 1] - gamma_plus_one[i] \
              for i in range(number_of_gammas)]

    for g, w in zip(gammas, weights):
      tf.logging.info(' V_{gamma=%0.5f} weight=%.3f', g, w)
    tf.logging.info('self.eval_gammas: %s', eval_gammas)
    tf.logging.info('self.gammas: %s', gammas)

    # Estimate the integral, which approximates the hyperbolic discounted
    # Q-values.
    for w, q_val in zip(weights, q_values):
      integral += w * q_val

  elif integral_estimate == 'upper':
    tf.logging.info('integral_estimate == upper')
    gamma = eval_gammas
    weights = [gamma[i + 1] - gamma[i] \
               for i in range(number_of_gammas - 1)]
    weights = [0.] + weights

    for g, w in zip(gammas, weights):
      tf.logging.info(' V_{gamma=%0.5f} weight=%.3f', g, w)
    tf.logging.info('self.eval_gammas: %s', eval_gammas)

    for w, q_val in zip(weights, q_values):
      integral += w * q_val

  else:
    raise NotImplementedError
  return integral
