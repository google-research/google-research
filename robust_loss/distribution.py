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

r"""Implements the distribution corresponding to the loss function.

This library implements the parts of Section 2 of "A General and Adaptive Robust
Loss Function", Jonathan T. Barron, https://arxiv.org/abs/1701.03077, that are
required for evaluating the negative log-likelihood (NLL) of the distribution
and for sampling from the distribution.
"""

import numbers

import mpmath
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from robust_loss import cubic_spline
from robust_loss import general
from robust_loss import util


def analytical_base_partition_function(numer, denom):
  r"""Accurately approximate the partition function Z(numer / denom).

  This uses the analytical formulation of the true partition function Z(alpha),
  as described in the paper (the math after Equation 18), where alpha is a
  positive rational value numer/denom. This is expensive to compute and not
  differentiable, so it's not implemented in TensorFlow and is only used for
  unit tests.

  Args:
    numer: the numerator of alpha, an integer >= 0.
    denom: the denominator of alpha, an integer > 0.

  Returns:
    Z(numer / denom), a double-precision float, accurate to around 9 digits
    of precision.

  Raises:
      ValueError: If `numer` is not a non-negative integer or if `denom` is not
        a positive integer.
  """
  if not isinstance(numer, numbers.Integral):
    raise ValueError('Expected `numer` of type int, but is of type {}'.format(
        type(numer)))
  if not isinstance(denom, numbers.Integral):
    raise ValueError('Expected `denom` of type int, but is of type {}'.format(
        type(denom)))
  if not numer >= 0:
    raise ValueError('Expected `numer` >= 0, but is = {}'.format(numer))
  if not denom > 0:
    raise ValueError('Expected `denom` > 0, but is = {}'.format(denom))

  alpha = numer / denom

  # The Meijer-G formulation of the partition function has singularities at
  # alpha = 0 and alpha = 2, but at those special cases the partition function
  # has simple closed forms which we special-case here.
  if alpha == 0:
    return np.pi * np.sqrt(2)
  if alpha == 2:
    return np.sqrt(2 * np.pi)

  # Z(n/d) as described in the paper.
  a_p = (np.arange(1, numer, dtype=np.float64) / numer).tolist()
  b_q = ((np.arange(-0.5, numer - 0.5, dtype=np.float64)) /
         numer).tolist() + (np.arange(1, 2 * denom, dtype=np.float64) /
                            (2 * denom)).tolist()
  z = (1. / numer - 1. / (2 * denom))**(2 * denom)
  mult = np.exp(np.abs(2 * denom / numer - 1.)) * np.sqrt(
      np.abs(2 * denom / numer - 1.)) * (2 * np.pi)**(1 - denom)
  return mult * np.float64(mpmath.meijerg([[], a_p], [b_q, []], z))


def partition_spline_curve(alpha):
  """Applies a curve to alpha >= 0 to compress its range before interpolation.

  This is a weird hand-crafted function designed to take in alpha values and
  curve them to occupy a short finite range that works well when using spline
  interpolation to model the partition function Z(alpha). Because Z(alpha)
  is only varied in [0, 4] and is especially interesting around alpha=2, this
  curve is roughly linear in [0, 4] with a slope of ~1 at alpha=0 and alpha=4
  but a slope of ~10 at alpha=2. When alpha > 4 the curve becomes logarithmic.
  Some (input, output) pairs for this function are:
    [(0, 0), (1, ~1.2), (2, 4), (3, ~6.8), (4, 8), (8, ~8.8), (400000, ~12)]
  This function is continuously differentiable.

  Args:
    alpha: A numpy array or TF tensor (float32 or float64) with values >= 0.

  Returns:
    An array/tensor of curved values >= 0 with the same type as `alpha`, to be
    used as input x-coordinates for spline interpolation.
  """
  c = lambda z: tf.cast(z, alpha.dtype)
  assert_ops = [tf.Assert(tf.reduce_all(alpha >= 0.), [alpha])]
  with tf.control_dependencies(assert_ops):
    x = tf.where(alpha < 4, (c(2.25) * alpha - c(4.5)) /
                 (tf.abs(alpha - c(2)) + c(0.25)) + alpha + c(2),
                 c(5) / c(18) * util.log_safe(c(4) * alpha - c(15)) + c(8))
    return x


def inv_partition_spline_curve(x):
  """The inverse of partition_spline_curve()."""
  c = lambda z: tf.cast(z, x.dtype)
  assert_ops = [tf.Assert(tf.reduce_all(x >= 0.), [x])]
  with tf.control_dependencies(assert_ops):
    alpha = tf.where(
        x < 8,
        c(0.5) * x + tf.where(
            x <= 4,
            c(1.25) - tf.sqrt(c(1.5625) - x + c(.25) * tf.square(x)),
            c(-1.25) + tf.sqrt(c(9.5625) - c(3) * x + c(.25) * tf.square(x))),
        c(3.75) + c(0.25) * util.exp_safe(x * c(3.6) - c(28.8)))
    return alpha


class Distribution(object):
  """A wrapper class around the distribution."""

  def __init__(self):
    """Initialize the distribution.

    Load the values, tangents, and x-coordinate scaling of a spline that
    approximates the partition function. The spline was produced by running
    the script in fit_partition_spline.py.
    """
    with util.get_resource_as_file(
        'robust_loss/data/partition_spline.npz') as spline_file:
      with np.load(spline_file, allow_pickle=False) as f:
        self._spline_x_scale = f['x_scale']
        self._spline_values = f['values']
        self._spline_tangents = f['tangents']

  def log_base_partition_function(self, alpha):
    r"""Approximate the distribution's log-partition function with a 1D spline.

    Because the partition function (Z(\alpha) in the paper) of the distribution
    is difficult to model analytically, we approximate it with a (transformed)
    cubic hermite spline: Each alpha is pushed through a nonlinearity before
    being used to interpolate into a spline, which allows us to use a relatively
    small spline to accurately model the log partition function over the range
    of all non-negative input values.

    Args:
      alpha: A tensor or scalar of single or double precision floats containing
        the set of alphas for which we would like an approximate log partition
        function. Must be non-negative, as the partition function is undefined
        when alpha < 0.

    Returns:
      An approximation of log(Z(alpha)) accurate to within 1e-6
    """
    float_dtype = alpha.dtype

    # The partition function is undefined when `alpha`< 0.
    assert_ops = [tf.Assert(tf.reduce_all(alpha >= 0.), [alpha])]
    with tf.control_dependencies(assert_ops):
      # Transform `alpha` to the form expected by the spline.
      x = partition_spline_curve(alpha)
      # Interpolate into the spline.
      return cubic_spline.interpolate1d(
          x * tf.cast(self._spline_x_scale, float_dtype),
          tf.cast(self._spline_values, float_dtype),
          tf.cast(self._spline_tangents, float_dtype))

  def nllfun(self, x, alpha, scale):
    r"""Implements the negative log-likelihood (NLL).

    Specifically, we implement -log(p(x | 0, \alpha, c) of Equation 16 in the
    paper as nllfun(x, alpha, shape).

    Args:
      x: The residual for which the NLL is being computed. x can have any shape,
        and alpha and scale will be broadcasted to match x's shape if necessary.
        Must be a tensorflow tensor or numpy array of floats.
      alpha: The shape parameter of the NLL (\alpha in the paper), where more
        negative values cause outliers to "cost" more and inliers to "cost"
        less. Alpha can be any non-negative value, but the gradient of the NLL
        with respect to alpha has singularities at 0 and 2 so you may want to
        limit usage to (0, 2) during gradient descent. Must be a tensorflow
        tensor or numpy array of floats. Varying alpha in that range allows for
        smooth interpolation between a Cauchy distribution (alpha = 0) and a
        Normal distribution (alpha = 2) similar to a Student's T distribution.
      scale: The scale parameter of the loss. When |x| < scale, the NLL is like
        that of a (possibly unnormalized) normal distribution, and when |x| >
        scale the NLL takes on a different shape according to alpha. Must be a
        tensorflow tensor or numpy array of floats.

    Returns:
      The NLLs for each element of x, in the same shape as x. This is returned
      as a TensorFlow graph node of floats with the same precision as x.
    """
    # `scale` and `alpha` must have the same type as `x`.
    tf.debugging.assert_type(scale, x.dtype)
    tf.debugging.assert_type(alpha, x.dtype)
    assert_ops = [
        # `scale` must be > 0.
        tf.Assert(tf.reduce_all(scale > 0.), [scale]),
        # `alpha` must be >= 0.
        tf.Assert(tf.reduce_all(alpha >= 0.), [alpha]),
    ]
    with tf.control_dependencies(assert_ops):
      loss = general.lossfun(x, alpha, scale, approximate=False)
      log_partition = (
          tf.math.log(scale) + self.log_base_partition_function(alpha))
      nll = loss + log_partition
      return nll

  def draw_samples(self, alpha, scale):
    r"""Draw samples from the robust distribution.

    This function implements Algorithm 1 the paper. This code is written to
    allow for sampling from a set of different distributions, each parametrized
    by its own alpha and scale values, as opposed to the more standard approach
    of drawing N samples from the same distribution. This is done by repeatedly
    performing N instances of rejection sampling for each of the N distributions
    until at least one proposal for each of the N distributions has been
    accepted. All samples assume a zero mean --- to get non-zero mean samples,
    just add each mean to each sample.

    Args:
      alpha: A TF tensor/scalar or numpy array/scalar of floats where each
        element is the shape parameter of that element's distribution.
      scale: A TF tensor/scalar or numpy array/scalar of floats where each
        element is the scale parameter of that element's distribution. Must be
        the same shape as `alpha`.

    Returns:
      A TF tensor with the same shape and precision as `alpha` and `scale` where
      each element is a sample drawn from the zero-mean distribution specified
      for that element by `alpha` and `scale`.
    """
    # `scale` must have the same type as `alpha`.
    float_dtype = alpha.dtype
    tf.debugging.assert_type(scale, float_dtype)
    assert_ops = [
        # `scale` must be > 0.
        tf.Assert(tf.reduce_all(scale > 0.), [scale]),
        # `alpha` must be >= 0.
        tf.Assert(tf.reduce_all(alpha >= 0.), [alpha]),
        # `alpha` and `scale` must have the same shape.
        tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(alpha), tf.shape(scale))),
            [tf.shape(alpha), tf.shape(scale)]),
    ]

    with tf.control_dependencies(assert_ops):
      shape = tf.shape(alpha)

      # The distributions we will need for rejection sampling. The sqrt(2)
      # scaling of the Cauchy distribution corrects for our differing
      # conventions for standardization.
      cauchy = tfp.distributions.Cauchy(loc=0., scale=tf.sqrt(2.))
      uniform = tfp.distributions.Uniform(low=0., high=1.)

      def while_cond(_, accepted):
        """Terminate the loop only when all samples have been accepted."""
        return ~tf.reduce_all(accepted)

      def while_body(samples, accepted):
        """Generate N proposal samples, and then perform rejection sampling."""
        # Draw N samples from a Cauchy, our proposal distribution.
        cauchy_sample = tf.cast(cauchy.sample(shape), float_dtype)

        # Compute the likelihood of each sample under its target distribution.
        nll = self.nllfun(cauchy_sample, alpha, tf.cast(1, float_dtype))
        # Bound the NLL. We don't use the approximate loss as it may cause
        # unpredictable behavior in the context of sampling.
        nll_bound = general.lossfun(
            cauchy_sample,
            tf.cast(0, float_dtype),
            tf.cast(1, float_dtype),
            approximate=False) + self.log_base_partition_function(alpha)

        # Draw N samples from a uniform distribution, and use each uniform
        # sample to decide whether or not to accept each proposal sample.
        uniform_sample = tf.cast(uniform.sample(shape), float_dtype)
        accept = uniform_sample <= tf.math.exp(nll_bound - nll)

        # If a sample is accepted, replace its element in `samples` with the
        # proposal sample, and set its bit in `accepted` to True.
        samples = tf.where(accept, cauchy_sample, samples)
        accepted = accept | accepted
        return (samples, accepted)

      # Initialize the loop. The first item does not matter as it will get
      # overwritten, the second item must be all False.
      while_loop_vars = (tf.zeros(shape,
                                  float_dtype), tf.zeros(shape, dtype=bool))

      # Perform rejection sampling until all N samples have been accepted.
      terminal_state = tf.while_loop(
          cond=while_cond, body=while_body, loop_vars=while_loop_vars)

      # Because our distribution is a location-scale family, we sample from
      # p(x | 0, \alpha, 1) and then scale each sample by `scale`.
      samples = tf.multiply(terminal_state[0], scale)

      return samples
