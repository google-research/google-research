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

r"""Implements the distribution corresponding to the loss function.

This library implements the parts of Section 2 of "A General and Adaptive Robust
Loss Function", Jonathan T. Barron, https://arxiv.org/abs/1701.03077, that are
required for evaluating the negative log-likelihood (NLL) of the distribution
and for sampling from the distribution.
"""

import jax.numpy as jnp
import jax.random as random
from robust_loss_jax import cubic_spline
from robust_loss_jax import general



def get_resource_as_file(path):
  """A uniform interface for internal/open-source files."""

  class NullContextManager(object):

    def __init__(self, dummy_resource=None):
      self.dummy_resource = dummy_resource

    def __enter__(self):
      return self.dummy_resource

    def __exit__(self, *args):
      pass

  return NullContextManager('./' + path)


def get_resource_filename(path):
  """A uniform interface for internal/open-source filenames."""
  return './' + path


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
    alpha: A tensor with values >= 0.

  Returns:
    A tensor of curved values >= 0 with the same type as `alpha`, to be
    used as input x-coordinates for spline interpolation.
  """
  log_safe = lambda z: jnp.log(jnp.minimum(z, 3e37))
  x = jnp.where(alpha < 4,
                (2.25 * alpha - 4.5) / (jnp.abs(alpha - 2) + 0.25) + alpha + 2,
                (5 / 18) * log_safe(4 * alpha - 15) + 8)
  return x


def inv_partition_spline_curve(x):
  """The inverse of partition_spline_curve()."""
  exp_safe = lambda z: jnp.exp(jnp.minimum(z, 87.5))
  alpha = jnp.where(
      x < 8,
      0.5 * x + jnp.where(x <= 4, 1.25 - jnp.sqrt(1.5625 - x + .25 * x**2),
                          -1.25 + jnp.sqrt(9.5625 - 3 * x + .25 * x**2)),
      3.75 + 0.25 * exp_safe(x * 3.6 - 28.8))
  return alpha


class Distribution(object):
  """A wrapper class around the distribution."""

  def __init__(self):
    """Initialize the distribution.

    Load the values, tangents, and x-coordinate scaling of a spline that
    approximates the partition function. The spline was produced by running
    the script in fit_partition_spline.py.
    """
    with get_resource_as_file(
        'robust_loss_jax/data/partition_spline.npz') as spline_file:
      with jnp.load(spline_file, allow_pickle=False) as f:
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
      alpha: A tensor containing the set of alphas for which we would like an
        approximate log partition function. Must be non-negative, as the
        partition function is undefined when alpha < 0.

    Returns:
      An approximation of log(Z(alpha)) accurate to within 1e-6
    """
    # The partition function is undefined when `alpha`< 0.
    alpha = jnp.maximum(0, alpha)
    # Transform `alpha` to the form expected by the spline.
    x = partition_spline_curve(alpha)
    # Interpolate into the spline.
    return cubic_spline.interpolate1d(x * self._spline_x_scale,
                                      self._spline_values,
                                      self._spline_tangents)

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
    alpha = jnp.maximum(0, alpha)
    scale = jnp.maximum(jnp.finfo(jnp.float32).eps, scale)
    loss = general.lossfun(x, alpha, scale)
    return loss + jnp.log(scale) + self.log_base_partition_function(alpha)

  def draw_samples(self, rng, alpha, scale):
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
      rng: A JAX pseudo random number generated, from random.PRNG().
      alpha: A tensor where each element is the shape parameter of that
        element's distribution. Must be > 0.
      scale: A tensor where each element is the scale parameter of that
        element's distribution. Must be >=0 and the same shape as `alpha`.

    Returns:
      A tensor with the same shape as `alpha` and `scale` where each element is
      a sample drawn from the zero-mean distribution specified for that element
      by `alpha` and `scale`.
    """
    assert jnp.all(scale > 0)
    assert jnp.all(alpha >= 0)
    assert jnp.all(jnp.array(alpha.shape) == jnp.array(scale.shape))
    shape = alpha.shape

    samples = jnp.zeros(shape)
    accepted = jnp.zeros(shape, dtype=bool)

    # Rejection sampling.
    while not jnp.all(accepted):

      # The sqrt(2) scaling of the Cauchy distribution corrects for our
      # differing conventions for standardization.
      rng, key = random.split(rng)
      cauchy_sample = random.cauchy(key, shape=shape) * jnp.sqrt(2)

      # Compute the likelihood of each sample under its target distribution.
      nll = self.nllfun(cauchy_sample, alpha, 1)

      # Bound the NLL. We don't use the approximate loss as it may cause
      # unpredictable behavior in the context of sampling.
      nll_bound = (
          general.lossfun(cauchy_sample, 0, 1) +
          self.log_base_partition_function(alpha))

      # Draw N samples from a uniform distribution, and use each uniform
      # sample to decide whether or not to accept each proposal sample.
      rng, key = random.split(rng)
      uniform_sample = random.uniform(key, shape=shape)
      accept = uniform_sample <= jnp.exp(nll_bound - nll)

      # If a sample is accepted, replace its element in `samples` with the
      # proposal sample, and set its bit in `accepted` to True.
      samples = jnp.where(accept, cauchy_sample, samples)
      accepted = accept | accepted

    # Because our distribution is a location-scale family, we sample from
    # p(x | 0, \alpha, 1) and then scale each sample by `scale`.
    samples *= scale

    return samples
