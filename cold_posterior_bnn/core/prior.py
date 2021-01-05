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

"""Priors for neural network parameters.

This file provides probabilistic priors usable with Keras's regularization API.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import math

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tensorflow_probability as tfp

tfd = tfp.distributions


class PriorRegularizer(tf.keras.regularizers.Regularizer):
  """Base class for regularizers based on proper priors."""

  def __init__(self, weight=1.0):
    """Initialize prior regularizer.

    Args:
      weight: Tensor, scalar, float, >=0.0, the negative log-likelihood is
        multiplied with this weight factor.  This can be used, for example,
        to ensure the log-prior is appropriately scaled with the total sample
        size in Bayesian neural networks: weight=1.0/total_sample_size.
    """
    self.weight = weight

  def get_config(self):
    return {'weight': self.weight}

  def logpdf(self, x):
    raise NotImplementedError('Derived classes need to implement '
                              'logpdf method.')

  def __call__(self, w):
    raise NotImplementedError('Derived classes need to implement '
                              '__call__ method.')


class NormalRegularizer(PriorRegularizer):
  """Zero mean Normal prior."""

  def __init__(self, stddev=0.1, **kwargs):
    """Initialize a Normal prior.

    Args:
      stddev: Tensor, scalar, the standard deviation of the Normal prior.
      **kwargs: keyword arguments passed to base class.
    """
    self.stddev = stddev
    super(NormalRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(NormalRegularizer, self).get_config()
    config.update({'stddev': self.stddev})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    reg = self(x)
    nelem = tf.cast(tf.size(x), x.dtype)
    logz = nelem * (-math.log(self.stddev) - 0.5*math.log(2.0*math.pi))
    ll = -reg + self.weight*logz   # weight already in reg

    return ll

  def __call__(self, x):
    return 0.5*self.weight*tf.reduce_sum(tf.square(x / self.stddev))


class ShiftedNormalRegularizer(PriorRegularizer):
  """Normal prior with non-zero mean.


  """

  def __init__(self, mean=0, stddev=0.1, **kwargs):
    """Initialize a Normal prior.

    The typical use case of this prior is to center the prior around a point
    estimate (e.g., obtained by SGD). This leads to a posthoc non-informative
    prior in the vicinity of a mode (point estimate).

    Use the method utils.center_shifted_normal_around_model_weights(model) to
    center all ShiftedNormalRegularizer in the given model around current
    model weights.

    Args:
      mean: Tensor, multi-dimensional or scalar, the mean of the Normal prior.
            If both mean and stddev are multi-dimensional tensors they must have
            the shame shape.
      stddev: Tensor, multi-dimensional or scalar, the standard deviation of
              the Normal prior. If both mean and stddev are multi-dimensional
              tensors they must have the shame shape.
      **kwargs: keyword arguments passed to base class.
    """
    self.mean = mean
    self.stddev = stddev
    super(ShiftedNormalRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(ShiftedNormalRegularizer, self).get_config()
    config.update({'mean': self.mean})
    config.update({'stddev': self.stddev})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    reg = self(x)
    nelem = tf.cast(tf.size(x), x.dtype)
    logz = nelem * (-math.log(self.stddev) - 0.5*math.log(2.0*math.pi))
    ll = -reg + self.weight*logz   # weight already in reg

    return ll

  def __call__(self, x):
    return 0.5 * self.weight * tf.reduce_sum(
        tf.square((x - self.mean) / self.stddev))


class StretchedNormalRegularizer(PriorRegularizer):
  """Stretched Normal regularization."""

  def __init__(self, offset=1.0, scale=1.0, **kwargs):
    """Stretched Normal prior regularization.

    The stretched Normal distribution has a flat part in the middle and then
    Gaussian tails to the left and right.  The univariate normalized density
    function is determined by an offset and a scale parameter as follows:

      p(x; offset, scale) = exp(-0.5*((|x|-offset)^2)/scale^2) /
                              (2*offset + sqrt(2*pi*scale^2))
                            { if |x|-a >= 0 },
                          = 1.0 / (2*offset + sqrt(2*pi*scale^2))
                            { if |x|-a < 0 }.

    For offset=0 this distribution becomes a zero mean Normal(0,scale^2)
    distribution.

    Args:
      offset: float, >= 0.0, the offset at which the Gaussian tails start.
      scale: float, > 0.0, the Normal tail standard deviation.
      **kwargs: keyword arguments passed to base class.
    """
    self.offset = offset
    self.scale = scale
    super(StretchedNormalRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(StretchedNormalRegularizer, self).get_config()
    config.update({'offset': self.offset,
                   'scale': self.scale})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    reg = self(x)
    nelem = tf.cast(tf.size(x), x.dtype)
    logz = nelem * (-math.log(
        2.0*self.offset + self.scale*math.sqrt(2.0*math.pi)))
    ll = -reg + self.weight*logz

    return ll

  def __call__(self, x):
    diff = tf.abs(x) - self.offset
    logp = tf.where(diff >= 0.0,
                    -0.5*tf.square(diff / self.scale),
                    tf.zeros_like(diff))

    # Do not normalize (not necessary as not dependent on x)
    # logp = logp - tf.math.log(2.0*self.offset +
    #                           self.scale*math.sqrt(2.0*math.pi))
    regularization = -self.weight * tf.reduce_sum(logp)

    return regularization


def _compute_fans(shape, data_format='channels_last'):
  """Computes the number of input and output units for a weight shape.

  Arguments:
    shape: Integer shape tuple.
    data_format: Image data format to use for convolution kernels.
      Note that all kernels in Keras are standardized on the
       `channels_last` ordering (even when inputs are set
       to `channels_first`).

  Returns:
    A tuple of scalars, `(fan_in, fan_out)`.

  Raises:
    ValueError: in case of invalid `data_format` argument.
  """
  if len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) in {3, 4, 5}:
    # Assuming convolution kernels (1D, 2D or 3D).
    # TH kernel shape: (depth, input_depth, ...)
    # TF kernel shape: (..., input_depth, depth)
    if data_format == 'channels_first':
      receptive_field_size = tf.reduce_prod(shape[2:])
      fan_in = shape[1] * receptive_field_size
      fan_out = shape[0] * receptive_field_size
    elif data_format == 'channels_last':
      receptive_field_size = tf.reduce_prod(shape[:-2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
    else:
      raise ValueError('Invalid data_format: ' + data_format)
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif not shape:  # len(shape) == 0
    fan_in = fan_out = 1
  else:
    raise NotImplementedError()
  return fan_in, fan_out


def _he_stddev(fan_in):
  """He-stddev scaling rule based on fan-in.

  The original He-scaling, see Section 2.2 in
  https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf

  This initialization also has the "edge of chaos" property if the ReLU
  activation is used, see Figure 4(b) in https://arxiv.org/pdf/1711.00165.pdf,
  and thus maximizes information propagation.

  Args:
    fan_in: int, or int Tensor, >= 1.

  Returns:
    stddev: He scaling standard deviation.
  """
  fan_in = tf.cast(fan_in, 'float32')
  return tf.sqrt(2.0/fan_in)


class HeNormalRegularizer(PriorRegularizer):
  """He-inspired Normal regularization."""

  def __init__(self, scale=1.0,
               data_format='channels_last', **kwargs):
    """Initialize a He Normal prior.

    Args:
      scale: float, > 0.0, the He standard deviation is scaled with this factor.
      data_format: Image data format to use for convolution kernels.
        Note that all kernels in Keras are standardized on the
        `channels_last` ordering (even when inputs are set
        to `channels_first`).
      **kwargs: keyword arguments passed to base class.
    """

    self.scale = scale
    self.data_format = data_format
    super(HeNormalRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(HeNormalRegularizer, self).get_config()
    config.update({'scale': self.scale,
                   'data_format': self.data_format})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    raise NotImplementedError('logpdf not implemented.')

  def __call__(self, x):
    fan_in, _ = _compute_fans(x.shape, self.data_format)
    stddev = self.scale*_he_stddev(fan_in)
    reg_lambda = 0.5 * self.weight / (stddev**2.0)
    regularization = reg_lambda * tf.reduce_sum(tf.square(x))
    return regularization


class GlorotNormalRegularizer(PriorRegularizer):
  """Glorot-inspired Normal regularization."""

  def __init__(self, scale=1.0,
               data_format='channels_last', **kwargs):
    """Initialize a Glorot Normal prior.

    Args:
      scale: float, > 0.0, the Glorot standard deviation is scaled with this
        factor.
      data_format: Image data format to use for convolution kernels.
        Note that all kernels in Keras are standardized on the
        `channels_last` ordering (even when inputs are set
        to `channels_first`).
      **kwargs: keyword arguments passed to base class.
    """

    self.scale = scale
    self.data_format = data_format
    super(GlorotNormalRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(GlorotNormalRegularizer, self).get_config()
    config.update({'scale': self.scale})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    raise NotImplementedError('logpdf not implemented.')

  def __call__(self, x):
    fan_in, fan_out = _compute_fans(x.shape, self.data_format)

    def glorot_stddev(fan_in, fan_out):
      fan_in = tf.cast(fan_in, 'float32')
      fan_out = tf.cast(fan_out, 'float32')
      return tf.sqrt(1.0/(0.5*(fan_in+fan_out)))

    stddev = self.scale*glorot_stddev(fan_in, fan_out)
    reg_lambda = 0.5 * self.weight / (stddev**2.0)
    regularization = reg_lambda * tf.reduce_sum(tf.square(x))
    return regularization


class LaplaceRegularizer(PriorRegularizer):
  """Zero mean Laplace prior."""

  def __init__(self, stddev=0.1, **kwargs):
    """Initialize a Laplace prior.

    Args:
      stddev: Tensor, scalar, the standard deviation of the Laplace prior.
      **kwargs: keyword arguments passed to base class.
    """
    self.stddev = stddev
    super(LaplaceRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(LaplaceRegularizer, self).get_config()
    config.update({'stddev': self.stddev})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    reg = self(x)
    nelem = tf.cast(tf.size(x), x.dtype)
    logz = nelem * (-math.log(math.sqrt(2.0)*self.stddev))
    ll = -reg + self.weight*logz

    return ll

  def __call__(self, x):
    laplace_b = self.stddev / math.sqrt(2.0)
    return self.weight*tf.reduce_sum(tf.abs(x / laplace_b))


class CauchyRegularizer(PriorRegularizer):
  """Zero mean Cauchy prior."""

  def __init__(self, scale=1.0, **kwargs):
    """Initialize a Cauchy prior.

    The [standard Cauchy
    distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)
    contains a location and scale parameter.  Here we fix the location to zero.

    Args:
      scale: float, > 0.0, the scale parameter of a zero-mean Cauchy
        distribution.
      **kwargs: keyword arguments passed to base class.
    """
    self.scale = scale
    super(CauchyRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(CauchyRegularizer, self).get_config()
    config.update({'scale': self.scale})
    return config

  def logpdf(self, x):
    """Return the log pdf of the density times weight."""
    reg = self(x)
    nelem = tf.cast(tf.size(x), x.dtype)
    logz = nelem * (-math.log(math.pi*self.scale))
    ll = -reg + self.weight*logz

    return ll

  def __call__(self, x):
    nll = tf.reduce_sum(tf.math.log1p(tf.math.square(x / self.scale)))
    regularization = self.weight * nll

    return regularization


class SpikeAndSlabRegularizer(PriorRegularizer):
  """Normal Spike-and-Slab prior."""

  def __init__(self, scale_spike=0.001, scale_slab=0.4,
               mass_spike=0.5, **kwargs):
    """Initialize a spike-and-slab prior.

    Args:
      scale_spike: Tensor, scalar, >0.0, the standard deviation of the Normal
        spike component.
      scale_slab: Tensor, scalar, >0.0, the standard deviation of the Normal
        slab component.
      mass_spike: Tensor, scalar, >0.0, <1.0, the probability mass associated
        with the spike component.
      **kwargs: keyword arguments passed to base class.
    """
    self.scale_spike = scale_spike
    self.scale_slab = scale_slab
    self.mass_spike = mass_spike
    super(SpikeAndSlabRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(SpikeAndSlabRegularizer, self).get_config()
    config.update({'scale_spike': self.scale_spike,
                   'scale_slab': self.scale_slab,
                   'mass_spike': self.mass_spike})
    return config

  def logpdf(self, x):
    return -self(x)

  def __call__(self, w):
    pss = tfd.Mixture(
        cat=tfd.Categorical(
            probs=[self.mass_spike, 1.0-self.mass_spike]),
        components=[
            tfd.Normal(loc=0.0, scale=self.scale_spike),
            tfd.Normal(loc=0.0, scale=self.scale_slab)])
    logp = tf.reduce_sum(pss.log_prob(w))

    return -self.weight*logp


def inverse_gamma_shape_scale_from_mean_stddev(mean, stddev):
  """Compute inverse Gamma shape and scale from mean and standard deviation.

  Args:
    mean: Tensor, scalar, >0.0, the mean of the Inverse Gamma variate.
    stddev: Tensor, scalar, >0.0, the standard deviation of the Inverse Gamma
      variate.

  Returns:
    ig_shape: Tensor, scalar, >0.0, the inverse Gamma shape parameter.
    ig_scale: Tensor, scalar, >0.0, the inverse Gamma scale parameter.
  """
  cvar = (mean / stddev)**2.0
  ig_shape = cvar + 2.0
  ig_scale = mean*(cvar + 1.0)

  return ig_shape, ig_scale


class EmpiricalBayesNormal(PriorRegularizer):
  r"""Empirical Bayes Normal prior.

  #### Mathematical details

  We assume a hierarchical prior:
    1. v ~ InverseGamma(ig_shape, ig_scale)
    2. w_i ~ Normal(0, v),  i=1,..,n.

  We then define the empirical Bayes choice
  \\(
    v_* = \frac{ig_scale + (1/2) \sum_i w_i^2}{ig_shape + n/2 + 1},
  \\)
  and use the empirical Bayes prior \\(p(w) := \prod_i Normal(w_i; 0, v_*).\\)

  Note that this is not guaranteed to be a proper prior for n == 1.
  """

  def __init__(self, ig_shape=2.01, ig_scale=0.101, **kwargs):
    r"""Construct an empirical Bayes Normal regularizer.

    Args:
      ig_shape: Tensor, scalar, float, >0.0, the shape parameter of the inverse
        Gamma distribution.
      ig_scale: Tensor, scalar, float, >0.0, the scale parameter of the
        inverse Gamma distribution.
      **kwargs: keyword arguments passed to base class.
    """
    self.ig_shape = ig_shape
    self.ig_scale = ig_scale
    super(EmpiricalBayesNormal, self).__init__(**kwargs)

  def get_config(self):
    config = super(EmpiricalBayesNormal, self).get_config()
    config.update({'ig_shape': float(self.ig_shape),
                   'ig_scale': float(self.ig_scale)})
    return config

  @staticmethod
  def from_stddev(stddev, weight=1.0):
    """Create Empirical Bayes Normal prior with specified marginal mean stddev.

    The distribution is constructed as:
      1. v ~ InverseGamma(ig_shape, ig_scale)
         So that E[v] = Var[v] = stddev^2, and
      2. w_i ~ Normal(0, v),  i=1,...,n.

    Args:
      stddev: Tensor, scalar, float, >0.0, the marginal mean variance of the
        distribution.
      weight: Tensor, scalar, float, >=0.0, the negative log-likelihood is
        multiplied with this weight factor.  This can be used, for example,
        to ensure the log-prior is appropriately scaled with the total sample
        size in Bayesian neural networks: weight=1.0/total_sample_size.

    Returns:
      prior: EmpiricalBayesNormal prior with suitable parameters.
    """
    variance = stddev**2.0
    ig_shape, ig_scale = inverse_gamma_shape_scale_from_mean_stddev(
        variance, stddev)

    return EmpiricalBayesNormal(ig_shape, ig_scale, weight=weight)

  def __call__(self, w):
    w2sum = tf.reduce_sum(tf.square(w))
    n = tf.cast(tf.size(w), tf.float32)

    # Posterior variance estimate
    vhat = (self.ig_scale + 0.5*w2sum) / (self.ig_shape + 0.5*n + 1.0)
    vhatsqrt = tf.math.sqrt(vhat)

    logp = -0.5*n*tf.math.log(2.0*math.pi)
    logp += -0.5*n*tf.math.log(vhat)
    logp += -0.5*tf.reduce_sum(tf.square(w / vhatsqrt))

    return -self.weight * logp


class HeNormalEBRegularizer(PriorRegularizer):
  """He-inspired Normal Empirical Bayes regularization."""

  def __init__(self, scale=1.0, data_format='channels_last', **kwargs):
    """Initialize a He Normal empirical Bayes prior.

    The empirical Bayes regularization is constructed as:
      1. v ~ InverseGamma(ig_shape, ig_scale)
         Where ig_shape and ig_scale are chosen such that
         E[v] = Var[v] = (scale * he_stddev)^2, and
      2. w_i ~ Normal(0, v),  i=1,...,n.

    The regularization is then
      - sum_i log Normal(w_i; 0, vhat(w)),
    where
      vhat(w) := argmax_v p(v | w) under the model above.

    We can solve vhat(w) analytically because of conjugacy in the above model.

    For tf.size(w) >= 2 the induced prior p*(w) is normalizable.

    Args:
      scale: float, > 0.0, the He standard deviation is scaled with this factor.
      data_format: Image data format to use for convolution kernels.
        Note that all kernels in Keras are standardized on the
        `channels_last` ordering (even when inputs are set
        to `channels_first`).
      **kwargs: keyword arguments passed to base class.
    """
    self.scale = scale
    self.data_format = data_format
    super(HeNormalEBRegularizer, self).__init__(**kwargs)

  def get_config(self):
    config = super(HeNormalEBRegularizer, self).get_config()
    config.update({'scale': self.scale,
                   'data_format': self.data_format})
    return config

  def logpdf(self, w):
    return -self(w)

  def __call__(self, w):
    n = tf.size(w)
    n = tf.cast(n, tf.float32)

    fan_in, _ = _compute_fans(w.shape, self.data_format)
    stddev = self.scale*_he_stddev(fan_in)
    variance = stddev**2.0
    ig_shape, ig_scale = inverse_gamma_shape_scale_from_mean_stddev(
        variance, variance)

    w2sum = tf.reduce_sum(tf.square(w))

    # Posterior variance estimate
    vhat = (ig_scale + 0.5*w2sum) / (ig_shape + 0.5*n + 1.0)
    vhatsqrt = tf.math.sqrt(vhat)

    logp = -0.5*n*tf.math.log(2.0*math.pi)
    logp += -0.5*n*tf.math.log(vhat)
    logp += -0.5*tf.reduce_sum(tf.square(w / vhatsqrt))

    return -self.weight * logp
