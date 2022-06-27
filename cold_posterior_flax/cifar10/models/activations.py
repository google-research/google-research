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

"""Experimental Activation Functions."""
from flax.deprecated import nn
from flax.deprecated.nn import initializers
import jax
import jax.numpy as jnp
import numpy as onp

from cold_posterior_flax.cifar10.models import new_initializers


class Capped(nn.Module):
  """Capped activation unit that converges roughly to standard normal in sequential application.
  """

  def apply(self, x):
    return nn.sigmoid(x) * nn.sigmoid(-x) * nn.tanh(x) * (1 / 0.15)


class TLU(nn.Module):
  """TLU activation following [1].

  [1] Singh, S., & Krishnan, S. (2020). Filter response normalization layer:
  Eliminating batch dependence in the training of deep neural networks.
  https://doi.org/10.1109/cvpr42600.2020.01125.
  """

  def apply(self, x):
    tau = self.param('tau', x.shape[-1:], initializers.zeros)
    return jnp.maximum(jnp.broadcast_to(tau, x.shape), x)


class BiasReluNorm(nn.Module):
  """ReLU activation normalizing for a learned scale and offset of input."""

  def apply(self,
            inputs,
            features,
            bias=True,
            scale=False,
            dtype=jnp.float32,
            precision=None,
            bias_init=initializers.zeros,
            scale_init=None,
            softplus=True):
    if scale_init is None:
      if softplus:
        scale_init = new_initializers.init_softplus_ones
      else:
        scale_init = initializers.ones
    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
    else:
      bias = 0.

    if scale:
      scale = self.param('scale', (features,), scale_init)
      scale = jnp.asarray(scale, dtype)
    else:
      scale = float(new_initializers.inv_softplus(1.0)) if softplus else 1.0

    if softplus:
      scale = nn.softplus(scale)

    y = inputs
    y *= scale
    y = y + bias
    relu_threshold = 0.0
    y = jnp.maximum(relu_threshold, y)

    # Normalize y analytically.
    mean = bias
    std = scale
    var = std**2
    # Kaiming initialized weights + bias + TLU
    # = mixture of delta peak + left-truncated gaussian
    # https://en.wikipedia.org/wiki/Mixture_distribution#Moments
    # https://en.wikipedia.org/wiki/Truncated_normal_distribution#One_sided_truncation_(of_lower_tail)[4]
    norm = jax.scipy.stats.norm
    t = (relu_threshold - mean) / std

    # If the distribution lies 4 stdev below the threshold, cap at t=4.
    t = jnp.minimum(4, t)
    z = 1 - norm.cdf(t)

    new_mean_non_cut = mean + (std * norm.pdf(t)) / z
    new_var_non_cut = (var) * (1 + t * norm.pdf(t) / z - (norm.pdf(t) / z)**2)

    # Psi function.
    # Compute mixture mean.
    new_mean = new_mean_non_cut * z + relu_threshold * norm.cdf(t)
    # Compute mixture variance.
    new_var = z * (new_var_non_cut + new_mean_non_cut**2 - new_mean**2)
    new_var += (1 - z) * (0 + relu_threshold**2 - new_mean**2)
    new_std = jnp.sqrt(new_var + 1e-8)
    new_std = jnp.maximum(0.01, new_std)

    # Normalize y.
    y_norm = y
    y_norm -= new_mean
    y_norm /= new_std
    return y_norm


class BiasSELUNorm(nn.Module):
  """SELU activation normalizing for a learned scale and offset of input.

  This is based in part on the derivations of the truncated lognormal by:
  [1] Zaninetti, L. (2017, August 21). A left and right truncated lognormal
  distribution for the stars. arXiv [astro-ph.SR].
  http://arxiv.org/abs/1708.06159.

  For original SELU, see:
  [2] `Self-Normalizing Neural Networks
  <https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf>`_.

  Full derivation to follow in future write-up.
  """

  def apply(self,
            inputs,
            features,
            bias=True,
            scale=True,
            dtype=jnp.float32,
            precision=None,
            bias_init=initializers.zeros,
            scale_init=None,
            softplus=True,
            norm_grad_block=False):
    if scale_init is None:
      if softplus:
        scale_init = new_initializers.init_softplus_ones
      else:
        scale_init = initializers.ones

    norm = jax.scipy.stats.norm
    erf = jax.scipy.special.erf  # Error function.

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
    else:
      bias = 0.

    if scale:
      scale = self.param('scale', (features,), scale_init)
      scale = jnp.asarray(scale, dtype)
    else:
      scale = float(new_initializers.inv_softplus(1.0)) if softplus else 1.0

    if softplus:
      scale = nn.softplus(scale)

    pre = inputs
    pre *= scale
    pre = pre + bias
    y = jax.nn.selu(pre)

    # Compute moments based in learned scale/bias.
    if norm_grad_block:
      scale = jax.lax.stop_gradient(scale)
      bias = jax.lax.stop_gradient(bias)
    std = scale
    mean = bias
    var = std**2

    # SELU magic numbers from SeLU paper [2] and jax.nn.selu.
    alpha = 1.6732632423543772848170429916717
    selu_scale = 1.0507009873554804934193349852946
    selu_threshold = 0

    # Compute moments of left and right side of split gaussian for x <=0 & x > 0
    t = (selu_threshold - mean) / std
    # If the distribution lies 4 stdev below the threshold, cap at t=4.
    t = jnp.maximum(-3, jnp.minimum(3, t))
    z = 1 - norm.cdf(t)
    new_mean_right = (mean + (std * norm.pdf(t)) / z)
    new_var_right = (var) * (1 + t * norm.pdf(t) / z - (norm.pdf(t) / z)**2)

    l_scale = jnp.exp(mean)  # Log normal scale parameter = exp(mean)
    log_scale = mean
    min_log = -5

    # Compute truncated log normal statistics for left part of SELU.
    # TODO(basv): improve numerical errors with np.exp1m?
    a1 = .5 * (1. / (std + 1e-5)) * jnp.sqrt(2) * (-var + min_log - log_scale)
    a2 = .5 * (1. / (std + 1e-5)) * jnp.sqrt(2) * (
        var + log_scale - selu_threshold)
    a3 = .5 * (1. / (std + 1e-5)) * jnp.sqrt(2) * (min_log - log_scale)
    a4 = .5 * (1. / (std + 1e-5)) * jnp.sqrt(2) * (-selu_threshold + log_scale)
    a5 = .5 * (1. /
               (std + 1e-5)) * jnp.sqrt(2) * (-2 * var + min_log - log_scale)
    a6 = .5 * (1. / (std + 1e-5)) * jnp.sqrt(2) * (2 * var + log_scale -
                                                   selu_threshold)
    e_a1 = erf(a1)
    e_a2 = erf(a2)
    e_a3 = erf(a3)
    e_a4 = erf(a4)
    e_a5 = erf(a5)
    e_a6 = erf(a6)
    exp_var = jnp.exp(var)

    # Equation 18 [1].
    trunc_lognorm_mean = (l_scale * jnp.exp(.5 * var) * (e_a1 + e_a2)) / (
        e_a3 + e_a4 + 1e-5)
    trunc_lognorm_mean_m1 = trunc_lognorm_mean - 1  # selu uses e^x - 1
    # Equation 20 [1].
    n = exp_var * (e_a3 * e_a5 * exp_var + e_a3 * e_a6 * exp_var +
                   e_a4 * e_a5 * exp_var + e_a4 * e_a6 * exp_var - e_a1**2 -
                   2 * e_a1 * e_a2 - e_a2**2) * l_scale**2
    # Equation 19 [1].
    trunc_lognorm_var = n / ((e_a3 + e_a4 + 1e-5)**2)

    selu_mean = alpha * trunc_lognorm_mean_m1
    selu_var = alpha**2 * trunc_lognorm_var

    # Compute mixture mean multiplied by selu_scale.
    new_mean = (selu_mean * (1 - z) + new_mean_right * z)

    # Compute mixture variance.
    new_var = z * (new_var_right + new_mean_right**2 - new_mean**2)
    new_var += (1 - z) * (selu_var + selu_mean**2 - new_mean**2)
    new_mean = selu_scale * new_mean
    new_std = jnp.sqrt(new_var + 1e-5) * selu_scale
    new_var *= selu_scale**2

    if norm_grad_block:
      new_mean = jax.lax.stop_gradient(new_mean)
      new_std = jax.lax.stop_gradient(new_std)

    new_std = jnp.maximum(1e-3, new_std)

    # Normalize y.
    y_norm = y
    y_norm -= new_mean
    y_norm /= new_std
    return y_norm


class SELUNormReBias(BiasSELUNorm):
  """SELU normalized activation with post-activation bias."""

  def apply(
      self,
      inputs,
      features,
      bias=True,
      dtype=jnp.float32,
      precision=None,
      bias_init=initializers.zeros,
  ):
    y_norm = super(SELUNormReBias, self).apply(inputs, features, bias, dtype,
                                               precision, bias_init)

    rebias = self.param('rebias', (features,), bias_init)
    rebias = jnp.asarray(rebias, dtype)

    y_norm += rebias
    return y_norm


class Swish(nn.Module):
  """Parametrized Swish as stated in https://arxiv.org/pdf/2004.02967.pdf."""

  def apply(self, x):
    tau = self.param('tau', x.shape[-1:], nn.initializers.ones)
    return x * nn.sigmoid(x * tau)


class TLUM(nn.Module):
  """TLU with mirror output."""

  def apply(self, x):
    tau = self.param('tau', x.shape[-1:], nn.initializers.zeros)
    return jnp.concatenate([
        jnp.maximum(jnp.broadcast_to(tau, x.shape), x),
        jnp.minimum(jnp.broadcast_to(tau, x.shape), x)
    ],
                           axis=-1)


class TLDU(nn.Module):
  """Two sided rectified learned linear unit."""

  def apply(self, x):
    tau_down = self.param('tau_down', x.shape[-1:], nn.initializers.zeros)
    tau_up = self.param('tau_up', x.shape[-1:], nn.initializers.ones)
    x = jnp.maximum(jnp.broadcast_to(tau_down, x.shape), x)
    x = jnp.minimum(jnp.broadcast_to(tau_up, x.shape), x)
    return x


def negative_ones(_, shape, dtype=jnp.float32):
  return -1 * jnp.ones(shape, dtype)


class TLDUZ(nn.Module):
  """Two sided rectified learned linear unit with zero mean."""

  def apply(self, x):
    tau_down = self.param('tau_down', x.shape[-1:], negative_ones)
    tau_up = self.param('tau_up', x.shape[-1:], nn.initializers.ones)
    x = jnp.maximum(jnp.broadcast_to(tau_down, x.shape), x)
    x = jnp.minimum(jnp.broadcast_to(tau_up, x.shape), x)
    return x


class relu_norm(nn.Module):  # pylint: disable=invalid-name
  """Relu + renormalized assuming gaussian inputs."""

  def apply(self, x):
    a = jnp.maximum(jnp.broadcast_to(0., x.shape), x)
    # Magic numbers for mean,std of relu(N(0,1))
    mean = 0.3989059089270267
    std = 0.5837662801636493
    return (a - mean) / std


class relu_unitvar(nn.Module):  # pylint: disable=invalid-name
  """ReLU with variance set to 1."""

  def apply(self, x):
    a = jnp.maximum(jnp.broadcast_to(0., x.shape), x)
    # Magic numbers for mean,std of relu(N(0,1))
    return a * onp.sqrt(2)
