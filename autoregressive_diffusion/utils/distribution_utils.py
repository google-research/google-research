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

"""Contains util functions that are needed often.

This contains often used transformations / functions that are very general, but
complicated enough that they warrant an implementation in this file.
"""
import functools
import jax
import jax.numpy as jnp
from autoregressive_diffusion.utils import util_fns


def sample_categorical_and_log_prob(key, logits):
  # If logits already normalized, operation does nothing as is desired.
  log_p = jax.nn.log_softmax(logits, axis=-1)

  category = sample_categorical(key, logits)

  category_onehot = util_fns.onehot(category, num_classes=logits.shape[-1])

  log_pcategory = (log_p * category_onehot).sum(axis=-1)
  return category, log_pcategory


def sample_categorical(key, logits):
  minval = jnp.finfo(logits.dtype).tiny
  unif = jax.random.uniform(key, logits.shape, minval=minval, maxval=1.0)
  gumbel = -jnp.log(-jnp.log(unif) + minval)
  category = jnp.argmax(logits + gumbel, -1)
  return category


@functools.partial(jax.jit, static_argnums=(2,))
def logits_discretized_mix_logistic_rgb(x, params, num_classes):
  """Computes the logits for discretized mix logistic for 3 channel images.

  The input x is used because the function is autoregressive over RGB channels.
  Args:
    x: input image, has values {0, ..., num_classes-1}
    params: parameters for the distribution.
    num_classes: the number of classes in x.

  Returns:
    logits for the distribution.

  """
  assert x.dtype == jnp.int32
  x = x / float(num_classes - 1) * 2 - 1
  assert len(x.shape) == 4
  batchsize, height, width, channels = x.shape
  assert channels == 3
  assert len(params.shape) == 4

  # 10 = [1 (mixtures) + 3 (means) + 3 (log_scales) + 3 (coeffs)].
  nr_mix = params.shape[3] // 10
  pi_logits = params[:, :, :, :nr_mix]   # mixture coefficients.
  remaining_params = params[:, :, :, nr_mix:].reshape(*x.shape, nr_mix * 3)
  means = remaining_params[:, :, :, :, :nr_mix]
  pre_act_scales = remaining_params[:, :, :, :, nr_mix:2*nr_mix]
  # log_scales = jnp.clip(log_scales, a_min=-7.)

  # Coeffs are used to autoregressively model the _mean_ parameter of the
  # distribution using the Red (variable x0) and Green (variable x1) channels.
  # For Green (x1) and Blue (variable x2).
  # There are 3 coeff channels, one for (x1 | x0) and two for (x2 | x0, x1)
  coeffs = jax.nn.tanh(remaining_params[:, :, :, :, 2*nr_mix:])

  x = x.reshape(*x.shape, 1)
  x = x.repeat(nr_mix, axis=-1)

  m1 = means[:, :, :, 0:1, :]
  m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).reshape(batchsize, height, width, 1, nr_mix)

  m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
        coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).reshape(
            batchsize, height, width, 1, nr_mix)

  means = jnp.concatenate([m1, m2, m3], axis=3)

  # In range [0, 1]
  bin_mid = jnp.arange(0, num_classes, dtype=jnp.float32) / (num_classes - 1)
  # Renormalize to [-1, +1]
  bin_mid = bin_mid * 2 - 1
  # Expand dims for bin locations
  data_axes = tuple(range(5))  # batch / height / width / channels / mixtures
  bin_mid = jnp.expand_dims(bin_mid, axis=data_axes)

  gridsize = 1. / (num_classes - 1.)

  # Expand dims for params
  pre_act_scales = jnp.expand_dims(pre_act_scales, axis=-1)
  means = jnp.expand_dims(means, axis=-1)

  bin_centered = bin_mid - means
  inv_stdv = jax.nn.softplus(pre_act_scales)
  plus_in = inv_stdv * (bin_centered + gridsize)
  cdf_plus = jax.nn.sigmoid(plus_in)
  min_in = inv_stdv * (bin_centered - gridsize)
  cdf_min = jax.nn.sigmoid(min_in)

  # log probability for rightmost grid value, tail of distribution
  log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
  # log probability for leftmost grid value, tail of distribution
  log_one_minus_cdf_min = -jax.nn.softplus(min_in)
  cdf_delta = cdf_plus - cdf_min  # probability for all other cases

  is_last_bin = (bin_mid > 0.9999).astype(jnp.float32)
  is_first_bin = (bin_mid < -0.9999).astype(jnp.float32)

  log_cdf_delta = jnp.log(jnp.clip(cdf_delta, a_min=1e-12))

  # Here the tails of the distribution are assigned, if applicable.
  log_probs = is_last_bin * log_one_minus_cdf_min + (
      1. - is_last_bin) * log_cdf_delta

  log_probs = is_first_bin * log_cdf_plus + (
      1. - is_first_bin) * log_probs

  assert log_probs.shape == (batchsize, height, width, channels, nr_mix,
                             num_classes)

  log_pi = jax.nn.log_softmax(pi_logits, axis=-1)
  log_pi = jnp.expand_dims(log_pi, axis=(3, 5))

  log_probs = jax.nn.logsumexp(log_probs + log_pi, axis=4)
  assert log_probs.shape == (batchsize, height, width, channels, num_classes)

  return log_probs


def discretized_mix_logistic_rgb(x, params, gridsize):
  """Computes discretized mix logistic for 3 channel images."""
  assert len(x.shape) == 4
  batchsize, height, width, channels = x.shape
  assert channels == 3
  assert len(params.shape) == 4

  # 10 = [1 (mixtures) + 3 (means) + 3 (log_scales) + 3 (coeffs)].
  nr_mix = params.shape[3] // 10
  pi_logits = params[:, :, :, :nr_mix]   # mixture coefficients.
  remaining_params = params[:, :, :, nr_mix:].reshape(*x.shape, nr_mix * 3)
  means = remaining_params[:, :, :, :, :nr_mix]
  pre_act_scale = remaining_params[:, :, :, :, nr_mix:2*nr_mix]

  # Coeffs are used to autoregressively model the _mean_ parameter of the
  # distribution using the Red (variable x0) and Green (variable x1) channels.
  # For Green (x1) and Blue (variable x2).
  # There are 3 coeff channels, one for (x1 | x0) and two for (x2 | x0, x1)
  coeffs = jax.nn.tanh(remaining_params[:, :, :, :, 2*nr_mix:])

  x = x.reshape(*x.shape, 1)
  x = x.repeat(nr_mix, axis=-1)

  m1 = means[:, :, :, 0:1, :]
  m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).reshape(batchsize, height, width, 1, nr_mix)

  m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
        coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).reshape(
            batchsize, height, width, 1, nr_mix)

  means = jnp.concatenate([m1, m2, m3], axis=3)

  centered_x = x - means
  inv_stdv = jax.nn.softplus(pre_act_scale)
  plus_in = inv_stdv * (centered_x + gridsize)
  cdf_plus = jax.nn.sigmoid(plus_in)
  min_in = inv_stdv * (centered_x - gridsize)
  cdf_min = jax.nn.sigmoid(min_in)

  # log probability for rightmost grid value, tail of distribution
  log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
  # log probability for leftmost grid value, tail of distribution
  log_one_minus_cdf_min = -jax.nn.softplus(min_in)
  cdf_delta = cdf_plus - cdf_min  # probability for all other cases

  is_last_bin = (x > 0.9999).astype(jnp.float32)
  is_first_bin = (x < -0.9999).astype(jnp.float32)

  log_cdf_delta = jnp.log(jnp.clip(cdf_delta, a_min=1e-12))

  # Here the tails of the distribution are assigned, if applicable.
  log_probs = is_last_bin * log_one_minus_cdf_min + (
      1. - is_last_bin) * log_cdf_delta

  log_probs = is_first_bin * log_cdf_plus + (
      1. - is_first_bin) * log_probs

  assert log_probs.shape == (batchsize, height, width, channels, nr_mix)

  log_pi = jax.nn.log_softmax(pi_logits, axis=-1)

  log_probs = jax.nn.logsumexp(log_probs + log_pi[Ellipsis, None, :], axis=-1)

  assert log_probs.shape == (batchsize, height, width, channels)

  log_probs = log_probs.sum(-1)

  assert log_probs.shape == (batchsize, height, width)

  return log_probs


def sample_from_discretized_mix_logistic_rgb(rng, params, nr_mix):
  """Sample from discretized mix logistic distribution."""
  xshape = params.shape[:-1] + (3,)
  batchsize, height, width, _ = xshape

  # unpack parameters
  pi_logits = params[:, :, :, :nr_mix]
  remaining_params = params[:, :, :, nr_mix:].reshape(*xshape, nr_mix * 3)

  # sample mixture indicator from softmax
  rng1, rng2 = jax.random.split(rng)
  mixture_idcs = sample_categorical(rng1, pi_logits)

  onehot_values = util_fns.onehot(mixture_idcs, nr_mix)

  assert onehot_values.shape == (batchsize, height, width, nr_mix)

  selection = onehot_values.reshape(xshape[:-1] + (1, nr_mix))

  # select logistic parameters
  means = jnp.sum(remaining_params[:, :, :, :, :nr_mix] * selection, axis=4)
  pre_act_scales = jnp.sum(
      remaining_params[:, :, :, :, nr_mix:2 * nr_mix] * selection, axis=4)

  coeffs = jnp.sum(jax.nn.tanh(
      remaining_params[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * selection, axis=4)

  u = jax.random.uniform(rng2, means.shape, minval=1e-5, maxval=1. - 1e-5)

  standard_logistic = jnp.log(u) - jnp.log(1. - u)
  scale = 1. / jax.nn.softplus(pre_act_scales)
  x = means + scale * standard_logistic

  x0 = jnp.clip(x[:, :, :, 0], a_min=-1., a_max=1.)
  # TODO(emielh) although this is typically how it is implemented, technically
  # one should first round x0 to the grid before using it. It does not matter
  # too much since it is only used linearly.
  x1 = jnp.clip(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, a_min=-1., a_max=1.)
  x2 = jnp.clip(
      x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1,
      a_min=-1., a_max=1.)

  sample_x = jnp.concatenate(
      [x0[:, :, :, None], x1[:, :, :, None], x2[:, :, :, None]], axis=3)

  return sample_x


def discretized_mix_logistic(x, params, gridsize):
  """Computes discretized mix logistic for images. Channels are independent."""
  shape = x.shape
  assert len(shape) >= 3
  batchsize, channels = shape[0], shape[-1]
  spatial_dims = shape[1:-1]
  assert len(params.shape) == len(shape)

  # The mixtures are shared over channels, and params contains a mean and scale
  # for every channel. Hence there are 1 + 2 * channels parameters
  # _per_ mixture.
  nr_mix = params.shape[-1] // (1 + channels * 2)
  pi_logits = params[Ellipsis, :nr_mix]  # Mixture coefficients.
  remaining_params = params[Ellipsis, nr_mix:].reshape(*x.shape, 2 * nr_mix)
  means = remaining_params[Ellipsis, :nr_mix]

  log_scales = remaining_params[Ellipsis, nr_mix:2*nr_mix]
  means = means + jnp.reshape(
      jnp.linspace(-1. + 1 / nr_mix, 1 - 1 / nr_mix, nr_mix),
      (1,) * x.ndim + (nr_mix,))
  log_scales = log_scales - jnp.log(nr_mix)
  log_scales = jnp.clip(log_scales, a_min=-7.)

  x = x.reshape(*x.shape, 1)
  x = x.repeat(nr_mix, axis=-1)

  centered_x = x - means
  inv_stdv = jnp.exp(-log_scales)
  plus_in = inv_stdv * (centered_x + gridsize)
  cdf_plus = jax.nn.sigmoid(plus_in)
  min_in = inv_stdv * (centered_x - gridsize)
  cdf_min = jax.nn.sigmoid(min_in)

  # Log probability for leftmost grid value.
  log_cdf_plus = plus_in - jax.nn.softplus(plus_in)
  # Log probability for leftmost grid value.
  log_one_minus_cdf_min = -jax.nn.softplus(min_in)
  cdf_delta = cdf_plus - cdf_min  # probability for all other cases

  is_last_bin = (x > 0.9999).astype(jnp.float32)
  is_first_bin = (x < -0.9999).astype(jnp.float32)

  log_cdf_delta = jnp.log(jnp.clip(cdf_delta, a_min=1e-12))

  log_probs = is_last_bin * log_one_minus_cdf_min + (
      1. - is_last_bin) * log_cdf_delta

  log_probs = is_first_bin * log_cdf_plus + (
      1. - is_first_bin) * log_probs

  assert log_probs.shape == ((batchsize,) + spatial_dims + (channels, nr_mix))

  log_pi = jax.nn.log_softmax(
      pi_logits, axis=-1).reshape(batchsize, *spatial_dims, 1, nr_mix)

  log_probs = jax.nn.logsumexp(log_probs + log_pi, axis=-1)
  assert log_probs.shape == (batchsize,) + spatial_dims + (channels,)

  return log_probs


def sample_from_discretized_mix_logistic(rng, params, nr_mix):
  """Sample from discretized mix logistic distribution, channel-independent."""
  channels = (params.shape[-1] // nr_mix - 1) // 2
  xshape = params.shape[:-1] + (channels,)
  batchsize = xshape[0]
  spatial_dims = xshape[1:-1]

  # Unpack parameters.
  pi_logits = params[Ellipsis, :nr_mix]
  remaining_params = params[Ellipsis, nr_mix:].reshape(*xshape, nr_mix * 2)

  means = remaining_params[Ellipsis, :nr_mix]
  means = means + jnp.reshape(
      jnp.linspace(-1. + 1 / nr_mix, 1 - 1 / nr_mix, nr_mix),
      (1,) * len(xshape) + (nr_mix,))

  log_scales = remaining_params[Ellipsis, nr_mix:2 * nr_mix]
  log_scales = log_scales - jnp.log(nr_mix)

  # Sample mixture indicator from softmax.
  rng1, rng2 = jax.random.split(rng)
  mixture_idcs = sample_categorical(rng1, pi_logits)

  onehot_values = util_fns.onehot(mixture_idcs, nr_mix)

  assert onehot_values.shape == (batchsize,) + spatial_dims + (nr_mix,)

  selection = onehot_values.reshape(xshape[:-1] + (1, nr_mix))

  # Select logistic parameters.
  means = jnp.sum(means * selection, axis=-1)
  log_scales = jnp.sum(log_scales * selection, axis=-1)
  log_scales = jnp.clip(log_scales, a_min=-7.)

  u = jax.random.uniform(rng2, means.shape, minval=1e-5, maxval=1. - 1e-5)

  standard_logistic = jnp.log(u) - jnp.log(1. - u)

  sample_x = means + jnp.exp(log_scales) * standard_logistic
  sample_x = jnp.clip(sample_x, a_min=-1., a_max=1.)

  return sample_x


@jax.jit
def get_safe_probs(logits, eps=2e-5):
  """Extract `safe` probs from logits by adding a small eps for each class.

  This function is used to make lossless compression work in a practical
  finite precision setting. We need actual probabilities (non-log) with
  non-zero probability.

  Args:
    logits: The logits of the categorical distribution.
    eps: The uniform probability that each class at least gets.

  Returns:
    The regularized probabilities, to be used for entropy coding.
  """
  num_classes = logits.shape[-1]
  discount = 1. - eps * num_classes
  probs = discount * jax.nn.softmax(logits, -1) + eps
  return probs
