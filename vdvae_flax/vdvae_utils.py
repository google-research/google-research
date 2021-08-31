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

"""Utils for building VDVAEs."""

import queue
import threading
from typing import Optional, Tuple, TypeVar

from absl import logging
import chex
import distrax
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from vdvae_flax import blocks as blocks_lib


def py_prefetch(iterable_function, buffer_size = 5):
  """Performs prefetching of elements from an iterable in a separate thread.

  Args:
    iterable_function: A python function that when called with no arguments
      returns an iterable. This is used to build a fresh iterable for each
      thread (crucial if working with tensorflow datasets because tf.graph
      objects are thread local).
    buffer_size (int): Number of elements to keep in the prefetch buffer.

  Yields:
    Prefetched elements from the original iterable.
  Raises:
    ValueError if the buffer_size <= 1.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  if buffer_size <= 1:
    raise ValueError('the buffer_size should be > 1')

  buffer = queue.Queue(maxsize=(buffer_size - 1))
  producer_error = []
  end = object()

  def producer():
    """Enques items from iterable on a given thread."""
    try:
      # Build a new iterable for each thread. This is crucial if working with
      # tensorflow datasets because tf.graph objects are thread local.
      iterable = iterable_function()
      for item in iterable:
        buffer.put(item)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error in producer thread for %s',
                        iterable_function.__name__)
      producer_error.append(e)
    finally:
      buffer.put(end)

  threading.Thread(target=producer, daemon=True).start()

  # Consumer.
  while True:
    value = buffer.get()
    if value is end:
      break
    yield value

  if producer_error:
    raise producer_error[0]


def allgather_and_reshape(x, axis_name='batch'):
  """Allgather and merge the newly inserted axis w/ the original batch axis."""
  y = jax.lax.all_gather(x, axis_name=axis_name)
  assert y.shape[1:] == x.shape
  return y.reshape(y.shape[0] * x.shape[0], *x.shape[1:])


def cast_to_float_center_and_normalize(inputs):
  """Casts inputs to float between -1 and 1.

  Args:
    inputs: a uint8 array.

  Returns:
    A float32 array with values between -1 and 1.
  """
  if inputs.dtype != jnp.uint8:
    raise ValueError('Expected inputs to be of type uint8 but got '
                     f'{inputs.dtype}')
  inputs = inputs.astype(jnp.float32) / 255.
  inputs = 2. * (inputs - 0.5)
  return inputs


BlockType = TypeVar('BlockType')


@flax.struct.dataclass
class QuantizedLogisticMixtureNetworkOutput:
  samples: chex.Array
  negative_log_likelihood: Optional[chex.Array]


class QuantizedLogisticMixtureNetwork(nn.Module):
  """Network computing a mixture of quantized logistic distributions."""
  num_mixtures: int
  low: int = 0
  high: int = 255
  num_output_channels: int = 3
  precision: Optional[jax.lax.Precision] = None
  #  name: str = 'logistic_mixture_sampler') -> None:

  @nn.compact
  def __call__(
      self,
      sample_rng,
      decoder_outputs,
      target_images = None,
  ):
    """Computes a mixture of logistic distributions from decoder output.

    Args:
      sample_rng: random key for sampling.
      decoder_outputs: the output of a Decoder, of shape [B, H, W, F] where B is
        the batch size, H=W are the height and width of the decoder output, and
        F its number of channels.
      target_images: if provided, the negative log likelihood of those images
        according to the computed distribution is reported in the output. The
        target images should have shape [B, H, W, C] where B, H and W have the
        same meaning and value as for decoder_output, and C is the number of
        output channels of the QuantizedLogisticMixtureNetwork. They should be
        of type uint8, i.e. with integer values between 0 and 255.

    Returns:
      A QuantizedLogisticMixtureNetwork containing a batch of samples and
      optionally the log likelihood of the target images if they were provided.
    """
    if len(decoder_outputs.shape) != 4:
      raise ValueError('Expected decoder_output to have rank 4 but got '
                       f'shape {decoder_outputs.shape}.')
    if target_images is not None:
      if len(target_images.shape) != 4:
        raise ValueError('Expected target_images to have rank 4 but got '
                         f'shape {target_images.shape}.')
      if target_images.shape[:3] != decoder_outputs.shape[:3]:
        raise ValueError('First three dimensions of target_images and '
                         'encoder_outputs should match but got '
                         f'{target_images.shape[:3]} and '
                         f'{decoder_outputs.shape[:3]}.')
      if target_images.shape[3] != self.num_output_channels:
        raise ValueError('Expected number of channels of target_images to '
                         'equal the number of output channels, but got '
                         f'{target_images.shape[3]} vs '
                         f'{self.num_output_channels}')

    num_output_channels = self.num_output_channels
    # For each channel, we have one parameter for mean, one for log-scale, and
    # one for mixture coefficients.
    num_params_per_mixture = 3 * num_output_channels + 1
    num_conv_output_channels = num_params_per_mixture * self.num_mixtures
    conv = blocks_lib.get_vdvae_convolution(
        num_conv_output_channels, (1, 1),
        weights_scale=0.,
        name='logistic_mixture_sampler_conv',
        precision=self.precision)
    outputs = conv(decoder_outputs)
    dmol_distribution = SpatiallyIndependentMixtureOfDiscreteLogistics(
        parameters=outputs,
        num_channels=num_output_channels,
        num_mixtures=self.num_mixtures,
        num_bits=int(np.log2(self.high - self.low + 1)))

    samples = dmol_distribution.sample(seed=sample_rng)

    if target_images is not None:
      negative_log_likelihood = -dmol_distribution.log_prob(target_images)
    else:
      negative_log_likelihood = None

    return QuantizedLogisticMixtureNetworkOutput(
        samples=samples, negative_log_likelihood=negative_log_likelihood)


class SpatiallyIndependentMixtureOfDiscreteLogistics(distrax.Distribution):
  """Spatially independent mixture of discrete logistic distributions."""

  def __init__(
      self,
      parameters,
      num_channels = 3,
      num_mixtures = 10,
      num_bits = 8,
  ):

    bs, height, width, feats = parameters.shape

    self._nb_channels = num_channels
    if feats == 3 * num_mixtures and self._nb_channels != 1:
      raise ValueError(f'If num_channels == 1, parameters should have '
                       f'3 * num_mixtures = {3 * num_mixtures} channels, '
                       f'but {feats} were provided.')
    elif feats == 10 * num_mixtures and self._nb_channels != 3:
      raise ValueError(f'If num_channels == 3, parameters should have '
                       f'10 * num_mixtures = {10 * num_mixtures} channels, '
                       f'but {feats} were provided.')
    elif feats not in [3 * num_mixtures, 10 * num_mixtures]:
      raise ValueError(f'Number of features in parameters should either be '
                       f'3 * num_mixtures: {3 * num_mixtures} or '
                       f'10 * num_mixtures: {10 * num_mixtures}, got {feats}.')

    self._num_mixtures = num_mixtures
    # [bs, height, width, num_mix]
    self._logits_probs = parameters[Ellipsis, :num_mixtures]
    # [bs, height, width, nb_channels, (2 or 3) * num_mix]
    sufficient_stats = jnp.reshape(parameters[Ellipsis, num_mixtures:],
                                   (bs, height, width, self._nb_channels, -1))
    # [bs, height, width, nb_channels, num_mix]
    self._means = sufficient_stats[Ellipsis, :num_mixtures]
    # [bs, height, width, nb_channels, num_mix]
    self._log_scales = sufficient_stats[Ellipsis, num_mixtures:2 * num_mixtures]
    # We only clip to a minimum, since we never exponentiate the scale directly
    # without a safeguard.
    self._log_scales = jnp.clip(self._log_scales, a_min=-7.)
    if self._nb_channels == 3:
      # [bs, height, width, nb_channels, num_mix]
      self._rgb_mix_coeffs = sufficient_stats[Ellipsis, 2 * num_mixtures:]
      self._rgb_mix_coeffs = jnp.tanh(self._rgb_mix_coeffs)

    self._max_val = 2.**num_bits - 1

  def log_prob(self, value):
    """Computes the log prob of the discrete mixture."""
    if value.ndim != 4:
      raise ValueError(f'Value is assumed to have 4 dims, got ' f'{value.ndim}')
    num_channels = value.shape[-1]

    if num_channels != self._nb_channels:
      raise ValueError(f'value should have {self._nb_channels} channels, '
                       f'got {num_channels}.')

    # [bs, height, width, num_channels]
    # bring to [-1, 1]
    value = value.astype(jnp.float32)
    value = value / 255.
    value = 2 * value - 1
    shape = value.shape
    value = value[Ellipsis, None]  # add a mixture axis

    # [bs, height, width, num_channels, num_mix]
    value = jnp.broadcast_to(value, shape + (self._num_mixtures,))

    if num_channels == 3:
      # RGB channels are produced autoregressively, thus the mean of the second
      # channel depends on the first channel of the value, and the mean of
      # the third depends on the 2 first channels.
      mean_r = self._means[:, :, :, 0, :]
      mean_g = (
          self._means[:, :, :, 1, :] +
          self._rgb_mix_coeffs[:, :, :, 0, :] * value[:, :, :, 0, :])
      mean_b = (
          self._means[:, :, :, 2, :] +
          self._rgb_mix_coeffs[:, :, :, 1, :] * value[:, :, :, 0, :] +
          self._rgb_mix_coeffs[:, :, :, 2, :] * value[:, :, :, 1, :])
      # [bs, height, width, num_channels, num_mix]
      means = jnp.stack([mean_r, mean_g, mean_b], axis=3)
    else:
      means = self._means

    centered = value - means

    # Each element of the mixture uses
    # sigmoid((x-m+1/max_val)/scale) - sigmoid((x-m-1/max_val)/scale)
    # as the probability for x, except for extremal xs, i.e. x = -1 and x = 1.
    # For x = 1, 1 - sigmoid((x - m - 1 / max_val) / scale) is taken,
    # For x = -1, sigmoid((x - m + 1 / max_val) / scale) is taken. One can
    # see that the final quantities sum to 1 when x ranges over -1, 1 with
    # steps 2. / max_val. See https://arxiv.org/pdf/1701.05517.pdf for a
    # more in depth discussion.
    inv_std = jnp.exp(-self._log_scales)
    plus_in = inv_std * (centered + 1. / self._max_val)
    cdf_plus = jax.nn.sigmoid(plus_in)
    minus_in = inv_std * (centered - 1. / self._max_val)
    cdf_min = jax.nn.sigmoid(minus_in)
    # For numerical stability, we only take x - softplus(x) when softplus(x)
    # is far from x, i.e. when x < 0. Otherwise we may end up computing
    # inaccurately small quantities.
    log_cdf_plus = jnp.where(plus_in > 0, -jax.nn.softplus(-plus_in),
                             plus_in - jax.nn.softplus(plus_in))
    log_one_minus_cdf_min = -jax.nn.softplus(minus_in)
    cdf_delta = cdf_plus - cdf_min

    # When the difference of CDFs get too small, numerical instabilities
    # can happen. To prevent this, a first order approximation of the
    # difference is taken in such cases.
    mid_in = inv_std * centered
    log_pdf_mid = mid_in - self._log_scales - 2 * jax.nn.softplus(mid_in)
    log_prob_mid_safe = jnp.where(cdf_delta > 1e-5,
                                  jnp.log(jnp.clip(cdf_delta, a_min=1e-10)),
                                  log_pdf_mid - jnp.log(self._max_val / 2))

    # Discrete values go from -1 to 1 with a 2 / max_val step.
    # To confidently catch all -1 values, we compare to the midpoint
    # between -1 and the closest discrete value -1 + 2 / max_val, i.e.
    # 1. / self.max_val - 1. We do the same for the max value.
    log_probs = jnp.where(
        value < 1. / self._max_val - 1.,
        log_cdf_plus,
        jnp.where(
            value > 1. - 1. / self._max_val,
            log_one_minus_cdf_min,
            log_prob_mid_safe,
        ),
    )

    # Compute the mixture log probs, i.e. sum log probs
    # along the channel axis, then add the mixture log probs
    # (each pixel as its own mixture, but not each channel)
    # [bs, height, width, num_mix]
    log_probs = (
        jnp.sum(log_probs, axis=3) +
        jax.nn.log_softmax(self._logits_probs, axis=-1))
    # [bs, height, width]
    log_probs = jax.nn.logsumexp(log_probs, axis=-1)
    log_probs = jnp.sum(log_probs, axis=(-1, -2))
    return log_probs

  @property
  def event_shape(self):
    sample_shape = self._means.shape[:-1]
    return sample_shape[1:]

  def _sample_n(self,
                key,
                n,
                mixture_temp = 1.,
                color_temp = 1.):
    """Sample with a temperature parameter."""
    cat_rng, unif_rng = jax.random.split(key, 2)
    # Add a sample axis and flatten
    bs, height, width, nb_channels, _ = self._means.shape

    means = jnp.tile(self._means, [n] + [1 for _ in self._means.shape[1:]])

    log_scales = jnp.tile(self._log_scales,
                          [n] + [1 for _ in self._log_scales.shape[1:]])

    logit_probs = jnp.tile(self._logits_probs,
                           [n] + [1 for _ in self._logits_probs.shape[1:]])

    if self._nb_channels == 3:
      rgb_mix_coeffs = jnp.tile(self._rgb_mix_coeffs, [n] +
                                [1 for _ in self._rgb_mix_coeffs.shape[1:]])

    # NVIDIA implementation uses a Gumbel parameterization.
    # JAX directly provides a categorical sampling function.
    # [bs, height, width]
    mixture_idx = jax.random.categorical(
        key=cat_rng,
        logits=logit_probs / mixture_temp,
        axis=-1,
    )
    # [bs, height, width, num_mix]
    mixture_idx_1h = jax.nn.one_hot(mixture_idx, self._num_mixtures)
    # [bs, height, width, 1, num_mix]
    mixture_idx_1h = mixture_idx_1h[:, :, :, None, :]

    # Get parameters of the corresponding mixture
    means = jnp.sum(means * mixture_idx_1h, axis=-1)
    log_scales = jnp.sum(log_scales * mixture_idx_1h, axis=-1)
    if self._nb_channels == 3:
      coeffs = jnp.sum(rgb_mix_coeffs * mixture_idx_1h, axis=-1)

    u = jax.random.uniform(
        key=unif_rng,
        shape=means.shape,
        dtype=jnp.float32,
        minval=1e-5,
        maxval=1 - 1e-5,
    )
    x = means + jnp.exp(log_scales) * color_temp * (jnp.log(u) - jnp.log(1 - u))

    # Autoregressive sampling on channels
    if self._nb_channels == 3:
      x0 = jnp.clip(x[Ellipsis, 0], -1., 1.)
      x1 = jnp.clip(x[Ellipsis, 1] + coeffs[Ellipsis, 0] * x0, -1., 1.)
      x2 = jnp.clip(
          x[Ellipsis, 2] + coeffs[Ellipsis, 1] * x0 + coeffs[Ellipsis, 2] * x1,
          -1.,
          1.,
      )

      x = jnp.stack([x0, x1, x2], axis=-1)
    else:
      x = jnp.clip(x, -1., 1.)
    x = (x / 2. + .5) * 255
    x = x.astype(jnp.uint8)

    # Reshape sample axis
    return jnp.reshape(x, (n, bs, height, width, nb_channels))
