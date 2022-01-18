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

"""Contains distribution to model the output of the ARM."""
import functools

import flax
import jax
import jax.numpy as jnp
import numpy as np

from autoregressive_diffusion.model.autoregressive_diffusion import ardm_utils
from autoregressive_diffusion.utils import distribution_utils
from autoregressive_diffusion.utils import util_fns


class SoftmaxCategorical(flax.linen.Module):
  """Class collection of softmax categorical functions."""
  n_channels: int
  n_classes: int

  def get_required_num_outputs(self):
    return self.n_channels * self.n_classes

  def log_prob(self, x, dist_params):
    """Computes log prob."""
    assert x.dtype == jnp.int32
    assert x.shape[:-1] == dist_params.shape[:-1], (
        f' for {x.shape} and {dist_params.shape}')
    assert dist_params.shape[-1] == self.n_classes * self.n_channels

    logits = dist_params.reshape(*x.shape, self.n_classes)

    x_onehot = util_fns.onehot(x, num_classes=self.n_classes)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_probs = jnp.sum(x_onehot * log_probs, axis=-1)

    return log_probs

  def sample(self, key, params):
    logits = params.reshape(*(params.shape[:-1]),
                            self.n_channels, self.n_classes)
    sample_x = distribution_utils.sample_categorical(key, logits)
    return sample_x

  @functools.partial(jax.jit, static_argnums=(0,))
  def get_probs_coding(self, params, selection, x=None):
    """Retrieves the probabilities used for entropy coding.

    Args:
      params: The distribution parameters.
      selection: A selection mask specifying the variable to encode.
      x: Optional x that should also be masked in a similar way.

    Returns:
      The probabilities to use, and optionally the selected x for encoding.
    """
    logits = params.reshape(*(params.shape[:-1]),
                            self.n_channels, self.n_classes)
    assert logits.shape[:-1] == selection.shape
    reduce_axes = tuple(range(1, len(selection.shape)))

    logits = jnp.sum(logits * selection[Ellipsis, None], axis=reduce_axes,
                     keepdims=True)

    probs = distribution_utils.get_safe_probs(logits)

    if x is not None:
      x = jnp.sum(x * selection, axis=reduce_axes, keepdims=True)
      return probs, x
    else:
      return probs

  def encode(self, streams, x, params, selection):
    """Encodes the variable x at location selection using the distribution.

    Args:
      streams: Batch of bitstream objects to encode to.
      x: Variable to encode.
      params: Distribution parameters for x.
      selection: Mask signifying which index to encode.

    Returns:
      The bitstream objects.
    """
    probs, x = self.get_probs_coding(params, selection, x=x)
    streams = ardm_utils.encode(streams, x, probs)
    return streams

  def decode(self, streams, params, selection):
    """Decodes a variable at location selection using the distribution.

    Args:
      streams: Batch of bitstream objects to decode from.
      params: Distribution parameters for x.
      selection: Mask signifying which index to decode.

    Returns:
      A tuple with the decoded variable, and the bitstream objects.
    """
    probs = self.get_probs_coding(params, selection)
    x_decoded, streams = ardm_utils.decode(streams, probs)
    return x_decoded, streams


class DiscretizedMixLogistic(flax.linen.Module):
  """Collection of discretized logistic functions (channel-independent)."""
  n_channels: int
  n_classes: int
  n_mixtures: int

  def get_required_num_outputs(self):
    """Return required number of channels in dist_params.

    The mixture parameters are shared over channels, the means/scales of the
    channels are not shared over channels. Hence we need 1 + (1 + 1) * channels
    parameters per mixture.

    Returns:
      num_required_outputs: the number of output channels that are needed for
        the distribution parameters.
    """
    return (1 + 2 * self.n_channels) * self.n_mixtures

  def log_prob(self, x, dist_params):
    """Computes log prob."""
    assert x.dtype == jnp.int32
    assert x.shape[:-1] == dist_params.shape[:-1]

    gridsize = 1. / (self.n_classes - 1.)
    x = x.astype(jnp.float32) / (self.n_classes - 1.)  # range [0, 1.]
    x = x * 2 - 1  # range [-1., 1.]

    log_probs = distribution_utils.discretized_mix_logistic(
        x, dist_params, gridsize)
    return log_probs

  def sample(self, key, params):
    sample_x = distribution_utils.sample_from_discretized_mix_logistic(
        key, params, self.n_mixtures)  # range [-1., 1.]

    sample_x = (sample_x + 1.) / 2.  # range [0, 1.]
    sample_x = sample_x * (self.n_classes - 1.)  # range [0, n_classes - 1]

    # Better round now, otherwise we get floor division when cast to int32.
    sample_x = jnp.round(sample_x)
    return sample_x

  def encode(self, streams, x, params, selection):
    raise NotImplementedError

  def decode(self, stream, params, selection):
    raise NotImplementedError


class DiscretizedMixLogisticRGB(flax.linen.Module):
  """Collection of discretized RGB functions."""
  n_channels: int
  n_classes: int
  n_mixtures: int

  def get_required_num_outputs(self):
    """Return required number of channels in dist_params.

    This module assumes n_channels = 3. So we need
    1 + (1 (mean) + 1 (scale) * 3) + 1 (channel-autoregressive coeff R->G)
      + 2 (channel-autoregressive coeff R, G->B) = 10.
    parameters per mixture.

    Returns:
      num_required_outputs: the number of output channels that are needed for
        the distribution parameters.
    """
    assert self.n_channels == 3
    return 10 * self.n_mixtures

  def log_prob(self, x, params):
    """Computes log prob."""
    assert x.dtype == jnp.int32
    assert x.shape[:-1] == params.shape[:-1]

    gridsize = 1. / (self.n_classes - 1.)
    x = x.astype(jnp.float32) / (self.n_classes - 1.)  # range [0, 1.]
    x = x * 2. - 1.  # range [-1., 1.]

    log_probs = distribution_utils.discretized_mix_logistic_rgb(
        x, params, gridsize)
    return log_probs

  def sample(self, key, params):
    sample_x = distribution_utils.sample_from_discretized_mix_logistic_rgb(
        key, params, self.n_mixtures)  # range [-1., 1.]

    sample_x = (sample_x + 1.) / 2.  # range [0, 1.]
    sample_x = sample_x * (self.n_classes - 1.)  # range [0, n_classes - 1]

    # Better round now, otherwise we get floor division when cast to int32.
    sample_x = jnp.round(sample_x)
    return sample_x

  def encode(self, streams, x, params, selection):
    """Encodes the variable x at location selection using the distribution.

    Args:
      streams: Batch of bitstream objects to encode to.
      x: Variable to encode.
      params: Distribution parameters for x.
      selection: Mask signifying which index to encode.

    Returns:
      The bitstream objects.
    """
    batch_size = x.shape[0]

    reduce_axes = tuple(range(1, len(x.shape)-1))

    # Selects the relevant variable via selection mask.
    params = jnp.sum(params * selection, axis=reduce_axes, keepdims=True)
    x = jnp.sum(x * selection, axis=reduce_axes, keepdims=True)
    x = np.asarray(x, dtype=np.int32)

    logits = distribution_utils.logits_discretized_mix_logistic_rgb(
        jnp.asarray(x), params, self.n_classes)

    probs = distribution_utils.get_safe_probs(logits)

    # Put RGB at first axis.
    x_flat = np.asarray(x, dtype=np.uint64).reshape(batch_size, 3)
    probs_flat = np.asarray(
        probs, dtype=np.float64).reshape(batch_size, 3, self.n_classes)

    for i in range(batch_size):
      for c in reversed(range(3)):
        streams[i].encode_cat(x=x_flat[i:i+1, c], probs=probs_flat[i:i+1, c])

    return streams

  def decode(self, streams, params, selection):
    """Decodes a variable at location selection using the distribution.

    Args:
      streams: Batch of bitstream objects to decode from.
      params: Distribution parameters for x.
      selection: Mask signifying which index to decode.

    Returns:
      A tuple with the decoded variable, and the bitstream objects.
    """
    batch_size = selection.shape[0]
    reduce_axes = tuple(range(1, len(selection.shape)-1))

    # Selects the relevant variable via selection mask.
    params = jnp.sum(params * selection, axis=reduce_axes, keepdims=True)
    x_decoded = np.zeros((batch_size, 1, 1, 3), dtype=np.int32)

    # Decode Red.
    logits_r = distribution_utils.logits_discretized_mix_logistic_rgb(
        jnp.asarray(x_decoded), params, self.n_classes)[Ellipsis, 0, :]
    probs_r = distribution_utils.get_safe_probs(logits_r)
    probs_r_flat = np.asarray(
        probs_r, dtype=np.float64).reshape(batch_size, self.n_classes)

    for i in range(batch_size):
      r = streams[i].decode_cat(probs=probs_r_flat[i:i+1])
      x_decoded[i, Ellipsis, 0] = r

    # Decode Green.
    logits_g = distribution_utils.logits_discretized_mix_logistic_rgb(
        jnp.asarray(x_decoded), params, self.n_classes)[Ellipsis, 1, :]
    probs_g = distribution_utils.get_safe_probs(logits_g)
    probs_g_flat = np.asarray(
        probs_g, dtype=np.float64).reshape(batch_size, self.n_classes)

    for i in range(batch_size):
      g = streams[i].decode_cat(probs=probs_g_flat[i:i+1])
      x_decoded[i, Ellipsis, 1] = g

    # Decode Blue.
    logits_b = distribution_utils.logits_discretized_mix_logistic_rgb(
        jnp.asarray(x_decoded), params, self.n_classes)[Ellipsis, 2, :]
    probs_b = distribution_utils.get_safe_probs(logits_b)
    probs_b_flat = np.asarray(
        probs_b, dtype=np.float64).reshape(batch_size, self.n_classes)

    for i in range(batch_size):
      blue = streams[i].decode_cat(probs=probs_b_flat[i:i+1])
      x_decoded[i, Ellipsis, 2] = blue

    return x_decoded, streams
