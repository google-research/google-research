# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Entropy bottlenecks that do not use the quantization heuristic.

We do _not_ return quantize(data - mean) + mean. This is required for
autoregressive models.

## Details

We transmit `round(data - round(mean)) == round(data) - round(mean)`,
using N(scale, mean - round(mean)).

Note that this is the same as transmitting round(data) with N(scale, mean),
since we just shift both the data and the predicted mean by a constant C =
round(mean). However, the benefit is that `mean - round(mean)` is in [-0.5,
0.5], which allows for a small table for range coding.

## Note on naming

Despite the name of this file, the main class has "shift" in its name.

## Gotchas

- With laplace_tail_mass, the bitrate as a function of data is only symmetric
  if the predicted mean is at 0, since the Laplacian is always centered on 0
- Even without laplace_tail_mass, the bitrate as a function of data is only
  symmetric if the mean is either an integer or exactly between integers.
"""

from typing import Optional, Tuple

from absl import logging
import tensorflow as tf
import tensorflow_compression as tfc


Tensor = tf.Tensor


def verysoftplus(inputs):
  """Evaluates a softplus function transitioning from 1/(1-x) to 1+x at x=0."""
  inputs_pos = tf.maximum(inputs, 0)
  inputs_neg = tf.minimum(inputs, 0)
  return tf.where(inputs > 0, inputs_pos + 1, 1 / (1 - inputs_neg))


class ContinuousIndexedEntropyModel(tfc.ContinuousIndexedEntropyModel):
  """Variant of the base class which uses _log_prob also for range coding."""

  def _build_tables(self, prior, precision, offset=None):
    # Prevent tensors from bouncing back and forth between host and GPU.

    with tf.device("/cpu:0"):
      # Variant of the base class where we use FOO
      precision = int(precision)
      offset = tf.cast(0 if offset is None else offset, prior.dtype)
      # Subclasses should have already caught this, but better be safe.
      assert not prior.event_shape.rank

      lower_tail = tfc.distributions.lower_tail(prior, self.tail_mass)
      upper_tail = tfc.distributions.upper_tail(prior, self.tail_mass)
      # Integers such that:
      # minima + offset < lower_tail
      # maxima + offset > upper_tail
      minima = tf.cast(tf.math.floor(lower_tail - offset), tf.int32)
      maxima = tf.cast(tf.math.ceil(upper_tail - offset), tf.int32)

      # PMF starting positions and lengths.
      pmf_start = tf.cast(minima, prior.dtype) + offset
      pmf_length = maxima - minima + 1

      # Sample the densities in the computed ranges, possibly computing more
      # samples than necessary at the upper end.
      max_length = tf.math.reduce_max(pmf_length)
      if tf.executing_eagerly() and max_length > 2048:
        logging.warning(
            "Very wide PMF with %d elements may lead to out of memory issues. "
            "Consider priors with smaller variance, or increasing `tail_mass` "
            "parameter.", int(max_length))
      samples = tf.range(tf.cast(max_length, prior.dtype), dtype=prior.dtype)
      samples = tf.reshape(samples, [-1] + pmf_length.shape.rank * [1])
      samples += pmf_start

      # TODO(b/251395094): Move this into TFC.
      # NOTE: This line is the main difference to the base class: Here we use
      # exp(log_prob(x)), which includes the Laplacian tail mass.
      pmf = tf.math.exp(self._log_prob(prior, samples))
      pmf_shape = tf.shape(pmf)[1:]
      num_pmfs = tf.reduce_prod(pmf_shape)

      # Collapse batch dimensions of distribution.
      pmf = tf.reshape(pmf, [max_length, num_pmfs])
      pmf = tf.transpose(pmf)

      logging.info("PMF shape: %s", pmf.shape)

      pmf_length = tf.broadcast_to(pmf_length, pmf_shape)
      pmf_length = tf.reshape(pmf_length, [num_pmfs])
      cdf_offset = tf.broadcast_to(minima, pmf_shape)
      cdf_offset = tf.reshape(cdf_offset, [num_pmfs])

      cdf = get_cdf(pmf, pmf_length, int(num_pmfs.numpy()), precision)

    return cdf, cdf_offset


@tf.function
def get_cdf(pmf, pmf_length, num_pmfs,
            precision):
  """Get CDF from PMFs."""
  precision_tensor = tf.constant([-precision], dtype=tf.int32)

  def fn(i):
    p = pmf[i, :pmf_length[i]]
    overflow = tf.math.maximum(1. - tf.reduce_sum(p, keepdims=True), 0.)
    p = tf.cast(tf.concat([p, overflow], 0), tf.float32)
    c = tfc.ops.pmf_to_quantized_cdf(p, precision=precision)
    # Each local cdf `c` has a different shape. This will become a RaggedTensor
    # in the output of the `map_fn` call.
    return tf.concat([precision_tensor, c], 0)

  cdf = tf.map_fn(
      fn,
      tf.stop_gradient(tf.range(num_pmfs)),
      dtype=tf.int32,
      back_prop=False,
      parallel_iterations=128,
      fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32))
  return cdf.values  # Flatten ragged output.


class Scaler:
  """Helper to scale `scale` before quantizing to indices.

  Introduces a log quantization grid that aligns well with
  how scales are distributed.

  The scale minimum is handled by adding `scales_min` to input scales and
  mapping the indices to `[scales_min, scales_max+scales_min]`. The effect of
  this is that `to_scale_idx` and `from_scale_idx` are not inverses of each
  other, rather: `x = to_scale_idx(from_scale_idx(x) - scales_min)`
  """

  def __init__(self, scales_min = 0.01,
               scales_max = 256.0,
               num_bins = 256,
               verify_valid_scales = True):
    self.scales_min = float(scales_min)
    self.scales_max = float(scales_max)
    self.num_bins = int(num_bins)
    self.verify_valid_scales = bool(verify_valid_scales)

  def to_scale_idx(self, scale, training = True):
    """Convert `scale` in [0, scales_max] to an index."""
    if self.verify_valid_scales:
      tf.debugging.assert_greater_equal(scale, 0.0)

    scale += self.scales_min

    # \in [log(scales_min), log(scales_max+scales_min)]
    idx = tf.math.log(scale)
    # \in [0, log(scales_max+scales_min) - log(scales_min)]
    idx = idx - tf.math.log(self.scales_min)
    # \in [0, 1]
    normalizer = self.scales_max / self.scales_min
    normalizer += 1  # log(max+min) - log(min) = log(max/min + 1)
    idx = idx / tf.math.log(normalizer)
    # \in [0, num_bins-1]
    idx = idx * (self.num_bins - 1)
    if not training:
      idx = tf.round(idx)
    return idx

  def from_scale_idx(self, idx):
    out = idx / (self.num_bins - 1)
    normalizer = self.scales_max / self.scales_min
    normalizer += 1  # log(max+min) - log(min) = log(max/min + 1)
    out = out * tf.math.log(normalizer)
    return tf.math.exp(out + tf.math.log(self.scales_min))


class ConditionalLocScaleShiftBottleneck(tf.Module):
  """Bottleneck that only operates on mu, sigma.

  transmit data - round(mean) using mean - round(mean).
  This allows for very high resolution mean tables.
  """

  def __init__(self,
               min_scale = 0.01,
               laplace_tail_mass = 1e-3,
               expected_grads = False,
               coding_rank = 3,
               compression = False,
               num_means = 100,
               num_scales = 256,
               round_indices = True,
               name = None):
    """Initializes the layer.

    Args:
      min_scale: Minimum scale.
      laplace_tail_mass: See tfc.ContinuousIndexedEntropyModel.
      expected_grads: See tfc.ContinuousIndexedEntropyModel.
      coding_rank: See tfc.ContinuousIndexedEntropyModel.
      compression: See tfc.ContinuousIndexedEntropyModel.
      num_means: Number of means to use.
      num_scales: Number of scales to use.
      round_indices: If True, round indices always.
      name: Layer name.
    """
    super().__init__(name=name)

    self._laplace_tail_mass = laplace_tail_mass
    self._num_means = num_means
    self._scaler = Scaler(num_bins=num_scales, scales_min=min_scale)
    self._round_indices = round_indices

    kwargs = dict(
        prior_fn=tfc.NoisyNormal,
        index_ranges=(self._num_means, num_scales),
        parameter_fns=dict(
            loc=lambda idx: idx[Ellipsis, 0] / self._num_means - 0.5,
            scale=lambda idx: self._scaler.from_scale_idx(idx[Ellipsis, 1])))

    self._entropy_model = ContinuousIndexedEntropyModel(
        **kwargs,
        coding_rank=coding_rank,
        laplace_tail_mass=laplace_tail_mass,
        expected_grads=expected_grads,
        compression=compression,
    )
    self._compression = compression

  @property
  def laplace_tail_mass(self):
    """Returns tail mass, see tfc.ContinuousIndexedEntropyModel."""
    return self._laplace_tail_mass

  def _get_indexes(self, mean, scale, training):
    """Returns indices for ContinuousIndexedEntropyModel."""
    scale = self._scaler.to_scale_idx(verysoftplus(scale))
    mean_i = (mean - tfc.round_st(mean) + 0.5) * self._num_means
    return tf.stack([mean_i, scale], axis=-1)

  def _get_shift(self, mean):
    return tfc.round_st(mean)

  @tf.Module.with_name_scope
  def __call__(
      self,
      latent,
      mean,
      scale,
      training,
  ):
    """Compresses `latent`, returns quantized version and number of bits."""
    indexes = self._get_indexes(mean, scale, training=training)
    if not training and self._round_indices:
      indexes = tf.round(indexes)
    shift = self._get_shift(mean)
    latent = latent - shift
    quantized_latent, bits = self._entropy_model(
        latent, indexes=indexes, training=training)
    if training:  # Round + STE
      quantized_latent = self._entropy_model.quantize(latent)
    return quantized_latent + shift, bits

  def compress(self, latent, mean,
               scale):
    """Compresses into a bytestring."""
    indexes = self._get_indexes(mean, scale, training=False)
    shift = self._get_shift(mean)
    latent = latent - shift
    with tf.device("cpu"):
      bytestring = self._entropy_model.compress(latent, indexes)
    quantized = self._entropy_model.quantize(latent)
    return quantized + shift, bytestring

  def decompress(self, bytesting, mean,
                 scale):
    """Decompresses `bytesting`."""
    indexes = self._get_indexes(mean, scale, training=False)
    with tf.device("cpu"):
      latent = self._entropy_model.decompress(bytesting, indexes)
    return latent + self._get_shift(mean)

  def quantize(self, latent, mean):
    shift = self._get_shift(mean)
    return tf.round(latent - shift) + shift
