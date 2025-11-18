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

"""Preprocess model inputs."""

import abc
import operator
from typing import Callable, Optional, Union

from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from global_metnet import geo_tensor
from global_metnet import normalizers
from global_metnet import utils


class StandardPreprocessor(abc.ABC):
  """Standard preprocessing operations like cropping."""

  def __init__(
      self,
      dataset_keys,
      replace_not_finite = False,
      output_normalizer_fn = lambda _: None,
      dtype=None,
      filter_channels = None,
      space_to_depth = 1,
      output_size_km = None,
      output_resolution_km = None,
      mask_out_values = None,
      mask_out_abs_tolerance = 1e-4,
      jit_compile=False,
  ):
    """Create a standard preprocessor.

    Args:
      dataset_keys: A list of dataset keys. The tensors corresponding to those
        keys will be operated together.
      replace_not_finite: Whether to replace NaNs and infinities with zeros.
      output_normalizer_fn: The desired way to normalize outputted values.
      dtype: Outputs are cast to this dtype.
      filter_channels: If not None, only the channels on this list will be kept.
      space_to_depth: If not None, each output pixel contains data for a
        space_to_depth x space_to_depth square at a higher resolution.
      output_size_km: This value overwrites size_km argument in preprocess(...)
        if present.
      output_resolution_km: This value overwrites resolution_km argument in
        preprocess(...) if present.
      mask_out_values: A list of values which are replaced with nan (for
        floating point) or -1 (for integers).
      mask_out_abs_tolerance: The absolute tolerance used to check which values
        to replace with nan/-1.
      jit_compile: Whether to jit the preprocess() method.
    """
    self.dataset_keys = dataset_keys
    self._replace_not_finite = replace_not_finite
    self.output_normalizer_fn = output_normalizer_fn
    self._dtype = dtype
    self._filter_channels = filter_channels
    self._space_to_depth = space_to_depth
    self.output_channels = None
    self.output_size_km = output_size_km
    self.output_resolution_km = output_resolution_km
    self._mask_out_values = mask_out_values
    self._mask_out_abs_tolerance = mask_out_abs_tolerance
    self._jit_compile = jit_compile

  def preprocess_maybe_jitted(self, size_km, resolution_km, is_eval, **kwargs):
    if self._jit_compile:
      return self.preprocess_jitted(size_km, resolution_km, is_eval, **kwargs)
    else:
      return self.preprocess(size_km, resolution_km, is_eval, **kwargs)

  @tf.function(jit_compile=True)
  def preprocess_jitted(self, size_km, resolution_km, is_eval, **kwargs):
    return self.preprocess(size_km, resolution_km, is_eval, **kwargs)

  def preprocess(self, size_km, resolution_km, is_eval, **kwargs):
    """Full preprocessing from dataset to network inputs/targets."""
    size_km = self.output_size_km or size_km
    resolution_km = self.output_resolution_km or resolution_km
    value = self._dataset_to_raw(size_km, resolution_km, **kwargs)
    value = self._preprocess_denormalized(value, is_eval=is_eval)
    value = self._final_preprocess(value)
    return value

  def _dataset_to_raw(self, size_km, resolution_km, **kwargs):
    """Concatenate, filter channels, pad/crop, and downsample."""
    # Concatenate.
    value = geo_tensor.GeoTensor.concat_channels(
        [kwargs[k] for k in self.dataset_keys]
    )
    assert len(value.shape) == 4  # [T, W, H, F]

    # Filter channels.
    if self._filter_channels:
      value = value.reorder_channels(self._filter_channels)

    # Crop.
    value = value.crop(new_size_km=size_km)
    if not utils.satisfies_op(value.size_km, size_km, operator.eq):
      raise ValueError(
          f'Size {value.size_km} not equal to {size_km} after cropping.'
      )

    # Replace custom values with nan/-1.
    if self._mask_out_values:
      replace_with = (
          np.nan if value.dtype.is_floating else tf.cast(-1, value.dtype)
      )
      denormalized = value.denormalize().data
      for value_to_mask in self._mask_out_values:
        diff = tf.abs(denormalized - value_to_mask)
        replace_mask = (
            (diff <= self._mask_out_abs_tolerance)
            if value.dtype.is_floating
            else (diff == 0)
        )
        data = tf.where(replace_mask, replace_with, value.data)
        value = value.replace_data(data)

    # Downsample or upsample.
    value = value.downsample_or_upsample_to(resolution_km)

    # Denormalize.
    value = value.astype(tf.float32).denormalize()
    return value

  def _preprocess_denormalized(self, value, is_eval):
    """Second part of preprocessing. Operates on unnormalized values."""
    del is_eval
    return value

  def _normalize(self, value):
    """Renormalizes the geo_tensor.GeoTensor."""
    output_normalizer = self.output_normalizer_fn(value.channels)
    if output_normalizer is not None:
      value = value.astype(tf.float32).renormalize(output_normalizer)
    return value

  def _final_preprocess(self, value):
    """Third part of preprocessing."""
    # Normalize.
    value = self._normalize(value)

    # Cast.
    if self._dtype:
      value = value.astype(self._dtype)

    # Handle NaNs.
    value = value.replace_not_finite(value=0.0)

    # Space to depth.
    if self._space_to_depth != 1:
      n = self._space_to_depth
      new_channels = []
      for i in range(n):
        for j in range(n):
          for ch in value.channels:
            new_channels.append(f'{i}_{j}_{ch}')
      value = geo_tensor.GeoTensor(
          tf.nn.space_to_depth(value.data, n),
          new_channels,
          value.resolution_km,
          normalizers.identity(len(new_channels)),
      )
    return value


class PrecipitationRatePreprocessor(StandardPreprocessor):
  """Squashes precip rates into a smaller range using tanh(log(rate)/4)."""

  def __init__(self, dataset_keys, rate_channel=None, **kwargs):
    self._rate_channel = rate_channel
    super().__init__(
        dataset_keys=dataset_keys, replace_not_finite=True, **kwargs
    )

  def _preprocess_denormalized(self, value, is_eval):
    channels = value.channels
    rate_channel = self._rate_channel
    if self._rate_channel is None:
      assert len(channels) == 1
      rate_channel = channels[0]
    rate = value.get_channel_denormalized(rate_channel)
    # This assumes that precip rates are in mm/h.
    # Filter out extreme values which are impossible.
    rate = tf.where(rate > 2000, np.nan, rate)
    rate = tf.where(rate < 0, np.nan, rate + 1)
    rate = tf.where(tf.math.is_finite(rate), rate, 0)
    rate = tf.tanh(tf.math.log(rate) / 4)
    if len(channels) == 1:
      new_value = value.replace_data(tf.expand_dims(rate, axis=-1))
    else:
      new_value = value.reorder_channels(
          [c for c in channels if c != rate_channel]
      ).append_channel_denormalized(rate_channel, rate)
    return super()._preprocess_denormalized(new_value, is_eval)


class PrecipitationTargetPreprocessor(StandardPreprocessor):
  """Filter precip rates which are impossibly large or negative."""

  def __init__(self, dataset_keys, **kwargs):
    super().__init__(dataset_keys=dataset_keys, **kwargs)

  def _preprocess_denormalized(self, value, is_eval):
    data = value.data
    data = tf.where(tf.math.logical_or(data > 2000, data < 0), np.nan, data)
    value = value.replace_data(data)
    return super()._preprocess_denormalized(value, is_eval)


class RestPreprocessor(StandardPreprocessor):
  """Preprocessor for Rest inputs."""

  def __init__(
      self,
      lon_lat_key=None,
      include_elevation=False,
      repeat_inputs=False,
      **kwargs,
  ):
    def output_normalizer_fn(channels):
      return normalizers.Normalizer(
          center=[0.0 for _ in channels],
          scale=[1000.0 if c in ['lon', 'lat'] else 1.0 for c in channels],
      )

    self._lon_lat_key = lon_lat_key or 'lonlat_input'
    dataset_keys = [self._lon_lat_key]
    self.repeat_inputs = repeat_inputs
    if include_elevation:
      dataset_keys.append('elevation_input')

    super().__init__(
        dataset_keys=dataset_keys,
        output_normalizer_fn=output_normalizer_fn,
        replace_not_finite=False,
        **kwargs,
    )


class Head(abc.ABC):
  """Head for continuous scalar targets which are discretized."""

  def __init__(
      self,
      bins,
      eval_bins,
      resolution_km,
      preprocessor,
      loss_weight = 1.0,
      early_stopping_weight = None,
      timedeltas_key = None,
      target_offsets = None,
      **kwargs,
  ):
    """Head for targets with continuous values we discretize.

    Args:
      bins: discretization bins. Values < bins[0] are in the same bin as values
        in the range [bins[0], bins[1]] and values > bins[-1] are in the same
        bin as values in the range [bins[-2], bins[-1]].
      eval_bins: these are the thresholds at which we compute metrics (e.g. F1)
        for binary prediction. Previously called 'rates'.
      resolution_km: The resolution of the network output and target.
      preprocessor: The preprocessor object which prepares the targets.
      loss_weight: The loss weight used for training.
      early_stopping_weight: The loss weight used for early stopping. None means
        same as loss_weight.
      timedeltas_key: The dataset key containing the target timedeltas. If None,
        it will be inferred from the preprocessor.
      target_offsets: The head predicts for len(target_offsets) timesteps
        simultaneously with target_offsets specifying the offset in mins between
        the single sampled lead time and lead times used for particular targets,
        e.g. range(-30, 31, 2) means that we predict for the whole hour centered
        on the sampled lead time. Defaults to [0].
      **kwargs: Additional keyword arguments. Supported kwargs are:
        - extra_logs: bool, whether to compute extra logs.
        - sample_non_nan_target_lead_times: bool, whether to sample non-nan
          target lead times.
    """
    self.resolution_km = resolution_km
    self._preprocessor = preprocessor
    self._timedeltas_key = timedeltas_key
    self.num_bins = np.asarray(bins).shape[0] - 1
    self.num_output_channels_per_timestep = self.num_bins
    self.loss_weight = loss_weight
    if early_stopping_weight is None:
      early_stopping_weight = self.loss_weight
    self.early_stopping_weight = early_stopping_weight

    assert bins.ndim == 1, f'{bins.ndim=}'
    bins = bins[Ellipsis, np.newaxis]
    assert bins.shape == (self.num_bins + 1, 1), f'{bins.shape=}'
    self.bins = bins
    self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
    self.bin_centers[0] = np.where(bins[0] == 0, 0, self.bin_centers[0])
    self.eval_bins = eval_bins
    self.regional_mask = None
    self.extra_logs = kwargs.get('extra_logs', False)
    self.sample_non_nan_target_lead_times = kwargs.get(
        'sample_non_nan_target_lead_times', False
    )
    self.target_offsets = target_offsets or [0]

  def timedeltas_key(self, is_eval):
    """Returns the dataset key containing the target timedeltas."""
    del is_eval
    if self._timedeltas_key:
      return self._timedeltas_key
    else:
      keys = self._preprocessor.dataset_keys
      assert len(keys) == 1, f'Was {keys}'
      return keys[0] + '_timedeltas'

  def preprocessor(self, is_eval):
    del is_eval
    return self._preprocessor

  @property
  def num_output_channels(self):
    return len(self.target_offsets) * self.num_output_channels_per_timestep

  def requires_model_output(self):
    """Whether the network should provide output for this head."""
    return self.num_output_channels > 0

  def preprocess_target(self, target, split):
    """Preprocesses the target tensor by discretizing it.

    Args:
      target: The input target tensor.
      split: The dataset split (e.g., 'train', 'eval').

    Returns:
      A tuple containing:
        - disc_target: The discretized target tensor.
        - target: The original target tensor.
    """
    del split
    bins = tf.reshape(
        self.bins,
        [self.num_bins + 1]
        + [1] * (len(target.shape) - 2)
        + [1],
    )
    bins = tf.cast(bins, target.dtype)
    bins = tf.broadcast_to(bins, [self.num_bins + 1] + target.shape[1:])
    disc_target = tfp.stats.find_bins(
        target,
        bins,
        extend_lower_interval=True,
        extend_upper_interval=True,
        dtype=tf.int32,
    )
    return disc_target, target

  def postprocess_output(self, output):
    """Called on outputs after the model is run."""
    assert output.ndim == 5, f'Shape: {output.shape}'
    b, t, w, h, c = output.shape
    assert t == 1
    error_msg = f'{output.shape=} {self.num_output_channels_per_timestep=}'
    assert (
        c == len(self.target_offsets) * self.num_output_channels_per_timestep
    ), error_msg
    output = output.reshape(
        b, w, h, len(self.target_offsets), self.num_output_channels_per_timestep
    )
    output = jnp.transpose(output, [0, 3, 1, 2, 4])
    return jnp.reshape(
        output, output.shape[:-1] + (1, self.num_bins)
    )

  def _process_mask(self, mask, real_target):
    """Processes the mask to account for regional mask and nans.

    Args:
      mask: The mask to process.
      real_target: The real target.

    Returns:
      The processed mask.
    """
    assert 'float' in str(real_target.dtype)
    if mask.ndim <= 1:
      mask = ~jnp.isnan(real_target)  # +/-inf targets are not masked out.
    else:
      mask = jnp.broadcast_to(mask, real_target.shape)
      mask *= ~jnp.isnan(real_target)  # ignore mask=1 if target=nan
    if self.regional_mask is not None:
      mask *= jnp.broadcast_to(self.regional_mask, mask.shape)
    mask = mask.astype(jnp.float32)
    return mask

  def compute_loss(self, output, target, mask):
    target, real_target = target
    mask = self._process_mask(mask, real_target)
    # bs, t, h, w, features, logits
    assert output.ndim == 6, f'Had shape {output.shape}'
    # bs, t, h, w, features
    assert target.ndim == 5, f'Had shape {target.shape}'
    # bs, t, h, w, features
    assert mask.ndim == 5, f'Had shape {mask.shape}'
    loss_fn = utils.masked_cross_entropy_loss
    return loss_fn(output, target, mask)
