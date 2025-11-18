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

"""GeoTensor class."""

from __future__ import annotations

from collections.abc import Sequence
import itertools
import operator
from typing import Any, Union

import numpy as np
import tensorflow as tf

from global_metnet import normalizers
from global_metnet import utils


def reorder_channels(t, input_channels, output_channels):
  """Reorder channels in a tensor."""
  if t.shape[-1] != len(input_channels):
    raise ValueError(f'The tensor has shape {t.shape} but '
                     f'{len(input_channels)} input channels provided.')
  assert len(set(input_channels)) == len(input_channels), (
      'Duplicated input channels.')
  perm = []
  for channel in output_channels:
    if channel not in input_channels:
      raise ValueError(f'Output channel {channel} not found in input channels. '
                       f'{input_channels=}.')
    perm.append(input_channels.index(channel))
  if isinstance(t, tf.Tensor):
    return tf.gather(t, perm, axis=-1)
  else:
    return t[Ellipsis, perm]


def factor(context_size, downsampling_ratio):
  context_size = utils.get_tuple(context_size)
  downsampling_ratio = utils.get_tuple(downsampling_ratio)
  return (int(context_size[0] / downsampling_ratio[0]),
          int(context_size[1] / downsampling_ratio[1]))


def center_crop(
    x,
    context_size,
    downsampling_ratio = 1,
    has_time_dim = True,
):
  """Return the center crop of a tensor."""
  if not x.shape:
    return x
  context_size = utils.get_tuple(factor(context_size, downsampling_ratio))
  if (context_size[0] != int(context_size[0]) or
      context_size[1] != int(context_size[1])):
    raise ValueError(f'Non-integer context_size: {context_size}')
  context_size = tuple(map(int, context_size))
  x_start = (x.shape[1] - context_size[0]) // 2
  y_start = (x.shape[2] - context_size[1]) // 2
  if x_start < 0:
    raise ValueError(
        f'x.shape[1]: {x.shape[1]} context_size: {context_size[0]}'
    )
  if y_start < 0:
    raise ValueError(
        f'x.shape[2]: {x.shape[2]} context_size: {context_size[1]}'
    )
  if has_time_dim:
    return x[
        :,
        x_start : x_start + context_size[0],
        y_start : y_start + context_size[1],
        Ellipsis,
    ]
  else:
    return x[
        x_start : x_start + context_size[0],
        y_start : y_start + context_size[1],
        Ellipsis,
    ]


class GeoTensorSpec(tf.TypeSpec):
  """TypeSpec for the GeoTensor class.

  This class defines the structure and metadata of a GeoTensor, allowing it to
  be used within TensorFlow's type system.
  """

  def __init__(
      self,
      channels,
      resolution_km,
      norm_kwargs,
  ):
    self._channels = list(channels)
    self._resolution_km = resolution_km
    self._norm_kwargs = norm_kwargs

  @property
  def channels(self):
    return self._channels

  @property
  def resolution_km(self):
    return self._resolution_km

  @property
  def norm_kwargs(self):
    return self._norm_kwargs

  @property
  def _component_specs(self):
    return (tf.TensorSpec(shape=None, dtype=tf.float32),)

  def _to_components(self, value):
    return (value.data,)

  def _from_components(self, components):
    (data,) = components
    if self._norm_kwargs:
      normalizer = normalizers.deserialize(self._norm_kwargs)
    else:
      normalizer = normalizers.identity(len(self.channels))
    return GeoTensor(
        data,
        self._channels,
        self._resolution_km,
        normalizer,
    )

  def _serialize(self):
    return (
        self._channels,
        self._resolution_km,
        self._norm_kwargs,
    )

  @classmethod
  def _deserialize(
      cls,
      serialization,
  ):
    (
        channels,
        resolution_km,
        norm_kwargs,
    ) = serialization
    return GeoTensorSpec(
        channels,
        resolution_km,
        norm_kwargs,
    )

  @property
  def value_type(self):
    return GeoTensor


class GeoTensor:
  """tf.Tensor with a specified list of channels, resolution and normalization.
  """

  def __init__(
      self,
      data,
      channels,
      resolution_km,
      normalizer,
  ):
    if len(channels) != data.shape[-1]:
      raise ValueError(f'Was {len(channels)} and {data.shape}.')
    if normalizer.num_channels != data.shape[-1]:
      raise ValueError(f'Was {normalizer.num_channels} and {data.shape}.')
    self._data = data
    self._channels = channels
    self._resolution_km = resolution_km
    self._normalizer = normalizer
    self.spec = GeoTensorSpec(
        self._channels,
        self._resolution_km,
        self._normalizer.serialize(),
    )

  @property
  def data(self):
    return self._data

  @property
  def channels(self):
    return self._channels

  @property
  def resolution_km(self):
    return self._resolution_km

  @property
  def normalizer(self):
    return self._normalizer

  def replace_data(self, new_data):
    return GeoTensor(new_data,
                     self._channels,
                     self._resolution_km,
                     self._normalizer)

  def reorder_channels(self, new_channels):
    """Reorder channels in a GeoTensor."""
    def reorder(t):
      if t is None:
        return None
      else:
        return reorder_channels(t, self._channels, new_channels)
    data = reorder(self._data)
    normalizer = normalizers.Normalizer(
        center=reorder(self._normalizer.center),
        scale=reorder(self._normalizer.scale),
        lower_bound=reorder(self._normalizer.lower_bound),
        upper_bound=reorder(self._normalizer.upper_bound))
    return GeoTensor(data, new_channels, self._resolution_km, normalizer)

  def get_channel_denormalized(self, channel):
    out = self.reorder_channels([channel]).denormalize().data
    out = tf.squeeze(out, axis=-1)
    return out

  def append_channel_denormalized(
      self, channel, value
  ):
    assert value.shape[-1] != 1, 'value should not have a dummy channel dim'
    value = value[Ellipsis, None]
    value = GeoTensor(value, [channel], self._resolution_km,
                      normalizers.identity(1))
    return GeoTensor.concat_channels([self, value])

  def renormalize(self, new_normalizer):
    raw = self._normalizer.denormalize(self._data)
    data = new_normalizer.normalize(raw)
    return GeoTensor(
        data=data,
        channels=self._channels,
        resolution_km=self._resolution_km,
        normalizer=new_normalizer,
    )

  def denormalize(self):
    return self.renormalize(normalizers.identity(len(self.channels)))

  def _is_multiple_of_resolution(
      self, new_size_km
  ):
    if isinstance(new_size_km, int) and isinstance(self._resolution_km, int):
      return new_size_km % self._resolution_km == 0
    new_size_km = utils.get_tuple(new_size_km)
    resolution_km = utils.get_tuple(self._resolution_km)
    return (
        new_size_km[0] % resolution_km[0] == 0
        and new_size_km[1] % resolution_km[1] == 0
    )

  def downsample_or_upsample_to(
      self,
      new_resolution_km,
  ):
    if new_resolution_km == self._resolution_km:
      return self
    raise NotImplementedError()

  def crop(self, new_size_km):
    """Crops the GeoTensor to the given size."""
    if len(self._data.shape) != 4:
      raise ValueError(
          f'Only 4 dimensional data can be cropped, found shape {self.shape}'
      )
    if not self._is_multiple_of_resolution(new_size_km):
      raise ValueError(
          f'New size {new_size_km} is not a multiple of {self._resolution_km}'
      )
    if not utils.satisfies_op(new_size_km, self.size_km, operator.le):
      raise ValueError(
          'Cannot crop when new size is greater than current size.'
      )
    if utils.satisfies_op(new_size_km, self.size_km, operator.eq):
      return self
    return self.replace_data(
        center_crop(self._data, new_size_km, self._resolution_km)
    )

  def replace_not_finite(self, value):
    value = tf.cast(value, self.dtype)
    is_nan = tf.math.is_nan(self._data) | tf.math.is_inf(self._data)
    data = tf.where(is_nan, value, self._data)
    return self.replace_data(data)

  def astype(self, dtype):
    return self.replace_data(tf.cast(self._data, dtype))

  @property
  def dtype(self):
    return self._data.dtype

  @property
  def shape(self):
    return self._data.shape

  @property
  def size_km(self):
    """Returns (height, width) of data in kms."""
    if isinstance(self._resolution_km, int) and self.shape[1] == self.shape[2]:
      return self.shape[1] * self._resolution_km
    resolution_km = utils.get_tuple(self._resolution_km)
    return (self.shape[1] * resolution_km[0], self.shape[2] * resolution_km[1])

  @classmethod
  def concat_channels(cls, xs):
    if len(set([utils.get_tuple(x.resolution_km) for x in xs])) != 1:
      raise ValueError('Resolutions of all tensors do not match')
    normalizer = normalizers.Normalizer(
        center=np.concatenate([x.normalizer.center for x in xs], axis=-1),
        scale=np.concatenate([x.normalizer.scale for x in xs], axis=-1),
    )
    return GeoTensor(
        tf.concat([x.data for x in xs], axis=-1),
        channels=list(itertools.chain.from_iterable([x.channels for x in xs])),
        resolution_km=xs[0].resolution_km,
        normalizer=normalizer,
    )
