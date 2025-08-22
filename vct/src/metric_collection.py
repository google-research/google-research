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

"""Helper to collect metrics."""

import collections
from collections.abc import Iterable
from collections.abc import Mapping
import contextlib
import enum
from typing import Any, Callable, Dict, NamedTuple, Optional, TypeVar, Union

import numpy as np
import tensorflow as tf


class CollectionType(enum.Enum):
  SCALARS = "scalars"
  IMAGES = "images"

Scalar = Union[int, float, np.number, np.ndarray, tf.Tensor]
Scalars = Mapping[str, Scalar]
Image = Union[np.ndarray, tf.Tensor]
Images = Mapping[str, Image]

_DELIMITER = "/"


def join(*keys):
  """Join keys with delimiter, and strip the delimiter from both sides."""
  return _DELIMITER.join(
      key.strip(_DELIMITER) for key in keys).strip(_DELIMITER)


_ValueT = TypeVar("_ValueT")

# Useful for type annotations, e.g., "-> WithMetrics[WarpOutputs]".
WithMetrics = tuple[_ValueT, "Metrics"]


def _value_repr(x):
  if isinstance(x, tf.Tensor):
    return f"<tf.Tensor: shape={x.shape}, dtype={repr(x.dtype)[3:]}>"
  else:
    return repr(x)


_ignore_all_recorded_values = False


@contextlib.contextmanager
def disable_recording():
  """Context to disable Metrics.record_{scalar,image,...} (not thread safe)."""
  global _ignore_all_recorded_values
  _ignore_all_recorded_values = True
  try:
    yield
  finally:
    _ignore_all_recorded_values = False


class Metrics(NamedTuple):
  """A simple container for collecting metrics into a flat dictionaries.

  We use tf.Tensor as the internal type, but accept anything that is convertible
  to tf.Tensor as input (see Scalar/Image type above).

  See unit tests for example usage.

  Note that we use a typing.NamedTuple instead of a dataclass to support
  interactions with tf.function.
  """

  # NOTE: `typing.NamedTuple` does not support default factories, so
  # we cannot add empty dicts as defaults. Users of this class can use
  # Metrics.make().

  scalars: Dict[str, tf.Tensor]

  # Images are stored as uint8 tensors of shape N, H, W, C.
  # Note that later passing metrics.images directly to `write_images` of a
  # `SummaryWriter` will show in TB as N summaries with the same name, one for
  # each element, but with a "sample i of N" indication.
  images: Dict[str, tf.Tensor]

  def __repr__(self):
    scalars_repr = repr({k: f"{v}" for k, v in self.scalars.items()})
    images_repr = repr({k: _value_repr(v) for k, v in self.images.items()})
    return f"Metrics(scalars={scalars_repr}, images={images_repr})"

  @property
  def scalars_np(self):
    return {k: np.float32(v) for k, v in self.scalars.items()}  # pytype: disable=bad-return-type  # numpy-scalars

  @property
  def scalars_float(self):
    return {k: float(v) for k, v in self.scalars.items()}

  @classmethod
  def make(cls):
    return cls({}, {})

  @classmethod
  def from_scalar(cls, tag, scalar):
    return cls(scalars={tag: scalar}, images={})

  @classmethod
  def from_image(cls, tag, image):
    return cls(scalars={}, images={tag: image})

  @classmethod
  def reduce(
      cls,
      metric_list,
      scalar_reduce_fn = None,
      image_reduce_fn = None,
  ):
    """Create a new metric with combined values from an iterable of metrics.

    Args:
      metric_list: The metrics to combine.
      scalar_reduce_fn: The function to combine scalars (e.g. tf.reduce_mean).
        If not provided, the result will have no scalars.
      image_reduce_fn: The function to combine images (e.g. tf.reduce_mean).
        If not provided, the result will have no images.

    Returns:
      A Metrics instance `reduced_metric` where the metrics have been combined.
      E.g.:
        reduced_metric.scalars[key] = scalar_reduce_fn([
            metric.scalars[key]
            for metric in metric_list
            if key in metric.scalars
        ])
    """
    reduce_fns = {
        CollectionType.SCALARS: scalar_reduce_fn,
        CollectionType.IMAGES: image_reduce_fn
    }
    reduced_collections = {
        collection_type.value: {} for collection_type in CollectionType
    }
    for collection_type in CollectionType:
      reduce_fn = reduce_fns[collection_type]
      if reduce_fn is None:
        continue
      all_values = collections.defaultdict(list)
      for metric in metric_list:
        for key, value in metric._collections[collection_type].items():  # pylint: disable=protected-access
          all_values[key].append(value)
      reduced_collection = {
          key: reduce_fn(values) for key, values in all_values.items()
      }
      reduced_collections[collection_type.value] = reduced_collection
    return cls(**reduced_collections)

  @property
  def _collections(self):
    return {
        CollectionType.SCALARS: self.scalars,
        CollectionType.IMAGES: self.images,
    }

  def _record(self, collection_type, key, value):
    """Helper to record a value into a collection."""
    if _ignore_all_recorded_values:
      return
    collection_dict = self._collections[collection_type]
    if key in collection_dict:
      raise ValueError(f"Duplicate value for key `{key}` in  {collection_type}`"
                       " collection")
    collection_dict[key] = value

  def record_scalar(self, key, value):
    """Record a scalar `value` into metric.scalars[key] as a tf.float32."""
    value_tf = tf.convert_to_tensor(value)
    if value_tf.shape.as_list():
      raise ValueError(f"Expected scalar, got `{value_tf}` with shape "
                       f"`{value_tf.shape}` (after conversion to tensor).")
    value_tf = tf.cast(value_tf, tf.float32)
    self._record(CollectionType.SCALARS, key, value_tf)

  def record_image(self, key, image):
    """Record an image batch `value` into metric.images[key].

    Args:
      key: The key of the summary.
      image: A [batch, height, width, channels] tensor of images. It can
        either have dtype tf.float32 (with values clipped to range [0...1]) or
        dtype tf.uint8.

    """
    image_tf = tf.convert_to_tensor(image)
    if len(image_tf.shape) != 4:
      raise ValueError(f"Expected a rank 4 image batch, got `{image_tf}` with "
                       f" shape `{image_tf.shape}` (after conversion to "
                       " tensor).")
    if image_tf.shape[-1] not in (1, 2, 3, 4):
      raise ValueError(
          f"Invalid num_channels={image_tf.shape[-1]} for `{key}`!")
    if image_tf.dtype == tf.float32:
      image_clipped = tf.clip_by_value(image_tf, 0.0, 1.0)
      image_tf_uint8 = tf.cast(tf.round(image_clipped*255.0), tf.uint8)
    else:
      if image_tf.dtype != tf.uint8:
        raise ValueError(
            "Images must be of dtype tf.uint8 or tf.float32")
      image_tf_uint8 = image_tf
    self._record(CollectionType.IMAGES, key, image_tf_uint8)

  def record_scalars(self, **scalars):
    for key, value in scalars.items():
      self.record_scalar(key, value)

  def record_images(self, **images):
    for key, value in images.items():
      self.record_image(key, value)

  def merge(self,
            prefix_or_other,
            other = None):
    """Merge another metric collection into this one.

    This can be called in one of the following ways:

    metrics.merge(other_metrics)  # prefix is ""

    or

    metrics.merge("my_prefix", other_metrics)

    Args:
      prefix_or_other: Either a string prefix, or a metrics instance.
      other: If `prefix_or_other` is a string, this is required.
    """
    if other is None:
      if isinstance(prefix_or_other, str):
        raise ValueError(
            f"Expected not-None `metrics` for prefix=`{prefix_or_other}`!")
      prefix = ""
      other = prefix_or_other
    else:
      prefix = prefix_or_other

    for collection_type, collection in other._collections.items():  # pylint: disable=protected-access
      for key, value in collection.items():
        full_key = join(prefix, key)
        self._record(collection_type, full_key, value)
