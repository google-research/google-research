# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Image FeatureConnectors."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class Image(tfds.features.Image):
  """Image `FeatureConnector`.

  Unlike tfds.features.Image this class has the following advantages:
  1. Support tf.uint16 images.
  3. Stores the image size, channels and format in addition.
  4. Supports rank2 image tensors of shape (H, W).

  Example:

  * In the DatasetInfo object:
    features=features.FeaturesDict({
        'image': features.Image(shape=(None, None, 3), dtype=tf.uint8),
    })

  * Internally stored as:
    {
      'image/encoded': 'encoded image string',
      'image/width': image width,
      'image/height': image height,
      'image/channels': image channels,
      'image/format': 'format string'
    }

  * During generation:
    yield {
        'image': np.ones(shape=(480, 640, 3), dtype=np.uint8),
    }

  * Decoding will return as dictionary of tensorflow tensors:
    {
      'image': tf.Tensor(shape=(480, 640, 3), dtype=tf.uint8)
    }
  """

  def __init__(self, shape=None, encoding_format='png', dtype=tf.uint8):
    self._shape = tuple(shape) if shape is not None else (None, None, 3)
    self._channels = self._shape[-1] if len(self._shape) > 2 else 0
    self._dtype = dtype

    encode_fn_map = {
        'png': tf.image.encode_png,
        'jpeg': tf.image.encode_jpeg,
    }
    supported = encode_fn_map.keys()
    if encoding_format not in supported:
      raise ValueError('`encoding_format` must be one of %s.' % supported)
    self._encoding_format = encoding_format
    self._encoding_fn = encode_fn_map[encoding_format]

  def get_serialized_info(self):
    return {
        'encoded':
            tfds.features.TensorInfo(shape=(), dtype=tf.string),
        'height':
            tfds.features.TensorInfo(
                shape=(), dtype=tf.int64, default_value=-1),
        'width':
            tfds.features.TensorInfo(
                shape=(), dtype=tf.int64, default_value=-1),
        'channels':
            tfds.features.TensorInfo(
                shape=(), dtype=tf.int64, default_value=-1),
        'format':
            tfds.features.TensorInfo(
                shape=(), dtype=tf.string, default_value='png'),
    }

  def encode_example(self, image_np):
    encoded_image = self._encode_image(image_np)
    return {
        'encoded': encoded_image,
        'height': image_np.shape[0],
        'width': image_np.shape[1],
        'channels': image_np.shape[2] if image_np.ndim == 3 else 0,
        'format': self._encoding_format
    }

  def decode_example(self, example):
    image = tf.image.decode_image(
        example['encoded'], channels=None, dtype=self._dtype)
    if self._channels == 0:
      image = tf.squeeze(image, axis=-1)
    image.set_shape(self._shape)
    return image

  def _encode_image(self, image_np):
    """Returns image_np encoded as jpeg or png."""
    tfds.core.utils.assert_shape_match(image_np.shape, self._shape)
    image_tf = tf.convert_to_tensor(image_np)
    if image_np.ndim == 2:
      image_tf = tf.expand_dims(image_tf, axis=2)
    return self._encoding_fn(image_tf).numpy()


class Depth(Image):
  """Depth Image `FeatureConnector` for storing depth maps.

  Given a floating point depth image, the encoder internally stores the depth
  map as a uint16/uint8 PNG image (after scaling with a provided shift value).
  During decoding the shift value is divided back to return a floating point
  image. As expected this process is hidden from user, but depth map will loose
  some accuracy because of the quantization.

  Example:

  * In the DatasetInfo object:
    features=features.FeaturesDict({
        'depth': features.Depth(shift=1000.0, dtype=tf.float32),
    })

  * Internally stored as:
    {
      'depth/encoded': 'encoded depth string',
      'depth/width': image width,
      'depth/height': image height,
      'depth/channels': image channels,
      'depth/format': 'format string'
      'depth/shift': depth shift value.
    }

  * During generation:
    yield {
        'depth': np.random.uniform(high=5.0, size=(480, 640)).astype('f'),
    }

  * Decoding will return as dictionary of tensorflow tensors:
    {
      'depth': tf.Tensor(shape=(480, 640), dtype=tf.float32)
    }
  """

  def __init__(self,
               height=None,
               width=None,
               shift=1000.0,
               dtype=tf.float32,
               encoding_dtype=tf.uint16):
    if not dtype.is_floating:
      raise ValueError('Requires floating point type but got %s.' % dtype)
    super(Depth, self).__init__(
        shape=(height, width), encoding_format='png', dtype=encoding_dtype)
    self._shift = shift
    self._target_dtype = dtype
    self._encoding_dtype = encoding_dtype.as_numpy_dtype

  def get_serialized_info(self):
    serialized_info = super(Depth, self).get_serialized_info()
    serialized_info.update({
        'shift':
            tfds.features.TensorInfo(
                shape=(), dtype=tf.float32, default_value=1000.0)
    })
    return serialized_info

  def encode_example(self, depth_np):
    shifted_depth = (depth_np * self._shift).astype(self._encoding_dtype)
    encoded = super(Depth, self).encode_example(shifted_depth)
    encoded.update({'shift': self._shift})
    return encoded

  def decode_example(self, example):
    shifted_depth = super(Depth, self).decode_example(example)
    scale = tf.cast(1.0 / example['shift'], self._target_dtype)
    return tf.cast(shifted_depth, self._target_dtype) * scale


class Normal(Image):
  """Normal Image `FeatureConnector` for storing normal maps.

  Given a floating point normal image, the encoder internally stores the normal
  image as a 3 channel uint16/uint8 PNG image. The dtype of the encoded image is
  determined by the encoding_dtype argument.

  Example:

  * In the DatasetInfo object:
    features=features.FeaturesDict({
        'normal': features.Normal(dtype=tf.float32),
    })

  * Internally stored as:
    {
      'normal/encoded': 'encoded normal string',
      'normal/width': image width,
      'normal/height': image height,
      'normal/channels': image channels,
      'normal/format': 'format string'
    }

  * During generation:
    yield {
        'normal': np.random.uniform(high=1.0, size=(480,640, 3)).astype('f'),
    }

  * Decoding will return as dictionary of tensorflow tensors:
    {
      'normal': tf.Tensor(shape=(480, 640, 3), dtype=tf.float32)
    }
  """

  def __init__(self,
               height=None,
               width=None,
               dtype=tf.float32,
               encoding_dtype=tf.uint16):
    if not dtype.is_floating:
      raise ValueError('Requires floating point type but got %s.' % dtype)
    super(Normal, self).__init__(
        shape=(height, width, 3), encoding_format='png', dtype=encoding_dtype)
    self._target_dtype = dtype
    self._encoding_dtype = encoding_dtype.as_numpy_dtype
    self._scale = np.iinfo(self._encoding_dtype).max / 2.0

  def encode_example(self, normal_np):
    normal_discrete = ((normal_np + 1.0) * self._scale).astype(
        self._encoding_dtype)
    return super(Normal, self).encode_example(normal_discrete)

  def decode_example(self, example):
    normal_discrete = super(Normal, self).decode_example(example)
    normal = (tf.cast(normal_discrete, self._target_dtype) / self._scale) - 1.0
    return normal


class Unary(Image):
  """Unary `FeatureConnector` for storing multiclass image unary maps.

  This FeatureConnector is used to store multi-class probability maps (e.g.
  image unary for semantic segmentation). The data is stored internally as a set
  of PNG16 images.

  Given a dense, per-pixel, multi-class unary (probability) map as a tensor of
  shape (H, W, C), the encoder internally stores the unary as per channel uint16
  PNG images (after scaling the valued for [0, 1] to [0, 65,535]).

  Example:

  * In the DatasetInfo object:
    features=features.FeaturesDict({
        'unary': Unary(dtype=tf.float32),
    })

  * Internally stored as:
    {
      'unary/encoded': ['class0 PNG string', 'class1 PNG string', ...]
      'unary/width': unary width,
      'unary/height': unary height,
      'unary/channels': unary channels,
      'unary/format': 'format string'
    }

  * During generation:
    yield {
        'unary': softmax(np.random.rand(480, 640, 10).astype('f'), axis=2),
    }

  * Decoding will return as dictionary of tensorflow tensors:
    {
      'unary': tf.Tensor(shape=(480, 640), dtype=tf.float32)
    }
  """

  def __init__(self, shape=(None, None, None), dtype=tf.float32):
    if not dtype.is_floating:
      raise ValueError('Requires floating point type but got %s.' % dtype)
    super(Unary, self).__init__(
        shape=shape, encoding_format='png', dtype=tf.uint16)
    self._target_dtype = dtype

  def get_serialized_info(self):
    serialized_info = super(Unary, self).get_serialized_info()
    serialized_info['encoded'] = tfds.features.TensorInfo(
        shape=(None,), dtype=tf.string)
    return serialized_info

  def encode_example(self, unary_prob):
    scale = np.iinfo(np.uint16).max
    unary_scaled = (unary_prob * scale).astype(np.uint16)
    channels = unary_prob.shape[2]
    encoded = [self._encode_image(x) for x in np.dsplit(unary_scaled, channels)]
    return {
        'encoded': encoded,
        'height': unary_prob.shape[0],
        'width': unary_prob.shape[1],
        'channels': unary_prob.shape[2],
        'format': self._encoding_format
    }

  def decode_example(self, example):
    enoded = example['encoded']
    unary_slices = [
        tf.squeeze(tf.image.decode_image(x, dtype=self._dtype), axis=2)
        for x in enoded
    ]
    unary = np.stack(unary_slices, axis=2)
    scale = tf.cast(1.0 / np.iinfo(np.uint16).max, self._target_dtype)
    unary = tf.cast(unary, self._target_dtype) * scale
    unary.set_shape(self._shape)
    return unary
