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

# Lint as: python3
"""Augment operators."""

import functools
import random

import tensorflow as tf

import deep_representation_one_class.simclr.data_util as simclr_ops


def base_augment(is_training=True, **kwargs):
  """Applies base (random resize and crop) augmentation."""
  size, crop_size = kwargs['size'], int(0.875 * kwargs['size'])
  if is_training:
    return [
        ('resize', {
            'size': size
        }),
        ('crop', {
            'size': crop_size
        }),
    ]
  return [('resize', {'size': size})]


def resize_augment(is_training=True, **kwargs):
  """Applies resize augmentation."""
  del is_training
  size = kwargs['size']
  return [('resize', {'size': size})]


def crop_augment(is_training=True, **kwargs):
  """Applies resize and random crop augmentation."""
  size, crop_size = kwargs['size'], kwargs['crop_size']
  if is_training:
    return [
        ('resize', {
            'size': size
        }),
        ('crop', {
            'size': crop_size
        }),
    ]
  return [('resize', {'size': size})]


def shift_augment(is_training=True, **kwargs):
  """Applies resize and random shift augmentation."""
  size, pad_size = kwargs['size'], int(0.125 * kwargs['size'])
  if is_training:
    return [
        ('resize', {
            'size': size
        }),
        ('shift', {
            'pad': pad_size
        }),
    ]
  return [('resize', {'size': size})]


def crop_and_resize_augment(is_training=True, **kwargs):
  """Applies random crop and resize augmentation."""
  size = kwargs['size']
  min_scale = kwargs['min_scale'] if 'min_scale' in kwargs else 0.5
  if is_training:
    return [
        ('crop_and_resize', {
            'size': size,
            'min_scale': min_scale
        }),
    ]
  return [('resize', {'size': size})]


def hflip_augment(is_training=True, **kwargs):
  """Applies random horizontal flip."""
  del kwargs
  if is_training:
    return [('hflip', {})]
  return []


def vflip_augment(is_training=True, **kwargs):
  """Applies random vertical flip."""
  del kwargs
  if is_training:
    return [('vflip', {})]
  return []


def rotate90_augment(is_training=True, **kwargs):
  """Applies rotation by 90 degree."""
  del kwargs
  if is_training:
    return [('rotate90', {})]
  return []


def rotate180_augment(is_training=True, **kwargs):
  """Applies rotation by 180 degree."""
  del kwargs
  if is_training:
    return [('rotate180', {})]
  return []


def rotate270_augment(is_training=True, **kwargs):
  """Applies rotation by 270 degree."""
  del kwargs
  if is_training:
    return [('rotate270', {})]
  return []


def jitter_augment(is_training=True, **kwargs):
  """Applies random color jitter augmentation."""
  if is_training:
    brightness = kwargs['brightness'] if 'brightness' in kwargs else 0.125
    contrast = kwargs['contrast'] if 'contrast' in kwargs else 0.4
    saturation = kwargs['saturation'] if 'saturation' in kwargs else 0.4
    hue = kwargs['hue'] if 'hue' in kwargs else 0
    return [('jitter', {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'hue': hue
    })]
  return []


def gray_augment(is_training=True, **kwargs):
  """Applies random grayscale augmentation."""
  if is_training:
    prob = kwargs['prob'] if 'prob' in kwargs else 0.2
    return [('gray', {'prob': prob})]
  return []


def blur_augment(is_training=True, **kwargs):
  """Applies random blur augmentation."""
  if is_training:
    prob = kwargs['prob'] if 'prob' in kwargs else 0.5
    return [('blur', {'prob': prob})]
  return []


class Resize(object):
  """Applies resize."""

  def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR):
    self.size = self._check_input(size)
    self.method = method

  def _check_input(self, size):
    if isinstance(size, int):
      size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 1:
      size = size * 2
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    return tf.image.resize(
        image, self.size, method=self.method) if is_training else image


class RandomCrop(object):
  """Applies random crop without padding."""

  def __init__(self, size):
    self.size = self._check_input(size)

  def _check_input(self, size):
    """Checks input size is valid."""
    if isinstance(size, int):
      size = (size, size, 3)
    elif isinstance(size, (list, tuple)):
      if len(size) == 1:
        size = tuple(size) * 2 + (3,)
      elif len(size) == 2:
        size = tuple(size) + (3,)
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    return tf.image.random_crop(image, self.size) if is_training else image


class RandomShift(object):
  """Applies random shift."""

  def __init__(self, pad):
    self.pad = self._check_input(pad)

  def _check_input(self, size):
    """Checks input size is valid."""
    if isinstance(size, int):
      size = (size, size)
    elif isinstance(size, (list, tuple)):
      if len(size) == 1:
        size = tuple(size) * 2
      elif len(size) > 2:
        size = tuple(size[:2])
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    if is_training:
      img_size = image.shape[-3:]
      image = tf.pad(
          image, [[self.pad[0]] * 2, [self.pad[1]] * 2, [0] * 2],
          mode='REFLECT')
      image = tf.image.random_crop(image, img_size)
    return image


class RandomCropAndResize(object):
  """Applies random crop and resize."""

  def __init__(self, size, min_scale=0.5):
    self.min_scale = min_scale
    self.size = self._check_input(size)

  def _check_input(self, size):
    """Checks input size is valid."""
    if isinstance(size, int):
      size = (size, size)
    elif isinstance(size, (list, tuple)) and len(size) == 1:
      size = size * 2
    else:
      raise TypeError('size must be an integer or list/tuple of integers')
    return size

  def __call__(self, image, is_training=True):
    if is_training:
      # crop and resize
      width = tf.random.uniform(
          shape=[],
          minval=tf.cast(image.shape[0] * self.min_scale, dtype=tf.int32),
          maxval=image.shape[0] + 1,
          dtype=tf.int32)
      size = (width, tf.minimum(width, image.shape[1]), image.shape[2])
      image = tf.image.random_crop(image, size)
      image = tf.image.resize(image, size=self.size)
    return image


class RandomFlipLeftRight(object):
  """Applies random horizontal flip."""

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.random_flip_left_right(image) if is_training else image


class RandomFlipUpDown(object):
  """Applies random vertical flip."""

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.random_flip_up_down(image) if is_training else image


class Rotate90(object):
  """Applies rotation by 90 degree."""

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.rot90(image, k=1) if is_training else image


class Rotate180(object):
  """Applies rotation by 180 degree."""

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.rot90(image, k=2) if is_training else image


class Rotate270(object):
  """Applies rotation by 270 degree."""

  def __init__(self):
    pass

  def __call__(self, image, is_training=True):
    return tf.image.rot90(image, k=3) if is_training else image


class ColorJitter(object):
  """Applies random color jittering."""

  def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
    self.brightness = self._check_input(brightness)
    self.contrast = self._check_input(contrast, center=1)
    self.saturation = self._check_input(saturation, center=1)
    self.hue = self._check_input(hue, bound=0.5)

  def _check_input(self, value, center=None, bound=None):
    if bound is not None:
      value = min(value, bound)
    if center is not None:
      value = [center - value, center + value]
      if value[0] == value[1] == center:
        return None
    elif value == 0:
      return None
    return value

  def _get_transforms(self):
    """Gets a randomly ordered sequence of color transformation."""
    transforms = []
    if self.brightness is not None:
      transforms.append(
          functools.partial(
              tf.image.random_brightness, max_delta=self.brightness))
    if self.contrast is not None:
      transforms.append(
          functools.partial(
              tf.image.random_contrast,
              lower=self.contrast[0],
              upper=self.contrast[1]))
    if self.saturation is not None:
      transforms.append(
          functools.partial(
              tf.image.random_saturation,
              lower=self.saturation[0],
              upper=self.saturation[1]))
    if self.hue is not None:
      transforms.append(
          functools.partial(tf.image.random_hue, max_delta=self.hue))
    random.shuffle(transforms)
    return transforms

  def __call__(self, image, is_training=True):
    if not is_training:
      return image
    num_concat = image.shape[2] // 3
    if num_concat == 1:
      for transform in self._get_transforms():
        image = transform(image)
    else:
      images = tf.split(image, num_concat, axis=-1)
      for transform in self._get_transforms():
        images = [transform(image) for image in images]
      image = tf.concat(images, axis=-1)
    return image


class RandomGrayScale(object):
  """Applies random grayscale augmentation."""

  def __init__(self, prob):
    self.prob = prob

  def __call__(self, image, is_training=True):
    return tf.cond(
        tf.random.uniform([]) > self.prob, lambda: image,
        lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image)))


class RandomBlur(object):
  """Applies random blur augmentation."""

  def __init__(self, prob=0.5):
    self.prob = prob

  def __call__(self, image, is_training=True):
    if is_training:
      return image
    return simclr_ops.random_blur(
        image, image.shape[0], image.shape[1], p=self.prob)
