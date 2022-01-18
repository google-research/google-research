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

"""Data proprocessing utilities.
"""

import random
import tensorflow as tf


def img_to_grayscale(image, keep_dims=True):
  image = tf.image.rgb_to_grayscale(image)
  if keep_dims:
    image = tf.tile(image, [1, 1, 3])
  return image


def random_brightness(image, lower, upper, seed=None):
  """Adjust brightness of image by multiplying with random factor.

  Args:
    image: Input image.
    lower: Lower bound for random factor (clipped to be at least zero).
    upper: Upper bound for random factor.
    seed: Random seed to use.

  Returns:
    Image with changed brightness.
  """
  lower = tf.math.maximum(lower, 0)
  rand_factor = tf.random.uniform([], lower, upper, seed=seed)
  return image * rand_factor


def crop_and_resize(image, crop_ratio, seed=None):
  """Randomly crops image and resizes it to original size.

  Args:
    image: Image tensor to be cropped.
    crop_ratio: Relative size of cropped image to original.
    seed: Random seed to use.

  Returns:
    Image of same size as input, randomly cropped and resized.
  """
  height, width = image.shape[0], image.shape[1]
  aspect_ratio = width / height
  aspect_ratio_range = (3/4 * aspect_ratio, 4/3 * aspect_ratio)

  bb_begin, bb_size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=tf.constant([0, 0, 1, 1],
                                 dtype=tf.float32,
                                 shape=[1, 1, 4]),
      min_object_covered=crop_ratio,
      aspect_ratio_range=aspect_ratio_range,
      area_range=(0.08, 1),
      seed=seed)

  image = tf.slice(image, bb_begin, bb_size)
  image = tf.image.resize([image], [height, width], method="bicubic")[0]
  return tf.cast(image, image.dtype)


def jitter_colors(image, jitter_strength=1.0, seed=None):
  """Randomly distorts brightness, contrast, saturation and hue of an image.

  Args:
    image: Image to be processed.
    jitter_strength: Strength of the color jittering in range [0,1].
    seed: Random seed to use.

  Returns:
    The color-jittered image.
  """

  if jitter_strength == 0:
    return image

  factor_brightness = 0.8
  factor_contrast = factor_brightness
  factor_saturation = factor_brightness
  factor_hue = 0.2

  def _apply_distort(i, image):
    def _distort_brightness(image):
      brightness = factor_brightness * jitter_strength
      return random_brightness(
          image, 1 - brightness, 1 + brightness, seed=seed)

    def _distort_contrast(image):
      contrast = factor_contrast * jitter_strength
      return tf.image.random_contrast(
          image, 1 - contrast, 1 + contrast, seed=seed)

    def _distort_saturation(image):
      saturation = factor_saturation * jitter_strength
      return tf.image.random_saturation(
          image, 1 - saturation, 1 + saturation, seed=seed)

    def _distort_hue(image):
      hue = factor_hue * jitter_strength
      return tf.image.random_hue(image, hue, seed=seed)

    if i == 0:
      return _distort_brightness(image)
    elif i == 1:
      return _distort_contrast(image)
    elif i == 2:
      return _distort_saturation(image)
    elif i == 3:
      return _distort_hue(image)

  rand_perm = [*range(4)]
  random.shuffle(rand_perm)
  for i in range(4):
    image = _apply_distort(rand_perm[i], image)
    image = tf.clip_by_value(image, 0, 1)
  return image
