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

"""General utility functions."""
_INTERNAL = False  # pylint: disable=g-statement-before-imports

import os.path

if not _INTERNAL:
  import cv2  # pylint: disable=g-import-not-at-top
import numpy as np
import skimage
import skimage.morphology
import tensorflow as tf
from tensorflow_addons import image as tfa_image
from tensorflow_addons.utils import types as tfa_types

# Small number added to near-zero quantities to avoid numerical instability.
_EPS = 1e-7


def _gaussian_kernel(kernel_size, sigma, n_channels,
                     dtype):
  x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
  g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
  g_norm2d = tf.pow(tf.reduce_sum(g), 2)
  g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
  g_kernel = tf.expand_dims(g_kernel, axis=-1)
  return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(im, sigma):
  """Applies a Gaussian blur to an image tensor."""
  blur = _gaussian_kernel(21, sigma, im.shape[-1], im.dtype)
  im = tf.nn.depthwise_conv2d(im, blur, [1, 1, 1, 1], 'SAME')
  return im


def remove_flare(combined, flare, gamma = 2.2):
  """Subtracts flare from the image in linear space.

  Args:
    combined: gamma-encoded image of a flare-polluted scene.
    flare: gamma-encoded image of the flare.
    gamma: [value in linear domain] = [gamma-encoded value] ^ gamma.

  Returns:
    Gamma-encoded flare-free scene.
  """
  # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
  # gamma < 1.
  combined = tf.clip_by_value(combined, _EPS, 1.0)
  flare = tf.clip_by_value(flare, _EPS, 1.0)

  combined_linear = tf.pow(combined, gamma)
  flare_linear = tf.pow(flare, gamma)

  scene_linear = combined_linear - flare_linear
  # Avoid zero. Otherwise, the gradient of pow() below will be undefined when
  # gamma > 1.
  scene_linear = tf.clip_by_value(scene_linear, _EPS, 1.0)
  scene = tf.pow(scene_linear, 1.0 / gamma)
  return scene


def quantize_8(image):
  """Converts and quantizes an image to 2^8 discrete levels in [0, 1]."""
  q8 = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
  return tf.cast(q8, tf.float32) * (1.0 / 255.0)


def write_image(image, path, overwrite = True):
  """Writes an image represented by a tensor to a PNG or JPG file."""
  if not os.path.basename(path):
    raise ValueError(f'The given path doesn\'t represent a file: {path}')
  if tf.io.gfile.exists(path):
    if tf.io.gfile.isdir(path):
      raise ValueError(f'The given path is an existing directory: {path}')
    if not overwrite:
      print(f'Not overwriting an existing file at {path}')
      return False
    tf.io.gfile.remove(path)
  else:
    tf.io.gfile.makedirs(os.path.dirname(path))

  image_u8 = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
  if path.lower().endswith('.png'):
    encoded = tf.io.encode_png(image_u8)
  elif path.lower().endswith('.jpg') or path.lower().endswith('.jpeg'):
    encoded = tf.io.encode_jpeg(image_u8, progressive=True)
  else:
    raise ValueError(f'Unsupported image format: {os.path.basename(path)}')
  with tf.io.gfile.GFile(path, 'wb') as f:
    f.write(encoded.numpy())
  return True


def _center_transform(t, height, width):
  """Modifies a homography such that the origin is at the image center.

  The transform matrices are represented using 8-vectors, following the
  `tensorflow_addons,image` package.

  Args:
    t: A [8]- or [B, 8]-tensor representing projective transform(s) defined
      relative to the origin (0, 0).
    height: Image height, in pixels.
    width: Image width, in pixels.

  Returns:
    The same transform(s), but applied relative to the image center (width / 2,
    height / 2) instead.
  """
  center_to_origin = tfa_image.translations_to_projective_transforms(
      [-width / 2, -height / 2])
  origin_to_center = tfa_image.translations_to_projective_transforms(
      [width / 2, height / 2])
  t = tfa_image.compose_transforms([center_to_origin, t, origin_to_center])
  return t


def scales_to_projective_transforms(scales, height,
                                    width):
  """Returns scaling transform matrices for a batched input.

  The scaling is applied relative to the image center, instead of (0, 0).

  Args:
    scales: 2-element tensor [sx, sy], or a [B, 2]-tensor reprenting a batch of
      such inputs. `sx` and `sy` are the scaling ratio in x and y respectively.
    height: Image height, in pixels.
    width: Image width, in pixels.

  Returns:
    A [B, 8]-tensor representing the transform that can be passed to
    `tensorflow_addons.image.transform`.
  """
  scales = tf.convert_to_tensor(scales)
  if tf.rank(scales) == 1:
    scales = scales[None, :]
  scales_x = tf.reshape(scales[:, 0], (-1, 1))
  scales_y = tf.reshape(scales[:, 1], (-1, 1))
  zeros = tf.zeros_like(scales_x)
  transform = tf.concat(
      [scales_x, zeros, zeros, zeros, scales_y, zeros, zeros, zeros], axis=-1)
  return _center_transform(transform, height, width)


def shears_to_projective_transforms(shears, height,
                                    width):
  """Returns shear transform matrices for a batched input.

  The shear is applied relative to the image center, instead of (0, 0).

  Args:
    shears: 2-element tensor [sx, sy], or a [B, 2]-tensor reprenting a batch of
      such inputs. `sx` and `sy` are the shear angle (in radians) in x and y
      respectively.
    height: Image height, in pixels.
    width: Image width, in pixels.

  Returns:
    A [B, 8]-tensor representing the transform that can be passed to
    `tensorflow_addons.image.transform`.
  """
  shears = tf.convert_to_tensor(shears)
  if tf.rank(shears) == 1:
    shears = shears[None, :]
  shears_x = tf.reshape(tf.tan(shears[:, 0]), (-1, 1))
  shears_y = tf.reshape(tf.tan(shears[:, 1]), (-1, 1))
  ones = tf.ones_like(shears_x)
  zeros = tf.zeros_like(shears_x)
  transform = tf.concat(
      [ones, shears_x, zeros, shears_y, ones, zeros, zeros, zeros], axis=-1)
  return _center_transform(transform, height, width)


def apply_affine_transform(image,
                           rotation = 0.,
                           shift_x = 0.,
                           shift_y = 0.,
                           shear_x = 0.,
                           shear_y = 0.,
                           scale_x = 1.,
                           scale_y = 1.,
                           interpolation = 'bilinear'):
  """Applies affine transform(s) on the input images.

  The rotation, shear, and scaling transforms are applied relative to the image
  center, instead of (0, 0). The transform parameters can either be scalars
  (applied to all images in the batch) or [B]-tensors (applied to each image
  individually).

  Args:
    image: Input images in [B, H, W, C] format.
    rotation: Rotation angle in radians. Positive value rotates the image
      counter-clockwise.
    shift_x: Translation in x direction, in pixels.
    shift_y: Translation in y direction, in pixels.
    shear_x: Shear angle (radians) in x direction.
    shear_y: Shear angle (radians) in y direction.
    scale_x: Scaling factor in x direction.
    scale_y: Scaling factor in y direction.
    interpolation: Interpolation mode. Supported values: 'nearest', 'bilinear'.

  Returns:
    The transformed images in [B, H, W, C] format.
  """
  height, width = image.shape[1:3]

  rotation = tfa_image.angles_to_projective_transforms(rotation, height, width)
  shear = shears_to_projective_transforms([shear_x, shear_y], height, width)
  scaling = scales_to_projective_transforms([scale_x, scale_y], height, width)
  translation = tfa_image.translations_to_projective_transforms(
      [shift_x, shift_y])

  t = tfa_image.compose_transforms([rotation, shear, scaling, translation])
  transformed = tfa_image.transform(image, t, interpolation=interpolation)

  return transformed


def get_highlight_mask(im,
                       threshold = 0.99,
                       dtype = tf.float32):
  """Returns a binary mask indicating the saturated regions in the input image.

  Args:
    im: Image tensor with shape [H, W, C], or [B, H, W, C].
    threshold: A pixel is considered saturated if its channel-averaged intensity
      is above this value.
    dtype: Expected output data type.

  Returns:
    A `dtype` tensor with shape [H, W, 1] or [B, H, W, 1].
  """
  binary_mask = tf.reduce_mean(im, axis=-1, keepdims=True) > threshold
  mask = tf.cast(binary_mask, dtype)
  return mask


def refine_mask(mask, morph_size = 0.01):
  """Refines a mask by applying mophological operations.

  Args:
    mask: A float array of shape [H, W] or [B, H, W].
    morph_size: Size of the morphological kernel relative to the long side of
      the image.

  Returns:
    Refined mask of shape [H, W] or [B, H, W].
  """
  mask_size = max(np.shape(mask))
  kernel_radius = .5 * morph_size * mask_size
  kernel = skimage.morphology.disk(np.ceil(kernel_radius))
  opened = skimage.morphology.binary_opening(mask, kernel)
  return opened


def _create_disk_kernel(kernel_size):
  x = np.arange(kernel_size) - (kernel_size - 1) / 2
  xx, yy = np.meshgrid(x, x)
  rr = np.sqrt(xx**2 + yy**2)
  kernel = np.float32(rr <= np.max(x)) + _EPS
  kernel = kernel / np.sum(kernel)
  return kernel


def blend_light_source(scene_input, scene_pred):
  """Adds suspected light source in the input to the flare-free image."""
  binary_mask = get_highlight_mask(scene_input, dtype=tf.bool).numpy()
  binary_mask = np.squeeze(binary_mask, axis=-1)
  binary_mask = refine_mask(binary_mask)

  labeled = skimage.measure.label(binary_mask)
  properties = skimage.measure.regionprops(labeled)
  max_diameter = 0
  for p in properties:
    max_diameter = max(max_diameter, p['equivalent_diameter'])

  mask = np.float32(binary_mask)

  kernel_size = round(1.5 * max_diameter)
  if kernel_size > 0:
    kernel = _create_disk_kernel(kernel_size)
    mask = cv2.filter2D(mask, -1, kernel)
    mask = np.clip(mask * 3.0, 0.0, 1.0)
    mask_rgb = np.stack([mask] * 3, axis=-1)
  else:
    mask_rgb = 0

  blend = scene_input * mask_rgb + scene_pred * (1 - mask_rgb)

  return blend


def normalize_white_balance(im):
  """Normalizes the RGB channels so the image appears neutral in color.

  Args:
    im: Image tensor with shape [H, W, C], or [B, H, W, C].

  Returns:
    Image(s) with equal channel mean. (The channel mean may be different across
    images for batched input.)
  """
  channel_mean = tf.reduce_mean(im, axis=(-3, -2), keepdims=True)
  max_of_mean = tf.reduce_max(channel_mean, axis=(-3, -2, -1), keepdims=True)
  normalized = max_of_mean * im / (channel_mean + _EPS)
  return normalized


def remove_background(im):
  """Removes the DC component in the background.

  Args:
    im: Image tensor with shape [H, W, C], or [B, H, W, C].

  Returns:
    Image(s) with DC background removed. The white level (maximum pixel value)
    stays the same.
  """
  im_min = tf.reduce_min(im, axis=(-3, -2), keepdims=True)
  im_max = tf.reduce_max(im, axis=(-3, -2), keepdims=True)
  return (im - im_min) * im_max / (im_max - im_min + _EPS)
