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

"""Image preprocessing."""
import abc
import functools
from typing import Optional, Tuple

import attr
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

from supcon import enums
from official.legacy.image_classification import augment

# Defaults for ImageNet.
IMAGE_SIZE = 224
CROP_PADDING = 32


@attr.s()
class DatasetOptions:
  """Parameters that are dataset-specific and not user-configurable."""
  # Optional tuple of 2 numpy arrays or Tensors that can be broadcast to the
  # preprocessed image shape, representing the mean and standard deviation of
  # the images in the dataset. If not None, these are used to whiten the image
  # after augmentation is applied, by first subtracting the mean and then
  # dividing by the standard deviation.
  image_mean_std: Optional[Tuple[np.ndarray,
                                 np.ndarray]] = attr.ib(default=None)
  # Whether the inputs should be decoded prior to preprocessing.
  decode_input: bool = attr.ib(default=True)


def _validate_image_dimensions(image):
  """Verify that the image dimensions are valid.

  Checks that the image has 3 color channels and is 3-dimensional. Updates
  the static shape to have 3 channels, if the channel dimension was previously
  unknown.

  Args:
    image: `Tensor` of raw image pixel data that is expected to be 3-dimensional
      and have 3 channels in the last dimension. Can be any numeric type.

  Returns:
    The input image with the final dimension statically set to 3 if it was
    previously None.

  Raises:
    tf.errors.InvalidArgumentError at runtime if the image is not 3-dimensional
      or does not have 3 channels in the last dimension.
  """
  with tf.name_scope('validate_image_dims'):
    image = tf.ensure_shape(image, [None, None, 3])
    return image


def _center_crop_window(image_shape, crop_dim=None, crop_frac=None):
  """Computes a centered square crop window.

  Args:
    image_shape: The shape of the image, expressed as a Tensor of shape [3], an
      iterable of length 3, or a tf.Shape with rank 3.
    crop_dim: The scalar side length of the cropped square. Only one of crop_dim
      and crop_frac should be set.
    crop_frac: The fraction of the minimum spatial dimension that should be used
      as the side length of the cropped square. Only one of crop_dim and
      crop_frac should be set.

  Returns:
    A Tensor of shape [6], representing the crop box in the format
    [offset_height, offset_width, offset_channel, crop_dim, crop_dim, channels].
    `offset_channel` is always 0.

  Raises:
    ValueError if both or neither of crop_frac and crop_dim are set.
  """
  if not (crop_frac is None) ^ (crop_dim is None):
    raise ValueError('Exactly one of crop_frac or crop_dim must be passed.')
  with tf.name_scope('center_crop_window'):
    if crop_frac is not None:
      crop_dim = tf.cast(
          crop_frac *
          tf.cast(tf.minimum(image_shape[0], image_shape[1]), tf.float32),
          tf.int32)
    offset_height = (image_shape[0] - crop_dim + 1) // 2
    offset_width = (image_shape[1] - crop_dim + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, 0, crop_dim, crop_dim, image_shape[2]])
    return crop_window


def _distorted_crop_window(image_shape,
                           min_object_covered=0.1,
                           aspect_ratio_range=(0.75, 1.33),
                           area_range=(0.08, 1.0),
                           max_attempts=100):
  """Computes a sampled distorted crop window from an input image shape.

  Calls into `tf.image.sample_distorted_bounding_box`, using the entire image as
  the bounding box. This can theoretically fail, in which case, we fall back to
  a deterministic center square crop.

  Args:
    image_shape: The shape of the image, expressed as a Tensor of shape [3], an
      iterable of length 3, or a tf.Shape with rank 3.
    min_object_covered: See `tf.image.sample_distorted_bounding_box`.
    aspect_ratio_range: See `tf.image.sample_distorted_bounding_box`.
    area_range: See `tf.image.sample_distorted_bounding_box`.
    max_attempts: See `tf.image.sample_distorted_bounding_box`.

  Returns:
    A Tensor of shape [6], representing the crop box in the format
    [offset_height, offset_width, offset_channel, crop_dim, crop_dim, channels].
    `offset_channel` is always 0.
  """
  with tf.name_scope('distorted_crop_window'):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=tf.zeros(shape=[1, 0, 4]),
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window_params = [
        offset_y, offset_x, 0, target_height, target_width, image_shape[2]
    ]
    # sample_distorted_bounding_box can fail, in which case it returns the input
    # image dimensions. In case of failure, fall back to central crop.
    success = tf.logical_or(
        tf.not_equal(target_height, image_shape[0]),
        tf.not_equal(target_width, image_shape[1]))
    crop_window = tf.cond(
        success, lambda: tf.stack(crop_window_params),
        lambda: _center_crop_window(image_shape, crop_frac=1.))
    return crop_window


def _resize_image(image, shape):
  """Resizes an image to a target shape.

  Ensures that the resulting image has the same dtype as the input.

  Args:
    image: Image Tensor with 3 or 4 dimensions and any numeric dtype.
    shape: The target shape to resize `image` to.

  Returns:
    The resized image, as a Tensor with the same dtype as `image`.
  """
  with tf.name_scope('resize_image'):
    resized = tf.image.resize(
        image, shape, method=tf.image.ResizeMethod.BILINEAR)
    # Resize casts to tf.float32, so cast back.
    return tf.cast(resized, image.dtype)


def _resize_to_min_dim(image, target_min_dim):
  """Resizes an image so that its minimum dimension is a specific size.

  Args:
    image: The image to resize, as a 3-dimensional Tensor with any numeric
      dtype.
    target_min_dim: The desired length of the minimum spatial dimension after
      resize.

  Returns:
    The resized image, as a Tensor with the same dtype as `image`.
  """
  with tf.name_scope('resize_to_min_dim'):
    original_shape = tf.cast(tf.shape(image)[:2], tf.float32)
    is_landscape = tf.less_equal(original_shape[0], original_shape[1])
    original_min_dim = tf.cond(is_landscape, lambda: original_shape[0],
                               lambda: original_shape[1])
    scale_factor = tf.cast(target_min_dim, tf.float32) / original_min_dim
    target_shape = tf.cast(scale_factor * original_shape, tf.int32)
    return _resize_image(image, target_shape)


def _decode_and_maybe_crop_image(image_bytes, crop_window=None):
  """Decodes an image and optionally crops it.

  It's useful to crop this way rather than decoding and then cropping, since we
  can read and decode just the cropped data. This provides substantial speedup
  over decoding the full image and then cropping.

  Args:
    image_bytes: The encoded JPEG image bytes as a string Tensor.
    crop_window: A crop window expressed as a Tensor of shape [4] in the format
      [offset_height, offset_width, crop_height, crop_width]. This is assumed to
      apply to all channels uniformly. If `crop_window` is None then the
      `image_bytes` are decoded without applying any cropping.

  Returns:
    The (possibly cropped) image as a 3D Tensor [height, width, channels] with
    dtype tf.uint8.
  """
  with tf.name_scope('decode_and_maybe_crop_image'):
    if crop_window is not None:
      image = tf.image.decode_and_crop_jpeg(
          image_bytes, crop_window, channels=3)
    else:
      image = tf.image.decode_image(image_bytes, channels=3)
    return _validate_image_dimensions(image)


def _convert_3d_crop_window_to_2d(crop_window):
  """Converts a 3D crop window to a 2D crop window.

  Extracts just the spatial parameters of the crop window and assumes that those
  apply uniformly across all channels.

  Args:
    crop_window: A 3D crop window, expressed as a Tensor in the format
      [offset_height, offset_width, offset_channel, crop_height, crop_width,
      crop_channels].

  Returns:
    A 2D crop window as a Tensor in the format [offset_height, offset_width,
    crop_height, crop_width].
  """
  with tf.name_scope('3d_crop_window_to_2d'):
    return tf.gather(crop_window, [0, 1, 3, 4])


def _crop_to_square(image,
                    decode_image=False,
                    side_length=IMAGE_SIZE,
                    crop_padding=CROP_PADDING,
                    area_range=(0.08, 1.0),
                    is_training=True,
                    resize_only=False,
                    eval_crop_method=enums.EvalCropMethod.RESIZE_THEN_CROP):
  """Produces a (possibly distorted) square crop of an image.

  Given an input image, either as an encoded bytes string or a decoded image
  Tensor, produces a square version of it with the desired side length, using a
  combination of cropping and resizing.

  If `resize_only` is True, simply resize the image to be
  `side_length`x`side_length`, possibly distorting it if the original image is
  not square.

  If `is_training` is True, then sample a random box to crop from the image and
  then resize the result to be `side_length`x`side_length`.

  If `is_training` is False then we follow `eval_crop_method` to determine the
  strategy of cropping and resizing. Generally the approach is to end up with a
  center crop of size `side_length`x`side_length` taken from the image resized
  to have a minimum dimension of `side_length` + `crop_padding`. By setting
  eval_crop_method appropriately, this can be accomplished by first resizing and
  then cropping, first cropping and then resizing, or a less common approach of
  cropping the central `side_length`/(`side_length`+`crop_padding`) pixels in
  each dimension followed by resizing (and distorting) to
  `side_length`x`side_length`.

  If `decode_image` is True (i.e., `image` is an encoded jpeg image string),
  when possible we crop before decoding, which can provide substantial speedups.

  Args:
    image: An image represented either as a 3D Tensor with any numeric DType or
      else as an encoded jpeg image string.
    decode_image: Whether `image` is an encoded jpeg image string or not.
    side_length: The side length, in both spatial dimentions, of the output
      image.
    crop_padding: When `is_training` is False, this determines how much padding
      to apply around the central square crop.
    area_range: List of floats. The cropped area of the image must contain a
      fraction of the supplied image within this range. Only relevant when
      `is_training` is True and `resize_only` is False.
    is_training: Whether this should operate in training (non-deterministic
      random crop window) or eval (deterministic central crop window) mode.
    resize_only: Whether to just resize the image to the target `side_length`
      without performing any cropping. This is likely to distort the image.
    eval_crop_method: The strategy for obtaining the desired square crop in eval
      mode. See EvalCropMethod for valid values.

  Returns:
    An image Tensor of shape [`side_length`, `side_length`, 3]. If `image` was
    provided then the output has the same dtype as `image`. If `image_bytes` was
    provided then the output dtype is tf.uint8.

  Raises:
    ValueError: If both or neither of `image` and `image_bytes` was passed.
  """
  with tf.name_scope('crop_to_square'):
    if not decode_image:
      image = _validate_image_dimensions(image)

    if resize_only:
      if decode_image:
        image = _decode_and_maybe_crop_image(image)
      resized = _resize_image(image, (side_length, side_length))
      return tf.ensure_shape(resized, [side_length, side_length, 3])

    image_shape = (
        tf.shape(image)
        if not decode_image else tf.image.extract_jpeg_shape(image))
    if is_training:
      # During training, always crop then resize.
      crop_window = _distorted_crop_window(image_shape, area_range=area_range)
      if decode_image:
        cropped = _decode_and_maybe_crop_image(
            image, _convert_3d_crop_window_to_2d(crop_window))
      else:
        cropped = tf.slice(image, crop_window[:3], crop_window[3:])
      resized = _resize_image(cropped, [side_length, side_length])
      return tf.ensure_shape(resized, [side_length, side_length, 3])
    else:
      # For eval, the ordering depends on eval_crop_method.
      crop_frac = (side_length / (side_length + crop_padding))
      if eval_crop_method == enums.EvalCropMethod.RESIZE_THEN_CROP:
        if decode_image:
          image = _decode_and_maybe_crop_image(image)
        resize_dim = side_length + crop_padding
        resized = _resize_to_min_dim(image, resize_dim)
        crop_window = _center_crop_window(
            tf.shape(resized), crop_dim=side_length)
        cropped = tf.slice(resized, crop_window[:3], crop_window[3:])
        return tf.ensure_shape(cropped, [side_length, side_length, 3])
      elif eval_crop_method == enums.EvalCropMethod.CROP_THEN_RESIZE:
        crop_window = _center_crop_window(image_shape, crop_frac=crop_frac)
        if decode_image:
          cropped = _decode_and_maybe_crop_image(
              image, _convert_3d_crop_window_to_2d(crop_window))
        else:
          cropped = tf.slice(image, crop_window[:3], crop_window[3:])
        resized = _resize_image(cropped, [side_length, side_length])
        return tf.ensure_shape(resized, [side_length, side_length, 3])
      elif eval_crop_method == enums.EvalCropMethod.CROP_THEN_DISTORT:
        if decode_image:
          image = _decode_and_maybe_crop_image(image)
        # Note that tf.image.central_crop does not produce a square crop. It
        # preserves the input aspect ratio.
        cropped = tf.image.central_crop(image, central_fraction=crop_frac)
        resized = _resize_image(cropped, [side_length, side_length])
        return tf.ensure_shape(resized, [side_length, side_length, 3])
      elif eval_crop_method == enums.EvalCropMethod.IDENTITY:
        if decode_image:
          image = _decode_and_maybe_crop_image(image)
        return tf.ensure_shape(image, [side_length, side_length, 3])


def _flip(image):
  """Apply a horizontal flip with 50% probability."""
  with tf.name_scope('random_flip'):
    image = tf.image.random_flip_left_right(image)
    return image


def _gaussian_blur(image, side_length, padding='SAME'):
  """Blurs the given image with separable convolution.

  Blurring with a 2D Gaussian kernel is equivalent to blurring with two 1D
  kernels. This is known as a separable convolution, and is significantly
  faster than using a 2D convolution [O(n^2) -> O(n)].

  Args:
    image: Tensor of shape either [height, width, channels] or [batch_size,
      height, width, channels] and float dtype to blur.
    side_length: A python integer. The length, in pixels, of the height and
      width dimensions of `image`.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    The blurred image.
  """
  with tf.name_scope('gaussian_blur'):
    allowed_image_dtypes = [tf.float16, tf.float32, tf.bfloat16]
    assert (image.dtype in allowed_image_dtypes), (
        f'Tensor dtype must be float. Was {image.dtype}.')

    kernel_size = side_length // 10
    radius = kernel_size // 2
    kernel_size = radius * 2 + 1
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    blur_filter = tf.exp(-tf.pow(x, 2.0) / (2.0 * tf.pow(sigma, 2.0)))
    blur_filter /= tf.reduce_sum(blur_filter)
    blur_filter = tf.cast(blur_filter, image.dtype)

    # One vertical and one horizontal filter.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
    num_channels = tf.shape(image)[-1]
    blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
    blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])

    expand_batch_dim = image.shape.ndims == 3
    if expand_batch_dim:
      # Tensorflow requires batched input to convolutions, which we can fake
      # with an extra dimension.
      image = tf.expand_dims(image, axis=0)
    blurred = tf.nn.depthwise_conv2d(
        image, blur_h, strides=[1, 1, 1, 1], padding=padding)
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
    if expand_batch_dim:
      blurred = tf.squeeze(blurred, axis=0)
    return blurred


def batch_random_blur(images, side_length=IMAGE_SIZE, blur_probability=0.5):
  """Probabilistically applies a Gaussian blur across a batch of images.

  Each image in the batch is blurred with probability `blur_probability`.

  Args:
    images: A Tensor representing a batch of images. Shape should be
      [batch_size, side_length, side_length, channels] and have float dtype.
    side_length: A python integer. The length, in pixels, of the height and
      width dimensions of `images`.
    blur_probability: The probaility with which to apply the blur operator to
      each image in the batch. A python float between 0 and 1.

  Returns:
    A batch of images of the same shape and dtype as the input `images`.
  """
  with tf.name_scope('batch_random_blur'):
    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    selector_shape = [batch_size, 1, 1, 1]
    blurred = _gaussian_blur(images, side_length=side_length, padding='SAME')

    selector = tf.less(
        tf.random_uniform(selector_shape, 0, 1, dtype=tf.float32),
        blur_probability)
    selector = tf.broadcast_to(selector, shape=images_shape)

    images = tf.where(selector, blurred, images)

    return images


def _warp(image,
          side_length=IMAGE_SIZE,
          max_control_length=23.,
          num_control_points_per_axis=2):
  """Apply a random warp of the image.

  Args:
    image: An image Tensor of shape [`side_length`, `side_length`, 3] and
      numeric dtype.
    side_length: A python integer. The length, in pixels, of the height and
      width dimensions of `image`.
    max_control_length: A python integer. The maximum distance that a single
      control point can be translated as part of the warping.
    num_control_points_per_axis: A python integer. The number of control points
      to use along each spatial dimension. These will be evenly distributed as a
      grid along the interior of the image.

  Returns:
    A warped image with the same shape and dtype as the input `image`.
  """
  with tf.name_scope('warp'):
    control_point_locations = []
    for i in range(num_control_points_per_axis):
      for j in range(num_control_points_per_axis):
        control_point_locations.append([
            (i + 1) * side_length / (num_control_points_per_axis + 1),
            (j + 1) * side_length / (num_control_points_per_axis + 1)
        ])

    control_point_locations_np = np.array([control_point_locations],
                                          dtype=np.float32)
    control_vector_length = tf.random.uniform((),
                                              minval=0,
                                              maxval=max_control_length,
                                              dtype=tf.float32)
    image = image[tf.newaxis, Ellipsis]
    rand_flow_vectors = tf.random.normal(
        control_point_locations_np.shape, dtype=tf.float32)
    rand_flow_vectors *= control_vector_length
    dest_locations = control_point_locations_np + rand_flow_vectors
    warped = tfa.image.sparse_image_warp(
        tf.cast(image, tf.float32), control_point_locations_np,
        dest_locations)[0][0]
    warped = tf.cast(warped, image.dtype)
    return warped


def _random_apply(func, p, x):
  """With probability p return func(x).

  With probability 1-p return x.

  Args:
    func: A function that takes a single argument (`x`) and returns a value of
      the same type as `x`.
    p: A scalar float probability between 0 and 1 of `func` being applied. Can
      be a Tensor or a python number.
    x: A Tensor of the shape and dtype expected by `func`.

  Returns:
    `x` or `func(x)`.
  """
  with tf.name_scope('random_apply'):
    return tf.cond(
        tf.less(
            tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


def _jitter_colors(image, strength=1.0, use_pytorch_impl=False):
  """Randomly jitter image colors.

  Applies random color jitter effects (brightness, contrast, saturation, and
  hue) in random order.

  Args:
    image: An image Tensor of shape [height, width, 3] and dtype tf.uint8.
    strength: A float controlling the maximum strength of the jitter. Generally,
      the range [0, 1] is expected, but higher positive values should work.
    use_pytorch_impl: Whether to use an implementation that is equivalent to
      `torchvision.transforms.ColorJitter` instead of the TensorFlow
      implementation.

  Returns:
    An image Tensor of the same shape and dtype as the input image.
  """
  if strength < 0.:
    raise ValueError(f'strength must be positive. Found {strength}.')

  with tf.name_scope('distort_color'):

    def jitter_brightness(im):
      s = 0.8 * strength  # Magic constant inherited from SimCLR.
      if use_pytorch_impl:
        v = tf.random.uniform((),
                              minval=max(0., 1. - s),
                              maxval=1. + s,
                              dtype=tf.float32)
        return v * im
      else:
        return tf.image.random_brightness(im, max_delta=s)

    def jitter_contrast(im):
      s = 0.8 * strength  # Magic constant inherited from SimCLR.
      if use_pytorch_impl:
        v = tf.random.uniform((),
                              minval=max(0., 1. - s),
                              maxval=1. + s,
                              dtype=tf.float32)
        gray = tf.image.rgb_to_grayscale(im)
        mean = tf.reduce_mean(gray, keepdims=True)
        return v * im + (1. - v) * mean
      else:
        return tf.image.random_contrast(im, lower=1 - s, upper=1 + s)

    def jitter_saturation(im):
      s = 0.8 * strength  # Magic constant inherited from SimCLR.
      if use_pytorch_impl:
        v = tf.random.uniform((),
                              minval=max(0., 1. - s),
                              maxval=1. + s,
                              dtype=tf.float32)
        gray = tf.image.rgb_to_grayscale(im)
        return v * im + (1. - v) * gray
      else:
        return tf.image.random_saturation(im, lower=1 - s, upper=1 + s)

    def jitter_hue(im):
      d = 0.2 * strength  # Magic constant inherited from SimCLR.
      if use_pytorch_impl:
        if d > 0.5:
          raise ValueError('Hue jitter strength must not be above 0.5')
        f = tf.random.uniform((), minval=-d, maxval=d, dtype=tf.float32)
        hsv = tf.image.rgb_to_hsv(im)
        channels = tf.unstack(hsv, 3, axis=-1)
        h, s, v = channels
        h = (h + f) % 1.0
        jittered_hsv = tf.stack([h, s, v], axis=-1)
        return tf.image.hsv_to_rgb(jittered_hsv)
      else:
        return tf.image.random_hue(im, max_delta=d)

    def apply_transform(i, im):
      """Apply the i-th transformation."""

      effects = [
          lambda: jitter_brightness(im), lambda: jitter_contrast(im),
          lambda: jitter_saturation(im), lambda: jitter_hue(im)
      ]
      return tf.cond(
          tf.less(i, 2), lambda: tf.cond(tf.less(i, 1), effects[0], effects[1]),
          lambda: tf.cond(tf.less(i, 3), effects[2], effects[3]))

    perm = tf.random_shuffle(tf.range(4))
    # Convert to float here, since each type of transform internally will
    # convert int types to float and then back, so this removes the extra
    # conversions in between transforms. Note that unlike most of the rest of
    # this code, these transformation expect floats in [0, 1] range, so we don't
    # use `_convert_image_dtype`.
    original_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    image = tf.image.convert_image_dtype(image, dtype=original_dtype)
    return image


def _rgb_to_gray(image):
  """Converts an RGB color image to grayscale, but still in RGB colorspace.

  Args:
    image: An image Tensor of shape [height, width, 3] and dtype tf.uint8.

  Returns:
    An grayscale image Tensor with the same shape and dtype as the input image.
  """
  with tf.name_scope('rgb_to_gray'):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 3])
    return image


def _simclr_augment(image,
                    warp_prob=0.5,
                    blur_prob=0.5,
                    strength=0.5,
                    side_length=IMAGE_SIZE,
                    use_pytorch_color_jitter=False):
  """Applies the data augmentation sequence from SimCLR.

  Citation: https://arxiv.org/abs/2002.05709

  Args:
    image: An image Tensor of shape [height, width, 3] and dtype tf.uint8.
    warp_prob: A Python float between 0 and 1. The probability of applying a
      warp transformation.
    blur_prob: A Python float between 0 and 1. The probability of applying a
      blur transformation.
    strength: A Python float in range [0,1] controlling the maximum
      strength of the augmentations.
    side_length: A Python integer. The length, in pixels, of the width and
      height of `image`.
    use_pytorch_color_jitter: A Python bool. Whether to use a color jittering
      algorithm that aims to replicate `torchvision.transforms.ColorJitter`
      rather than the standard TensorFlow color jittering.

  Returns:
    An image with the same shape and dtype as the input image.
  """
  with tf.name_scope('simclr_augment'):
    image = _random_apply(
        functools.partial(
            _jitter_colors,
            strength=strength,
            use_pytorch_impl=use_pytorch_color_jitter),
        p=0.8,
        x=image)
    image = _random_apply(_rgb_to_gray, p=0.2, x=image)
    if warp_prob > 0.:
      image = _random_apply(
          functools.partial(_warp, side_length=side_length),
          p=warp_prob,
          x=image)
    if blur_prob > 0.:
      image = _convert_image_dtype(image, tf.float32)
      image = _random_apply(
          functools.partial(_gaussian_blur, side_length=side_length),
          p=blur_prob,
          x=image)
    return image


def _stacked_simclr_randaugment(image,
                                warp_prob=0.5,
                                blur_prob=0.5,
                                strength=0.5,
                                side_length=IMAGE_SIZE,
                                use_pytorch_color_jitter=False):
  """A combination the data augmentation sequences from SimCLR and RandAugment.

  Citations:
    SimCLR: https://arxiv.org/abs/2002.05709
    RandAugment: https://arxiv.org/abs/1909.13719

  Args:
    image: An image Tensor of shape [height, width, 3] and dtype tf.uint8.
    warp_prob: A Python float between 0 and 1. The probability of applying a
      warp transformation.
    blur_prob: A Python float between 0 and 1. The probability of applying a
      blur transformation.
    strength: strength: A Python float in range [0,1] controlling the maximum
      strength of the augmentations.
    side_length: A Python integer. The length, in pixels, of the width and
      height of `image`.
    use_pytorch_color_jitter: A Python bool. Whether to use a color jittering
      algorithm that aims to replicate `torchvision.transforms.ColorJitter`
      rather than the standard TensorFlow color jittering.

  Returns:
    An image with the same shape and dtype as the input image.
  """
  with tf.name_scope('stacked_simclr_randaugment'):
    image = _random_apply(
        functools.partial(
            _jitter_colors,
            strength=strength,
            use_pytorch_impl=use_pytorch_color_jitter),
        p=0.8,
        x=image)
    image = augment.RandAugment().distort(image)
    image = _random_apply(_rgb_to_gray, p=0.2, x=image)
    if warp_prob > 0.:
      image = _random_apply(
          functools.partial(_warp, side_length=side_length),
          p=warp_prob,
          x=image)
    if blur_prob > 0.:
      image = _convert_image_dtype(image, tf.float32)
      image = _random_apply(
          functools.partial(_gaussian_blur, side_length=side_length),
          p=blur_prob,
          x=image)
    return image


def _augment_image(image,
                   augmentation_type=enums.AugmentationType.SIMCLR,
                   warp_prob=0.5,
                   blur_prob=0.5,
                   augmentation_magnitude=0.5,
                   use_pytorch_color_jitter=False):
  """Applies data augmentation to an image.

  Args:
    image: An image Tensor of shape [height, width, 3] and dtype tf.uint8.
    augmentation_type: An enums.AugmentationType.
    warp_prob: The probability of applying a warp augmentation at the end (only
      applies to augmentation_types SIMCLR and STACKED_RANDAUGMENT).
    blur_prob: A Python float between 0 and 1. The probability of applying a
      blur transformation.
    augmentation_magnitude: The magnitude of augmentation. Valid range differs
      depending on the augmentation_type.
    use_pytorch_color_jitter: A Python bool. Whether to use a color jittering
      algorithm that aims to replicate `torchvision.transforms.ColorJitter`
      rather than the standard TensorFlow color jittering. Only used for
      augmentation_types SIMCLR and STACKED_RANDAUGMENT.

  Returns:
    The augmented image.

  Raises:
    ValueError if `augmentation_type` is unknown.
  """
  augmentation_type_map = {
      enums.AugmentationType.SIMCLR:
          functools.partial(
              _simclr_augment,
              warp_prob=warp_prob,
              blur_prob=blur_prob,
              strength=augmentation_magnitude,
              use_pytorch_color_jitter=use_pytorch_color_jitter),
      enums.AugmentationType.STACKED_RANDAUGMENT:
          functools.partial(
              _stacked_simclr_randaugment,
              warp_prob=warp_prob,
              blur_prob=blur_prob,
              strength=augmentation_magnitude,
              use_pytorch_color_jitter=use_pytorch_color_jitter),
      enums.AugmentationType.RANDAUGMENT:
          augment.RandAugment(magnitude=augmentation_magnitude).distort,
      enums.AugmentationType.AUTOAUGMENT:
          augment.AutoAugment().distort,
      enums.AugmentationType.IDENTITY: (lambda x: x),
  }
  if augmentation_type not in augmentation_type_map:
    raise ValueError(f'Invalid augmentation_type: {augmentation_type}.')

  if image.dtype != tf.uint8:
    raise TypeError(f'Image must have dtype tf.uint8. Was {image.dtype}.')

  if augmentation_magnitude <= 0.:
    return image

  with tf.name_scope('augment_image'):
    return augmentation_type_map[augmentation_type](image)


def _convert_image_dtype(image, dtype):
  """Converts an image to a different DType.

  Also transforms it into the expected range for the target DType.

  This only handles floating point DTypes and tf.uint8.

  Floating point DTypes are in range [-1, 1] and tf.uint8 is in range [0, 255].
  This does not verify that the inputs are in the expected range.

  If the image already has the target dtype then this just returns it as is.

  Args:
    image: A Tensor of any shape and DType that is castable to `dtype`.
    dtype: An instance of tf.DType. The target DType that image should be
      converted to.

  Returns:
    A tensor the same shape as image, but having dtype `dtype` and its values
    transformed into the expected range for `dtype`.

  Raises:
    TypeError: If `dtype` is not tf.uint8 or a floating dtype.
  """
  with tf.name_scope('convert_image_dtype'):
    if image.dtype == dtype:
      return image

    if dtype == tf.uint8:
      out_image = tf.cast((image + 1.0) * 127.5, dtype)
    elif dtype.is_floating:
      out_image = tf.cast(image, dtype)
      if not image.dtype.is_floating:
        out_image = out_image / 127.5 - 1.0
    else:
      raise TypeError(f'Invalid image dtype: {dtype}')

    assert out_image.dtype == dtype

    return out_image


def _preprocess_image_for_train(image,
                                preprocessing_options,
                                dataset_options,
                                bfloat16_supported=False):
  """Preprocesses an image for training before feeding it into the network.

  Args:
    image: Either a rank 3 `Tensor` representing an image with shape [height,
      width, channels=3] or else an encoded jpeg image string. The value of
      `dataset_options.decode_input` should be set accordingly.
    preprocessing_options: An instance of hparams.ImagePreprocessing.
    dataset_options: An instance of DatasetOptions.
    bfloat16_supported: Whether bfloat16 is supported on this platform.

  Returns:
    A preprocessed image `Tensor` of shape
    [preprocessing_options.image_size, preprocessing_options.image_size, 3],
    The image dtype is either tf.float32 or tf.bfloat16, depending on
    `preprocessing_options.allow_mixed_precision` and `bfloat16_supported`. The
    image values are in the range [-1, 1].
  """
  with tf.name_scope('preprocess_for_train'):
    image_patch = _crop_to_square(
        image,
        decode_image=dataset_options.decode_input,
        side_length=preprocessing_options.image_size,
        crop_padding=preprocessing_options.crop_padding,
        is_training=True,
        resize_only=False,
        area_range=preprocessing_options.crop_area_range)
    image_patch = _flip(image_patch)
    image_patch = _augment_image(
        image_patch,
        augmentation_type=preprocessing_options.augmentation_type,
        warp_prob=preprocessing_options.warp_probability,
        blur_prob=(0. if preprocessing_options.defer_blurring else
                   preprocessing_options.blur_probability),
        augmentation_magnitude=preprocessing_options.augmentation_magnitude,
        use_pytorch_color_jitter=preprocessing_options.use_pytorch_color_jitter)
    image_patch = _convert_image_dtype(
        image_patch,
        tf.bfloat16 if preprocessing_options.allow_mixed_precision and
        bfloat16_supported else tf.float32)
    return image_patch


def _preprocess_image_for_eval(image,
                               preprocessing_options,
                               dataset_options,
                               bfloat16_supported=False):
  """Preprocesses an image for evaluation.

  Args:
    image: Either a rank 3 `Tensor` representing an image with shape [height,
      width, channels=3] or else an encoded jpeg image string. The value of
      `preprocessing_options.decode_input` should be set accordingly.
    preprocessing_options: An instance of hparams.ImagePreprocessing.
    dataset_options: An instance of DatasetOptions.
    bfloat16_supported: Whether bfloat16 is supported on this platform.

  Returns:
    A preprocessed image `Tensor` of shape
    [preprocessing_options.image_size, preprocessing_options.image_size, 3]
    and dtype is tf.float32 or tf.bfloat16 depending on
    `preprocessing_options.allow_mixed_precision` and `bfloat16_supported`. The
    image values are in the range [-1, 1].
  """
  with tf.name_scope('preprocess_for_eval'):
    image_patch = _crop_to_square(
        image,
        decode_image=dataset_options.decode_input,
        side_length=preprocessing_options.image_size,
        crop_padding=preprocessing_options.crop_padding,
        is_training=False,
        resize_only=False,
        eval_crop_method=preprocessing_options.eval_crop_method)
    assert image_patch.dtype == tf.uint8, (
        f'Expected uint8, was {image_patch.dtype}')

    image_patch = _convert_image_dtype(
        image_patch,
        tf.bfloat16 if preprocessing_options.allow_mixed_precision and
        bfloat16_supported else tf.float32)
    return tf.ensure_shape(
        image_patch,
        [preprocessing_options.image_size, preprocessing_options.image_size, 3])


def preprocess_image(image,
                     preprocessing_options,
                     dataset_options=None,
                     bfloat16_supported=False,
                     is_training=False):
  """Preprocesses an image before feeding it into the network.

  Args:
    image: Either a rank 3 `Tensor` representing an image of arbitrary size or
      else an encoded jpeg image string. The value of
      `dataset_options.decode_input` should be set accordingly.
    preprocessing_options: An instance of hparams.ImagePreprocessing.
    dataset_options: An instance of DatasetOptions.
    bfloat16_supported: Whether bfloat16 is supported on this platform.
    is_training: Whether we are in training mode.

  Returns:
    A preprocessed image `Tensor` of shape [image_size, image_size, 3]. The
    image dtype is either tf.float32 or tf.bfloat16, depending on
    `preprocessing_options.allow_mixed_precision` and `bfloat16_supported`.
    Images are in the range [-1, 1].
  """
  with tf.name_scope('preprocess_image'):
    dataset_options = dataset_options or DatasetOptions()
    if dataset_options.decode_input:
      image = tf.convert_to_tensor(image)
      assert image.dtype == tf.string, (
          f'image should be tf.string. Was {image.dtype}')
    else:
      assert image.dtype == tf.uint8, (
          f'image should be tf.uint8. Was {image.dtype}')

    preprocess_fn = (
        _preprocess_image_for_train
        if is_training else _preprocess_image_for_eval)
    image = preprocess_fn(image, preprocessing_options, dataset_options,
                          bfloat16_supported)

    if preprocessing_options.apply_whitening:
      if dataset_options.image_mean_std is None:
        tf.logging.warning(
            '`apply_whitening` was requested, but the dataset does not specify '
            'whitening parameters in `image_mean_std`. Skipping whitening.')
      else:
        mean, std = dataset_options.image_mean_std
        mean = tf.cast(mean, image.dtype)
        std = tf.cast(std, image.dtype)
        image = (image - mean) / std

    return image


class Preprocessor(metaclass=abc.ABCMeta):
  """Preprocessor interface.

  Attrs:
    preprocessing_options: An hparams.ImagePreprocessing instance.
    dataset_options: A DatasetOptions instance.
    is_training: Whether to use training-style preprocessing instead of
      eval-style.
  """

  def __init__(self,
               preprocessing_options,
               dataset_options=None,
               bfloat16_supported=False,
               is_training=False):
    self.preprocessing_options = preprocessing_options
    self.dataset_options = dataset_options or DatasetOptions()
    if self.preprocessing_options.num_views < 1:
      raise ValueError(
          'preprocessing_options.num_views'
          f'(= {self.preprocessing_options.num_views}) must be >= 1')
    self.is_training = is_training
    self.bfloat16_supported = bfloat16_supported

  @abc.abstractmethod
  def preprocess(self, input_data):
    """Applies preprocessing to `input_data`.

    Args:
      input_data: Generic input data Tensor corresponding to a single data
        sample.

    Returns:
      `Tensor` containing the preprocessed version of `input_data`.
    """
    pass


class ImageToMultiViewedImagePreprocessor(Preprocessor):
  """Preprocessor that converts an image to a multiviewed image."""

  def __init__(self, *args, **kwargs):
    super(ImageToMultiViewedImagePreprocessor, self).__init__(*args, **kwargs)

  def preprocess(self, image):
    """Preprocesses `image` into a multiviewed image.

    Args:
      image: Either a rank 3 `Tensor` of shape [height, weight, channels=3]
        representing an RGB image of arbitrary size or else an encoded jpeg
        image string. The value of `dataset_options.decode_input` should
        be set accordingly.

    Returns:
      A preprocessed mulitiviewed image `Tensor` of shape
      [image_size, image_size, 3 * num_views] corresponding to `num_views` RGB
      images stacked along the channels dimension. The image dtype is either
      tf.float32 or tf.bfloat16, depending on
      `preprocessing_options.allow_mixed_precision` and `bfloat16_supported`.
      Images are in the range [-1, 1].
    """
    image_views = []
    for i in range(self.preprocessing_options.num_views):
      image_view = preprocess_image(
          image,
          # Since the eval preprocessing is deterministic, we use train mode
          # preprocessing for views after the first even at eval time so that a
          # non-trivial contrastive loss can be computed.
          is_training=self.is_training if i == 0 else True,
          preprocessing_options=self.preprocessing_options,
          dataset_options=self.dataset_options,
          bfloat16_supported=self.bfloat16_supported)
      image_views.append(image_view)

    return tf.concat(image_views, axis=2)
