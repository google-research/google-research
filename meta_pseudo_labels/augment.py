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

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=g-long-lambda
# pylint: disable=g-bare-generic
# pylint: disable=g-complex-comprehension

r"""AutoAugment and RandAugment policies for enhanced image preprocessing.

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import math
from typing import Text, Dict, Any, Tuple, List

from absl import logging
import tensorflow.compat.v1 as tf

from tensorflow.contrib import image as image_ops


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
    'Identity': identity,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})


def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  def _blend():
    image_1 = tf.image.convert_image_dtype(image1, tf.float32)
    image_2 = tf.image.convert_image_dtype(image2, tf.float32)
    output = image_1 + factor * (image_2 - image_1)
    output = tf.where_v2(
        tf.logical_and(tf.less(0., factor), tf.less(factor, 1.)),
        x=output, y=tf.clip_by_value(output, 0., 255.))
    return tf.image.convert_image_dtype(output, tf.uint8)

  pred_fn_pairs = [
      (tf.equal(factor, 0.), lambda: image1),
      (tf.equal(factor, 1.), lambda: image2),
  ]
  return tf.case(
      pred_fn_pairs, default=_blend, exclusive=True, strict=True, name='blend')


def cutout(image, pad_size, replace = 0):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Samples the center location in the image where the zero mask is applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims, constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where_v2(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace,
      image)
  return image


def solarize(image, threshold = 128):
  """Keep a pixel if `pixel < threshold`, otherwise subtract 225 from it."""
  threshold = tf.cast(threshold, image.dtype)
  return tf.where_v2(image < threshold, image, 255 - image)


def solarize_add(image,
                 addition = 0,
                 threshold = 128):
  """If `pixel < threshold`, add `addition` to it and clip between 0 and 255."""
  threshold = tf.cast(threshold, image.dtype)
  added_image = tf.cast(image, tf.int32) + addition
  added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
  return tf.where_v2(image < threshold, added_image, image)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = tf.cast(8 - bits, image.dtype)
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees, replace):
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.
    replace: A one or three value 1D tensor to fill empty pixels caused by
      the rotate operation.

  Returns:
    The rotated version of image.
  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = image_ops.rotate(wrap(image), radians)
  return unwrap(image, replace)


def translate_x(image, pixels, replace):
  """Equivalent of PIL Translate in X dimension."""
  image = image_ops.translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  image = image_ops.translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def shear_x(image, level, replace):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = image_ops.transform(
      wrap(image), [1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image, level, replace):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = image_ops.transform(
      wrap(image), [1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def identity(image):
  """Identity."""
  return tf.identity(image)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where_v2(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where_v2(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def invert(image):
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], axis=2)
  return extended


def unwrap(image, replace):
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)
  alpha_channel = tf.tile(alpha_channel, [1, 4])

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where_v2(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  return tf.cond(
      tf.equal(level, 0.), lambda: 1., lambda: 2. / (_MAX_LEVEL / level) + 0.9)


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _mult_to_arg(level, multiplier = 1.):
  # return (int(float(level) / _MAX_LEVEL * multiplier),)
  return (
      tf.cast(tf.cast(level, tf.float32) / _MAX_LEVEL * multiplier, tf.int32),)


def _apply_func_with_prob(func,
                          image,
                          args,
                          prob):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(
      should_apply_op,
      lambda: func(image, *args),
      lambda: image)
  return augmented_image


def select_and_apply_random_policy(policies, image):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image


def level_to_arg(cutout_const, translate_const):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

  args = {
      'Identity': no_arg,
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
  }
  return args


def _parse_policy_info(name,
                       prob,
                       level,
                       replace_value,
                       cutout_const,
                       translate_const):
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(self, image):
    """Given an image tensor, returns a distorted image with the same shape.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()


class RandAugment(ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """

  def __init__(self,
               num_layers = 2,
               magnitude = None,
               cutout_const = 40.,
               translate_const = 100.,
               available_ops = None):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 30].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
      available_ops: allow to override the default option of using all ops.
    """
    super(RandAugment, self).__init__()

    self.num_layers = num_layers
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    if available_ops is None:
      available_ops = [
          'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize',
          'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
          'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout',
      ]
    self.available_ops = available_ops
    self.magnitude = magnitude

  def distort(self, image):
    """Applies the RandAugment policy to `image`.

    Args:
      image: `Tensor` of shape `h, w, 3` representing an image.

    Returns:
      The augmented version of `image`.
    """
    input_image_type = image.dtype
    image = tf.clip_by_value(
        image, tf.cast(0, image.dtype), tf.cast(255, image.dtype))
    image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3
    prob = tf.random.uniform([], 0.2, 0.8, tf.float32)
    for _ in range(self.num_layers):
      op_to_select = tf.random.uniform(
          [], minval=0, maxval=len(self.available_ops), dtype=tf.int32)

      branch_fns = []
      for (i, op_name) in enumerate(self.available_ops):
        func, _, args = _parse_policy_info(op_name, prob, self.magnitude,
                                           replace_value, self.cutout_const,
                                           self.translate_const)
        def branch_fn(selected_func=func, selected_args=args):
          return tf.cond(tf.random.uniform([], 0., 1., prob.dtype) <= prob,
                         lambda: selected_func(image, *selected_args),
                         lambda: image)
        branch_fns.append((i, branch_fn))

      image = tf.switch_case(branch_index=op_to_select, branch_fns=branch_fns)

    image = tf.cast(image, dtype=input_image_type)
    return image
