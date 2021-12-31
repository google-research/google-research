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

"""SMURF augmentation.

This library contains various augmentation functions.
"""
# pylint:skip-file
import math
from typing import Tuple, Union, Dict

import gin
import gin.tf
import tensorflow as tf
from tensorflow_addons import image as tfa_image
from functools import partial

from smurf import smurf_utils

_TensorTuple2 = Tuple[tf.Tensor, tf.Tensor]
_TensorTuple3 = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
_TensorTuple4 = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]


def apply_augmentation(
    inputs,
    crop_height = 640,
    crop_width = 640,
    return_full_scale=False):
  """Applies photometric and geometric augmentations to images and flow.

  Args:
    inputs: dictionary of data to perform augmentation on.
    crop_height: Height of the final augmented output.
    crop_width: Width of the final augmented output.
    return_full_scale: bool, if True, include the full size images.

  Returns:
    Augmented images and possibly flow, mask (if provided).
  """
  # Ensure sequence length of two to be able to unstack images.
  images = inputs['images']
  flow = inputs.get('flow')
  mask = inputs.get('flow_valid')
  images = tf.ensure_shape(images, (2, None, None, None))
  # Apply geometric augmentation functions.
  if return_full_scale:  # Perform "full-scale warping."
    images, flow, mask, full_size_images, crop_h, crop_w, pad_h, pad_w = geometric_augmentation(
        images, flow, mask, crop_height, crop_width, return_full_scale=True)
  else:
    images, flow, mask = geometric_augmentation(
        images, flow, mask, crop_height, crop_width, return_full_scale=False)

  images_aug = photometric_augmentation(images)

  if flow is not None:
    inputs['flow'] = flow
    inputs['flow_valid'] = mask

  if return_full_scale:
    inputs['crop_h'] = crop_h
    inputs['crop_w'] = crop_w
    inputs['pad_h'] = pad_h
    inputs['pad_w'] = pad_w
    inputs['full_size_images'] = full_size_images

  inputs['images'] = images
  inputs['augmented_images'] = images_aug

  return inputs


@gin.configurable
def photometric_augmentation(
    images,
    probability_color_swap = 0.0,
    probability_hue_shift = 1.0,
    probability_saturation = 1.0,
    probability_brightness = 1.0,
    probability_contrast = 1.0,
    probability_gaussian_noise = 0.0,
    probability_brightness_individual = 0.0,
    probability_contrast_individual = 0.0,
    probability_eraser = 0.5,
    probability_eraser_additional_operations = 0.5,
    probability_assymetric = 0.2,
    max_delta_hue = 0.5 / 3.14,
    min_bound_saturation = 0.6,
    max_bound_saturation = 1.4,
    max_delta_brightness = 0.4,
    min_bound_contrast = 0.6,
    max_bound_contrast = 1.4,
    min_bound_gaussian_noise = 0.0,
    max_bound_gaussian_noise = 0.02,
    max_delta_brightness_individual = 0.02,
    min_bound_contrast_individual = 0.95,
    max_bound_contrast_individual = 1.05,
    min_size_eraser = 50,
    max_size_eraser = 100,
    max_operations_eraser = 3):
  """Applies photometric augmentations to an image pair.

  Args:
    images: Image pair of shape [2, height, width, channels].
    probability_color_swap: Probability of applying color swap augmentation.
    probability_hue_shift: Probability of applying hue shift augmentation.
    probability_saturation: Probability of applying saturation augmentation.
    probability_brightness: Probability of applying brightness augmentation.
    probability_contrast: Probability of applying contrast augmentation.
    probability_gaussian_noise: Probability of applying gaussian noise
      augmentation.
    probability_brightness_individual: Probability of applying brightness
      augmentation individually to each image of the image pair.
    probability_contrast_individual: Probability of applying contrast
      augmentation individually to each image of the image pair.
    probability_eraser: Probability of applying the eraser augmentation.
    probability_eraser_additional_operations: Probability of applying additional
      erase operations within the eraser augmentation.
    probability_assymetric: Probability of applying some photomoteric
      augmentations invidiually per frame (hue_shift, brightness,
      saturation, contrast, gaussian noise).
    max_delta_hue: Must be in the interval [0, 0.5]. Defines the interval
      [-max_delta_hue, max_delta_hue] in which to pick a random hue offset.
    min_bound_saturation: Lower bound for the randomly picked saturation factor
      in the interval [lower, upper].
    max_bound_saturation: Upper bound for the randomly picked saturation factor
      in the interval [lower, upper].
    max_delta_brightness: Delta defines the interval [-max_delta, max_delta) in
      which a random value is picked that will be added to the image values.
    min_bound_contrast: Lower bound for the randomly picked contrast factor in
      the interval [lower, upper]. It will be applied per channel via (x - mean)
      * contrast_factor + mean.
    max_bound_contrast: Upper bound for the randomly picked contrast factor in
      the interval [lower, upper]. It will be applied per channel via (x - mean)
      * contrast_factor + mean.
    min_bound_gaussian_noise: Lower bound for the randomly picked sigma in the
      interval [lower, upper].
    max_bound_gaussian_noise: Upper bound for the randomly picked sigma in the
      interval [lower, upper].
    max_delta_brightness_individual: Same as max_delta_brightness, but for the
      augmentation applied individually per frame.
    min_bound_contrast_individual: Same as min_bound_contrast, but for the
      augmentation applied individually per frame.
    max_bound_contrast_individual: Same as max_bound_contrast, but for the
      augmentation applied individually per frame.
    min_size_eraser: Minimal side length of the rectangle shaped region that
      will be removed.
    max_size_eraser: Maximal side length of the rectangle shaped region that
      will be removed.
    max_operations_eraser: Maximal number of rectangle shaped regions that will
      be removed.

  Returns:
    Augmented images and possibly flow, mask (if provided).
  """
  # All photometric augmentation that could be applied individually per frame.
  def potential_asymmetric_augmentations(images):
    if probability_hue_shift > 0:
      images = random_hue_shift(images, probability_hue_shift, max_delta_hue)
    if probability_saturation > 0:
      images = random_saturation(images, probability_saturation,
                                 min_bound_saturation, max_bound_saturation)
    if probability_brightness > 0:
      images = random_brightness(images, probability_brightness,
                                 max_delta_brightness)
    if probability_contrast > 0:
      images = random_contrast(images, probability_contrast, min_bound_contrast,
                               max_bound_contrast)
    if probability_gaussian_noise > 0:
      images = random_gaussian_noise(images, probability_gaussian_noise,
                                     min_bound_gaussian_noise,
                                     max_bound_gaussian_noise)
    return images

  perform_assymetric = tf.random.uniform([]) < probability_assymetric
  def true_fn(images):
    image_1, image_2 = tf.unstack(images)
    image_1 = potential_asymmetric_augmentations(image_1)
    image_2 = potential_asymmetric_augmentations(image_2)
    return tf.stack([image_1, image_2])
  def false_fn(images):
    return images

  images = tf.cond(perform_assymetric, lambda: true_fn(images),
                   lambda: false_fn(images))

  # Photometric augmentations applied to all frames of a pair.
  if probability_color_swap > 0:
    images = random_color_swap(images, probability_color_swap)

  # Photometric augmentations applied individually per image frame.
  if probability_contrast_individual > 0:
    images = random_contrast_individual(images, probability_contrast_individual,
                                        min_bound_contrast_individual,
                                        max_bound_contrast_individual)
  if probability_brightness_individual > 0:
    images = random_brightness_individual(images,
                                          probability_brightness_individual,
                                          max_delta_brightness_individual)

  # Crop values to ensure values are within [0,1], some augmentations may
  # violate this.
  images = tf.clip_by_value(images, 0.0, 1.0)

  # Apply special photometric augmentations.
  if probability_eraser > 0:
    images = random_eraser(
        images,
        min_size=min_size_eraser,
        max_size=max_size_eraser,
        probability=probability_eraser,
        max_operations=max_operations_eraser,
        probability_additional_operations=probability_eraser_additional_operations
    )
  return images


@gin.configurable
def geometric_augmentation(images,
                           flow = None,
                           mask = None,
                           crop_height = 640,
                           crop_width = 640,
                           probability_flip_left_right = 0.5,
                           probability_flip_up_down = 0.1,
                           probability_scale = 0.8,
                           probability_relative_scale = 0.,
                           probability_stretch = 0.8,
                           probability_rotation = 0.0,
                           probability_relative_rotation = 0.0,
                           probability_crop_offset = 0.0,
                           min_bound_scale = -0.2,
                           max_bound_scale = 0.6,
                           max_strech_scale = 0.2,
                           min_bound_relative_scale = -0.1,
                           max_bound_relative_scale = 0.1,
                           max_rotation_deg = 15,
                           max_relative_rotation_deg = 3,
                           max_relative_crop_offset = 5,
                           return_full_scale=False):

  """Applies geometric augmentations to an image pair and corresponding flow.

  Args:
    images: Image pair of shape [2, height, width, channels].
    flow: Corresponding forward flow field of shape [height, width, 2].
    mask: Mask indicating which positions in the flow field hold valid flow
      vectors of shape [height, width, 1]. Non-valid poisitions are encoded with
      0, valid positions with 1.
    crop_height: Height of the final augmented output.
    crop_width: Width of the final augmented output.
    probability_flip_left_right: Probability of applying left/right flip.
    probability_flip_up_down: Probability of applying up/down flip
    probability_scale: Probability of applying scale augmentation.
    probability_relative_scale: Probability of applying scale augmentation to
      only the second frame of the the image pair.
    probability_stretch: Probability of applying stretch augmentation (scale
      without keeping the aspect ratio).
    probability_rotation: Probability of applying rotation augmentation.
    probability_relative_rotation: Probability of applying rotation augmentation
      to only the second frame of the the image pair.
    probability_crop_offset: Probability of applying a relative offset while
      cropping.
    min_bound_scale: Defines the smallest possible scaling factor as
      2**min_bound_scale.
    max_bound_scale: Defines the largest possible scaling factor as
      2**max_bound_scale.
    max_strech_scale: Defines the smallest and largest possible streching factor
      as 2**-max_strech_scale and 2**max_strech_scale.
    min_bound_relative_scale: Defines the smallest possible scaling factor for
      the relative scaling as 2**min_bound_relative_scale.
    max_bound_relative_scale: Defines the largest possible scaling factor for
      the relative scaling as 2**max_bound_relative_scale.
    max_rotation_deg: Defines the maximum angle of rotation in degrees.
    max_relative_rotation_deg: Defines the maximum angle of rotation in degrees
      for the relative rotation.
    max_relative_crop_offset: Defines the maximum relative offset in pixels for
      cropping.
    return_full_scale: bool. If this is passed, the full size images will be
      returned in addition to the geometrically augmented (cropped and / or
      resized) images. In addition to the resized images, the crop height,
      width, and any padding applied will be returned.

  Returns:
    if return_full_scale is False:
      Augmented images, flow and mask (if not None).
    if return_full_scale is True:
      Augmented images, flow, mask, full_size_images, crop_h, crop_w, pad_h,
       and pad_w.
  """

  # apply geometric augmentation
  if probability_flip_left_right > 0:
    images, flow, mask = random_flip_left_right(
        images, flow, mask, probability_flip_left_right)

  if probability_flip_up_down > 0:
    images, flow, mask = random_flip_up_down(
        images, flow, mask, probability_flip_up_down)

  if probability_scale > 0 or probability_stretch > 0:
    images, flow, mask = random_scale(
        images,
        flow,
        mask,
        min_scale=min_bound_scale,
        max_scale=max_bound_scale,
        max_strech=max_strech_scale,
        probability_scale=probability_scale,
        probability_strech=probability_stretch)

  if probability_relative_scale > 0:
    images, flow, mask = random_scale_second(
        images, flow, mask,
        min_scale=min_bound_relative_scale,
        max_scale=max_bound_relative_scale,
        probability_scale=probability_relative_scale)

  if probability_rotation > 0:
    images, flow, mask = random_rotation(
        images, flow, mask,
        probability=probability_rotation,
        max_rotation=max_rotation_deg, not_empty_crop=True)

  if probability_relative_rotation > 0:
    images, flow, mask = random_rotation_second(
        images, flow, mask,
        probability=probability_relative_rotation,
        max_rotation=max_relative_rotation_deg, not_empty_crop=True)

  images_uncropped = images
  images, flow, mask, offset_h, offset_w = random_crop(
      images, flow, mask, crop_height, crop_width,
      relative_offset=max_relative_crop_offset,
      probability_crop_offset=probability_crop_offset)
  # Add 100 / 200 pixels to crop height / width for full scale warp
  pad_to_size_h = crop_height + 200
  pad_to_size_w = crop_width + 400
  if return_full_scale:
    if pad_to_size_w:
      uncropped_shape = tf.shape(images_uncropped)
      if images.shape[1] > uncropped_shape[1] or images.shape[
          2] > uncropped_shape[2]:
        images_uncropped = images
        uncropped_shape = tf.shape(images_uncropped)
        offset_h = tf.zeros_like(offset_h)
        offset_w = tf.zeros_like(offset_w)

      if uncropped_shape[1] > pad_to_size_h:
        crop_ht = offset_h - (200 // 2)
        crop_hb = offset_h + crop_height + (200 // 2)
        crop_hb += tf.maximum(0, -crop_ht)
        crop_ht -= tf.maximum(0, -(uncropped_shape[1] - crop_hb))
        crop_ht = tf.maximum(crop_ht, 0)
        crop_hb = tf.minimum(crop_hb, uncropped_shape[1])
        offset_h -= crop_ht
        images_uncropped = images_uncropped[:, crop_ht:crop_hb, :, :]

      if uncropped_shape[2] > pad_to_size_w:
        crop_wt = offset_w - (400 // 2)
        crop_wb = offset_w + crop_width + (400 // 2)
        crop_wb += tf.maximum(0, -crop_wt)
        crop_wt -= tf.maximum(0, -(uncropped_shape[2] - crop_wb))
        crop_wt = tf.maximum(crop_wt, 0)
        crop_wb = tf.minimum(crop_wb, uncropped_shape[2])
        offset_w -= crop_wt
        images_uncropped = images_uncropped[:, :, crop_wt:crop_wb, :]

      uncropped_shape = tf.shape(images_uncropped)
      # remove remove_pixels_w from the width while keeping the crop centered
      pad_h = pad_to_size_h - uncropped_shape[1]
      pad_w = pad_to_size_w - uncropped_shape[2]
      with tf.control_dependencies([
          tf.compat.v1.assert_greater_equal(pad_h, 0),
          tf.compat.v1.assert_greater_equal(pad_w, 0)
      ]):
        images_uncropped = tf.pad(images_uncropped,
                                  [[0, 0], [pad_h, 0], [pad_w, 0], [0, 0]])
      images_uncropped = tf.ensure_shape(images_uncropped,
                                         [2, pad_to_size_h, pad_to_size_w, 3])
    return images, flow, mask, images_uncropped, offset_h, offset_w, pad_h, pad_w

  return images, flow, mask


def _center_crop(images, height, width):
  """Performs a center crop with the given heights and width."""
  # ensure height, width to be int
  height = tf.cast(height, tf.int32)
  width = tf.cast(width, tf.int32)
  # get current size
  images_shape = tf.shape(images)
  current_height = images_shape[-3]
  current_width = images_shape[-2]
  # compute required offset
  offset_height = tf.cast((current_height - height) / 2, tf.int32)
  offset_width = tf.cast((current_width - width) / 2, tf.int32)
  # perform the crop
  images = tf.image.crop_to_bounding_box(
      images, offset_height, offset_width, height, width)
  return images


def _positions_center_origin(height, width):
  """Returns image coordinates where the origin at the image center."""
  h = tf.range(0.0, height, 1)
  w = tf.range(0.0, width, 1)
  center_h = tf.cast(height, tf.float32) / 2.0 - 0.5
  center_w = tf.cast(width, tf.float32) / 2.0 - 0.5
  return tf.stack(tf.meshgrid(h - center_h, w - center_w, indexing='ij'), -1)


def rotate(img,
           angle_radian,
           is_flow,
           mask = None):
  """Rotate an image or flow field."""
  def _rotate(img, mask=None):
    if angle_radian == 0.0:
      # early return if no resizing is required
      if mask is not None:
        return img, mask
      else:
        return img

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      img = tf.math.multiply(img, mask)
      # rotate img
      img_rotated = tfa_image.rotate(
          img, angle_radian, interpolation='BILINEAR')
      # rotate mask (will serve as normalization weights)
      mask_rotated = tfa_image.rotate(
          mask, angle_radian, interpolation='BILINEAR')
      # normalize sparse flow field and mask
      img_rotated = tf.math.multiply(
          img_rotated, tf.math.reciprocal_no_nan(mask_rotated))
      mask_rotated = tf.math.multiply(
          mask_rotated, tf.math.reciprocal_no_nan(mask_rotated))
    else:
      img_rotated = tfa_image.rotate(
          img, angle_radian, interpolation='BILINEAR')

    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # rotation.
      cos = tf.math.cos(angle_radian)
      sin = tf.math.sin(angle_radian)
      rotation_matrix = tf.reshape([cos, sin, -sin, cos], [2, 2])
      img_rotated = tf.linalg.matmul(img_rotated, rotation_matrix)

    if mask is not None:
      return img_rotated, mask_rotated
    return img_rotated

  # Apply resizing at the right shape.
  shape = img.shape.as_list()
  if len(shape) == 3:
    if mask is not None:
      img_rotated, mask_rotated = _rotate(img[None], mask[None])
      return img_rotated[0], mask_rotated[0]
    else:
      return _rotate(img[None])[0]
  elif len(shape) == 4:
    # Input at the right shape.
    return _rotate(img, mask)
  else:
    raise ValueError('Cannot rotate an image of shape', shape)


def random_flip_left_right(images,
                           flow,
                           mask,
                           probability):
  """Performs a random left/right flip."""
  perform_flip = tf.less(tf.random.uniform([]), probability)
  # apply flip
  images = tf.cond(pred=perform_flip,
                   true_fn=lambda: tf.reverse(images, axis=[-2]),
                   false_fn=lambda: images)
  if flow is not None:
    flow = tf.cond(pred=perform_flip,
                   true_fn=lambda: tf.reverse(flow, axis=[-2]),
                   false_fn=lambda: flow)
    mask = tf.cond(pred=perform_flip,
                   true_fn=lambda: tf.reverse(mask, axis=[-2]),
                   false_fn=lambda: mask)
    # correct sign of flow
    sign_correction = tf.reshape([1.0, -1.0], [1, 1, 2])
    flow = tf.cond(pred=perform_flip,
                   true_fn=lambda: flow * sign_correction,
                   false_fn=lambda: flow)
  return images, flow, mask


def random_flip_up_down(images,
                        flow,
                        mask,
                        probability):
  """Performs a random up/down flip."""
  # 50/50 chance
  perform_flip = tf.less(tf.random.uniform([]), probability)
  # apply flip
  images = tf.cond(pred=perform_flip,
                   true_fn=lambda: tf.reverse(images, axis=[-3]),
                   false_fn=lambda: images)
  if flow is not None:
    flow = tf.cond(pred=perform_flip,
                   true_fn=lambda: tf.reverse(flow, axis=[-3]),
                   false_fn=lambda: flow)
    mask = tf.cond(pred=perform_flip,
                   true_fn=lambda: tf.reverse(mask, axis=[-3]),
                   false_fn=lambda: mask)
    # correct sign of flow
    sign_correction = tf.reshape([-1.0, 1.0], [1, 1, 2])
    flow = tf.cond(pred=perform_flip,
                   true_fn=lambda: flow * sign_correction,
                   false_fn=lambda: flow)
  return images, flow, mask


def _get_random_scaled_resolution(
    orig_height,
    orig_width,
    min_scale,
    max_scale,
    max_strech,
    probability_strech):
  """Computes a new random resolution."""
  # Choose a random scale factor and compute new resolution.
  scale = 2 ** tf.random.uniform([],
                                 minval=min_scale,
                                 maxval=max_scale,
                                 dtype=tf.float32)
  scale_height = scale
  scale_width = scale

  # Possibly change scale values individually to perform strech
  def true_fn(scale_height, scale_width):
    scale_height *= 2 ** tf.random.uniform([], -max_strech, max_strech)
    scale_width *= 2 ** tf.random.uniform([], -max_strech, max_strech)
    return tf.stack((scale_height, scale_width), axis=0)
  def false_fn(scale_height, scale_width):
    return tf.stack((scale_height, scale_width), axis=0)
  perform_strech = tf.random.uniform([]) < probability_strech
  scales = tf.cond(perform_strech,
                   lambda: true_fn(scale_height, scale_width),
                   lambda: false_fn(scale_height, scale_width))
  scale_height = scales[0]
  scale_width = scales[1]

  # Compute scaled image resolution.
  new_height = tf.cast(
      tf.math.ceil(tf.cast(orig_height, tf.float32) * scale_height), tf.int32)
  new_width = tf.cast(
      tf.math.ceil(tf.cast(orig_width, tf.float32) * scale_width), tf.int32)
  return new_height, new_width, scale


def random_scale(images,
                 flow,
                 mask,
                 min_scale,
                 max_scale,
                 max_strech,
                 probability_scale,
                 probability_strech):
  """Performs a random scaling in the given range."""
  perform_scale = tf.random.uniform([]) < probability_scale
  def true_fn(images, flow, mask):
    # Get a random new resolution to which the images will be scaled.
    orig_height = tf.shape(images)[-3]
    orig_width = tf.shape(images)[-2]
    new_height, new_width, _ = _get_random_scaled_resolution(
        orig_height=orig_height,
        orig_width=orig_width,
        min_scale=min_scale,
        max_scale=max_scale,
        max_strech=max_strech,
        probability_strech=probability_strech)

    # rescale the images (and flow)
    images = smurf_utils.resize(images, new_height, new_width, is_flow=False)

    if flow is not None:
      flow, mask = smurf_utils.resize(
          flow, new_height, new_width, is_flow=True, mask=mask)
    return images, flow, mask
  def false_fn(images, flow, mask):
    return images, flow, mask

  return tf.cond(perform_scale, lambda: true_fn(images, flow, mask),
                 lambda: false_fn(images, flow, mask))


def random_scale_second(images,
                        flow,
                        mask,
                        min_scale,
                        max_scale,
                        probability_scale):
  """Performs a random scaling on the second image in the given range."""
  perform_scale = tf.random.uniform([]) < probability_scale

  def true_fn(images, flow, mask):
    # choose a random scale factor and compute new resolution
    orig_height = tf.shape(images)[-3]
    orig_width = tf.shape(images)[-2]
    new_height, new_width, scale = _get_random_scaled_resolution(
        orig_height=orig_height,
        orig_width=orig_width,
        min_scale=min_scale,
        max_scale=max_scale,
        max_strech=0.0,
        probability_strech=0.0)

    # rescale only the second image
    image_1, image_2 = tf.unstack(images)
    image_2 = smurf_utils.resize(image_2, new_height, new_width, is_flow=False)
    # Crop either first or second image to have matching dimensions
    if scale < 1.0:
      image_1 = _center_crop(image_1, new_height, new_width)
    else:
      image_2 = _center_crop(image_2, orig_height, orig_width)
    images = tf.stack([image_1, image_2])

    if flow is not None:
      # get current locations (with the origin in the image center)
      positions = _positions_center_origin(orig_height, orig_width)

      # compute scale factor of the actual new image resolution
      scale_flow_h = tf.cast(new_height, tf.float32) / tf.cast(
          orig_height, tf.float32)
      scale_flow_w = tf.cast(new_width, tf.float32) / tf.cast(
          orig_width, tf.float32)
      scale_flow = tf.stack([scale_flow_h, scale_flow_w])

      # compute augmented flow (multiply by mask to zero invalid flow locations)
      flow = ((positions + flow) * scale_flow - positions) * mask

      if scale < 1.0:
        # in case we downsample the image we crop the reference image to keep
        # the same shape
        flow = _center_crop(flow, new_height, new_width)
        mask = _center_crop(mask, new_height, new_width)
    return images, flow, mask
  def false_fn(images, flow, mask):
    return images, flow, mask

  return tf.cond(perform_scale, lambda: true_fn(images, flow, mask),
                 lambda: false_fn(images, flow, mask))


def random_crop(images,
                flow,
                mask,
                crop_height,
                crop_width,
                relative_offset,
                probability_crop_offset):
  """Performs a random crop with the given height and width."""
  # early return if crop_height or crop_width is not specified
  if crop_height is None or crop_width is None:
    return images, flow, mask

  orig_height = tf.shape(images)[-3]
  orig_width = tf.shape(images)[-2]

  # check if crop size fits the image size
  scale = 1.0
  ratio = tf.cast(crop_height, tf.float32) / tf.cast(orig_height, tf.float32)
  scale = tf.math.maximum(scale, ratio)
  ratio = tf.cast(crop_width, tf.float32) / tf.cast(orig_width, tf.float32)
  scale = tf.math.maximum(scale, ratio)
  # compute minimum required hight
  new_height = tf.cast(
      tf.math.ceil(tf.cast(orig_height, tf.float32) * scale), tf.int32)
  new_width = tf.cast(
      tf.math.ceil(tf.cast(orig_width, tf.float32) * scale), tf.int32)
  # perform resize (scales with 1 if not required)
  images = smurf_utils.resize(images, new_height, new_width, is_flow=False)

  # compute joint offset
  max_offset_h = new_height - tf.cast(crop_height, dtype=tf.int32)
  max_offset_w = new_width - tf.cast(crop_width, dtype=tf.int32)
  joint_offset_h = tf.random.uniform([], maxval=max_offset_h+1, dtype=tf.int32)
  joint_offset_w = tf.random.uniform([], maxval=max_offset_w+1, dtype=tf.int32)

  # compute relative offset
  min_relative_offset_h = tf.math.maximum(
      joint_offset_h - relative_offset, 0)
  max_relative_offset_h = tf.math.minimum(
      joint_offset_h + relative_offset, max_offset_h)
  min_relative_offset_w = tf.math.maximum(
      joint_offset_w - relative_offset, 0)
  max_relative_offset_w = tf.math.minimum(
      joint_offset_w + relative_offset, max_offset_w)

  relative_offset_h = tf.random.uniform(
      [], minval=min_relative_offset_h, maxval=max_relative_offset_h+1,
      dtype=tf.int32)
  relative_offset_w = tf.random.uniform(
      [], minval=min_relative_offset_w, maxval=max_relative_offset_w+1,
      dtype=tf.int32)

  set_crop_offset = tf.random.uniform([]) < probability_crop_offset
  relative_offset_h = tf.cond(
      set_crop_offset, lambda: relative_offset_h, lambda: joint_offset_h)
  relative_offset_w = tf.cond(
      set_crop_offset, lambda: relative_offset_w, lambda: joint_offset_w)

  # crop both images
  image_1, image_2 = tf.unstack(images)
  image_1 = tf.image.crop_to_bounding_box(
      image_1, offset_height=joint_offset_h, offset_width=joint_offset_w,
      target_height=crop_height, target_width=crop_width)
  image_2 = tf.image.crop_to_bounding_box(
      image_2, offset_height=relative_offset_h, offset_width=relative_offset_w,
      target_height=crop_height, target_width=crop_width)
  images = tf.stack([image_1, image_2])

  if flow is not None:
    # perform resize (scales with 1 if not required)
    flow, mask = smurf_utils.resize(
        flow, new_height, new_width, is_flow=True, mask=mask)

    # crop flow and mask
    flow = tf.image.crop_to_bounding_box(
        flow,
        offset_height=joint_offset_h,
        offset_width=joint_offset_w,
        target_height=crop_height,
        target_width=crop_width)
    mask = tf.image.crop_to_bounding_box(
        mask,
        offset_height=joint_offset_h,
        offset_width=joint_offset_w,
        target_height=crop_height,
        target_width=crop_width)

    # correct flow for relative shift (/crop)
    flow_delta = tf.stack(
        [tf.cast(relative_offset_h - joint_offset_h, tf.float32),
         tf.cast(relative_offset_w - joint_offset_w, tf.float32)])
    flow = (flow - flow_delta) * mask
  return images, flow, mask, joint_offset_h, joint_offset_w


def random_rotation(images,
                    flow,
                    mask,
                    probability,
                    max_rotation,
                    not_empty_crop = True):
  """Performs a random rotation with the specified maximum rotation."""
  perform_rotation = tf.random.uniform([]) < probability
  def true_fn(images, flow, mask):

    angle_radian = tf.random.uniform(
        [], minval=-max_rotation, maxval=max_rotation,
        dtype=tf.float32) * math.pi / 180.0
    images = rotate(images, angle_radian, is_flow=False, mask=None)

    if not_empty_crop:
      orig_height = tf.shape(images)[-3]
      orig_width = tf.shape(images)[-2]
      # introduce abbreviations for shorter notation
      cos = tf.math.cos(angle_radian % math.pi)
      sin = tf.math.sin(angle_radian % math.pi)
      h = tf.cast(orig_height, tf.float32)
      w = tf.cast(orig_width, tf.float32)

      # compute required scale factor
      scale = tf.cond(tf.math.less(angle_radian % math.pi, math.pi/2.0),
                      lambda: tf.math.maximum((w/h)*sin+cos, (h/w)*sin+cos),
                      lambda: tf.math.maximum((w/h)*sin-cos, (h/w)*sin-cos))
      new_height = tf.math.floor(h / scale)
      new_width = tf.math.floor(w / scale)

      # crop image again to original size
      offset_height = tf.cast((h - new_height) / 2, tf.int32)
      offset_width = tf.cast((w - new_width) / 2, tf.int32)
      images = tf.image.crop_to_bounding_box(
          images,
          offset_height=offset_height,
          offset_width=offset_width,
          target_height=tf.cast(new_height, tf.int32),
          target_width=tf.cast(new_width, tf.int32))

    if flow is not None:
      flow, mask = rotate(flow, angle_radian, is_flow=True, mask=mask)

      if not_empty_crop:
        # crop flow and mask again to original size
        flow = tf.image.crop_to_bounding_box(
            flow,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=tf.cast(new_height, tf.int32),
            target_width=tf.cast(new_width, tf.int32))
        mask = tf.image.crop_to_bounding_box(
            mask,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=tf.cast(new_height, tf.int32),
            target_width=tf.cast(new_width, tf.int32))
    return images, flow, mask
  def false_fn(images, flow, mask):
    return images, flow, mask

  return tf.cond(perform_rotation, lambda: true_fn(images, flow, mask),
                 lambda: false_fn(images, flow, mask))


def random_rotation_second(images,
                           flow,
                           mask,
                           probability,
                           max_rotation,
                           not_empty_crop=True):
  """Performs a random rotation on only the second image."""
  perform_rotation = tf.random.uniform([]) < probability
  def true_fn(images, flow, mask):
    angle_radian = tf.random.uniform(
        [], minval=-max_rotation, maxval=max_rotation,
        dtype=tf.float32) * math.pi / 180.0

    image_1, image_2 = tf.unstack(images)
    image_2 = rotate(image_2, angle_radian, is_flow=False, mask=None)
    images = tf.stack([image_1, image_2])

    if not_empty_crop:
      orig_height = tf.shape(images)[-3]
      orig_width = tf.shape(images)[-2]
      # introduce abbreviations for shorter notation
      cos = tf.math.cos(angle_radian % math.pi)
      sin = tf.math.sin(angle_radian % math.pi)
      h = tf.cast(orig_height, tf.float32)
      w = tf.cast(orig_width, tf.float32)

      # compute required scale factor
      scale = tf.cond(tf.math.less(angle_radian % math.pi, math.pi/2.0),
                      lambda: tf.math.maximum((w/h)*sin+cos, (h/w)*sin+cos),
                      lambda: tf.math.maximum((w/h)*sin-cos, (h/w)*sin-cos))
      new_height = tf.math.floor(h / scale)
      new_width = tf.math.floor(w / scale)

      # crop image again to original size
      offset_height = tf.cast((h-new_height)/2, tf.int32)
      offset_width = tf.cast((w-new_width)/2, tf.int32)
      images = tf.image.crop_to_bounding_box(
          images,
          offset_height=offset_height,
          offset_width=offset_width,
          target_height=tf.cast(new_height, tf.int32),
          target_width=tf.cast(new_width, tf.int32))

    if flow is not None:
      # get current locations (with the origin in the image center)
      positions = _positions_center_origin(orig_height, orig_width)

      # compute augmented flow (multiply by mask to zero invalid flow locations)
      cos = tf.math.cos(angle_radian)
      sin = tf.math.sin(angle_radian)
      rotation_matrix = tf.reshape([cos, sin, -sin, cos], [2, 2])
      flow = (tf.linalg.matmul(
          (positions + flow), rotation_matrix) - positions) * mask

      if not_empty_crop:
        # crop flow and mask again to original size
        flow = tf.image.crop_to_bounding_box(
            flow,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=tf.cast(new_height, tf.int32),
            target_width=tf.cast(new_width, tf.int32))
        mask = tf.image.crop_to_bounding_box(
            mask,
            offset_height=offset_height,
            offset_width=offset_width,
            target_height=tf.cast(new_height, tf.int32),
            target_width=tf.cast(new_width, tf.int32))
    return images, flow, mask
  def false_fn(images, flow, mask):
    return images, flow, mask

  return tf.cond(perform_rotation, lambda: true_fn(images, flow, mask),
                 lambda: false_fn(images, flow, mask))


def random_color_swap(images, probability):
  """Randomly permute colors (rolling and reversing covers all permutations)."""
  perform_color_swap = tf.random.uniform([]) < probability
  def true_fn(images):
    r = tf.random.uniform([], maxval=3, dtype=tf.int32)
    images = tf.roll(images, r, axis=-1)
    r = tf.equal(tf.random.uniform([], maxval=2, dtype=tf.int32), 1)
    return tf.reverse(images, axis=[-1])
  def false_fn(images):
    return images
  return tf.cond(perform_color_swap,
                 lambda: true_fn(images),
                 lambda: false_fn(images))


def random_hue_shift(images,
                     probability,
                     max_delta):
  perform_hue_shift = tf.random.uniform([]) < probability
  return tf.cond(perform_hue_shift,
                 lambda: tf.image.random_hue(images, max_delta), lambda: images)


def random_saturation(images,
                      probability,
                      min_bound,
                      max_bound):
  perform_saturation = tf.random.uniform([]) < probability
  return tf.cond(
      perform_saturation,
      lambda: tf.image.random_saturation(images, min_bound, max_bound),
      lambda: images)


def random_brightness(images,
                      probability,
                      max_delta):
  perform_brightness = tf.random.uniform([]) < probability
  return tf.cond(
      perform_brightness,
      lambda: tf.image.random_brightness(images, max_delta),
      lambda: images)


def random_contrast(images,
                    probability,
                    min_bound,
                    max_bound):
  perform_contrast = tf.random.uniform([]) < probability
  return tf.cond(
      perform_contrast,
      lambda: tf.image.random_contrast(images, min_bound, max_bound),
      lambda: images)


def random_contrast_individual(images,
                               probability,
                               min_bound,
                               max_bound):
  perform_augmentation = tf.random.uniform([]) < probability
  def true_fn(images):
    image_1, image_2 = tf.unstack(images)
    image_1 = tf.image.random_contrast(image_1, min_bound, max_bound)
    image_2 = tf.image.random_contrast(image_2, min_bound, max_bound)
    return tf.stack([image_1, image_2])
  def false_fn(images):
    return images
  return tf.cond(perform_augmentation,
                 lambda: true_fn(images),
                 lambda: false_fn(images))


def random_brightness_individual(images,
                                 probability,
                                 max_delta):
  perform_augmentation = tf.random.uniform([]) < probability
  def true_fn(images):
    image_1, image_2 = tf.unstack(images)
    image_1 = tf.image.random_brightness(image_1, max_delta)
    image_2 = tf.image.random_brightness(image_2, max_delta)
    return tf.stack([image_1, image_2])
  def false_fn(images):
    return images
  return tf.cond(perform_augmentation,
                 lambda: true_fn(images),
                 lambda: false_fn(images))


def random_gaussian_noise(images,
                          probability,
                          min_bound,
                          max_bound):
  """Augments images by adding gaussian noise."""
  perform_gaussian_noise = tf.random.uniform([]) < probability
  def true_fn(images):
    sigma = tf.random.uniform([],
                              minval=min_bound,
                              maxval=max_bound,
                              dtype=tf.float32)
    noise = tf.random.normal(
        tf.shape(input=images), stddev=sigma, dtype=tf.float32)
    images = images + noise
  def false_fn(images):
    return images
  return tf.cond(perform_gaussian_noise,
                 lambda: true_fn(images),
                 lambda: false_fn(images))


def random_eraser(images,
                  min_size,
                  max_size,
                  probability,
                  max_operations,
                  probability_additional_operations,
                  augment_entire_batch = False):
  """Earses a random rectangle shaped areas in the second image or image batch.

  Args:
    images: Stacked image pair that should be augmented with shape
      [2, height, width, 3] or a batch of images that should be augmented with
      shape [batch, height, width, 3].
    min_size: Minimum size of erased rectangle.
    max_size: Maximum size of erased rectangle.
    probability: Probability of applying this augementation function.
    max_operations: Maximum number total areas that should be erased.
    probability_additional_operations: Probability for each additional area to
      be erased if augementation is applied.
    augment_entire_batch: If true the input is treated as batch of images to
      which the augmentation should be applid.

  Returns:
    Possibly augemented images.
  """
  perform_erase = tf.less(tf.random.uniform([]), probability)
  height = tf.shape(images)[-3]
  width = tf.shape(images)[-2]

  # Returns augemented images.
  def true_fn(images):
    if augment_entire_batch:
      image_2 = images
      mean_color = tf.reduce_mean(image_2, axis=[1, 2], keepdims=True)
      print(mean_color.shape)
    else:
      image_1, image_2 = tf.unstack(images)
      mean_color = tf.reduce_mean(image_2, axis=[0, 1], keepdims=True)
    def body(var_img, mean_color):
      x0 = tf.random.uniform([], 0, width, dtype=tf.int32)
      y0 = tf.random.uniform([], 0, height, dtype=tf.int32)
      dx = tf.random.uniform([], min_size, max_size, dtype=tf.int32)
      dy = tf.random.uniform([], min_size, max_size, dtype=tf.int32)
      x = tf.range(width)
      x_mask = (x0 <= x) & (x < x0+dx)
      y = tf.range(height)
      y_mask = (y0 <= y) & (y < y0+dy)
      mask = x_mask & y_mask[:, tf.newaxis]
      mask = tf.cast(mask[:, :, tf.newaxis], image_2.dtype)
      result = var_img * (1 - mask) + mean_color * mask
      return result
    # Perform at least one erase operation.
    image_2 = body(image_2, mean_color)
    # Perform additional erase operations.
    for _ in range(max_operations - 1):
      perform_erase = tf.less(
          tf.random.uniform([]), probability_additional_operations)
      image_2 = tf.cond(perform_erase, lambda: body(image_2, mean_color),
                        lambda: image_2)
    if augment_entire_batch:
      images = image_2
    else:
      images = tf.stack([image_1, image_2])
    return images

  # Returns unaugmented images.
  def false_fn(images):
    return images

  return tf.cond(perform_erase,
                 lambda: true_fn(images),
                 lambda: false_fn(images))


def build_selfsup_transformations(num_flow_levels=3,
                                  crop_height=0,
                                  crop_width=0,
                                  resize=True):
  """Apply augmentations to a list of student images."""
  def transform(images, is_flow, crop_height, crop_width, resize):

    height = images.shape[-3]
    width = images.shape[-2]

    op5 = tf.compat.v1.assert_greater(
        height,
        2 * crop_height,
        message='Image height is too small for cropping.')
    op6 = tf.compat.v1.assert_greater(
        width, 2 * crop_width, message='Image width is too small for cropping.')
    with tf.control_dependencies([op5, op6]):
      images = images[:, crop_height:height - crop_height,
                      crop_width:width - crop_width, :]
    if resize:
      images = smurf_utils.resize(images, height, width, is_flow=is_flow)
      images.set_shape((images.shape[0], height, width, images.shape[3]))
    else:
      images.set_shape((images.shape[0], height - 2 * crop_height,
                        width - 2 * crop_width, images.shape[3]))
    return images

  max_divisor = 2**(num_flow_levels - 1)
  assert crop_height % max_divisor == 0
  assert crop_width % max_divisor == 0
  # Compute random shifts for different images in a sequence.
  return partial(
      transform,
      crop_height=crop_height,
      crop_width=crop_width,
      resize=resize)
