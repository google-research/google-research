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

"""UFlow augmentation.

This library contains various augmentation functions.
"""

# pylint:disable=g-importing-member
from functools import partial
from math import pi

import gin
import gin.tf
import tensorflow as tf
from tensorflow_addons import image as tfa_image

from uflow import uflow_utils


def apply_augmentation(images, flow=None, mask=None,
                       crop_height=640, crop_width=640):
  """Applies photometric and geometric augmentations to images and flow."""
  # ensure sequence length of two, to be able to unstack images
  images = tf.ensure_shape(images, (2, None, None, None))
  # apply geometric augmentation functions
  images, flow, mask = geometric_augmentation(
      images, flow, mask, crop_height, crop_width)
  # apply photometric augmentation functions
  images_aug = photometric_augmentation(images)

  # return flow and mask if available
  if flow is not None:
    return images_aug, images, flow, mask
  return images_aug, images


@gin.configurable
def photometric_augmentation(images,
                             augment_color_swap=True,
                             augment_hue_shift=True,
                             augment_saturation=False,
                             augment_brightness=False,
                             augment_contrast=False,
                             augment_gaussian_noise=False,
                             augment_brightness_individual=False,
                             augment_contrast_individual=False,
                             max_delta_hue=0.5,
                             min_bound_saturation=0.8,
                             max_bound_saturation=1.2,
                             max_delta_brightness=0.1,
                             min_bound_contrast=0.8,
                             max_bound_contrast=1.2,
                             min_bound_gaussian_noise=0.0,
                             max_bound_gaussian_noise=0.02,
                             max_delta_brightness_individual=0.02,
                             min_bound_contrast_individual=0.95,
                             max_bound_contrast_individual=1.05):
  """Applies photometric augmentations to an image pair."""
  # Randomly permute colors by rolling and reversing.
  # This covers all permutations.
  if augment_color_swap:
    r = tf.random.uniform([], maxval=3, dtype=tf.int32)
    images = tf.roll(images, r, axis=-1)
    r = tf.equal(tf.random.uniform([], maxval=2, dtype=tf.int32), 1)
    images = tf.cond(pred=r,
                     true_fn=lambda: tf.reverse(images, axis=[-1]),
                     false_fn=lambda: images)

  if augment_hue_shift:
    images = tf.image.random_hue(images, max_delta_hue)

  if augment_saturation:
    images = tf.image.random_saturation(
        images, min_bound_saturation, max_bound_saturation)

  if augment_brightness:
    images = tf.image.random_brightness(images, max_delta_brightness)

  if augment_contrast:
    images = tf.image.random_contrast(
        images, min_bound_contrast, max_bound_contrast)

  if augment_gaussian_noise:
    sigma = tf.random.uniform([],
                              minval=min_bound_gaussian_noise,
                              maxval=max_bound_gaussian_noise,
                              dtype=tf.float32)
    noise = tf.random.normal(
        tf.shape(input=images), stddev=sigma, dtype=tf.float32)
    images = images + noise

  # perform relative photometric augmentation (individually per image)
  image_1, image_2 = tf.unstack(images)
  if augment_brightness_individual:
    image_1 = tf.image.random_contrast(
        image_1, min_bound_contrast_individual, max_bound_contrast_individual)
    image_2 = tf.image.random_contrast(
        image_2, min_bound_contrast_individual, max_bound_contrast_individual)

  if augment_contrast_individual:
    image_1 = tf.image.random_brightness(
        image_1, max_delta_brightness_individual)
    image_2 = tf.image.random_brightness(
        image_2, max_delta_brightness_individual)

  # crop values to ensure values in [0,1] (some augmentations can violate this)
  image_1 = tf.clip_by_value(image_1, 0.0, 1.0)
  image_2 = tf.clip_by_value(image_2, 0.0, 1.0)
  return tf.stack([image_1, image_2])


@gin.configurable
def geometric_augmentation(images,
                           flow=None,
                           mask=None,
                           crop_height=640,
                           crop_width=640,
                           augment_flip_left_right=False,
                           augment_flip_up_down=False,
                           augment_scale=False,
                           augment_relative_scale=False,
                           augment_rotation=False,
                           augment_relative_rotation=False,
                           augment_crop_offset=False,
                           min_bound_scale=0.9,
                           max_bound_scale=1.5,
                           min_bound_relative_scale=0.95,
                           max_bound_relative_scale=1.05,
                           max_rotation_deg=15,
                           max_relative_rotation_deg=3,
                           max_relative_crop_offset=5):
  """Apply geometric augmentations to an image pair and corresponding flow."""

  # apply geometric augmentation
  if augment_flip_left_right:
    images, flow, mask = random_flip_left_right(images, flow, mask)

  if augment_flip_up_down:
    images, flow, mask = random_flip_up_down(images, flow, mask)

  if augment_scale:
    images, flow, mask = random_scale(
        images,
        flow,
        mask,
        min_scale=min_bound_scale,
        max_scale=max_bound_scale)

  if augment_relative_scale:
    images, flow, mask = random_scale_second(
        images, flow, mask,
        min_scale=min_bound_relative_scale, max_scale=max_bound_relative_scale)

  if augment_rotation:
    images, flow, mask = random_rotation(
        images, flow, mask,
        max_rotation=max_rotation_deg, not_empty_crop=True)

  if augment_relative_rotation:
    images, flow, mask = random_rotation_second(
        images, flow, mask,
        max_rotation=max_relative_rotation_deg, not_empty_crop=True)

  # always perform random cropping
  if not augment_crop_offset:
    max_relative_crop_offset = 0
  images, flow, mask = random_crop(
      images, flow, mask, crop_height, crop_width,
      relative_offset=max_relative_crop_offset)

  # return flow and mask if available
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


def rotate(img, angle_radian, is_flow, mask=None):
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


def random_flip_left_right(images, flow=None, mask=None):
  """Performs a random left/right flip."""
  # 50/50 chance
  perform_flip = tf.equal(tf.random.uniform([], maxval=2, dtype=tf.int32), 1)
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


def random_flip_up_down(images, flow=None, mask=None):
  """Performs a random up/down flip."""
  # 50/50 chance
  perform_flip = tf.equal(tf.random.uniform([], maxval=2, dtype=tf.int32), 1)
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


def random_scale(images, flow=None, mask=None, min_scale=1.0, max_scale=1.0):
  """Performs a random scaling in the given range."""
  # choose a random scale factor and compute new resolution
  orig_height = tf.shape(images)[-3]
  orig_width = tf.shape(images)[-2]
  scale = tf.random.uniform([],
                            minval=min_scale,
                            maxval=max_scale,
                            dtype=tf.float32)
  new_height = tf.cast(
      tf.math.ceil(tf.cast(orig_height, tf.float32) * scale), tf.int32)
  new_width = tf.cast(
      tf.math.ceil(tf.cast(orig_width, tf.float32) * scale), tf.int32)

  # rescale the images (and flow)
  images = uflow_utils.resize(images, new_height, new_width, is_flow=False)
  if flow is not None:
    flow, mask = uflow_utils.resize(
        flow, new_height, new_width, is_flow=True, mask=mask)
  return images, flow, mask


def random_scale_second(
    images, flow=None, mask=None, min_scale=1.0, max_scale=1.0):
  """Performs a random scaling on the second image in the given range."""
  # choose a random scale factor and compute new resolution
  orig_height = tf.shape(images)[-3]
  orig_width = tf.shape(images)[-2]
  scale = tf.random.uniform(
      [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  new_height = tf.cast(
      tf.math.ceil(tf.cast(orig_height, tf.float32) * scale), tf.int32)
  new_width = tf.cast(
      tf.math.ceil(tf.cast(orig_width, tf.float32) * scale), tf.int32)

  # rescale only the second image
  image_1, image_2 = tf.unstack(images)
  image_2 = uflow_utils.resize(image_2, new_height, new_width, is_flow=False)
  # crop either first or second image to have matching dimensions
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
      # in case we downsample the image we crop the reference image to keep the
      # same shape
      flow = _center_crop(flow, new_height, new_width)
      mask = _center_crop(mask, new_height, new_width)
  return images, flow, mask


def random_crop(images, flow=None, mask=None, crop_height=None, crop_width=None,
                relative_offset=0):
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
  images = uflow_utils.resize(images, new_height, new_width, is_flow=False)

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
    flow, mask = uflow_utils.resize(
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
  return images, flow, mask


def random_rotation(
    images, flow=None, mask=None, max_rotation=10, not_empty_crop=True):
  """Performs a random rotation with the specified maximum rotation."""

  angle_radian = tf.random.uniform(
      [], minval=-max_rotation, maxval=max_rotation,
      dtype=tf.float32) * pi / 180.0
  images = rotate(images, angle_radian, is_flow=False, mask=None)

  if not_empty_crop:
    orig_height = tf.shape(images)[-3]
    orig_width = tf.shape(images)[-2]
    # introduce abbreviations for shorter notation
    cos = tf.math.cos(angle_radian % pi)
    sin = tf.math.sin(angle_radian % pi)
    h = tf.cast(orig_height, tf.float32)
    w = tf.cast(orig_width, tf.float32)

    # compute required scale factor
    scale = tf.cond(tf.math.less(angle_radian % pi, pi/2.0),
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


def random_rotation_second(
    images, flow=None, mask=None, max_rotation=10, not_empty_crop=True):
  """Performs a random rotation on only the second image."""

  angle_radian = tf.random.uniform(
      [], minval=-max_rotation, maxval=max_rotation, dtype=tf.float32)*pi/180.0

  image_1, image_2 = tf.unstack(images)
  image_2 = rotate(image_2, angle_radian, is_flow=False, mask=None)
  images = tf.stack([image_1, image_2])

  if not_empty_crop:
    orig_height = tf.shape(images)[-3]
    orig_width = tf.shape(images)[-2]
    # introduce abbreviations for shorter notation
    cos = tf.math.cos(angle_radian % pi)
    sin = tf.math.sin(angle_radian % pi)
    h = tf.cast(orig_height, tf.float32)
    w = tf.cast(orig_width, tf.float32)

    # compute required scale factor
    scale = tf.cond(tf.math.less(angle_radian % pi, pi/2.0),
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
    flow = (tf.linalg.matmul((positions+flow), rotation_matrix)-positions)*mask

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


def build_selfsup_transformations(num_flow_levels=3,
                                  seq_len=2,
                                  crop_height=0,
                                  crop_width=0,
                                  max_shift_height=0,
                                  max_shift_width=0,
                                  resize=True):
  """Apply augmentations to a list of student images."""

  def transform(images, i_or_ij, is_flow, crop_height, crop_width,
                shift_heights, shift_widths, resize):
    # Expect (i, j) for flows and masks and i for images.
    if isinstance(i_or_ij, int):
      i = i_or_ij
      # Flow needs i and j.
      assert not is_flow
    else:
      i, j = i_or_ij

    if is_flow:
      shifts = tf.stack([shift_heights, shift_widths], axis=-1)
      flow_offset = shifts[i] - shifts[j]
      images = images + tf.cast(flow_offset, tf.float32)

    shift_height = shift_heights[i]
    shift_width = shift_widths[i]
    height = images.shape[-3]
    width = images.shape[-2]

    # Assert that the cropped bounding box does not go out of the image frame.
    op1 = tf.compat.v1.assert_greater_equal(crop_height + shift_height, 0)
    op2 = tf.compat.v1.assert_greater_equal(crop_width + shift_width, 0)
    op3 = tf.compat.v1.assert_less_equal(height - crop_height + shift_height,
                                         height)
    op4 = tf.compat.v1.assert_less_equal(width - crop_width + shift_width,
                                         width)
    op5 = tf.compat.v1.assert_greater(
        height,
        2 * crop_height,
        message='Image height is too small for cropping.')
    op6 = tf.compat.v1.assert_greater(
        width, 2 * crop_width, message='Image width is too small for cropping.')
    with tf.control_dependencies([op1, op2, op3, op4, op5, op6]):
      images = images[:, crop_height + shift_height:height - crop_height +
                      shift_height, crop_width + shift_width:width -
                      crop_width + shift_width, :]
    if resize:
      images = uflow_utils.resize(images, height, width, is_flow=is_flow)
      images.set_shape((images.shape[0], height, width, images.shape[3]))
    else:
      images.set_shape((images.shape[0], height - 2 * crop_height,
                        width - 2 * crop_width, images.shape[3]))
    return images

  max_divisor = 2**(num_flow_levels - 1)
  assert crop_height % max_divisor == 0
  assert crop_width % max_divisor == 0
  assert max_shift_height <= crop_height
  assert max_shift_width <= crop_width
  # Compute random shifts for different images in a sequence.
  if max_shift_height > 0 or max_shift_width > 0:
    max_rand = max_shift_height // max_divisor
    shift_height_at_highest_level = tf.random.uniform([seq_len],
                                                      minval=-max_rand,
                                                      maxval=max_rand + 1,
                                                      dtype=tf.int32)
    shift_heights = shift_height_at_highest_level * max_divisor

    max_rand = max_shift_height // max_divisor
    shift_width_at_highest_level = tf.random.uniform([seq_len],
                                                     minval=-max_rand,
                                                     maxval=max_rand + 1,
                                                     dtype=tf.int32)
    shift_widths = shift_width_at_highest_level * max_divisor
  transform_fns = []
  for level in range(num_flow_levels):

    if max_shift_height == 0 and max_shift_width == 0:
      shift_heights = [0, 0]
      shift_widths = [0, 0]
    else:
      shift_heights = shift_heights // (2**level)
      shift_widths = shift_widths // (2**level)

    fn = partial(
        transform,
        crop_height=crop_height // (2**level),
        crop_width=crop_width // (2**level),
        shift_heights=shift_heights,
        shift_widths=shift_widths,
        resize=resize)
    transform_fns.append(fn)
  assert len(transform_fns) == num_flow_levels
  return transform_fns
