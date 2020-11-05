# coding=utf-8
"""Utility functions for supporting the code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ieg.dataset_utils import augmentation_transforms
from ieg.dataset_utils import autoaugment
from ieg.dataset_utils import randaugment

import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

GODD_POLICIES = autoaugment.cifar10_policies()
RANDOM_POLICY_OPS = randaugment.RANDOM_POLICY_OPS
_IMAGE_SIZE = 224
_CROP_PADDING = 32
# Does multiprocessing speed things up?
POOL = None


def cifar_process(image, augmentation=True):
  """Map function for cifar dataset.

  Args:
    image: An image tensor.
    augmentation: If True, process train images.

  Returns:
    A processed image tensor.
  """
  # label = tf.cast(label, dtype=tf.int32)
  image = tf.math.divide(tf.cast(image, dtype=tf.float32), 255.0)

  if augmentation:
    image = tf.image.resize_image_with_crop_or_pad(image, 32 + 4, 32 + 4)
    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [32, 32, 3])
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, 0, 1)

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image


def apply_autoaugment(data, no_policy=False):
  """Python written Autoaugment for preprocessing image.

  Args:
    data: can be
      A list: (3D image, policy)
      A list: (4D image, policy) A numpy image
    no_policy: If True, does not used learned AutoAugment policies

  Returns:
    A 3D or 4D processed images
  """
  if not isinstance(data, (list, tuple)):
    epoch_policy = GODD_POLICIES[np.random.choice(len(GODD_POLICIES))]
    image = data
  else:
    image, epoch_policy = data

  if len(image.shape) == 3:
    images = [image]
  else:
    images = image
  res = []
  for img in images:
    assert img.max() <= 1 and img.min() >= -1
    # ! image is assumed to be normalized to [-1, 1]
    if no_policy:
      final_img = img
    else:
      final_img = augmentation_transforms.apply_policy(epoch_policy, img)

    final_img = augmentation_transforms.random_flip(
        augmentation_transforms.zero_pad_and_crop(final_img, 4))
    final_img = augmentation_transforms.cutout_numpy(final_img)
    res.append(final_img.astype(np.float32))

  res = np.concatenate(res, 0)

  return res


def apply_randomaugment(data):
  """Apply random augmentations."""
  image, epoch_policy = data

  if len(image.shape) == 3:
    images = [image]
  else:
    images = image
  res = []
  for img in images:
    # ! image is assumed to be normalized to [-1, 1]
    final_img = randaugment.apply_policy(epoch_policy, img)
    final_img = randaugment.random_flip(
        randaugment.zero_pad_and_crop(final_img, 4))
    final_img = randaugment.cutout_numpy(final_img)
    res.append(final_img.astype(np.float32))

  res = np.concatenate(res, 0)

  return res


def pool_policy_augmentation(images):
  """Batch AutoAugment.

  Given a 4D numpy tensor of images,
  perform AutoAugment using apply_autoaugment().

  Args:
    images: 4D numpy tensor

  Returns:
    A 4D numpy tensor of processed images.

  """
  # Use the same policy for all batch data seems work better.
  policies = [GODD_POLICIES[np.random.choice(len(GODD_POLICIES))]
             ] * images.shape[0]
  jobs = [(image.squeeze(), policy) for image, policy in zip(
      np.split(images.copy(), images.shape[0], axis=0), policies)]
  if POOL is None:
    jobs = np.split(images.copy(), images.shape[0], axis=0)
    augmented_images = map(apply_autoaugment, jobs)
  else:
    augmented_images = POOL.map(apply_autoaugment, jobs)
  augmented_images = np.stack(augmented_images, axis=0)

  return augmented_images


def random_augmentation(images, magnitude=10, nops=2):
  """Apply random augmentations for a batch of data."""
  # using shared policies are better
  policies = [(policy, 0.5, mag) for (policy, mag) in zip(
      np.random.choice(RANDOM_POLICY_OPS, nops),
      np.random.randint(1, magnitude, nops))]
  policies = [policies] * images.shape[0]
  if POOL is not None:
    jobs = [(image.squeeze(), policy) for image, policy in zip(
        np.split(images.copy(), images.shape[0], axis=0), policies)]
    augmented_images = POOL.map(apply_randomaugment, jobs)
  else:
    augmented_images = []
    for image, policy in zip(images.copy(), policies):
      final_img = apply_randomaugment((image, policy))
      augmented_images.append(final_img)

  augmented_images = np.stack(augmented_images, axis=0)

  return augmented_images


def autoaug_batch_process_map_fn(images, labels):
  """tf.data.Dataset map function to enable python AutoAugmnet with tf.py_func.

  It is usually called after tf.data.Dataset is batched.

  Args:
    images: A 4D tensor of a batch of images.
    labels: labels of images.

  Returns:
    A 5D tensor of processed images [Bx2xHxWx3].
  """
  if FLAGS.aug_type == 'autoaug':
    aa_images = tf.py_func(pool_policy_augmentation, [tf.identity(images)],
                           [tf.float32])
  elif FLAGS.aug_type == 'randaug':
    aa_images = tf.py_func(random_augmentation, [tf.identity(images)],
                           [tf.float32])
  elif FLAGS.aug_type == 'default':
    aa_images = tf.py_func(cifar_process, [tf.identity(images)], [tf.float32])
  else:
    raise NotImplementedError('{} aug_type does not exist'.format(
        FLAGS.aug_type))
  aa_images = tf.reshape(aa_images, [-1] + images.shape.as_list()[1:])
  images = tf.concat([tf.expand_dims(images, 1),
                      tf.expand_dims(aa_images, 1)],
                     axis=1)
  return images, labels


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
      each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
      xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
      of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image must
      contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional `str` for name scope.

  Returns:
    cropped image `Tensor`
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """Checks if at least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Makes a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes, image_size),
      lambda: tf.image.resize_bicubic(  # pylint: disable=g-long-lambda
          [image], [image_size, image_size])[0])

  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + _CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def preprocess_for_train(image_bytes,
                         use_bfloat16,
                         image_size=_IMAGE_SIZE,
                         autoaugment_name=None):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.
    autoaugment_name: `string` that is the name of the autoaugment policy to
      apply to the image. If the value is `None` autoaugment will not be
      applied.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

  if autoaugment_name:
    tf.logging.info('Apply AutoAugment policy {}'.format(autoaugment_name))
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)
    # Random aug shouldo also work.
    image = autoaugment.distort_image_with_autoaugment(image, autoaugment_name)
    image = tf.cast(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_for_eval(image_bytes, use_bfloat16, image_size=_IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def cutout(image, pad_size, replace=0):
  """Applies cutout (https://arxiv.org/abs/1708.04552) to image."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied
  cutout_center_height = tf.random_uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random_uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace, image)
  return image


def preprocess_image(image_bytes,
                     is_training=False,
                     use_bfloat16=False,
                     image_size=_IMAGE_SIZE,
                     autoaugment_name=None):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.
    autoaugment_name: `string` that is the name of the autoaugment policy to
      apply to the image. If the value is `None` autoaugment will not be
      applied.

  Returns:
    A preprocessed image `Tensor` with value range of [0, 1].
  """

  if is_training:
    image = preprocess_for_train(image_bytes, use_bfloat16, image_size,
                                 autoaugment_name)
  else:
    image = preprocess_for_eval(image_bytes, use_bfloat16, image_size)

  # rescale to [-1, 1]
  image = tf.math.divide(image, 255.0)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image


def imagenet_preprocess_image(image_bytes,
                              is_training=False,
                              use_bfloat16=False,
                              image_size=_IMAGE_SIZE,
                              autoaugment_name=None,
                              use_cutout=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.
    autoaugment_name: `string` that is the name of the autoaugment policy to
      apply to the image. If the value is `None` autoaugment will not be
      applied.
    use_cutout: 'bool' for whether use cutout.

  Returns:
    A preprocessed image `Tensor` with value range of [0, 1].
  """

  if is_training:
    image = preprocess_for_train(image_bytes, use_bfloat16, image_size,
                                 autoaugment_name)
    if use_cutout:
      image = cutout(image, pad_size=8)
  else:
    image = preprocess_for_eval(image_bytes, use_bfloat16, image_size)

  # clip the extra values
  image = tf.clip_by_value(image, 0.0, 255.0)
  image = tf.math.divide(image, 255.0)
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  return image
