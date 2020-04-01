# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

# Does multiprocessing speed things up?
POOL = None
# POOL = multiprocessing.Pool(5)


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
