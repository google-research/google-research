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

r"""Functions for reading and adding noise to CIFAR datasets."""

import functools
import copy
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Define Coarse classes for CIFAR-100
_COARSE_CLASSES = [
    [4, 72, 55, 30, 95],  #pylint: disable
    [32, 1, 67, 73, 91],
    [70, 82, 54, 92, 62],
    [9, 10, 16, 28, 61],
    [0, 83, 51, 53, 57],
    [39, 40, 86, 22, 87],
    [5, 20, 84, 25, 94],
    [6, 7, 14, 18, 24],
    [97, 3, 42, 43, 88],
    [68, 37, 12, 76, 17],
    [33, 71, 49, 23, 60],
    [38, 15, 19, 21, 31],
    [64, 66, 34, 75, 63],
    [99, 77, 45, 79, 26],
    [2, 35, 98, 11, 46],
    [44, 78, 93, 27, 29],
    [65, 36, 74, 80, 50],
    [96, 47, 52, 56, 59],
    [90, 8, 13, 48, 58],
    [69, 41, 81, 85, 89]
]


def preprocess_fn(*features,
                  mean,
                  std,
                  image_size=32,
                  augment=False,
                  noise_type='none'):
  """Preprocess CIFAR-10 dataset.

  Args:
    features: Tuple of original features and corrupted labels.
    mean: Channel-wise mean for normalizing image pixels.
    std: Channel-wise standard deviation for normalizing image pixels.
    image_size: Spatial height (=width) of the image.
    augment: A `Boolean` indicating whether to do data augmentation.
    noise_type: Noise type (`none` indicates clean data).

  Returns:
    A dict of preprocessed images and labels
  """

  if noise_type != 'none':
    image = features[0]['image']
    label = tf.cast(features[1], tf.int32)  # corrupted label
  else:
    features = features[0]
    image = features['image']
    label = tf.cast(features['label'], tf.int32)
  image = tf.cast(image, tf.float32) / 255.0
  if augment:
    image = tf.image.resize_with_crop_or_pad(image, image_size + 4,
                                             image_size + 4)
    image = tf.image.random_crop(image,
                                 [image.shape[0], image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = (image - mean) / std
  else:
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = (image - mean) / std
  return dict(image=image, label=label)


def get_corrupted_labels(ds,
                         noise_type,
                         noisy_frac=0.2,
                         num_classes=10,
                         seed=1335):
  """Simulate corrupted or noisy labels.

  Args:
    ds: A Tensorflow dataset object.
    noise_type: A string specifying noise type. One of
      none/random/random_flip/random_flip_asym/random_flip_next.
    noisy_frac: A float specifying the fraction of noisy examples.
    seed: Random seed.

  Returns:
    A `numpy` 1-D array containing noisy labels.
  """
  rng = np.random.RandomState(seed)  # fix the random seed
  ds = ds.batch(
      2048, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  labels_noisy = np.zeros(50000)
  count = 0
  for batch in ds:
    label = batch['label']
    label_c = label
    if noise_type == 'random_flip':
      # noisy samples have randomly flipped label (always incorrect)
      label_c = label + rng.choice(
          num_classes,
          size=len(label),
          replace=True,
          p=np.concatenate([[1 - noisy_frac],
                            np.ones(num_classes - 1) * noisy_frac /
                            (num_classes - 1)]))
      label_c = tf.math.floormod(label_c, num_classes)
    elif noise_type == 'random':
      # noisy samples have random label (following
      # https://arxiv.org/pdf/1904.11238.pdf Sec 4.1)
      noisy_ids = rng.binomial(1, noisy_frac, len(label))
      label_c = tf.where(noisy_ids, rng.choice(num_classes, size=len(label)),
                         label)
    elif noise_type == 'random_flip_next':
      corrupted = rng.choice([0, 1],
                             size=len(label),
                             replace=True,
                             p=[1. - noisy_frac, noisy_frac])
      label_c = (label + corrupted) % num_classes
    elif noise_type == 'random_flip_asym':
      corrupted = rng.choice([0, 1],
                             size=len(label),
                             replace=True,
                             p=[1. - noisy_frac, noisy_frac])
      if num_classes == 10:  # cifar-10
        label_c = label
      elif num_classes == 100:  # cifar-100
        coarse_label = batch['coarse_label']
        label_c = []
        for ll, cc in zip(label, coarse_label):
          choices = copy.deepcopy(_COARSE_CLASSES[cc])
          choices.remove(ll)
          label_c.append(rng.choice(choices))
        label_c = np.array(label_c)
        label_c = np.where(corrupted == 0, label, label_c)
    else:
      raise ValueError('Unknown noisy type: {}'.format(noise_type))
    labels_noisy[count:count + len(label)] = label_c
    count += len(label)
  labels_noisy = labels_noisy[:count]
  if noise_type == 'random_flip_asym' and num_classes == 10:
    # cifar-10 classes: airplane : 0, automobile : 1, bird : 2, cat : 3,
    # deer : 4, dog : 5, frog : 6, horse : 7, ship : 8, truck : 9
    # noise:  truck → automobile, bird → airplane, deer → horse, cat ↔ dog
    # Also, only the subset of classes that can be mapped to other classes
    # are noisified. This corresponds to 50% of the total training examples.
    # Therefore, noisy_frac is divided by 2 to be consistent with CIFAR-100.
    # ref: https://arxiv.org/pdf/2006.13554.pdf
    noisy_frac = noisy_frac / 2.0
    noisy_examples_idx = np.arange(len(labels_noisy))[np.in1d(
        labels_noisy, [2, 3, 4, 5, 9])]
    noisy_examples_idx = noisy_examples_idx[rng.permutation(
        len(noisy_examples_idx))]
    noisy_examples_idx = noisy_examples_idx[:int(noisy_frac *
                                                 len(labels_noisy))]
    label_c = labels_noisy[noisy_examples_idx]
    label_c[label_c == 2] = 0  #  bird → airplane
    label_c[label_c == 4] = 7  #  deer → horse
    idx_cat = label_c == 3
    label_c[label_c == 5] = 3  #  dog → cat
    label_c[idx_cat] = 5  #  cat → dog
    label_c[label_c == 9] = 1  #  truck → automobile
    labels_noisy[noisy_examples_idx] = label_c
  print('total examples:', count)
  return labels_noisy


def get_dataset(batch_size,
                data='cifar10',
                num_classes=10,
                image_size=32,
                noise_type='none',
                noisy_frac=0.,
                train_on_full=False):
  r"""Create Tensorflow dataset object for CIFAR.

  Args:
    batch_size: Size of the minibatches.
    data: A string specifying the dataset (cifar10/cifar100)/
    image_size: Spatial height (=width) of the image.
    noise_type: A string specifying Noise type
      (none/random/random_flip/random_flip_asym/random_flip_next).
    train_on_full: A `Boolean` specifying whether to train on full dataset
      (True) or 90% of the dataset (False).

  Returns:
    Tensorflow dataset objects for train, validation, and test.
  """
  if data == 'cifar10':
    mean = tf.constant(
        np.reshape([0.4914, 0.4822, 0.4465], [1, 1, 1, 3]), dtype=tf.float32)
    std = tf.constant(
        np.reshape([0.2023, 0.1994, 0.2010], [1, 1, 1, 3]), dtype=tf.float32)
  elif data == 'cifar100':
    mean = tf.constant(
        np.reshape([0.5071, 0.4865, 0.4409], [1, 1, 1, 3]), dtype=tf.float32)
    std = tf.constant(
        np.reshape([0.2673, 0.2564, 0.2762], [1, 1, 1, 3]), dtype=tf.float32)
  preproc_fn_train = functools.partial(
      preprocess_fn,
      mean=mean,
      std=std,
      image_size=image_size,
      augment=True,
      noise_type=noise_type)
  if train_on_full:
    ds = tfds.load(data, split='train', with_info=False)
  else:
    ds = tfds.load(data, split='train[:90%]', with_info=False)
  if noise_type != 'none':
    labels_noisy = get_corrupted_labels(ds, noise_type, noisy_frac, num_classes)
    labels_noisy = tf.data.Dataset.from_tensor_slices(labels_noisy)
    ds = tf.data.Dataset.zip((ds, labels_noisy))
  ds = ds.repeat().shuffle(
      batch_size * 4, seed=1).batch(
          batch_size,
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.map(preproc_fn_train)

  ds_valid = tfds.load(data, split='train[90%:]', with_info=False)
  if noise_type != 'none':
    labels_noisy = get_corrupted_labels(
        ds_valid, noise_type, noisy_frac, num_classes, seed=1338)
    labels_noisy = tf.data.Dataset.from_tensor_slices(labels_noisy)
    ds_valid = tf.data.Dataset.zip((ds_valid, labels_noisy))
  ds_valid = ds_valid.shuffle(
      10000, seed=1).batch(
          batch_size,
          drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  ds_valid = ds_valid.map(
      functools.partial(
          preprocess_fn,
          mean=mean,
          std=std,
          image_size=image_size,
          noise_type=noise_type))

  ds_tst = tfds.load(data, split='test', with_info=False)
  ds_tst = ds_tst.shuffle(
      10000, seed=1).batch(
          batch_size,
          drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  ds_tst = ds_tst.map(
      functools.partial(
          preprocess_fn, mean=mean, std=std, image_size=image_size))
  return ds, ds_valid, ds_tst
