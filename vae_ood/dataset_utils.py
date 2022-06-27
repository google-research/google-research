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

"""This module contains some utility functions for loading data."""

import csv
import functools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

# pylint: disable=g-bad-import-order

partial = functools.partial


def noise_generator(split=b'val', mode=b'grayscale'):
  """Generator function for the noise dataest.

  Args:
    split: Data split to load - "train", "val" or "test"
    mode: Load in "grayscale" or "color" modes
  Yields:
    An noise image
  """
  # Keeping support for train to prevent AttributeError in get_dataset(),
  # but it's not used for training
  if split == b'train':
    np.random.seed(0)
  if split == b'val':
    np.random.seed(1)
  else:
    np.random.seed(2)
  for _ in range(10000):
    if mode == b'grayscale':
      yield np.random.randint(low=0, high=256, size=(32, 32, 1))
    else:
      yield np.random.randint(low=0, high=256, size=(32, 32, 3))


def hand_sign_mnist_builder():
  """Generator function for the grayscale Hand Sign MNIST dataest.

  Source: https://www.kaggle.com/ash2703/handsignimages

  Returns:
    A dataset builder object
  """
  rootpath = 'vae_ood/datasets/SignLang'

  random.seed(42)
  if not os.path.exists(os.path.join(rootpath, 'Val')):
    os.mkdir(os.path.join(rootpath, 'Val'))
    classes = sorted(os.listdir(os.path.join(rootpath, 'Train')))
    for cls in classes:
      os.mkdir(os.path.join(rootpath, 'Val', cls))
      imgs = sorted(os.listdir(os.path.join(rootpath, 'Train', cls)))
      random.shuffle(imgs)
      for val_img in imgs[-int(len(imgs)*0.1):]:
        os.rename(os.path.join(rootpath, 'Train', cls, val_img),
                  os.path.join(rootpath, 'Val', cls, val_img))
  return tfds.folder_dataset.ImageFolder(rootpath)


def compcars_generator(split=b'train'):
  """Generator function for the CompCars Surveillance dataest.

  Source: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/

  Args:
    split: Data split to load - "train", "val" or "test".

  Yields:
    An image
  """

  rootpath = 'vae_ood/datasets/sv_data'

  random.seed(42)
  # random.seed(43) # split 2

  if split in [b'train', b'val']:
    split_path = os.path.join(rootpath, 'train_surveillance.txt')
    with open(split_path) as f:
      all_images = f.read().split()
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath, 'image', image_name))

  elif split == b'test':
    split_path = os.path.join(rootpath, 'test_surveillance.txt')
    with open(split_path) as f:
      all_images = f.read().split()
    for image_name in all_images:
      yield plt.imread(os.path.join(rootpath, 'image', image_name))


def gtsrb_generator(split=b'train'):
  """Generator function for the GTSRB Dataset.

  Source: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

  Args:
    split: Data split to load - "train", "val" or "test".

  Yields:
    An image
  """

  rootpath = 'vae_ood/datasets/GTSRB'

  random.seed(42)
  # random.seed(43) # split 2

  if split in [b'train', b'val']:
    rootpath = os.path.join(rootpath, 'Final_Training', 'Images')
    all_images = []
    for c in range(0, 43):
      prefix = rootpath + '/' + format(c, '05d') + '/'
      gt_file = open(prefix + 'GT-'+ format(c, '05d') + '.csv')
      gt_reader = csv.reader(gt_file, delimiter=';')
      next(gt_reader)

      for row in gt_reader:
        all_images.append((prefix + row[0],
                           (int(row[3]), int(row[4]), int(row[5]), int(row[6]))
                          ))
      gt_file.close()
    random.shuffle(all_images)
    if split == b'train':
      all_images = all_images[:-(len(all_images)//10)]
    else:
      all_images = all_images[-(len(all_images)//10):]
    for image, _ in all_images:
      img = plt.imread(image)
      yield img

  elif split == b'test':
    rootpath = os.path.join(rootpath, 'Final_Test', 'Images/')
    gt_file = open(rootpath + '/GT-final_test.test.csv')
    gt_reader = csv.reader(gt_file, delimiter=';')
    next(gt_reader)
    for row in gt_reader:
      img = plt.imread(rootpath + row[0])
      yield img
    gt_file.close()


def cifar10_class_generator(split, cls):
  """Generator function of class wise CIFAR10 dataset.

  Args:
    split: Data split to load - "train", "val" or "test".
    cls: The target class to load examples from.

  Yields:
    An image
  """
  (ds_train, ds_val, ds_test) = tfds.load('cifar10',
                                          split=['train[:90%]',
                                                 'train[90%:]',
                                                 'test'
                                                 ],
                                          as_supervised=True)
  # split 2
  # (ds_train, ds_val, ds_test) = tfds.load('cifar10',
  #                                         split=['train[10%:]',
  #                                                'train[:10%]',
  #                                                'test'
  #                                                ],
  #                                         as_supervised=True)

  if split == b'train':
    ds = ds_train
  elif split == b'val':
    ds = ds_val
  else:
    ds = ds_test

  for x, y in ds:
    if y == cls:
      yield x


def get_dataset(name,
                batch_size,
                mode,
                normalize=None,
                dequantize=False,
                shuffle_train=True,
                visible_dist='cont_bernoulli'
                ):
  """Returns the required dataset with custom pre-processing.

  Args:
    name: Name of the dataset. Supported names are:
      svhn_cropped
      cifar10
      celeb_a
      gtsrb
      compcars
      mnist
      fashion_mnist
      emnist/letters
      sign_lang
      noise
    batch_size: Batch Size
    mode: Load in "grayscale" or "color" modes
    normalize: Type of normalization to apply. Supported values are:
      None
      pctile-x (x is an integer)
      histeq (for color datasets)
      adhisteq (for grayscale datasets)
    dequantize: Whether to apply : dequantization
    shuffle_train: Whether to shuffle examples in the train split
    visible_dist: Visible dist of the model

  Returns:
    The train, val and test splits respectively
  """

  def preprocess(image, inverted, mode, normalize, dequantize, visible_dist):
    # pylint: disable=g-long-lambda
    if isinstance(image, dict):
      image = image['image']
    image = tf.cast(image, tf.float32)
    if dequantize:
      image += tf.random.uniform(image.shape)
      image = image / 256.0
    else:
      image = image / 255.0
    image = tf.image.resize(image, [32, 32], antialias=True)
    if mode == 'grayscale':
      if image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)
    else:
      if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)

    if isinstance(normalize, str) and normalize.startswith('pctile'):
      pct = float(normalize.split('-')[1])
      mn = tfp.stats.percentile(image, pct)
      mx = tfp.stats.percentile(image, 100-pct)
      if mx == mn:
        mn = tfp.stats.percentile(image, 0)
        mx = tfp.stats.percentile(image, 100)
      image = (image-mn)/(mx-mn)
    elif normalize == 'histeq':
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(exposure.equalize_hist(x.numpy()),
                                         dtype=tf.float32),
          [image],
          tf.float32
      )
    elif normalize == 'adhisteq':
      image = tf.py_function(
          lambda x: tf.convert_to_tensor(
              exposure.equalize_adapthist(x.numpy().squeeze())[:, :, None],
              dtype=tf.float32), [image], tf.float32)
    elif normalize is not None:
      raise NotImplementedError(
          f'Normalization method {normalize} not implemented')

    if inverted:
      image = 1 - image
    image = tf.clip_by_value(image, 0., 1.)

    target = image
    if visible_dist == 'categorical':
      target = tf.round(target*255)

    return image, target

  assert name in [
      'svhn_cropped', 'cifar10', 'celeb_a', 'gtsrb', 'compcars', 'mnist',
      'fashion_mnist', 'sign_lang', 'emnist/letters', 'noise',
      *[f'cifar10-{i}' for i in range(10)]
  ], f'Dataset {name} not supported'

  inverted = False
  if name.endswith('inverted'):
    name = name[:-9]
    inverted = True

  if name == 'noise':
    n_channels = 1 if mode == 'grayscale' else 3
    ds_train = tf.data.Dataset.from_generator(
        noise_generator,
        args=['train', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_val = tf.data.Dataset.from_generator(
        noise_generator,
        args=['val', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    ds_test = tf.data.Dataset.from_generator(
        noise_generator,
        args=['test', mode],
        output_types=tf.int32,
        output_shapes=(None, None, n_channels))
    n_examples = 1024
  elif name.startswith('gtsrb'):
    ds_train = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        gtsrb_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name == 'compcars':
    ds_train = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['train'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['val'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        compcars_generator,
        args=['test'],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    n_examples = 1024
  elif name.startswith('cifar10-'):
    n_examples = 1024
    cls = int(name.split('-')[1])
    ds_train = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['train', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_val = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['val', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
    ds_test = tf.data.Dataset.from_generator(
        cifar10_class_generator,
        args=['test', cls],
        output_types=tf.int32,
        output_shapes=(None, None, 3))
  elif name == 'sign_lang':
    builder = hand_sign_mnist_builder()
    ds_train = builder.as_dataset('Train')
    ds_val = builder.as_dataset('Val')
    ds_test = builder.as_dataset('Test')
    n_examples = 1024
  else:
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        name, split=['train[:90%]', 'train[90%:]', 'test'], with_info=True)
    # split 2
    # (ds_train, ds_val, ds_test), ds_info = tfds.load(
    #     name, split=['train[10%:]', 'train[:10%]', 'test'], with_info=True)
    n_examples = ds_info.splits['train'].num_examples

  ds_train = ds_train.map(
      partial(preprocess, inverted=inverted, mode=mode,
              normalize=normalize, dequantize=dequantize,
              visible_dist=visible_dist,),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.cache()
  if shuffle_train:
    ds_train = ds_train.shuffle(n_examples)
  ds_train = ds_train.batch(batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_val = ds_val.map(
      partial(preprocess, inverted=inverted, mode=mode,
              normalize=normalize, dequantize=dequantize,
              visible_dist=visible_dist,),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_val = ds_val.cache()
  ds_val = ds_val.batch(batch_size)
  ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.map(
      partial(preprocess, inverted=inverted, mode=mode,
              normalize=normalize, dequantize=dequantize,
              visible_dist=visible_dist,),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.batch(batch_size)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  return ds_train, ds_val, ds_test

