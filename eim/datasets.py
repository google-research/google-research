# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Datasets for EIM experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfd = tfp.distributions
flags = tf.flags

flags.DEFINE_string("ROOT_PATH", "/tmp", "The root directory of datasets.")
FLAGS = flags.FLAGS

ROOT_PATH = lambda: FLAGS.ROOT_PATH
MNIST_PATH = "data/mnist"
STATIC_BINARIZED_MNIST_PATH = "data/static_binarized_mnist"
CELEBA_PATH = "data/celeba"
FASHION_MNIST_PATH = "data/fashion_mnist"
CELEBA_IMAGE_SIZE = 64


def get_nine_gaussians(batch_size, scale=0.1, spacing=1.0):
  """Creates a mixture of 9 2-D gaussians on a 3x3 grid centered at 0."""
  components = []
  for i in [-spacing, 0., spacing]:
    for j in [-spacing, 0., spacing]:
      loc = tf.constant([i, j], dtype=tf.float32)
      scale = tf.ones_like(loc) * scale
      components.append(tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale))

  dist = tfd.Mixture(
      cat=tfd.Categorical(probs=tf.ones([9], dtype=tf.float32) / 9.),
      components=components)
  batch = dist.sample(batch_size)
  return batch


def get_mnist(split="train", data_dir=MNIST_PATH, shuffle_files=None):
  """Get MNIST dataset."""
  del shuffle_files  # Ignored
  path = os.path.join(ROOT_PATH(), data_dir, split + ".npy")
  with tf.io.gfile.GFile(path, "rb") as f:
    np_ims = np.load(f)
  # Always load the train mean, no matter what split.
  mean_path = os.path.join(ROOT_PATH(), data_dir, "train_mean.npy")
  with tf.io.gfile.GFile(mean_path, "rb") as f:
    mean = np.load(f).astype(np.float32)
  dataset = tf.data.Dataset.from_tensor_slices(np_ims)

  mean *= 255.
  dataset = dataset.map(lambda im: tf.to_float(im) * 255.)

  return dataset, mean


def get_static_mnist(split="train", shuffle_files=None):
  return get_mnist(split, data_dir=STATIC_BINARIZED_MNIST_PATH,
                   shuffle_files=shuffle_files)


def get_celeba(split="train", shuffle_files=False):
  """Get CelebA dataset."""
  split_map = {
      "train": "train",
      "valid": "validation",
      "test": "test",
  }
  datasets = tfds.load("celeb_a", shuffle_files=shuffle_files)

  mean_path = os.path.join(ROOT_PATH(), CELEBA_PATH, "train_mean.npy")
  with tf.io.gfile.GFile(mean_path, "rb") as f:
    train_mean = np.load(f).astype(np.float32)

  def _preprocess(sample, crop_width=80, image_size=CELEBA_IMAGE_SIZE):
    """Output images are in [0, 255]."""
    image_shape = sample["image"].shape
    crop_slices = [
        slice(w // 2 - crop_width, w // 2 + crop_width) for w in image_shape[:2]
    ] + [slice(None)]
    image_cropped = sample["image"][crop_slices]
    image_resized = tf.image.resize_images(image_cropped, [image_size] * 2)
    x = tf.to_float(image_resized)
    return x

  data = datasets[split_map[split]].map(_preprocess)
  return data, train_mean


def get_fashion_mnist(split="train", shuffle_files=False):
  """Get FashionMNIST dataset."""
  split_map = {
      "train": "train",
      "valid": "train",  # No validation set, so reuse train.
      "test": "test",
  }
  dataset = (
      tfds.load(name="fashion_mnist",
                split=split_map[split],
                shuffle_files=shuffle_files,
               ).map(lambda x: tf.to_float(x["image"])))

  train_mean_path = os.path.join(ROOT_PATH(), FASHION_MNIST_PATH,
                                 "train_mean.npy")
  with tf.io.gfile.GFile(train_mean_path, "rb") as f:
    train_mean = np.load(f).astype(np.float32)
  return dataset, train_mean


def dataset_and_mean_to_batch(dataset,
                              train_mean,
                              batch_size,
                              binarize=False,
                              repeat=True,
                              shuffle=True,
                              initializable=False,
                              flatten=False,
                              jitter=False):
  """Transforms data based on args (assumes images in [0, 255])."""

  def jitter_im(im):
    jitter_noise = tfd.Uniform(
        low=tf.zeros_like(im), high=tf.ones_like(im)).sample()
    jittered_im = im + jitter_noise
    return jittered_im

  def _preprocess(im):
    """Preprocess the image."""
    assert not (jitter and
                binarize), "Can only choose binarize or jitter, not both."

    if jitter:
      im = jitter_im(im)
    elif binarize:
      im = tfd.Bernoulli(probs=im / 255.).sample()
    else:  # [0, 1]
      im /= 255.

    if flatten:
      im = tf.reshape(im, [-1])
    return im

  dataset = dataset.map(_preprocess)

  if repeat:
    dataset = dataset.repeat()

  dataset = dataset.cache()

  if shuffle:
    dataset = dataset.shuffle(1024)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  if initializable:
    itr = dataset.make_initializable_iterator()
  else:
    itr = dataset.make_one_shot_iterator()

  ims = itr.get_next()

  if flatten:
    train_mean = tf.reshape(train_mean, [-1])
  if jitter:
    train_mean += 0.5
  elif binarize:
    train_mean /= 255.
  else:
    train_mean /= 255.

  return ims, train_mean[None], itr


def get_dataset(dataset,
                batch_size,
                split,
                repeat=True,
                shuffle=True,
                initializable=False):
  """Return the reference dataset with options."""
  dataset_map = {
      "dynamic_mnist": (get_mnist, {
          "binarize": True
      }),
      "raw_mnist": (get_mnist, {}),
      "static_mnist": (get_static_mnist, {}),
      "jittered_mnist": (get_mnist, {
          "jitter": True
      }),
      "jittered_celeba": (get_celeba, {
          "jitter": True
      }),
      "jittered_flat_celeba": (get_celeba, {
          "jitter": True,
          "flatten": True
      }),
      "fashion_mnist": (get_fashion_mnist, {
          "binarize": True
      }),
      "flat_fashion_mnist": (get_fashion_mnist, {
          "binarize": True,
          "flatten": True
      }),
      "jittered_flat_fashion_mnist": (get_fashion_mnist, {
          "jitter": True,
          "flatten": True
      }),
  }

  dataset_fn, dataset_kwargs = dataset_map[dataset]
  raw_dataset, mean = dataset_fn(split, shuffle_files=shuffle)
  data_batch, mean, itr = dataset_and_mean_to_batch(
      raw_dataset,
      mean,
      batch_size=batch_size,
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable,
      **dataset_kwargs)

  return tf.cast(data_batch, tf.float32), mean, itr
