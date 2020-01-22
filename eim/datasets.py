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

"""Datasets for EIM experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfd = tfp.distributions
flags = tf.flags

flags.DEFINE_string("data_dir", None, "Directory to store datasets.")
FLAGS = flags.FLAGS
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


def compute_mean(dataset):
  def _helper(aggregate, x):
    total, n = aggregate
    return total + x, n + 1

  total, n = tfds.as_numpy(dataset.reduce((0., 0), _helper))
  return tf.to_float(total / n)


def get_mnist(split="train", shuffle_files=False):
  """Get FashionMNIST dataset."""
  split_map = {
      "train": "train",
      "valid": "validation",
      "test": "test",
  }
  datasets = dict(
      zip(["train", "validation", "test"],
          tfds.load(
              "mnist:3.*.*",
              split=["train[:50000]", "train[50000:]", "test"],
              shuffle_files=shuffle_files,
              data_dir=FLAGS.data_dir)))
  preprocess = lambda x: tf.to_float(x["image"])
  train_mean = compute_mean(datasets[split_map["train"]].map(preprocess))
  return datasets[split_map[split]].map(preprocess), train_mean


def get_static_mnist(split="train", shuffle_files=False):
  """Get Static Binarized MNIST dataset."""
  split_map = {
      "train": "train",
      "valid": "validation",
      "test": "test",
  }
  preprocess = lambda x: tf.cast(x["image"], tf.float32) * 255.
  datasets = tfds.load(name="binarized_mnist",
                       shuffle_files=shuffle_files,
                       data_dir=FLAGS.data_dir)
  train_mean = compute_mean(datasets[split_map["train"]].map(preprocess))
  return datasets[split_map[split]].map(preprocess), train_mean


def get_celeba(split="train", shuffle_files=False):
  """Get CelebA dataset."""
  split_map = {
      "train": "train",
      "valid": "validation",
      "test": "test",
  }
  datasets = tfds.load("celeb_a:2.*.*",
                       shuffle_files=shuffle_files,
                       data_dir=FLAGS.data_dir)
  def preprocess(sample, crop_width=80, image_size=CELEBA_IMAGE_SIZE):
    """Output images are in [0, 255]."""
    image_shape = sample["image"].shape
    crop_slices = [
        slice(w // 2 - crop_width, w // 2 + crop_width) for w in image_shape[:2]
    ] + [slice(None)]
    image_cropped = sample["image"][crop_slices]
    image_resized = tf.image.resize_images(image_cropped, [image_size] * 2)
    x = tf.to_float(image_resized)
    return x
  train_mean = compute_mean(datasets[split_map["train"]].map(preprocess))
  return datasets[split_map[split]].map(preprocess), train_mean


def get_fashion_mnist(split="train", shuffle_files=False):
  """Get FashionMNIST dataset."""
  split_map = {
      "train": "train",
      "valid": "train",  # No validation set, so reuse train.
      "test": "test",
  }
  datasets = tfds.load("fashion_mnist",
                       shuffle_files=shuffle_files,
                       data_dir=FLAGS.data_dir)
  preprocess = lambda x: tf.to_float(x["image"])
  train_mean = compute_mean(datasets[split_map["train"]].map(preprocess))
  return datasets[split_map[split]].map(preprocess), train_mean


def dataset_and_mean_to_batch(dataset,
                              train_mean,
                              batch_size,
                              binarize=False,
                              repeat=True,
                              shuffle=True,
                              initializable=False,
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

    return im

  dataset = dataset.map(_preprocess)

  if repeat:
    dataset = dataset.repeat()

  if shuffle:
    dataset = dataset.shuffle(1024)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  if initializable:
    itr = dataset.make_initializable_iterator()
  else:
    itr = dataset.make_one_shot_iterator()

  ims = itr.get_next()

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
          "binarize": True,
      }),
      "raw_mnist": (get_mnist, {}),
      "static_mnist": (get_static_mnist, {}),
      "jittered_mnist": (get_mnist, {
          "jitter": True,
      }),
      "jittered_celeba": (get_celeba, {
          "jitter": True
      }),
      "fashion_mnist": (get_fashion_mnist, {
          "binarize": True
      }),
      "jittered_fashion_mnist": (get_fashion_mnist, {
          "jitter": True,
      }),
  }

  dataset_fn, dataset_kwargs = dataset_map[dataset]
  raw_dataset, mean = dataset_fn(split, shuffle_files=False)
  data_batch, mean, itr = dataset_and_mean_to_batch(
      raw_dataset,
      mean,
      batch_size=batch_size,
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable,
      **dataset_kwargs)

  return tf.cast(data_batch, tf.float32), mean, itr
