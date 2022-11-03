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

"""Dataset for DisARM experiments."""
import scipy.io
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


tfd = tfp.distributions


default_omniglot_url = (
    "https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat")


def get_binarized_mnist_batch(batch_size):
  """Get MNIST that is binarized by tf.cast(x > .5, tf.float32)."""
  def _preprocess(x):
    return tf.cast(
        (tf.cast(x["image"], tf.float32) / 255.) > 0.5,
        tf.float32)

  train, valid, test = tfds.load(
      "mnist:3.*.*",
      split=["train[:50000]", "train[50000:]", "test"],
      shuffle_files=False)

  train = (train.map(_preprocess)
           .repeat()
           .shuffle(1024)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (valid.map(_preprocess)
           .shuffle(1024)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (test.map(_preprocess)
          .shuffle(1024)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_dynamic_mnist_batch(batch_size, fashion_mnist=False):
  """Transforms data based on args (assumes images in [0, 255])."""

  def _preprocess(x):
    """Sample dynamic image."""
    return tfd.Bernoulli(probs=tf.cast(x["image"], tf.float32) / 255.).sample()

  if fashion_mnist:
    dataset_name = "fashion_mnist"
  else:
    dataset_name = "mnist:3.*.*"
  train, valid, test = tfds.load(
      dataset_name,
      split=["train[:50000]", "train[50000:]", "test"],
      shuffle_files=False)

  train = (train.map(_preprocess)
           .repeat()
           .shuffle(1024)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (valid.map(_preprocess)
           .shuffle(1024)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (test.map(_preprocess)
          .shuffle(1024)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_static_mnist_batch(batch_size):
  """Get static MNIST dataset with tfds."""
  preprocess = lambda x: tf.cast(x["image"], tf.float32)
  mnist_dataset = tfds.load("binarized_mnist")
  train_ds, valid_ds, test_ds = [
      mnist_dataset[tag].map(preprocess)
      for tag in ["train", "validation", "test"]]
  train_ds = train_ds.repeat().shuffle(1024).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  valid_ds = valid_ds.shuffle(1024).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  test_ds = test_ds.shuffle(1024).batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  return train_ds, valid_ds, test_ds


def get_omniglot_batch(
    batch_size,
    omniglot_url=default_omniglot_url):
  """Load omnigload (assumes images in [0., 1.])."""
  def _preprocess(x):
    """Sample dynamic image."""
    return tfd.Bernoulli(probs=x).sample()

  with open(omniglot_url, "rb") as f:
    omni_raw = scipy.io.loadmat(f)
  num_valid = 1345  # number of validation sample
  train_data, test = omni_raw["data"], omni_raw["testdata"]
  train_data = train_data.T.reshape([-1, 28, 28, 1])
  test = test.T.reshape([-1, 28, 28, 1])
  train, valid = train_data[:-num_valid], train_data[-num_valid:]

  train = (tf.data.Dataset.from_tensor_slices(train)
           .map(_preprocess)
           .repeat()
           .shuffle(1024)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  valid = (tf.data.Dataset.from_tensor_slices(valid)
           .map(_preprocess)
           .shuffle(1024)
           .batch(batch_size)
           .prefetch(tf.data.experimental.AUTOTUNE))
  test = (tf.data.Dataset.from_tensor_slices(test)
          .map(_preprocess)
          .shuffle(1024)
          .batch(batch_size)
          .prefetch(tf.data.experimental.AUTOTUNE))
  return train, valid, test


def get_mean_from_iterator(train_ds, batch_size, dataset_size):
  """Calculate the mean of dataset."""
  train_iter = iter(train_ds)
  init_batch = tf.reshape(
      tf.cast(next(train_iter), tf.float32), [batch_size, -1])
  mean_xs = tf.reduce_mean(init_batch, axis=0)
  for i in range(1, int(dataset_size/batch_size)):
    current_batch = tf.reshape(
        tf.cast(next(train_iter), tf.float32), [batch_size, -1])
    batch_mean = tf.reduce_mean(current_batch, axis=0)
    mean_xs = (i * mean_xs + batch_mean)/(i+1)
  return mean_xs
