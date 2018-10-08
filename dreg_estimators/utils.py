# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Utility functions for DReGs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.io
import tensorflow as tf

MNIST_LOCATION = ""
OMNIGLOT_LOCATION = ""



def binarize_batch_xs(batch_xs):
  """Randomly binarize a batch of data."""
  return (batch_xs > np.random.random(size=batch_xs.shape)).astype(
      batch_xs.dtype)


def summarize_grads(grads):
  """Summarize the gradient vector."""
  grad_ema = tf.train.ExponentialMovingAverage(decay=0.99)
  vectorized_grads = tf.concat(
      [tf.reshape(g, [-1]) for g, _ in grads if g is not None], axis=0)
  new_second_moments = tf.square(vectorized_grads)
  new_first_moments = vectorized_grads
  maintain_grad_ema_op = grad_ema.apply([new_first_moments, new_second_moments])
  first_moments = grad_ema.average(new_first_moments)
  second_moments = grad_ema.average(new_second_moments)
  variances = second_moments - tf.square(first_moments)
  return (maintain_grad_ema_op, tf.reduce_mean(variances),
          tf.reduce_mean(tf.square(first_moments)) / tf.reduce_mean(variances))


def load_omniglot(dynamic_binarization=True, shuffle=True, shuffle_seed=123):
  """Load Omniglot dataset.

  Args:
    dynamic_binarization: Return the data as floats, or return the data
      binarized with a fixed seed.
    shuffle: Shuffle the train set before extracting the last examples for the
      validation set.
    shuffle_seed: Seed for the shuffling.

  Returns:
    Tuple of (train, valid, test).
  """
  n_validation = 1345  # Default magic number

  def reshape_data(data):
    return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order="fortran")

  # Try to load data locally
  if tf.gfile.Exists(os.path.join("/tmp", "omniglot.mat")):
    omni_raw = scipy.io.loadmat(os.path.join("/tmp", "omniglot.mat"))
  else:
    # Fall back to external
    with tf.gfile.GFile(OMNIGLOT_LOCATION, "rb") as f:
      omni_raw = scipy.io.loadmat(f)

  train_data = reshape_data(omni_raw["data"].T.astype("float32"))
  test_data = reshape_data(omni_raw["testdata"].T.astype("float32"))

  if not dynamic_binarization:
    # Binarize the data with a fixed seed
    np.random.seed(5)
    train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
    test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

  if shuffle:
    permutation = np.random.RandomState(seed=shuffle_seed).permutation(
        train_data.shape[0])
    train_data = train_data[permutation]

  train_data, valid_data = (train_data[:-n_validation],
                            train_data[-n_validation:])

  return train_data, valid_data, test_data


def load_mnist():
  """Load the MNIST training set."""

  def load_dataset(dataset="train_xs"):
    if os.path.exists("/tmp/%s.npy" % dataset):
      with tf.gfile.Open("/tmp/%s.npy" % dataset, "rb") as f:
        xs = np.load(f).reshape(-1, 784)
    else:
      with tf.gfile.Open(
          os.path.join(MNIST_LOCATION, "%s.npy" % dataset), "rb") as f:
        xs = np.load(f).reshape(-1, 784)

    return xs.astype(np.float32)

  train_xs = load_dataset("train_xs")
  test_xs = load_dataset("test_xs")
  valid_xs = load_dataset("valid_xs")

  return train_xs, valid_xs, test_xs
