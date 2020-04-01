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

"""Dataset-related utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


_THRESHOLD = 0.7
HASH_ARGS = {"train": {"split_name": "train", "hash_args": [5, (0, 1, 2, 3)]},
             "valid": {"split_name": "train", "hash_args": [5, (4,)]},
             "test": {"split_name": "test", "hash_args": [1, (0,)]}}


def filter_label_function(classes):
  """Return a function which returns true if an examples label is in classes.

  Args:
    classes (list): which classes to keep.
  Returns:
    f (function): a function for filtering to keep only examples in classes.
  """
  classes_array = np.array(classes).astype(np.int64)
  def f(x):
    return tf.math.reduce_any(tf.equal(classes_array, x["label"]))
  return f


def filter_example_function(example):
  def f(x, _):
    return tf.reduce_any(tf.math.not_equal(x, example))
  return f


def get_hash(x, buckets=5):
  s = tf.strings.as_string(x)
  s_joined = tf.strings.reduce_join(s)
  return tf.strings.to_hash_bucket(s_joined, num_buckets=buckets)


def get_hash_filter_function(buckets, keep_vals):
  """Return a function which filters an (x, y) pair by its hash value.

  This is used to randomly split out a validation set from the training set
  defined in tfds. The validation set will be of size
  (len(keep_vals) / buckets) * (original training set size).

  Args:
    buckets (int): how many buckets to hash into.
    keep_vals (list of ints): which hash values we want to keep.
  Returns:
    f (function): a function which filters an (x, y) pair by its hash value.
  """
  def f(x):
    xinp = tf.concat([tf.reshape(tf.cast(x["image"], tf.float32), [-1, 1]),
                      tf.reshape(tf.cast(x["label"], tf.float32), [-1, 1])],
                     axis=0)
    hash_value = get_hash(xinp, buckets=buckets)
    return tf.reduce_any(tf.equal(hash_value, keep_vals))
  return f


def process_image_tuple_function(threshold):
  """Returns a function for processing image from raw data.

  Args:
    threshold (float): cutoff for binarizing pixels, between 0 and 1.
  Returns:
    f (function): image processing function.
  """
  def f(x):
    """Returns a binarized version of a greyscale image and its label.

    Args:
      x (float): where to cut off between white and black.
    Returns:
      image, label (tuple): binarized image and its associated label.
    """
    return (tf.round(tf.cast(x["image"], "float") / 255. / 2. / threshold),
            x["label"])
  return f


def process_image_tuple_function_onehot(threshold):
  """Returns a function for processing image from raw data.

  Args:
    threshold (float): cutoff for binarizing pixels, between 0 and 1.
  Returns:
    f (function): image processing function.
  """
  proc_function = process_image_tuple_function(threshold)
  def f(x):
    """Returns a binarized version of a greyscale image and its label.

    Args:
      x (float): where to cut off between white and black.
    Returns:
      image, label (tuple): binarized image and its associated one-hot label.
    """
    img, label = proc_function(x)
    return (img, make_onehot(label, 10))
  return f


def process_image_function(threshold):
  """Returns a function for processing image from raw data.

  Args:
    threshold (float): cutoff for binarizing pixels, between 0 and 1.
  Returns:
    f (function): image processing function.
  """
  processing_function = process_image_tuple_function(threshold)
  def f(x):
    """Returns a binarized version of a greyscale image.

    Args:
      x (float): where to cut off between white and black.
    Returns:
      image (tuple): binarized image.
    """
    img, _ = processing_function(x)
    return img
  return f


def make_labels_noisy_function(noise_prob=0.0):
  """Returns a function for adding label noise to one-hot vectors.

  This noise consists of selecting a random one-hot vector from the uniform
  categorical distribution with probability noise_prob.

  Args:
    noise_prob (float): probability of corrupting label.
  Returns:
    f (function): label-corruption function for one-hot labels.
  """
  def f(x, y):
    """Returns an image with a possibly corrupted version of its label.

    Args:
      x (tensor): an image.
      y (tensor): a one-hot label.
    Returns:
      noisy_x (datpoint): image with potentially corrupted label.
    """
    make_noisy = tf.reduce_all(tf.random.uniform((1, 1)) < noise_prob)
    return (x, tf.cond(make_noisy,
                       true_fn=lambda: tf.random.shuffle(y),
                       false_fn=lambda: y))
  return f


def make_onehot(label, depth):
  """Make integer tensor label into a one-hot tensor.

  Args:
    label (tensor): a single-column tensor of integers.
    depth (int): the number of categories for the one-hot tensor.
  Returns:
    onehot (tensor): a one-hot float tensor of the same length as label.
  """
  return tf.squeeze(tf.one_hot(label, depth=depth))


def get_iterator(dataset_name, phase, process_function, noise_prob=0.):
  """Get an iterator for phase = {train, valid, test}."""
  dataset_itr = (
      tfds.load(dataset_name, split=HASH_ARGS[phase]["split_name"],
                as_dataset_kwargs={"shuffle_files": False})
      .filter(get_hash_filter_function(*HASH_ARGS[phase]["hash_args"]))
      .map(process_function(_THRESHOLD))
      .map(make_labels_noisy_function(noise_prob))
      .shuffle(1024).repeat().batch(64).prefetch(4))
  return tf.compat.v1.data.make_one_shot_iterator(dataset_itr)


def get_iterator_by_class(dataset_name, phase, classes,
                          process_function, filter_function, noise_prob=0.):
  """Get an iterator for phase = {train, valid, test}."""
  dataset_itr = (
      tfds.load(dataset_name, split=HASH_ARGS[phase]["split_name"],
                as_dataset_kwargs={"shuffle_files": False})
      .filter(get_hash_filter_function(*HASH_ARGS[phase]["hash_args"]))
      .filter(filter_function(classes))
      .map(process_function(_THRESHOLD))
      .map(make_labels_noisy_function(noise_prob))
      .shuffle(1024).repeat().batch(64).prefetch(4))
  return tf.compat.v1.data.make_one_shot_iterator(dataset_itr)


def get_iterator_filtered(dataset_name, phase, process_function,
                          filter_function):
  """Get an iterator for phase = {train, valid, test}."""
  dataset_itr = (
      tfds.load(dataset_name, split=HASH_ARGS[phase]["split_name"],
                as_dataset_kwargs={"shuffle_files": False})
      .filter(get_hash_filter_function(*HASH_ARGS[phase]["hash_args"]))
      .map(process_function(_THRESHOLD))
      .filter(filter_function).shuffle(1024).repeat().batch(64).prefetch(4))
  return tf.compat.v1.data.make_one_shot_iterator(dataset_itr)


def load_dataset(dataset_name, function_args):
  """Load dataset. Return iterators for training, validation, and test."""

  return (get_iterator(dataset_name, "train", **function_args),
          get_iterator(dataset_name, "valid", **function_args),
          get_iterator(dataset_name, "test", **function_args))


def load_dataset_ood(dataset_name, ind_classes, ood_classes, function_args):
  """Load dataset with OOD splits.

  Args:
    dataset_name (str): the name of this dataset.
    ind_classes (list): ints of which classes we want to train on.
    ood_classes (list): ints of which classes we want evaluate OOD on.
    function_args (dict): parameters for loading iterators.
  Returns:
    four iterators, for training, validation, testing, and OOD evaluation.
  """
  return (get_iterator_by_class(dataset_name, "train", ind_classes,
                                **function_args),
          get_iterator_by_class(dataset_name, "valid", ind_classes,
                                **function_args),
          get_iterator_by_class(dataset_name, "test", ind_classes,
                                **function_args),
          get_iterator_by_class(dataset_name, "test", ood_classes,
                                **function_args))


def load_dataset_exclude_example(dataset_name, function_args):
  """Load dataset with OOD splits.

  Args:
    dataset_name (str): the name of this dataset.
    function_args (dict): parameters for loading iterators.
  Returns:
    four iterators, for training, validation, testing, and OOD evaluation.
  """
  train_function_args = function_args.copy()
  function_args.pop("filter_function")
  return (get_iterator_filtered(dataset_name, "train", **train_function_args),
          get_iterator(dataset_name, "valid", **function_args),
          get_iterator(dataset_name, "test", **function_args))


def load_dataset_supervised(dataset_name="mnist"):
  return load_dataset(dataset_name,
                      {"process_function": process_image_tuple_function})


def load_dataset_supervised_onehot(dataset_name="mnist", label_noise=0.):
  return load_dataset(dataset_name,
                      {"process_function": process_image_tuple_function_onehot,
                       "noise_prob": label_noise})


def load_dataset_unsupervised(dataset_name="mnist"):
  """Load dataset. Return iterators for training, validation, and test."""

  return load_dataset(dataset_name,
                      {"process_function": process_image_function})


def load_dataset_ood_unsupervised(ind_classes, ood_classes,
                                  dataset_name="mnist"):
  """Load dataset with OOD splits.

  Args:
    ind_classes (list): ints of which classes we want to train on.
    ood_classes (list): ints of which classes we want evaluate OOD on.
    dataset_name (str): the name of this dataset.
  Returns:
    four iterators, for training, validation, testing, and OOD evaluation.
  """
  function_args = {"process_function": process_image_function,
                   "filter_function": filter_label_function}
  return load_dataset_ood(dataset_name, ind_classes, ood_classes, function_args)


def load_dataset_ood_supervised(ind_classes, ood_classes,
                                dataset_name="mnist", label_noise=0.):
  """Load dataset with OOD splits.

  Args:
    ind_classes (list): ints of which classes we want to train on.
    ood_classes (list): ints of which classes we want evaluate OOD on.
    dataset_name (str): the name of this dataset.
    label_noise (float): probability of changing the label.
  Returns:
    Four iterators, for training, validation, testing, and OOD evaluation.
  """

  function_args = {"process_function": process_image_tuple_function,
                   "filter_function": filter_label_function,
                   "noise_prob": label_noise}
  return load_dataset_ood(dataset_name, ind_classes, ood_classes, function_args)


def load_dataset_ood_supervised_onehot(ind_classes, ood_classes,
                                       dataset_name="mnist", label_noise=0.):
  """Load dataset with OOD splits.

  Args:
    ind_classes (list): ints of which classes we want to train on.
    ood_classes (list): ints of which classes we want evaluate OOD on.
    dataset_name (str): the name of this dataset.
    label_noise (float): percentage of data we want to add label noise for.
  Returns:
    Four iterators, for training, validation, testing, and OOD evaluation.
  """

  function_args = {"process_function": process_image_tuple_function_onehot,
                   "filter_function": filter_label_function,
                   "noise_prob": label_noise}
  return load_dataset_ood(dataset_name, ind_classes, ood_classes, function_args)


def load_dataset_exclude_example_supervised_onehot(example,
                                                   dataset_name="mnist"):
  """Load dataset, excluding one training example.

  Args:
    example (tensor): a training example to exclude.
    dataset_name (str): the name of this dataset.
  Returns:
    Four iterators, for training, validation, and testing.
  """

  filter_exclude_example_function = filter_example_function(example)
  function_args = {"process_function": process_image_tuple_function_onehot,
                   "filter_function": filter_exclude_example_function}
  return load_dataset_exclude_example(dataset_name, function_args)


def get_supervised_batch_noise(x_shape, y_shape):
  # For one-hot encoding, we want to test float tensors.
  # For integer labeling, we want to test integer tensors.
  y_type = tf.dtypes.int32 if y_shape[1] == 1 else tf.dtypes.float32
  return (tf.random.uniform(x_shape), tf.ones(y_shape, dtype=y_type))


def get_supervised_batch_noise_iterator(x_shape, y_shape=None):
  """Return an iterator which returns noise, for testing purposes.

  Args:
    x_shape (tuple): shape of x data to return.
    y_shape (tuple): shape of y data to return.
  Returns:
    itr (TestIterator): an iterator which returns batches in the given shapes.
  """

  class TestIterator(object):
    """An iterator which returns noise, for testing purposes."""

    def __init__(self, x_shape, y_shape):
      self.x_shape = x_shape
      self.y_shape = y_shape

    def __iter__(self):
      return self

    def next(self):
      if self.y_shape is not None:
        return get_supervised_batch_noise(x_shape, y_shape)
      else:
        x, _ = get_supervised_batch_noise(x_shape, (1, 1))
        return x

  return TestIterator(x_shape, y_shape)

