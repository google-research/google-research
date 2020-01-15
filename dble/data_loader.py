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

"""Data loading and processing.

Contains dataloaders and augmentation functions of MNIST,
CIFAR-10, CIFAR-100 and Tiny-ImageNet.
Also contains the implementation of dataset interface of prototypical learning.
"""
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
import numpy as np
import tensorflow.compat.v1 as tf
from dble import tiny_imagenet


def load_mnist():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = x_train.astype(float) / 255.
  x_test = x_test.astype(float) / 255.
  fields = x_train, np.squeeze(y_train)
  fields_test = x_test, np.squeeze(y_test)

  return fields, fields_test


def load_cifar10():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  fields = x_train, np.squeeze(y_train)
  fields_test = x_test, np.squeeze(y_test)

  return fields, fields_test


def load_cifar100():
  (x_train, y_train), (x_test, y_test) = cifar100.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  fields = x_train, np.squeeze(y_train)
  fields_test = x_test, np.squeeze(y_test)

  return fields, fields_test


def load_tiny_imagenet(data_dir, val_data_dir):
  """Loads the training and validation data of Tiny-ImageNet.

  Args:
    data_dir: The directory of raw training data of Tiny-ImageNet.
    val_data_dir: The directory of raw validation data of Tiny-ImageNet.
  Returns:
    fields: the tuple of (x_train, y_train). x_train is a numpy array with
    shape (num_samples, height, width, num_channels). y_train is a numpy array
    with shape (num_samples, ).
    fields_test: the tuple of (x_test, y_test).
  """
  (x_train, y_train, _), annotations = tiny_imagenet.load_training_images(
      data_dir)
  (x_test, y_test, _) = tiny_imagenet.load_validation_images(
      val_data_dir, annotations)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  indx = np.random.choice([i for i in range(98179)], size=100000, replace=True)
  x_train = x_train[indx]
  y_train = np.squeeze(y_train)[indx]
  indx_2 = np.argsort(y_train)
  fields = x_train[indx_2], y_train[indx_2]
  indx_test = np.random.choice([i for i in range(9832)],
                               size=10000,
                               replace=True)
  y_test = np.squeeze(y_test)[indx_test]
  x_test = x_test[indx_test]
  indx_test2 = np.argsort(y_test)
  fields_test = x_test[indx_test2], y_test[indx_test2]

  return fields, fields_test


def augment_cifar(batch_data, is_training=False):
  image = batch_data
  if is_training:
    image = tf.image.resize_image_with_crop_or_pad(batch_data, 32 + 8, 32 + 8)
    i = image.get_shape().as_list()[0]
    image = tf.random_crop(image, [i, 32, 32, 3])
    image = tf.image.random_flip_left_right(image)
  image = tf.image.per_image_standardization(image)

  return image


def augment_tinyimagenet(batch_data, is_training=False):
  image = batch_data
  if is_training:
    image = tf.image.random_flip_left_right(image)
  image = tf.image.per_image_standardization(image)

  return image


def get_image_size(dataset_name):
  if dataset_name == 'cifar10' or dataset_name == 'cifar100':
    image_size = 32
  else:
    image_size = 64
  return image_size


def uniform(n):

  def sampler(n_samples, rng=np.random):
    return rng.choice(n, n_samples)

  return sampler


class Dataset(object):
  """Basic dataset interface for prototypical learning."""

  def __init__(self, fields):
    """Store a tuple of fields and access it through next_batch interface.

    Args:
      fields: field[0] and field[1] are considered to be x and y.
    """
    self.n_samples = len(fields[0])
    self.fields = fields
    self.sampler = uniform(self.n_samples)

  @property
  def x(self):
    return self.fields[0]

  @property
  def y(self):
    return self.fields[1]

  def next_batch(self, n, rng=np.random):
    idx = self.sampler(n, rng)
    return tuple(field[idx] for field in self.fields)

  def get_few_shot_idxs(self, classes, num_supports):
    """Samples the supports and queries given classes and return their indexs.

    Args:
      classes: The list of indexs of classes in the episode.
      num_supports: A scalar describing the number of support samples needed for
      every class.
    Returns:
      np.array(support_idxs): indexs of the supports sampled.
      np.array(query_idxs): indexs of the queries sampled.
    """
    support_idxs, query_idxs = [], []
    idxs = np.arange(len(self.y))
    for cl in classes:
      class_idxs = idxs[self.y == cl]
      class_idxs_support = np.random.choice(
          class_idxs, size=num_supports, replace=False)
      class_idxs_query = np.setxor1d(class_idxs, class_idxs_support)

      support_idxs.extend(class_idxs_support)
      query_idxs.extend(class_idxs_query)

    return np.array(support_idxs), np.array(query_idxs)

  def next_few_shot_batch(self, query_batch_size_per_task, num_classes_per_task,
                          num_supports_per_class, num_tasks):
    """Samples the few-shot batch for prototypical training.

    Args:
      query_batch_size_per_task: The number of queries required
      for every task(episode).
      num_classes_per_task: The number of classes for every task.
      num_supports_per_class: The number of support samples required for every
      class.
      num_tasks: Task number of the batch.
    Returns:
      np.concatenate(query_images, axis=0): numpy array of query images with
      shape (query_batch_size_per_task*num_tasks, height, width, num_channels).
      np.concatenate(query_labels, axis=0): numpy array of query labels with
      shape (query_batch_size_per_task*num_tasks, ).
      np.concatenate(support_images, axis=0): numpy array of support images with
      shape (num_classes_per_task*num_supports_per_class*num_tasks,
      height, width, num_channels).
      np.concatenate(support_labels, axis=0): numpy array of support labels with
      shape (num_classes_per_task*num_supports_per_class*num_tasks, ).
    """
    labels = self.y
    classes = np.unique(labels)

    query_images = []
    query_labels = []
    support_images = []
    support_labels = []
    for _ in range(num_tasks):
      task_classes = np.random.choice(
          classes, size=num_classes_per_task, replace=False)

      support_idxs, query_idxs = self.get_few_shot_idxs(
          classes=task_classes, num_supports=num_supports_per_class)
      query_idxs = np.random.choice(
          query_idxs, size=query_batch_size_per_task, replace=False)

      labels_query = labels[query_idxs]
      labels_support = labels[support_idxs]

      class_map = {c: i for i, c in enumerate(task_classes)}
      # pylint: disable=cell-var-from-loop
      class_map_fn = np.vectorize(lambda t: class_map[t])

      query_images.append(self.x[query_idxs])
      query_labels.append(class_map_fn(labels_query))
      support_images.append(self.x[support_idxs])
      support_labels.append(class_map_fn(labels_support))

    return np.concatenate(query_images, axis=0), \
        np.concatenate(query_labels, axis=0), \
        np.concatenate(support_images, axis=0), \
        np.concatenate(support_labels, axis=0)
