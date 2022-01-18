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

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains code for loading and preprocessing the CIFAR data."""

import cifar100_dataset
import cifar10_dataset
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider
from tensorflow.python.ops import control_flow_ops


datasets_map = {
    'cifar10': cifar10_dataset,
    'cifar100': cifar100_dataset,
}


def provide_resnet_data(dataset_name,
                        split_name,
                        batch_size,
                        dataset_dir=None,
                        num_epochs=None):
  """Provides batches of CIFAR images for resnet.

  Args:
    dataset_name: Eiether 'cifar10' or 'cifar100'.
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the MNIST data can be found.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 1]
    one_hot_labels: A `Tensor` of size [batch_size, NUM_CLASSES], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of total samples in the dataset.
    num_classes: The number of total classes in the dataset.


  Raises:
    ValueError: If `split_name` is not either 'train' or 'test'.
  """
  dataset = _get_dataset(dataset_name, split_name, dataset_dir=dataset_dir)

  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'),
      num_epochs=num_epochs)

  [image, label] = provider.get(['image', 'label'])

  image = tf.to_float(image)

  image_size = 32
  if split_name == 'train':
    image = tf.image.resize_image_with_crop_or_pad(image, image_size + 4,
                                                   image_size + 4)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image /= 255
    # pylint: disable=unnecessary-lambda
    image = _apply_with_random_selector(
        image, lambda x, ordering: distort_color(x, ordering), num_cases=2)
    image = 2 * (image - 0.5)

  else:
    image = tf.image.resize_image_with_crop_or_pad(image, image_size,
                                                   image_size)
    image = (image - 127.5) / 127.5

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=1,
      capacity=5 * batch_size,
      allow_smaller_final_batch=True)

  one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
  one_hot_labels = tf.squeeze(one_hot_labels, 1)
  return images, one_hot_labels, dataset.num_samples, dataset.num_classes


def _get_dataset(name, split_name, **kwargs):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, name of the dataset.
    split_name: A train/test split name.
    **kwargs: Extra kwargs for get_split, for example dataset_dir.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if dataset unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  dataset = datasets_map[name].get_split(split_name, **kwargs)
  dataset.name = name
  return dataset


def _apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from

  Returns:
    The result of func(x, sel), where func receives the value of
    the selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, scope=None):
  """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    color_ordering: Python int, a type of distortion (valid values: 0, 1).
    scope: Optional scope for name_scope.

  Returns:
    color-distorted image
  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
