# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Training library and binary."""
import dataclasses
import functools
import os
import random

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from hypertransformer.tf.core import common_ht
from hypertransformer.tf.core import datasets

NUMPY_BATCH_SIZE = 128


@dataclasses.dataclass
class ModelState(object):
  """Model state."""
  loss: Optional[tf.Tensor] = None


@dataclasses.dataclass
class DatasetState:
  meta_ds: Optional[Any] = None


def make_augmentation_config(data_config,
                             num_labels):
  """Returns dataset augmentation configuration."""
  random_config = datasets.RandomizedAugmentationConfig(
      rotation_probability=data_config.rotation_probability,
      smooth_probability=data_config.smooth_probability,
      contrast_probability=data_config.contrast_probability,
      resize_probability=data_config.resize_probability,
      negate_probability=data_config.negate_probability,
      roll_probability=data_config.roll_probability,
      angle_range=data_config.angle_range,
      rotate_by_90=data_config.rotate_by_90)
  if data_config.per_label_augmentation:
    with tf.variable_scope('augmentations'):
      return datasets.AugmentationConfig(
          children=[datasets.AugmentationConfig(random_config=random_config)
                    for _ in range(num_labels)])
  else:
    return datasets.AugmentationConfig(random_config=random_config)


def _convert_bool(arr):
  return arr.astype(np.int8) * 255


def _load_cache(data_config
                ):
  """Loads cached dataset from a saved NumPy array."""
  folder = os.path.join(data_config.cache_path, data_config.dataset_name)
  path = os.path.join(data_config.cache_path, data_config.dataset_name + '.npy')
  print(f'Looking for cache in "{data_config.cache_path}"')
  if os.path.exists(path):
    # Reading a NumPy cache.
    with open(path, 'rb') as dev:
      data = np.load(dev)
    if len(data.shape) < 4:
      data = np.expand_dims(data, axis=-1)
    # Converting a 4D tensor [Label, Batch, W, H, C] to a dictionary by label.
    if data.dtype == bool:
      return {k: _convert_bool(data[k]) for k in range(data.shape[0])}
    else:
      return {k: data[k] for k in range(data.shape[0])}
  elif os.path.exists(folder):
    # Reading from a folder with NumPy cache files.
    names = os.listdir(folder)
    output = {}
    index = 0
    for name in sorted(names):
      # Each file contains a list of image sets for different labels.
      # File names are sorted to keep a proper label order.
      with open(os.path.join(folder, name), 'rb') as data_file:
        file_records = np.load(data_file, allow_pickle=True)
      for record in file_records:
        output[index] = record
        index += 1
    return output
  print(f'No cache files for {data_config.dataset_name} found. Falling back '
        'to TF dataset.')
  return None


def _make_numpy_array(data_config,
                      batch_size,
                      sess = None):
  """Makes a NumPy array for given dataset configuration."""
  output = None
  if sess is None:
    sess = tf.Session()
  ds = data_config.ds
  if ds is None:
    output = _load_cache(data_config)
    if output is None:
      ds = tfds.load(data_config.dataset_name,
                     data_dir=data_config.data_dir)[data_config.tfds_split]

  dataset_info = data_config.dataset_info
  if dataset_info is None:
    dataset_info = datasets.get_dataset_info(data_config.dataset_name)

  if output is None:
    assert dataset_info.num_samples_per_label is not None
    output = datasets.make_numpy_data(
        sess,
        ds=ds,
        batch_size=batch_size,
        num_labels=dataset_info.num_labels,
        samples_per_label=dataset_info.num_samples_per_label,
        transpose=dataset_info.transpose_images)

  if data_config.shuffle_labels_seed > 0:
    keys = list(output.keys())
    orig_keys = keys[:]
    random.seed(data_config.shuffle_labels_seed)
    random.shuffle(keys)
    output = {orig_keys[i]: output[keys[i]] for i in range(len(keys))}

  return output


def _resize(imgs, image_size):
  if len(imgs.shape) < 4:
    imgs = tf.expand_dims(imgs, axis=-1)
  return tf.image.resize_images(
      imgs, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)


def make_dataset_helper_unbalanced(
    batch_size,
    image_size,
    num_labels,
    data_config,
    always_same_labels = False,
    sess = None
    ):
  """Helper function for creating a dataset."""
  numpy_arr = _make_numpy_array(data_config, batch_size, sess)
  config = make_augmentation_config(data_config=data_config,
                                    num_labels=num_labels)

  with tf.name_scope(None, default_name='data'):
    gen = datasets.TaskGenerator(
        numpy_arr,
        num_labels=num_labels,
        image_size=image_size,
        use_label_subset=data_config.use_label_subset,
        always_same_labels=always_same_labels)
    randomize_op = config.randomize_op()
    images, labels, classes = gen.get_batch(
        batch_size=batch_size, config=config,
        num_unlabeled_per_class=data_config.num_unlabeled_per_class)
    if image_size is not None:
      images = _resize(images, image_size)
    images = images / 128.0 - 1.0

  # Stopping gradients to avoid backpropagation through random shuffling
  # operation.
  images = tf.stop_gradient(images)
  labels = tf.stop_gradient(labels)
  classes = tf.stop_gradient(classes)

  return images, labels, classes, randomize_op


def make_dataset_helper_balanced(
    batch_sizes,
    num_unlabeled_per_class,
    image_size,
    num_labels,
    data_config,
    always_same_labels = False,
    sess = None
    ):
  """Helper function for creating a balanced dataset."""
  numpy_arr = _make_numpy_array(data_config, NUMPY_BATCH_SIZE, sess)
  config = make_augmentation_config(data_config=data_config,
                                    num_labels=num_labels)

  with tf.name_scope(None, default_name='data'):
    gen = datasets.TaskGenerator(
        numpy_arr,
        num_labels=num_labels,
        image_size=image_size,
        use_label_subset=data_config.use_label_subset,
        always_same_labels=always_same_labels)
    randomize_op = config.randomize_op()
    images_labels = gen.get_batches(
        batch_sizes=batch_sizes, config=config,
        num_unlabeled_per_class=num_unlabeled_per_class)
    output = []
    for images, labels, classes in images_labels:
      if image_size is not None:
        images = _resize(images, image_size)
      images = images / 128.0 - 1.0
      # Stopping gradients to avoid backpropagation through random shuffling
      # operation.
      labels.set_shape((images.shape[0],))
      output.append((tf.stop_gradient(images),
                     tf.stop_gradient(labels),
                     tf.stop_gradient(classes)))

  return output, randomize_op


def _get_class_bounds(data_config
                      ):
  if (data_config.use_label_subset is None or
      callable(data_config.use_label_subset)):
    return None, None
  return min(data_config.use_label_subset), max(data_config.use_label_subset)


def make_dataset_unbalanced(model_config,
                            data_config,
                            shuffle_labels = True
                            ):
  """Creates data for Transformer and CNN.

  Arguments:
    model_config: Model configuration.
    data_config: Dataset configuration.
    shuffle_labels: True if should subsample random labels from
        `data_config.use_label_subset` for each new mini-dataset.

  Returns:
    `DatasetSamples` structure containing Transformer and CNN samples.
  """
  batch_size = model_config.num_transformer_samples
  batch_size += model_config.num_cnn_samples

  images, labels, classes, randomize_op = make_dataset_helper_unbalanced(
      batch_size=batch_size,
      image_size=model_config.image_size,
      num_labels=model_config.num_labels,
      data_config=data_config,
      always_same_labels=not shuffle_labels)

  transformer_samples = model_config.num_transformer_samples
  transformer_images = images[:transformer_samples]
  if len(transformer_images.shape) == 3:
    transformer_images = tf.expand_dims(transformer_images, axis=-1)
  cnn_images = images[transformer_samples:]
  if len(cnn_images.shape) == 3:
    cnn_images = tf.expand_dims(cnn_images, axis=-1)

  real_class_min, real_class_max = _get_class_bounds(data_config)

  return common_ht.DatasetSamples(
      transformer_images=transformer_images,
      transformer_labels=labels[:transformer_samples],
      transformer_real_classes=classes[:transformer_samples],
      cnn_images=cnn_images,
      cnn_labels=labels[transformer_samples:],
      cnn_real_classes=classes[transformer_samples:],
      randomize_op=randomize_op,
      real_class_min=real_class_min,
      real_class_max=real_class_max)


def make_dataset_balanced(
    model_config,
    data_config,
    shuffle_labels = True):
  """Creates data for Transformer and CNN.

  Arguments:
    model_config: Model configuration.
    data_config: Dataset configuration.
    shuffle_labels: True if should subsample random labels from
        `data_config.use_label_subset` for each new mini-dataset.

  Returns:
    `DatasetSamples` structure containing Transformer and CNN samples.
  """
  batch_sizes = [model_config.num_transformer_samples,
                 model_config.num_cnn_samples]
  # Removing labels only from the Transformer batch.
  num_unlabeled_per_class = [data_config.num_unlabeled_per_class, 0]

  batches, randomize_op = make_dataset_helper_balanced(
      batch_sizes=batch_sizes,
      num_unlabeled_per_class=num_unlabeled_per_class,
      image_size=model_config.image_size,
      num_labels=model_config.num_labels,
      data_config=data_config,
      always_same_labels=not shuffle_labels)

  transformer_images, transformer_labels, transformer_classes = batches[0]
  cnn_images, cnn_labels, cnn_classes = batches[1]

  if len(transformer_images.shape) == 3:
    transformer_images = tf.expand_dims(transformer_images, axis=-1)
  if len(cnn_images.shape) == 3:
    cnn_images = tf.expand_dims(cnn_images, axis=-1)

  real_class_min, real_class_max = _get_class_bounds(data_config)

  return common_ht.DatasetSamples(
      transformer_images=transformer_images,
      transformer_labels=transformer_labels,
      transformer_real_classes=transformer_classes,
      cnn_images=cnn_images,
      cnn_labels=cnn_labels,
      cnn_real_classes=cnn_classes,
      randomize_op=randomize_op,
      real_class_min=real_class_min,
      real_class_max=real_class_max)


def make_dataset(model_config,
                 data_config,
                 dataset_state = None,
                 **kwargs):
  """Makes dataset given dataset and model configuration."""
  augment = functools.partial(
      datasets.augment_images,
      augment_individually=data_config.augment_individually)

  if data_config.balanced_batches:
    dataset_maker = make_dataset_balanced
  else:
    dataset_maker = make_dataset_unbalanced
  output = dataset_maker(model_config=model_config,
                         data_config=data_config,
                         **kwargs)
  if data_config.apply_image_augmentations:
    image_size = model_config.image_size
    output = dataclasses.replace(
        output,
        transformer_images=augment(output.transformer_images, image_size),
        cnn_images=augment(output.cnn_images, image_size))
  return output, dataset_state
