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

"""Data utils."""

import math
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DATA_DIR = '~/tensorflow_datasets/'


def get_color_mnist_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    cache = True,
    buffer_size = 10000,
):
  """Gets Color MNIST datasets."""

  def preprocess_data(
      image, label
  ):
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.pad_to_bounding_box(image, 2, 2, 32, 32)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  builder = tfds.builder(name='mnist', data_dir=DATA_DIR)
  ds = builder.as_dataset(split=split, as_supervised=True)
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_svhn_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    cache = True,
    buffer_size = 10000,
):
  """Gets SVHN datasets."""

  def preprocess_data(
      image, label
  ):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  builder = tfds.builder(
      name='svhn_cropped',
      data_dir=DATA_DIR,
  )
  ds = builder.as_dataset(split=split, as_supervised=True)
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_cifar10_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    data_augment = False,
    cache = True,
    buffer_size = 10000,
):
  """Gets CIFAR-10 datasets."""

  def preprocess_data(
      image, label
  ):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  def augment(
      image, label
  ):
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height=4,
        offset_width=4,
        target_height=40,
        target_width=40,
    )
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label

  builder = tfds.builder(
      name='cifar10',
      data_dir=DATA_DIR,
  )
  ds = builder.as_dataset(split=split, as_supervised=True)
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if data_augment:
    ds = ds.map(
        augment,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_cinic10_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    data_augment = False,
    cache = True,
    max_size = 20000,
    buffer_size = 10000,
):
  """Gets CINIC-10 datasets."""

  def preprocess_data(
      image, label
  ):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  def augment(
      image, label
  ):
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height=4,
        offset_width=4,
        target_height=40,
        target_width=40,
    )
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label
  builder = tfds.builder(
      name='cinic10',
      data_dir=DATA_DIR,
  )
  ds = builder.as_dataset(split=split, as_supervised=True)
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if max_size > 0:
    ds = ds.take(max_size)
  if cache:
    ds = ds.cache()
  if data_augment:
    ds = ds.map(
        augment,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_domainnet_dataset(
    domain_name,
    split,
    batch_size,
    shuffle,
    drop_remainder,
    data_augment = False,
    cache = True,
    buffer_size = 10000,
    max_size = -1,
):
  """Gets DomainNet datasets."""

  def preprocess_data(
      data, image_size = 96
  ):
    image = data['image']
    label = data['label']
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(
        image, (image_size, image_size), preserve_aspect_ratio=True
    )
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    return image, label

  def augment(
      image, label
  ):
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height=4,
        offset_width=4,
        target_height=100,
        target_width=100,
    )
    image = tf.image.random_crop(image, [96, 96, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label
  builder = tfds.builder(
      name=f'domainnet/{domain_name}', data_dir=DATA_DIR
  )
  ds = builder.as_dataset(
      split=split, as_supervised=False, shuffle_files=shuffle
  )
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if max_size > 0:
    ds = ds.take(max_size)
  if cache:
    ds = ds.cache()
  if data_augment:
    ds = ds.map(
        augment,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_fmow_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    data_augment = False,
    include_meta = False,
    cache = True,
    buffer_size = 10000,
):
  """Gets FMoW datasets."""

  def preprocess_data(
      data, image_size = 96
  ):
    image = data['image']
    label = data['label']
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(
        image, (image_size, image_size), preserve_aspect_ratio=True
    )
    if include_meta:
      meta = data['meta_data']
      return image, label, meta
    else:
      return image, label

  def augment(
      image, label, meta = None
  ):
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height=4,
        offset_width=4,
        target_height=100,
        target_width=100,
    )
    image = tf.image.random_crop(image, [96, 96, 3])
    image = tf.image.random_flip_left_right(image)
    if include_meta:
      return image, label, meta
    else:
      return image, label
  builder = tfds.builder(
      name='fmow', data_dir=DATA_DIR
  )
  ds = builder.as_dataset(
      split=split, as_supervised=False, shuffle_files=shuffle
  )
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if data_augment:
    ds = ds.map(
        augment,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_amazon_review_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    include_meta = False,
    cache = True,
    buffer_size = 10000,
):
  """Gets Amazon review datasets."""

  def preprocess_data(data):
    embedding = data['embedding']
    label = data['label']
    if include_meta:
      meta = data['meta_data']
      return embedding, label, meta
    else:
      return embedding, label
  builder = tfds.builder(
      name='amazon_review',
      data_dir=DATA_DIR,
  )
  ds = builder.as_dataset(
      split=split, as_supervised=False, shuffle_files=shuffle
  )
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_amazon_review_test_sub_dataset(
    subset_index,
    batch_size,
    shuffle,
    drop_remainder,
    num_reviewers_per_subset = 300,
    cache = True,
    buffer_size = 10000,
):
  """Gets Amazon Review test sub-datasets."""

  def preprocess_data(data):
    embedding = data['embedding']
    label = data['label']
    return embedding, label

  def filter_func(data):
    return (
        data['meta_data'][0]
        >= 2586 + (subset_index - 1) * num_reviewers_per_subset
    ) and (
        data['meta_data'][0] < 2586 + subset_index * num_reviewers_per_subset
    )

  builder = tfds.builder(
      name='amazon_review',
      data_dir=DATA_DIR,
  )
  ds = builder.as_dataset(
      split='test', as_supervised=False, shuffle_files=shuffle
  )
  ds = ds.filter(filter_func)
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_otto_dataset(
    split,
    batch_size,
    shuffle,
    drop_remainder,
    cache = True,
    buffer_size = 10000,
):
  """Gets Otto datasets."""

  def preprocess_data(data):
    input_feature = data['input_feature']
    label = data['label']
    return input_feature, label

  builder = tfds.builder(
      name='otto',
      data_dir=DATA_DIR,
  )
  ds = builder.as_dataset(
      split=split, as_supervised=False, shuffle_files=shuffle
  )
  ds = ds.map(
      preprocess_data,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if cache:
    ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=drop_remainder)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def get_ds_data(ds):
  """Gets dataset data."""
  inputs = []
  labels = []
  is_dict = False
  for batch_x, batch_y in ds:
    if isinstance(batch_x, dict) and (not is_dict):
      is_dict = True
      inputs = {}
      for key in batch_x:
        inputs[key] = []
    if is_dict:
      for key in inputs:
        inputs[key].extend(batch_x[key].numpy())
    else:
      inputs.extend(batch_x.numpy())
    labels.extend(batch_y.numpy())
  if is_dict:
    for key in inputs:
      inputs[key] = np.array(inputs[key])
  else:
    inputs = np.array(inputs)
  labels = np.array(labels)
  data_dict = {
      'inputs': inputs,
      'labels': labels,
  }
  return data_dict


def construct_dataset(
    data_dict,
    batch_size,
    shuffle,
    include_label = True,
    cache = True,
    buffer_size = 10000,
):
  """Constructs a dataset using given data."""

  def gen(n, include_label):
    if isinstance(inputs, dict):
      for i in range(n):
        input_dict = {}
        for key in inputs:
          input_dict[key] = inputs[key][i]
        if include_label:
          yield input_dict, labels[i]
        else:
          yield input_dict
    else:
      for i in range(n):
        if include_label:
          yield inputs[i], labels[i]
        else:
          yield inputs[i]
  inputs = data_dict['inputs']
  if isinstance(inputs, dict):
    input_signature = {}
    n = 0
    for key in inputs:
      if n == 0:
        n = inputs[key].shape[0]
      input_signature[key] = tf.TensorSpec(
          shape=inputs[key].shape[1:], dtype=inputs[key].dtype
      )
  else:
    n = inputs.shape[0]
    input_signature = tf.TensorSpec(shape=inputs.shape[1:], dtype=inputs.dtype)
  if include_label:
    labels = data_dict['labels']
    label_signature = tf.TensorSpec(shape=labels.shape[1:], dtype=labels.dtype)
  ds = tf.data.Dataset.from_generator(
      gen,
      args=(n, include_label),
      output_signature=(input_signature, label_signature)
      if include_label
      else input_signature,
  )
  if cache:
    ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size)
  ds = ds.batch(batch_size, drop_remainder=False)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  # Since `tf.data.Dataset` is a generator, we cannot find its size.
  # This is a hack to add a `__len__` attribute.
  ds.__class__ = type(
      ds.__class__.__name__,
      (ds.__class__,),
      {'__len__': lambda _: int(math.ceil(n / batch_size))},
  )
  return ds


def construct_sub_dataset(
    data_dict,
    selected_indices,
    batch_size,
    shuffle,
    include_label = True,
    cache = True,
    max_size = -1,
    return_raw_ds = False,
    buffer_size = 10000,
):
  """Constructs subset of the data."""

  def gen(n, include_label):
    if isinstance(selected_inputs, dict):
      for i in range(n):
        input_dict = {}
        for key in selected_inputs:
          input_dict[key] = selected_inputs[key][i]
        if include_label:
          yield input_dict, selected_labels[i]
        else:
          yield input_dict
    else:
      for i in range(n):
        if include_label:
          yield selected_inputs[i], selected_labels[i]
        else:
          yield selected_inputs[i]

  n = selected_indices.shape[0]
  if isinstance(data_dict['inputs'], dict):
    selected_inputs = {}
    input_signature = {}
    for key in data_dict['inputs']:
      selected_inputs[key] = data_dict['inputs'][key][selected_indices]
      input_signature[key] = tf.TensorSpec(
          shape=selected_inputs[key].shape[1:], dtype=selected_inputs[key].dtype
      )
  else:
    selected_inputs = data_dict['inputs'][selected_indices]
    input_signature = tf.TensorSpec(
        shape=selected_inputs.shape[1:], dtype=selected_inputs.dtype
    )
  if include_label:
    selected_labels = data_dict['labels'][selected_indices]
    label_signature = tf.TensorSpec(
        shape=selected_labels.shape[1:], dtype=selected_labels.dtype
    )
  sub_ds = tf.data.Dataset.from_generator(
      gen,
      args=(n, include_label),
      output_signature=(input_signature, label_signature)
      if include_label
      else input_signature,
  )
  if cache:
    sub_ds = sub_ds.cache()
  if return_raw_ds:
    return sub_ds
  if shuffle:
    sub_ds = sub_ds.shuffle(buffer_size)
  if max_size > 0:
    sub_ds = sub_ds.take(max_size)
  sub_ds = sub_ds.batch(batch_size, drop_remainder=False)
  sub_ds = sub_ds.prefetch(tf.data.AUTOTUNE)
  # Since `tf.data.Dataset` is a generator, we cannot find its size.
  # This is a hack to add a `__len__` attribute.
  sub_ds.__class__ = type(
      sub_ds.__class__.__name__,
      (sub_ds.__class__,),
      {'__len__': lambda _: int(math.ceil(n / batch_size))},
  )
  return sub_ds
