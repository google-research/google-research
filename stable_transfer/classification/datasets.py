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

"""Dataset functions for Transferability Experiments."""

from typing import List, Optional, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from stable_transfer.classification import networks


DATASETS_WITH_VALIDATION_AS_TEST = ['food101', 'imagenette']


def load_dataset(dataset_name,
                 split = 'train',
                 with_info = False):
  """Wrapper around tfds load dataset to correct for test split name."""
  if split == 'test' and dataset_name in DATASETS_WITH_VALIDATION_AS_TEST:
    split = 'validation'
  return tfds.load(dataset_name, split=split, with_info=with_info)


def ds_preprocess_input(ds, network_architecture):
  """tf.data.Dataset preprocessing inputs based on the network architecture."""
  image = tf.image.resize(ds['image'], network_architecture.target_shape)
  ds['image'] = network_architecture.preprocessing(image)
  return ds


def ds_as_supervised(ds):
  """tf.data.Dataset maps dict to (image, label) pairs."""
  return ds['image'], ds['label']


def ds_filter_target_classes(d, target_classes):
  """tf.data.Dataset filter images belonging to target class."""
  return tf.reduce_any(tf.equal(d['label'], target_classes))


def ds_relabel_target_classes(d, target_classes):
  """tf.data.Dataset map class labels to new range."""
  label = d['label']
  d['label_original'] = label
  d['label'] = tf.boolean_mask(
      tf.range(len(target_classes)), tf.equal(label, target_classes))[0]
  return d


def get_experiment_dataset(
    ds,
    network_architecture,
    target_classes = None,
    do_shuffle = True,
    as_supervised = True,
    shuffle_seed = 789,
    shuffle_reshuffle_each_iteration = True,
    batch_size=64
    ):
  """Get (preprocessed) target dataset."""

  # Filter target classes
  if target_classes is not None:
    ds_target = ds.filter(lambda d: ds_filter_target_classes(d, target_classes))
    ds_target = ds_target.map(
        lambda d: ds_relabel_target_classes(d, target_classes))
  else:
    ds_target = ds.map(lambda d: d)  # Otherwise it might change input ds.

  # Preprocess input
  ds_target = ds_target.map(
      lambda d: ds_preprocess_input(d, network_architecture))

  if as_supervised:
    ds_target = ds_target.map(ds_as_supervised)

  if do_shuffle:
    shuffle_buffer = 32 * batch_size  # 32 * 64 = 2048
    ds_target = ds_target.shuffle(
        shuffle_buffer,
        seed=shuffle_seed,
        reshuffle_each_iteration=shuffle_reshuffle_each_iteration)
  if batch_size != 1:
    ds_target = ds_target.batch(batch_size)

  return ds_target
