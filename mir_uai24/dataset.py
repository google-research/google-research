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

"""Dataset loading and transform utils."""

import collections
import functools

import numpy as np
import pandas as pd
import tensorflow as tf

from mir_uai24 import enum_utils
from mir_uai24 import synthetic
from mir_uai24 import us_census

SHUFFLE_BUFFER_SIZE = 100000


def instancemir_transform(
    batch, dataset_info
):
  """Replaces instance labels in a batch with the respective bag labels.

  Args:
    batch: Batch of instances.
    dataset_info: Dataset info object.

  Returns:
    The transformed batch
  """
  bag_ids = tf.unique(batch[dataset_info.bag_id])[1]
  one_hot_bag_ids = tf.cast(
      tf.one_hot(bag_ids, depth=tf.reduce_max(bag_ids) + 1), tf.float64)
  bag_labels = tf.cast(
      batch[dataset_info.label][
          batch[dataset_info.bag_id] == batch[dataset_info.instance_id][:, 0]],
      tf.float64)
  batch[dataset_info.label] = tf.matmul(one_hot_bag_ids, bag_labels)
  return batch


def aggregatedmir_transform(
    batch, dataset_info
):
  """Aggregates individual instances in a bag into a single instance.

  Args:
    batch: Batch of instances.
    dataset_info: Dataset info object.

  Returns:
    The transformed batch
  """
  bag_ids = tf.unique(batch[dataset_info.bag_id])[1]
  one_hot_bag_ids = tf.transpose(
      tf.cast(tf.one_hot(bag_ids, depth=tf.reduce_max(bag_ids) + 1), tf.float32)
  )
  batch[dataset_info.label] = batch[dataset_info.label][
      batch[dataset_info.bag_id] == batch[dataset_info.instance_id][:, 0]]
  for feature in dataset_info.features:
    batch[feature.key] = tf.matmul(one_hot_bag_ids, batch[feature.key])
  batch[dataset_info.bag_id] = tf.unique(batch[dataset_info.bag_id])[0]
  batch[dataset_info.instance_id] = batch[dataset_info.bag_id][:, None]
  batch[dataset_info.bag_id_x_instance_id] = batch[dataset_info.bag_id][:, None]
  return batch


def get_memberships(df):
  """Computes dataset membership info from a bags dataframe.

  Args:
    df: Bags dataframe.

  Returns:
    Dataset membership info object.
  """
  instances = collections.defaultdict(list)
  bags = collections.defaultdict(list)

  for bag_id in df.bag_id:
    for instance_id, bag_id_x_instance_id in zip(
        df.loc[bag_id].instance_id, df.loc[bag_id].bag_id_X_instance_id
    ):
      instance = enum_utils.DatasetInstance(
          bag_id=bag_id,
          instance_id=instance_id,
          bag_id_x_instance_id=bag_id_x_instance_id
      )
      instances[instance_id].append(instance)
      bags[bag_id].append(instance)
  memberships = enum_utils.DatasetMembershipInfo(instances=instances, bags=bags)
  return memberships


def load_instances(path, batch_size):
  """Loads instance-level training data.

  Args:
    path: Data path.
    batch_size: Batch size.

  Returns:
    A tf.data.Dataset object containing the instance-level training data.
  """
  def make_2d(batch):
    for key, value in batch.items():
      if key != 'bag_id':
        batch[key] = value[:, None]
    return batch
  df = pd.read_feather(path)
  dataset = tf.data.Dataset.from_tensor_slices(dict(df))
  dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(make_2d)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def make_instances(
    batch, bag_size):
  batch['bag_id'] = tf.repeat(batch['bag_id'], repeats=bag_size)
  for key in batch:
    if key != 'bag_id':
      batch[key] = tf.reshape(batch[key], shape=[-1, 1])
  return batch


def load_bags(
    path_or_df, bag_size, batch_size
):
  """Loads bag-level training data.

  Args:
    path_or_df: Path to load the data from or a preloaded dataframe.
    bag_size: Bag size.
    batch_size: Batch size.

  Returns:
    A tf.data.Dataset object containing the bag-level training data.
  """
  bags_df = path_or_df
  if not isinstance(path_or_df, pd.DataFrame):
    bags_df = pd.read_feather(path_or_df)
  bags_dict = dict(bags_df)
  for key in bags_dict:
    bags_dict[key] = np.stack(
        bags_dict[key].to_numpy(),
        dtype=bags_dict[key][0].dtype)
  dataset = tf.data.Dataset.from_tensor_slices(bags_dict)
  dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(functools.partial(make_instances, bag_size=bag_size))
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def load(
    dataset,
    train_instance=False,
    batch_size=32,
    with_info=True,
):
  """Loads the desired dataset.

  Args:
    dataset: Dataset to load.
    train_instance: If True, loads instance-level training data.
    batch_size: Batch size.
    with_info: If True, returns dataset info object.

  Returns:
    If with_info is True, returns a tuple of (train, val, test) datasets and
    dataset info. Otherwise, returns a tuple of (train, val, test) datasets.
  """
  dataset = {
      enum_utils.Dataset.SYNTHETIC: synthetic,
      enum_utils.Dataset.US_CENSUS: us_census,
  }[dataset]
  bags_df = None
  dataset_info = None
  if with_info:
    get_info_out = dataset.get_info(return_bags_df=True)
    assert isinstance(get_info_out, tuple)
    dataset_info, bags_df = get_info_out
  dataset_info.memberships = get_memberships(bags_df)

  if train_instance:
    ds_train = load_instances(
        dataset.TRAIN_INSTANCE_DATA_PATH, batch_size)
  else:
    ds_train = load_bags(
        dataset.TRAIN_BAGS_DATA_PATH if bags_df is None else bags_df,
        dataset.BAG_SIZE,
        batch_size)

  ds_val = load_instances(dataset.VAL_DATA_PATH, batch_size)
  ds_test = load_instances(dataset.TEST_DATA_PATH, batch_size)

  if with_info:
    return (ds_train, ds_val, ds_test), dataset_info
  return (ds_train, ds_val, ds_test)
