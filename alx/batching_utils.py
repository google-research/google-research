# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Contains utilities for common batching needs across datasets."""

import collections
import dataclasses
import random
import typing
from typing import Any
from typing import List

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class Batch():
  """Holds a densely packed batch of examples.

  This data structure will be used as an intermediate state before converting to
  TF Example format.

  Examples:

  1. Densely pack a single example:

  row_id=r0, history=(h0, ..., hm), ground_truth=(g0, ..., gn)

  Both history and ground truth will be packed into a 2D-tensor of size
  [batch_size, seq_len] and [ground_truth_batch_size, seq_len] respectively.

  If s = seq_len, batch may look like

  [h0, ..., h(s-1)
   hs, ..., h(2*s-1)
   .
   .
   .
   -1, ..., -1]

  and ground_truth_batch may look like

  [g0, ..., g(s-1)
   gs, ..., g(2*s-1)
   .
   .
   .
   -1, ..., -1]

  2. Densely pack multiple examples in a single batch:

  In this case, we would pack the history and ground truth as before but we
  would also have to keep track of which rows in batch (ground_truth_batch)
  correspond to the same example in the pre-batched set.

  We keep track of this in batch_ids (ground_truth_batch_ids) tensor.

  If batch_ids = [0, 1, 1, 2, 2, 2, -1], then first row in batch tensor
  corresponds to a single distinct example before dense batching. Rows 1 and 2
  jointly corresponds to another distinct example and so forth.

  We use -1 as padding value to make sure we can keep the shapes of all
  batched tensors constant across the whole dataset.
  """
  is_test: bool
  seq_len: int
  num_rows_per_batch: int

  # Densely packed history.
  row_ids: List[int]  # (num_rows_per_batch)
  item_lengths: List[int]  # (num_rows_per_batch)
  batch_size: int
  batch_ids: List[int]  # (batch_size)
  batch: List[List[int]]  # (batch_size, seq_len)

  # Densely packed ground truth. Non empty only if is_test=True.
  ground_truth_batch_size: int
  ground_truth_batch_ids: List[int]  # (ground_truth_batch_size)
  ground_truth_batch: List[
      List[int]]  # (ground_truth_batch_size, seq_len)


def batch_with_batch_size(xs,
                          batch_size,
                          num_rows_per_batch,
                          seq_len = 16,
                          is_test = False,
                          ground_truth_batch_size = 0,
                          shuffle = False):
  """Create batches from a list of examples.

  Splits long user histories and ground truth into multiple examples and
  pads any extra space with -1.

  This function assumes that no user history is large enough to overfill the
  space in a single batch i.e. all user histories <= batch_size * seq_len.

  Args:
    xs: list of examples. (row_id, history, ground_truth) if is_test if True,
      (row_id, history) otherwise.
    batch_size: size of the batch.
    num_rows_per_batch: fixed number of rows per batch. We keep this static to
      not change the shapes of tensors every iteration.
    seq_len: sequence length of the user history.
    is_test: if this the test set, we process batch ground_truth as well.
    ground_truth_batch_size: size of ground truth batch.
    shuffle: is True, we shuffle xs before batching.

  Returns:
    Batches for the whole dataset.
  """
  batches = []

  if shuffle:
    # We first check if its a multi-process TPU setup, each process will end up
    # having a different order and dataset.shard will NOT work correctly.
    if jax.process_count() > 1:
      raise ValueError(
          "Shuffling is not allowed in multi-process setup since dataset.shard"
          "will stop working correctly later on."
      )

    random.shuffle(xs)

  batch = Batch(
      is_test=is_test,
      batch_size=batch_size,
      seq_len=seq_len,
      num_rows_per_batch=num_rows_per_batch,
      batch=[],
      batch_ids=[],
      row_ids=[],
      item_lengths=[],
      ground_truth_batch_size=ground_truth_batch_size,
      ground_truth_batch_ids=[],
      ground_truth_batch=[])
  current_batch_id = 0
  for x in xs:
    row_id = x[0]
    history = x[1]

    if batch_size * seq_len < len(history):
      raise ValueError(
          "Increase batch size or seq len so that we can fit the largest "
          "example in a single batch."
      )

    is_over_ground_truth_batch_size = False
    if is_test:
      ground_truth = x[2]

      if ground_truth_batch_size * seq_len < len(ground_truth):
        raise ValueError(
            "Increase ground_truth_batch_size or seq len so that we can fit the"
            "largest ground truth in a single batch."
        )

      ground_truth_capacity_needed = (len(ground_truth) // seq_len) + 1
      is_over_ground_truth_batch_size = len(
          batch.ground_truth_batch
      ) + ground_truth_capacity_needed > ground_truth_batch_size

    capacity_needed = (len(history) // seq_len) + 1
    is_over_batch_size = len(batch.batch) + capacity_needed > batch_size
    is_over_num_users_per_batch = current_batch_id + 1 > num_rows_per_batch

    if (is_over_batch_size or is_over_num_users_per_batch or
        is_over_ground_truth_batch_size):
      # Append null user histrories to last user and make sure number of users
      # stay constant.
      num_remaining = batch_size - len(batch.batch)
      batch.batch.extend([[-1] * seq_len] * num_remaining)
      batch.batch_ids.extend([current_batch_id - 1] * num_remaining)
      batch.row_ids.extend([-1] * (num_rows_per_batch - current_batch_id))
      batch.item_lengths.extend([0] * (num_rows_per_batch - current_batch_id))
      if is_test:
        ground_truth_num_remaining = ground_truth_batch_size - len(
            batch.ground_truth_batch)
        batch.ground_truth_batch.extend([[-1] * seq_len] *
                                        ground_truth_num_remaining)
        batch.ground_truth_batch_ids.extend([current_batch_id - 1] *
                                            ground_truth_num_remaining)
      batches.append(batch)

      # Reset.
      batch = Batch(
          is_test=is_test,
          batch_size=batch_size,
          seq_len=seq_len,
          num_rows_per_batch=num_rows_per_batch,
          batch=[],
          batch_ids=[],
          row_ids=[],
          item_lengths=[],
          ground_truth_batch_size=ground_truth_batch_size,
          ground_truth_batch_ids=[],
          ground_truth_batch=[])
      current_batch_id = 0
    batch.row_ids.append(row_id)
    batch.item_lengths.append(len(history))

    start_index = 0
    while start_index < len(history):
      end_index = min(len(history), start_index + seq_len)
      history_to_be_appended = history[start_index:end_index]
      history_to_be_appended += [-1] * (
          seq_len - len(history_to_be_appended))
      batch.batch.append(history_to_be_appended)
      batch.batch_ids.append(current_batch_id)
      start_index += seq_len

    if is_test:
      ground_truth_start_index = 0
      while ground_truth_start_index < len(ground_truth):
        ground_truth_end_index = min(
            len(ground_truth), ground_truth_start_index + seq_len)
        ground_truth_to_be_appended = ground_truth[
            ground_truth_start_index:ground_truth_end_index]
        ground_truth_to_be_appended += [-1] * (
            seq_len - len(ground_truth_to_be_appended))
        batch.ground_truth_batch.append(ground_truth_to_be_appended)
        batch.ground_truth_batch_ids.append(current_batch_id)
        ground_truth_start_index += seq_len

    current_batch_id += 1

  # Fill the last batch.
  num_remaining = batch_size - len(batch.batch)
  batch.batch.extend([[-1] * seq_len] * num_remaining)
  batch.batch_ids.extend([current_batch_id - 1] * num_remaining)
  batch.row_ids.extend([-1] * (num_rows_per_batch - current_batch_id))
  batch.item_lengths.extend([0] * (num_rows_per_batch - current_batch_id))
  if is_test:
    ground_truth_num_remaining = ground_truth_batch_size - len(
        batch.ground_truth_batch)
    batch.ground_truth_batch.extend([[-1] * seq_len] *
                                    ground_truth_num_remaining)
    batch.ground_truth_batch_ids.extend([current_batch_id - 1] *
                                        ground_truth_num_remaining)
  batches.append(batch)
  return batches


# TODO(harshm): make these methods private, including
# create_tf_example_from_batch and process_dataset.
def get_bytes_feature(ex, name):
  return list(ex.features.feature[name].bytes_list.value)


def get_ints_feature(ex, name):
  return list(ex.features.feature[name].int64_list.value)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


# TODO(harshm): make the keys module level constants.
def create_tf_example_from_batch(batch):
  """Converts from Batch to serialized TF example."""
  features = collections.OrderedDict()
  batched_history = np.asarray(batch.batch).reshape(-1)
  features["batched_history"] = create_int_feature(batched_history.tolist())
  features["item_lengths"] = create_int_feature(batch.item_lengths)
  features["row_ids"] = create_int_feature(batch.row_ids)
  features["batch_ids"] = create_int_feature(batch.batch_ids)
  if batch.is_test:
    batched_ground_truths = np.asarray(batch.ground_truth_batch).reshape(-1)
    features["batched_ground_truths"] = create_int_feature(
        batched_ground_truths.tolist())
    features["ground_truth_batch_ids"] = create_int_feature(
        batch.ground_truth_batch_ids)
  tf_example = tf.train.Example(
      features=tf.train.Features(feature=features))
  return tf_example.SerializeToString()


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name, t in example.items():
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t
  return example


# TODO(harshm): return Dataset[Batch] instead of Dataset[Dict].
def process_dataset(
    ds,
    is_test = False,
    batch_size = 1024,
    seq_len = 32,
    num_rows_per_batch = 64,
    ground_truth_batch_size = 2048,
    num_devices = 8
):
  """Returns the dataset as a numpy Dataset object.

  Args:
    ds: input tf Dataset.
    is_test: if is_test is True, we decode additional features.
    batch_size: size of the batch.
    seq_len: sequence length of the user history.
    num_rows_per_batch: fixed number of rows per batch. We keep this static to
      not change the shapes of tensors every iteration.
    ground_truth_batch_size: size of ground truth batch.
      Each examples is padded to this size.
    num_devices: the dataset is already batched, but we batch num_devices worth
      of batched examples so that we each device gets one batched example.

  Returns:
    Batches for the whole dataset.
  """
  name_to_features = {
      "batched_history":
          tf.io.FixedLenFeature([batch_size * seq_len], tf.int64),
      "item_lengths":
          tf.io.FixedLenFeature([num_rows_per_batch], tf.int64),
      "row_ids":
          tf.io.FixedLenFeature([num_rows_per_batch], tf.int64),
      "batch_ids":
          tf.io.FixedLenFeature([batch_size], tf.int64)
  }
  if is_test:
    name_to_features["batched_ground_truths"] = tf.io.FixedLenFeature(
        [ground_truth_batch_size, seq_len], tf.int64)
    name_to_features["ground_truth_batch_ids"] = tf.io.FixedLenFeature(
        [ground_truth_batch_size], tf.int64)

  ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())
  ds = ds.map(
      lambda record: _decode_record(record, name_to_features),
      num_parallel_calls=tf.data.AUTOTUNE)
  print(f"Num devices in dataset: {num_devices}")
  ds = ds.batch(batch_size=num_devices, drop_remainder=False)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return tfds.as_numpy(ds)


def load_dataset_from_generator(
    list_of_tf_examples,
    is_test = False,
    batch_size = 1024,
    seq_len = 32,
    num_rows_per_batch = 64,
    ground_truth_batch_size = 1024,
    num_devices = 8
):
  """Returns the dataset as a numpy Dataset object.

  Args:
    list_of_tf_examples: input list of examples.
    is_test: if is_test is True, we decode additional features.
    batch_size: size of the batch.
    seq_len: sequence length of the user history.
    num_rows_per_batch: fixed number of rows per batch. We keep this static to
      not change the shapes of tensors every iteration.
    ground_truth_batch_size: size of ground truth batch.
      Each examples is padded to this size.
    num_devices: the dataset is already batched, but we batch num_devices worth
      of batched examples so that we each device gets one batched example.

  Returns:
    Batches for the whole dataset.
  """

  generator = lambda: (x for x in list_of_tf_examples)
  ds = tf.data.Dataset.from_generator(generator, output_types=tf.string)
  return process_dataset(
      ds=ds,
      is_test=is_test,
      batch_size=batch_size,
      seq_len=seq_len,
      num_rows_per_batch=num_rows_per_batch,
      ground_truth_batch_size=ground_truth_batch_size,
      num_devices=num_devices)


def load_dataset_from_files(
    data_pattern,
    is_test = False,
    batch_size = 1024,
    seq_len = 32,
    num_rows_per_batch = 64,
    ground_truth_batch_size = 2048,
    num_devices = 8
):
  """Returns the dataset as a numpy Dataset object.

  Args:
    data_pattern: input tfrecord file pattern.
    is_test: if is_test is True, we decode additional features.
    batch_size: size of the batch.
    seq_len: sequence length of the user history.
    num_rows_per_batch: fixed number of rows per batch. We keep this static to
      not change the shapes of tensors every iteration.
    ground_truth_batch_size: size of ground truth batch.
    num_devices: the dataset is already batched, but we batch num_devices worth
      of batched examples so that we each device gets one batched example.

  Returns:
    Batches for the whole dataset.
  """

  files = tf.data.Dataset.list_files(tf.io.gfile.glob(data_pattern))
  ds = files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE)
  return process_dataset(
      ds=ds,
      is_test=is_test,
      batch_size=batch_size,
      seq_len=seq_len,
      num_rows_per_batch=num_rows_per_batch,
      ground_truth_batch_size=ground_truth_batch_size,
      num_devices=num_devices)


def batch_and_create_tf_examples(xs,
                                 batch_size,
                                 num_rows_per_batch,
                                 seq_len=16,
                                 is_test=False,
                                 ground_truth_batch_size=0,
                                 num_devices=None,
                                 to_pad=False):
  """Given a list of unbatched examples, batch them and create TF examples."""
  batched_list = batch_with_batch_size(
      xs,
      batch_size=batch_size,
      num_rows_per_batch=num_rows_per_batch,
      seq_len=seq_len,
      is_test=is_test,
      ground_truth_batch_size=ground_truth_batch_size)

  def batch_density(batch):
    np_batch = np.asarray(batch.batch)
    np_ground_truth_batch = np.asarray(batch.ground_truth_batch)
    return (np.sum(np_batch != -1) / np_batch.size,
            np.sum(np_ground_truth_batch != -1) / np_ground_truth_batch.size)

  densities = list(map(batch_density, batched_list))
  history_densities, ground_truth_densities = zip(*densities)

  print(f"History density: {sum(history_densities)/len(history_densities)}")
  if is_test:
    print(
        f"Ground truth density: {sum(ground_truth_densities)/len(ground_truth_densities)}"
    )
  batched_tf_list = list(map(create_tf_example_from_batch, batched_list))

  if to_pad:
    if not num_devices:
      raise ValueError("Set valid num_devices in order to pad.")

    padding_examples_list = list(
        padding_examples(
            len(batched_tf_list), num_devices, batch_size, num_rows_per_batch,
            seq_len, is_test, ground_truth_batch_size))
    batched_tf_list.extend(padding_examples_list)

  return batched_tf_list


def padding_examples(
    list_size,
    num_devices,
    batch_size,
    num_rows_per_batch,
    seq_len=16,
    is_test=False,
    ground_truth_batch_size=0):
  """Generates padding examples to make each batch of same shape."""
  empty_example = create_empty_tf_example(batch_size, num_rows_per_batch,
                                          seq_len, is_test,
                                          ground_truth_batch_size)
  num = num_to_append(list_size, num_devices=num_devices)
  for _ in range(num):
    yield empty_example


def num_to_append(list_size, num_devices):
  return num_devices - list_size % num_devices


def create_empty_tf_example(batch_size,
                            num_rows_per_batch,
                            seq_len=16,
                            is_test=False,
                            ground_truth_batch_size=0):
  """Creates a batch from a single dummy example."""
  if is_test:
    example_for_empty_batch = [(-1, [-1], [-1])]
  else:
    example_for_empty_batch = [(-1, [-1])]
  empty_batch = batch_with_batch_size(
      example_for_empty_batch,
      batch_size=batch_size,
      num_rows_per_batch=num_rows_per_batch,
      seq_len=seq_len,
      is_test=is_test,
      ground_truth_batch_size=ground_truth_batch_size)
  assert len(empty_batch) == 1
  return create_tf_example_from_batch(empty_batch[0])


def tf_examples_to_examples(serialized_examples, is_test=False):
  """Deserializes TF examples and coverts to List[user_id, List[item_id]]."""
  def parse_example(serialized_example):
    example = tf.train.Example()
    example.ParseFromString(serialized_example.numpy())
    return example
  examples = list(map(parse_example, serialized_examples))

  def get_row_id(example):
    return get_ints_feature(example, "row_tag")[0]

  def get_histories(example):
    return get_ints_feature(example, "col_tag")

  def get_ground_truth(example):
    return get_ints_feature(example, "gt_tag")

  row_ids = list(map(get_row_id, examples))
  histories = list(map(get_histories, examples))

  if is_test:
    gts = list(map(get_ground_truth, examples))
    processed_examples = list(zip(row_ids, histories, gts))
  else:
    processed_examples = list(zip(row_ids, histories))
  return processed_examples
