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

"""Data preprocessing functions and dataset operations for Stackoverflow next word prediction task."""

import collections

import logging

from typing import Callable, List, Tuple

import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

SUFFLE_SIZE = 10000
MAX_ELEMENTS_PER_USER = 1000


def count_batches(dataset):
  cnt = 0
  for _ in dataset:
    cnt += 1
  return int(cnt)


def split_time(dataset, idx):
  batch_idx = tf.cast(idx, tf.int64)
  train_data_time = dataset.take(batch_idx)
  test_data_time = dataset.skip(batch_idx).take(1)

  return train_data_time, test_data_time


@attr.s(eq=False, frozen=True)
class SpecialTokens(object):
  """Structure for Special tokens.

  Attributes:
    pad: int - Special token for padding.
    oov: list - Special tokens for out of vocabulary tokens.
    bos: int - Special token for beginning of sentence.
    eos: int - Special token for end of sentence.
  """
  pad = attr.ib()
  oov = attr.ib()
  bos = attr.ib()
  eos = attr.ib()


def create_vocab(vocab_size):
  """Creates vocab from `vocab_size` most common words in Stackoverflow."""
  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts(
      cache_dir='/tmp')
  return list(vocab_dict.keys())[:vocab_size]


def split_input_target(chunk):
  """Generate input and target data.

  The task of language model is to predict the next word.

  Args:
    chunk: A Tensor of text data.

  Returns:
    A tuple of input and target data.
  """
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


def build_to_ids_fn(
    vocab,
    max_sequence_length,
    num_oov_buckets = 1):
  """Constructs function mapping examples to sequences of token indices."""
  special_tokens = get_special_tokens(len(vocab), num_oov_buckets)
  bos = special_tokens.bos
  eos = special_tokens.eos

  table_values = np.arange(len(vocab), dtype=np.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=num_oov_buckets)

  def to_ids(example):

    sentence = tf.reshape(example['tokens'], shape=[1])
    words = tf.strings.split(sentence, sep=' ').values
    truncated_words = words[:max_sequence_length]
    tokens = table.lookup(truncated_words) + 1
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_sequence_length),
        lambda: tf.concat([tokens, [eos]], 0), lambda: tokens)

    return tf.concat([[bos], tokens], 0)

  return to_ids


def batch_and_split(dataset, max_sequence_length,
                    batch_size):
  return dataset.padded_batch(
      batch_size, padded_shapes=[max_sequence_length + 1]).map(
          split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_special_tokens(vocab_size,
                       num_oov_buckets = 1):
  """Gets tokens dataset preprocessing code will add to Stackoverflow."""
  return SpecialTokens(
      pad=0,
      oov=[vocab_size + 1 + n for n in range(num_oov_buckets)],
      bos=vocab_size + num_oov_buckets + 1,
      eos=vocab_size + num_oov_buckets + 2)


def _creation_date_string_to_integer(dates):
  """Converts ISO date string to integer that can be sorted.

  Returned integers retain the property that sorting the integers results in
  dates sorted in chronological order, so these integers can be used to sort
  examples by date. Ignores fractional seconds if provided.
  Assumes standard time offset.

  For example:
    2009-06-15T13:45:30 -> 20090615134530
    2009-06-15T13:45:30.345Z -> 20090615134530

  Args:
    dates: A tf.string tensor of dates in simplified ISO 8601 format. The data
      produced by `tff.simulation.datasets.stackoverflow.load_data` conforms
      to this format.

  Returns:
    A tf.int64 tensor of integers representing dates.
  """
  year = tf.strings.to_number(
      tf.strings.substr(dates, 0, 4), out_type=tf.int64)
  month = tf.strings.to_number(
      tf.strings.substr(dates, 5, 2), out_type=tf.int64)
  day = tf.strings.to_number(
      tf.strings.substr(dates, 8, 2), out_type=tf.int64)
  hour = tf.strings.to_number(
      tf.strings.substr(dates, 11, 2), out_type=tf.int64)
  minute = tf.strings.to_number(
      tf.strings.substr(dates, 14, 2), out_type=tf.int64)
  second = tf.strings.to_number(
      tf.strings.substr(dates, 17, 2), out_type=tf.int64)

  timestamp = 0
  timestamp = (timestamp + year) * 100
  timestamp = (timestamp + month) * 100
  timestamp = (timestamp + day) * 100
  timestamp = (timestamp + hour) * 100
  timestamp = (timestamp + minute) * 100
  timestamp = timestamp + second
  return timestamp


def _sort_examples_by_date(
    examples):
  """Sorts a batch of dataset elements by increasing creation date.

  Sorting is stable, so original ordering is consistently retained for ties.

  Args:
    examples: A batch of examples.

  Returns:
    Output batch, sorted by creation date.
  """
  date_integers = _creation_date_string_to_integer(examples['creation_date'])
  sorted_indices = tf.argsort(date_integers, stable=True)
  new_examples = collections.OrderedDict()
  for key in examples:
    new_examples[key] = tf.gather(examples[key], sorted_indices)
  return new_examples


def build_preprocess_fn(vocab,
                        so_nwp_sequence_length=20,
                        batch_size=128,
                        so_nwp_num_oov_buckets=1,
                        debug=False):
  """Builds a preprocessing function.

  Args:
    vocab: A list of word vocaburay to include in the task.
    so_nwp_sequence_length: Sequence length to be padded per example.
    batch_size: Batch size for the dataset.
    so_nwp_num_oov_buckets: Number of buckets for the OOVs.
    debug: Degug mode or not.

  Returns:
    A preprocessing function for tf.data.Dataset.
  """

  def preprocess_fn(org_dataset):
    to_ids = build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=so_nwp_sequence_length,
        num_oov_buckets=so_nwp_num_oov_buckets)
    dataset = org_dataset.map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    id_dataset = org_dataset.map(
        lambda x: x['client_id'],
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=[so_nwp_sequence_length + 1])
    dataset = dataset.map(
        split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((dataset, id_dataset.batch(batch_size)))

    def _reorder_id(x, idx):
      return (x[0], idx), x[1]

    dataset = dataset.map(_reorder_id)

    if debug:
      logging.info('Test run ...')
      return dataset.take(100)
    else:
      return dataset

  return preprocess_fn


def create_centralized_datasets(preprocess_fn,
                                to_embedding_id,
                                sample_valid_client_ids):
  """Creates centralized dataset for both training and testing.

  Args:
    preprocess_fn: A preprocess function.
    to_embedding_id: An encoding function maps client id string to id number.
    sample_valid_client_ids: Client ids for validation.

  Returns:
    Centralized, preprocessed, and batched training dataset and
    validation dataset.
  """
  def build_centralized_dataset(clientdata, sample_valid_client_ids=None):
    def _create_dataset_with_id(client_id):
      def add_id(x):
        x['client_id'] = to_embedding_id(client_id)
        return x

      # pylint: disable=protected-access
      return clientdata._create_dataset(client_id).map(
          add_id, num_parallel_calls=tf.data.experimental.AUTOTUNE).take(
              MAX_ELEMENTS_PER_USER)

    if sample_valid_client_ids is not None:
      extract_ids = sample_valid_client_ids
    else:
      extract_ids = clientdata.client_ids

    nested_dataset = tf.data.Dataset.from_tensor_slices(extract_ids)
    cent_train = nested_dataset.flat_map(_create_dataset_with_id)
    logging.info(cent_train.element_spec)

    return cent_train

  train_set, valid_set, _ = tff.simulation.datasets.stackoverflow.load_data()
  train_dataset = build_centralized_dataset(train_set)
  val_dataset = build_centralized_dataset(valid_set, sample_valid_client_ids)

  train_dataset = preprocess_fn(train_dataset).cache().shuffle(
      SUFFLE_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
  val_dataset = preprocess_fn(val_dataset).cache().prefetch(
      tf.data.experimental.AUTOTUNE)

  return train_dataset, val_dataset


# Personalization data
def sort_by_date_pipe(dataset):
  return dataset.take(MAX_ELEMENTS_PER_USER).batch(MAX_ELEMENTS_PER_USER).map(
      _sort_examples_by_date).unbatch()


def random_split_pipe(dataset, num_shards, index):
  return dataset.take(MAX_ELEMENTS_PER_USER).shard(num_shards, index)


def build_split_centralized_dataset(clientdata,
                                    preprocess_fn,
                                    to_embedding_id,
                                    sample_client_ids,
                                    split_by='date'):
  """Creates centralized dataset for both training and testing.

  The following functions are created to flat_map to avoid loading all
  dataset tensor into the memory.

  Args:
    clientdata: tff client datasets object.
    preprocess_fn: A data preprocessing function.
    to_embedding_id: An encoding function maps client id string to id number.
    sample_client_ids: Client ids for validation.
    split_by: setup of client data split, choose 'data' or 'random'.

  Returns:
    A training set for fine-tuning, a split training set with smaller size, a
    test set.
  """

  # The number of split for train/test sets
  # Default to 50%/50%, that is the test size is the last 50%
  # We further have a smaller split, which is the first 1/num_small_split.
  num_split = 2
  num_small_split = 4

  # Splits by date
  def _create_dataset_with_id_train(client_id):
    # add_in must be created in the _create_dataset function.
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = sort_by_date_pipe(
        clientdata._create_dataset(client_id)).map(add_id)
    total_size = client_ds.reduce(0, lambda x, _: x + 1)
    num_elements_half = tf.cast((total_size - 1) / num_split, dtype=tf.int64)
    return client_ds.take(num_elements_half)

  def _create_dataset_with_id_train_sp(client_id):
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = sort_by_date_pipe(
        clientdata._create_dataset(client_id)).map(add_id)
    total_size = client_ds.reduce(0, lambda x, _: x + 1)
    num_elements_ten_percent = tf.cast(
        (total_size - 1) / num_small_split, dtype=tf.int64)
    return client_ds.take(num_elements_ten_percent)

  def _create_dataset_with_id_test(client_id):
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = sort_by_date_pipe(
        clientdata._create_dataset(client_id)).map(add_id)
    total_size = client_ds.reduce(0, lambda x, _: x + 1)
    num_elements_half = tf.cast((total_size - 1) / num_split, dtype=tf.int64)
    return client_ds.skip(num_elements_half)

  # split by random
  def _random_split_create_dataset_with_id_train(client_id):
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = clientdata._create_dataset(client_id)
    client_ds = random_split_pipe(client_ds, num_split, 0).map(add_id)
    return client_ds

  def _random_split_create_dataset_with_id_train_sp(client_id):
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = clientdata._create_dataset(client_id)
    client_ds = random_split_pipe(
        random_split_pipe(client_ds, num_split, 0), num_small_split,
        0).map(add_id)
    return client_ds

  def _random_split_create_dataset_with_id_test(client_id):
    def add_id(x):
      x['client_id'] = to_embedding_id(client_id)
      return x

    # pylint: disable=protected-access
    client_ds = clientdata._create_dataset(client_id)
    client_ds = random_split_pipe(client_ds, num_split, 1).map(add_id)
    return client_ds

  nested_dataset = tf.data.Dataset.from_tensor_slices(sample_client_ids)

  if split_by == 'date':
    train_fn = _create_dataset_with_id_train
    split_fn = _create_dataset_with_id_train_sp
    test_fn = _create_dataset_with_id_test

  elif split_by == 'random':
    train_fn = _random_split_create_dataset_with_id_train
    split_fn = _random_split_create_dataset_with_id_train_sp
    test_fn = _random_split_create_dataset_with_id_test

  per_train = nested_dataset.flat_map(train_fn)
  per_train_split = nested_dataset.flat_map(split_fn)
  per_test = nested_dataset.flat_map(test_fn)

  per_train = preprocess_fn(per_train).cache()
  per_train_split = preprocess_fn(per_train_split).cache()
  per_test = preprocess_fn(per_test).cache()

  return per_train, per_train_split, per_test


