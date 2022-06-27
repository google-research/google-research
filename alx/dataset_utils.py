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

"""Utilities to build TF dataset objects."""

import tensorflow as tf

from alx import als
from alx import batching_utils


def build_datasets(cfg):
  local_device_count = len(als.get_local_devices(cfg))
  device_count = len(als.get_devices(cfg))
  if cfg.is_pre_batched:
    return _build_pre_batched_datasets(cfg, local_device_count)
  else:
    return _batch_and_build_datasets(cfg, local_device_count, device_count)


def _batch_and_build_datasets(cfg, local_device_count,
                              device_count):
  """Creates TF dataset objects for train, train_t and test files."""
  files = tf.data.Dataset.list_files(tf.io.gfile.glob(cfg.train_files))
  tf_examples = list(tf.data.TFRecordDataset(files))
  examples = batching_utils.tf_examples_to_examples(tf_examples)
  train_tf_list = batching_utils.batch_and_create_tf_examples(
      examples,
      batch_size=cfg.batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.num_rows_per_batch,
      num_devices=device_count,
      to_pad=True)

  files = tf.data.Dataset.list_files(
      tf.io.gfile.glob(cfg.train_transpose_files))
  tf_examples = list(tf.data.TFRecordDataset(files))
  examples = batching_utils.tf_examples_to_examples(tf_examples)
  train_transpose_tf_list = batching_utils.batch_and_create_tf_examples(
      examples,
      batch_size=cfg.transpose_batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.transpose_num_rows_per_batch,
      num_devices=device_count,
      to_pad=True)

  files = tf.data.Dataset.list_files(tf.io.gfile.glob(cfg.test_files))
  tf_examples = list(tf.data.TFRecordDataset(files))
  examples = batching_utils.tf_examples_to_examples(tf_examples, is_test=True)
  test_tf_list = batching_utils.batch_and_create_tf_examples(
      examples,
      is_test=True,
      batch_size=cfg.eval_batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.eval_num_rows_per_batch,
      num_devices=device_count,
      ground_truth_batch_size=cfg.ground_truth_batch_size,
      to_pad=True)

  ds = batching_utils.load_dataset_from_generator(
      train_tf_list,
      batch_size=cfg.batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.num_rows_per_batch,
      num_devices=local_device_count)
  tds = batching_utils.load_dataset_from_generator(
      train_transpose_tf_list,
      batch_size=cfg.transpose_batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.transpose_num_rows_per_batch,
      num_devices=local_device_count)
  test_ds = batching_utils.load_dataset_from_generator(
      test_tf_list,
      is_test=True,
      batch_size=cfg.eval_batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.eval_num_rows_per_batch,
      ground_truth_batch_size=cfg.ground_truth_batch_size,
      num_devices=local_device_count)
  return ds, tds, test_ds


def _build_pre_batched_datasets(cfg, local_device_count):
  """Creates TF datasets for pre-batched train, train_t and test files."""
  ds = batching_utils.load_dataset_from_files(
      cfg.train_files,
      batch_size=cfg.batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.num_rows_per_batch,
      num_devices=local_device_count)
  tds = batching_utils.load_dataset_from_files(
      cfg.train_transpose_files,
      batch_size=cfg.transpose_batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.transpose_num_rows_per_batch,
      num_devices=local_device_count)
  test_ds = batching_utils.load_dataset_from_files(
      cfg.test_files,
      is_test=True,
      batch_size=cfg.eval_batch_size,
      seq_len=cfg.seq_len,
      num_rows_per_batch=cfg.eval_num_rows_per_batch,
      ground_truth_batch_size=cfg.ground_truth_batch_size,
      num_devices=local_device_count)
  return ds, tds, test_ds
