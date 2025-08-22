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

# Copyright 2024 Google LLC
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

"""Dataset utilities."""
from typing import Dict, Tuple

from absl import logging
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import transformers

# Buffer size on already batched data.
_BUFFER_SIZE = 8
_DATA_DIR = "."
X_KEY = "x"
TARGET_KEY = "y"
_EVAL_BATCH_SIZE_MULTIPLIER = 1


def _make_dataset_transform_fn(
    tokenizer,
    text_key,
    target_key,
    max_length = 512,
):
  """Creates a dataset transformation function."""

  @tf.function
  def transform_fn(inputs):
    # This function cannot execute as a graph as the tokenizer expects a list
    # of strings and not symbolic tensors.
    @tf.py_function(Tout=tf.int32)
    def tokenize(x):
      x = x.numpy().tolist()
      if isinstance(x[0], bytes):
        x = [b.decode('utf-8') for b in x]

      toks = tokenizer.batch_encode_plus(
          x,
          max_length=max_length,
          padding="max_length",
          truncation=True,
          return_tensors="tf",
      )
      return toks['input_ids']

    return {X_KEY: tokenize(inputs[text_key]), TARGET_KEY: inputs[target_key]}

  return transform_fn


def process_datasets_before_training(
    dataset,
    # test_ds: tf.data.Dataset,
    # epochs: int,
    is_train = True,
):
  """Processes train and test datasets before training."""

  # filename = tempfile.mkdtemp() # Re-enable after disk OOM is resolved.
  # train_ds = train_ds.cache(filename)
  # This gives a false warning: b/194670791
  dataset = dataset.cache()
  if is_train:
    dataset = dataset.shuffle(
        buffer_size=_BUFFER_SIZE,
        reshuffle_each_iteration=True,
    )
    # dataset = dataset.repeat(epochs)
    dataset = dataset.repeat(-1)

  # Batch again for distributed evaluation.
  # test_ds = test_ds.batch(jax.local_device_count(), drop_remainder=True)

  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  # test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
  return dataset


def _carve_test_dataset(
    train_dataset,
    batch_size,
    test_set_size = 1024,
):
  """Carves test dataset from training dataset, returns new train and test sets."""
  if test_set_size < batch_size:
    logging.warn("Test set size must be >= batch size. Increasing test size.")
    test_set_size = batch_size
  test_dataset = train_dataset.take(test_set_size)
  # Note that this may result in distribution shift between train and test.
  # train_dataset = train_dataset.skip(test_set_size).take_while(lambda x: True)
  return test_dataset


def get_train_dataset(
    dataset_name,
    tokenizer,
    batch_size,
    # epochs: int,
    text_key,
    target_key,
    test_set_size = 1024,
    download_data = True,
):
  """Retrieves the specified train dataset."""
  # Do not shuffle train_ds as we sometimes create test and validation sets out
  # of it. Each training iteration will shuffle the data.
  train_ds = tfds.load(
      dataset_name,
      split="train",
      shuffle_files=False,
      download=download_data,
      data_dir=_DATA_DIR,
  )
  try:
    tfds.load(
        dataset_name,
        split="test",
        shuffle_files=False,
        download=download_data,
        data_dir=_DATA_DIR,
    )
  except ValueError:
    # Test set not available in this case.
    train_ds = train_ds.skip(test_set_size).take_while(lambda x: True)

  # Uncomment the sampling while debugging / testing the pipeline.
  # train_ds = train_ds.take(32768)

  # Distributed strategy requires datasets to be batched per-replica.
  per_device_batch_size = batch_size // jax.local_device_count()
  train_ds = train_ds.batch(per_device_batch_size, drop_remainder=True)

  transform_fn = _make_dataset_transform_fn(tokenizer, text_key, target_key)
  train_ds = train_ds.map(transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
  return process_datasets_before_training(train_ds, is_train=True)


def get_test_and_validation_dataset(
    dataset_name,
    tokenizer,
    batch_size,
    # epochs: int,
    text_key,
    target_key,
    validation_set_size = 256,
    test_set_size = 1024,
    download_data = True,
):
  """Retrieves specified test and validation datasets."""
  try:
    test_ds = tfds.load(
        dataset_name,
        split="test",
        shuffle_files=False,
        download=download_data,
        data_dir=_DATA_DIR,
    )
  except ValueError:
    train_ds = tfds.load(
        dataset_name,
        split="train",
        shuffle_files=False,
        download=download_data,
        data_dir=_DATA_DIR,
    )
    test_ds = _carve_test_dataset(
        train_ds, batch_size=batch_size, test_set_size=test_set_size
    )
    logging.warning(
        "Failed to load test dataset. Using part of training set for test"
        " evaluation"
    )

  # test_ds = test_ds.take(10240)
  validation_ds = test_ds.take(validation_set_size)
  test_ds = test_ds.skip(validation_set_size).take_while(lambda x: True)
  # Distributed strategy requires datasets to be batched per-replica.
  # per_device_batch_size = batch_size // jax.local_device_count()
  test_ds = test_ds.batch(
      batch_size * _EVAL_BATCH_SIZE_MULTIPLIER, drop_remainder=True
  )
  validation_ds = validation_ds.batch(
      batch_size * _EVAL_BATCH_SIZE_MULTIPLIER, drop_remainder=True
  )
  # test_ds = test_ds.batch(
  #    batch_size // jax.local_device_count(), drop_remainder=True
  # )

  transform_fn = _make_dataset_transform_fn(tokenizer, text_key, target_key)
  # train_ds = train_ds.map(transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.map(transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
  validation_ds = validation_ds.map(
      transform_fn, num_parallel_calls=tf.data.AUTOTUNE
  )
  test_ds = process_datasets_before_training(test_ds, is_train=False)
  validation_ds = process_datasets_before_training(
      validation_ds, is_train=False
  )
  return test_ds, validation_ds


def get_dataset(
    dataset_name,
    tokenizer,
    batch_size,
    text_key,
    target_key,
    test_set_size = 1024,
    download_data = True,
):
  """Retrieves specified train and test datasets."""
  train_ds = tfds.load(
      dataset_name,
      split="train",
      shuffle_files=True,
      download=download_data,
      data_dir=_DATA_DIR,
  )
  try:
    test_ds = tfds.load(
        dataset_name,
        split="test",
        shuffle_files=False,
        download=download_data,
        data_dir=_DATA_DIR,
    )
  except ValueError:
    train_ds, test_ds = _carve_test_dataset(
        train_ds, batch_size=batch_size, test_set_size=test_set_size
    )
    logging.warning(
        "Failed to load test dataset. Using part of training set for test"
        " evaluation"
    )
  # Temporary sampling while testing the pipeline.
  # train_ds = train_ds.take(32768)
  # test_ds = test_ds.take(10240)
  # Distributed strategy requires datasets to be batched per-replica.
  per_device_batch_size = batch_size // jax.local_device_count()
  train_ds = train_ds.batch(per_device_batch_size, drop_remainder=True)
  # test_ds = test_ds.batch(
  #    batch_size // jax.local_device_count(), drop_remainder=True
  # )
  test_ds = test_ds.batch(
      batch_size * _EVAL_BATCH_SIZE_MULTIPLIER, drop_remainder=False
  )

  transform_fn = _make_dataset_transform_fn(tokenizer, text_key, target_key)
  train_ds = train_ds.map(transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
  test_ds = test_ds.map(transform_fn, num_parallel_calls=tf.data.AUTOTUNE)
  train_ds = process_datasets_before_training(train_ds, is_train=True)
  test_ds = process_datasets_before_training(test_ds, is_train=False)
  return train_ds, test_ds
