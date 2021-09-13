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

"""Dataset loader and processor."""
from typing import Tuple

from clu import deterministic_data
import jax
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_train_dataset(
    task,
    batch_size,
    substeps,
    data_rng):
  """Create datasets for training."""
  # Compute batch size per device from global batch size..
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)

  train_split = deterministic_data.get_read_instruction_for_host(
      "train", dataset_builder.info.splits["train"].num_examples)
  batch_dims = [jax.local_device_count(), substeps, per_device_batch_size]
  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=_preprocess_cifar10,
      rng=data_rng)

  return dataset_builder.info, train_ds


def create_eval_dataset(
    task,
    batch_size,
    subset):
  """Create datasets for evaluation."""
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)

  eval_split = deterministic_data.get_read_instruction_for_host(
      subset, dataset_builder.info.splits[subset].num_examples)
  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      num_epochs=1,
      shuffle=False,
      batch_dims=[jax.local_device_count(), per_device_batch_size],
      preprocess_fn=_preprocess_cifar10)

  return dataset_builder.info, eval_ds


def _preprocess_cifar10(features):
  """Helper to extract images from dict."""
  return features["image"]
