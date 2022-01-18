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

"""Deterministic input pipeline."""

import dataclasses
from typing import Dict
from clu import deterministic_data
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# Register spherical_mnist so that tfds.load works.
import spin_spherical_cnns.spherical_mnist.spherical_mnist  # pylint: disable=unused-import


# Dataset creation functions return info, train, validation and test sets.
@dataclasses.dataclass
class DatasetSplits:
  info: tfds.core.DatasetInfo
  train: tf.data.Dataset
  validation: tf.data.Dataset
  test: tf.data.Dataset


def create_datasets(config,
                    data_rng):
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations.

  Args:
    config: Configuration to use.
    data_rng: PRNGKey for seeding operations in the training dataset.

  Returns:
    A DatasetSplits object containing the dataset info, and the train,
    validation, and test splits.
  """
  if config.dataset == "tiny_dummy":
    return _create_dataset_tiny_dummy(config)
  if config.dataset in ["spherical_mnist/rotated", "spherical_mnist/canonical"]:
    return _create_dataset_spherical_mnist(config, data_rng)
  else:
    raise ValueError(f"Dataset {config.dataset} not supported.")


def _create_dataset_tiny_dummy(
    config):
  """Create low-resolution dataset for testing. See create_datasets()."""
  size = 100
  resolution = 8
  n_spins = 1
  n_channels = 1
  num_classes = 10
  shape = (size, resolution, resolution, n_spins, n_channels)
  entries = np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
  labels = np.resize(np.arange(num_classes), [size])
  train_dataset = tf.data.Dataset.from_tensor_slices({"input": entries,
                                                      "label": labels})
  train_dataset = train_dataset.batch(config.per_device_batch_size,
                                      drop_remainder=True)
  train_dataset = train_dataset.batch(jax.local_device_count(),
                                      drop_remainder=True)

  features = tfds.features.FeaturesDict(
      {"label": tfds.features.ClassLabel(num_classes=num_classes)})
  builder = tfds.testing.DummyDataset()
  dataset_info = tfds.core.DatasetInfo(builder=builder, features=features)

  # We don't really care about the difference between train, validation and test
  # and for dummy data.
  return DatasetSplits(info=dataset_info,
                       train=train_dataset,
                       validation=train_dataset.take(5),
                       test=train_dataset.take(5))


def _preprocess_spherical_mnist(
    features):
  features["input"] = tf.cast(features["image"], tf.float32) / 255.0
  # Add dummy spin dimension.
  features["input"] = features["input"][Ellipsis, None, :]
  features.pop("image")
  return features


def _create_train_dataset(config,
                          dataset_builder,
                          split,
                          data_rng):
  """Create train dataset."""
  # This ensures determinism in distributed setting.
  train_split = deterministic_data.get_read_instruction_for_host(
      split, dataset_info=dataset_builder.info)
  train_dataset = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      rng=data_rng,
      preprocess_fn=_preprocess_spherical_mnist,
      shuffle_buffer_size=config.shuffle_buffer_size,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      num_epochs=config.num_epochs,
      shuffle=True,
  )
  options = tf.data.Options()
  options.experimental_external_state_policy = (
      tf.data.experimental.ExternalStatePolicy.WARN)
  train_dataset = train_dataset.with_options(options)

  return train_dataset


def _create_eval_dataset(config,
                         dataset_builder,
                         split):
  """Create evaluation dataset (validation or test sets)."""
  # This ensures the correct number of elements in the validation sets.
  num_validation_examples = (
      dataset_builder.info.splits[split].num_examples)
  eval_split = deterministic_data.get_read_instruction_for_host(
      split, dataset_info=dataset_builder.info, drop_remainder=False)

  eval_num_batches = None
  if config.eval_pad_last_batch:
    # This is doing some extra work to get exactly all examples in the
    # validation split. Without this the dataset would first be split between
    # the different hosts and then into batches (both times dropping the
    # remainder). If you don't mind dropping a few extra examples you can omit
    # the `pad_up_to_batches` argument.
    eval_batch_size = jax.local_device_count() * config.per_device_batch_size
    eval_num_batches = int(
        np.ceil(num_validation_examples / eval_batch_size / jax.host_count()))
  return deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      # Only cache dataset in distributed setup to avoid consuming a lot of
      # memory in Colab and unit tests.
      cache=jax.host_count() > 1,
      batch_dims=[jax.local_device_count(), config.per_device_batch_size],
      num_epochs=1,
      shuffle=False,
      preprocess_fn=_preprocess_spherical_mnist,
      pad_up_to_batches=eval_num_batches,
  )


def _create_dataset_spherical_mnist(
    config,
    data_rng):
  """Create Spherical MNIST. See create_datasets()."""

  dataset_loaded = False
  if not dataset_loaded:
    dataset_builder = tfds.builder("spherical_mnist")

  if config.dataset == "spherical_mnist/rotated":
    train_split = "train_rotated"
    validation_split = "validation_rotated"
    test_split = "test_rotated"
  elif config.dataset == "spherical_mnist/canonical":
    train_split = "train_canonical"
    validation_split = "validation_canonical"
    test_split = "test_canonical"
  else:
    raise ValueError(f"Unrecognized dataset: {config.dataset}")

  if config.combine_train_val_and_eval_on_test:
    train_split = f"{train_split} + {validation_split}"

  train_dataset = _create_train_dataset(config,
                                        dataset_builder,
                                        train_split,
                                        data_rng)
  validation_dataset = _create_eval_dataset(config,
                                            dataset_builder,
                                            validation_split)
  test_dataset = _create_eval_dataset(config, dataset_builder, test_split)

  return DatasetSplits(info=dataset_builder.info,
                       train=train_dataset,
                       validation=validation_dataset,
                       test=test_dataset)
