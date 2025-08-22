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

"""Input pipeline for TFDS datasets."""

import functools
import os
from typing import Dict, List, Tuple

from clu import deterministic_data
from clu import preprocess_spec

import jax
import jax.numpy as jnp
import ml_collections

import sunds
import tensorflow as tf
import tensorflow_datasets as tfds

from invariant_slot_attention.lib import preprocessing

Array = jnp.ndarray
PRNGKey = Array


PATH_CLEVR_WITH_MASKS = "gs://multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords"
FEATURES_CLEVR_WITH_MASKS = {
    "image": tf.io.FixedLenFeature([240, 320, 3], tf.string),
    "mask": tf.io.FixedLenFeature([11, 240, 320, 1], tf.string),
    "x": tf.io.FixedLenFeature([11], tf.float32),
    "y": tf.io.FixedLenFeature([11], tf.float32),
    "z": tf.io.FixedLenFeature([11], tf.float32),
    "pixel_coords": tf.io.FixedLenFeature([11, 3], tf.float32),
    "rotation": tf.io.FixedLenFeature([11], tf.float32),
    "size": tf.io.FixedLenFeature([11], tf.string),
    "material": tf.io.FixedLenFeature([11], tf.string),
    "shape": tf.io.FixedLenFeature([11], tf.string),
    "color": tf.io.FixedLenFeature([11], tf.string),
    "visibility": tf.io.FixedLenFeature([11], tf.float32),
}

PATH_TETROMINOES = "gs://multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords"
FEATURES_TETROMINOES = {
    "image": tf.io.FixedLenFeature([35, 35, 3], tf.string),
    "mask": tf.io.FixedLenFeature([4, 35, 35, 1], tf.string),
    "x": tf.io.FixedLenFeature([4], tf.float32),
    "y": tf.io.FixedLenFeature([4], tf.float32),
    "shape": tf.io.FixedLenFeature([4], tf.float32),
    "color": tf.io.FixedLenFeature([4, 3], tf.float32),
    "visibility": tf.io.FixedLenFeature([4], tf.float32),
}

PATH_OBJECTS_ROOM = "gs://multi-object-datasets/objects_room/objects_room_train.tfrecords"
FEATURES_OBJECTS_ROOM = {
    "image": tf.io.FixedLenFeature([64, 64, 3], tf.string),
    "mask": tf.io.FixedLenFeature([7, 64, 64, 1], tf.string),
}

PATH_WAYMO_OPEN = "datasets/waymo_v_1_4_0_images/tfrecords"

FEATURES_WAYMO_OPEN = {
    "image": tf.io.FixedLenFeature([128, 192, 3], tf.string),
    "segmentations": tf.io.FixedLenFeature([128, 192], tf.string),
    "depth": tf.io.FixedLenFeature([128, 192], tf.float32),
    "num_objects": tf.io.FixedLenFeature([1], tf.int64),
    "has_mask": tf.io.FixedLenFeature([1], tf.int64),
    "camera": tf.io.FixedLenFeature([1], tf.int64),
}


def _decode_tetrominoes(example_proto):
  single_example = tf.io.parse_single_example(
      example_proto, FEATURES_TETROMINOES)
  for k in ["mask", "image"]:
    single_example[k] = tf.squeeze(
        tf.io.decode_raw(single_example[k], tf.uint8), axis=-1)
  return single_example


def _decode_objects_room(example_proto):
  single_example = tf.io.parse_single_example(
      example_proto, FEATURES_OBJECTS_ROOM)
  for k in ["mask", "image"]:
    single_example[k] = tf.squeeze(
        tf.io.decode_raw(single_example[k], tf.uint8), axis=-1)
  return single_example


def _decode_clevr_with_masks(example_proto):
  single_example = tf.io.parse_single_example(
      example_proto, FEATURES_CLEVR_WITH_MASKS)
  for k in ["mask", "image", "color", "material", "shape", "size"]:
    single_example[k] = tf.squeeze(
        tf.io.decode_raw(single_example[k], tf.uint8), axis=-1)
  return single_example


def _decode_waymo_open(example_proto):
  """Unserializes a serialized tf.train.Example sample."""
  single_example = tf.io.parse_single_example(
      example_proto, FEATURES_WAYMO_OPEN)
  for k in ["image", "segmentations"]:
    single_example[k] = tf.squeeze(
        tf.io.decode_raw(single_example[k], tf.uint8), axis=-1)
  single_example["segmentations"] = tf.expand_dims(
      single_example["segmentations"], axis=-1)
  single_example["depth"] = tf.expand_dims(
      single_example["depth"], axis=-1)
  return single_example


def _preprocess_minimal(example):
  return {
      "image": example["image"],
      "segmentations": tf.cast(tf.argmax(example["mask"], axis=0), tf.uint8),
  }


def _sunds_create_task():
  """Create a sunds task to return images and instance segmentation."""
  return sunds.tasks.Nerf(
      yield_mode=sunds.tasks.YieldMode.IMAGE,
      additional_camera_specs={
          "depth_image": False,  # Not available in the dataset.
          "category_image": False,  # Not available in the dataset.
          "instance_image": True,
          "extrinsics": True,
      },
      additional_frame_specs={"pose": True},
      add_name=True
  )


def preprocess_example(features,
                       preprocess_strs):
  """Processes a single data example.

  Args:
    features: A dictionary containing the tensors of a single data example.
    preprocess_strs: List of strings, describing one preprocessing operation
      each, in clu.preprocess_spec format.

  Returns:
    Dictionary containing the preprocessed tensors of a single data example.
  """
  all_ops = preprocessing.all_ops()
  preprocess_fn = preprocess_spec.parse("|".join(preprocess_strs), all_ops)
  return preprocess_fn(features)  # pytype: disable=bad-return-type  # allow-recursive-types


def get_batch_dims(global_batch_size):
  """Gets the first two axis sizes for data batches.

  Args:
    global_batch_size: Integer, the global batch size (across all devices).

  Returns:
    List of batch dimensions

  Raises:
    ValueError if the requested dimensions don't make sense with the
      number of devices.
  """
  num_local_devices = jax.local_device_count()
  if global_batch_size % jax.host_count() != 0:
    raise ValueError(f"Global batch size {global_batch_size} not evenly "
                     f"divisble with {jax.host_count()}.")
  per_host_batch_size = global_batch_size // jax.host_count()
  if per_host_batch_size % num_local_devices != 0:
    raise ValueError(f"Global batch size {global_batch_size} not evenly "
                     f"divisible with {jax.host_count()} hosts with a per host "
                     f"batch size of {per_host_batch_size} and "
                     f"{num_local_devices} local devices. ")
  return [num_local_devices, per_host_batch_size // num_local_devices]


def create_datasets(
    config,
    data_rng):
  """Create datasets for training and evaluation.

  For the same data_rng and config this will return the same datasets. The
  datasets only contain stateless operations.

  Args:
    config: Configuration to use.
    data_rng: JAX PRNGKey for dataset pipeline.

  Returns:
    A tuple with the training dataset and the evaluation dataset.
  """

  if config.data.dataset_name == "tetrominoes":
    ds = tf.data.TFRecordDataset(
        PATH_TETROMINOES,
        compression_type="GZIP", buffer_size=2*(2**20))
    ds = ds.map(_decode_tetrominoes,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(_preprocess_minimal,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    class TetrominoesBuilder:
      """Builder for tentrominoes dataset."""

      def as_dataset(self, split, *unused_args, ds=ds, **unused_kwargs):
        """Simple function to conform to the builder api."""
        if split == "train":
          # We use 512 training examples.
          ds = ds.skip(100)
          ds = ds.take(512)
          return tf.data.experimental.assert_cardinality(512)(ds)
        elif split == "validation":
          # 100 validation examples.
          ds = ds.take(100)
          return tf.data.experimental.assert_cardinality(100)(ds)
        else:
          raise ValueError("Invalid split.")

    dataset_builder = TetrominoesBuilder()
  elif config.data.dataset_name == "objects_room":
    ds = tf.data.TFRecordDataset(
        PATH_OBJECTS_ROOM,
        compression_type="GZIP", buffer_size=2*(2**20))
    ds = ds.map(_decode_objects_room,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(_preprocess_minimal,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    class ObjectsRoomBuilder:
      """Builder for objects room dataset."""

      def as_dataset(self, split, *unused_args, ds=ds, **unused_kwargs):
        """Simple function to conform to the builder api."""
        if split == "train":
          # 1M - 100 training examples.
          ds = ds.skip(100)
          return tf.data.experimental.assert_cardinality(999900)(ds)
        elif split == "validation":
          # 100 validation examples.
          ds = ds.take(100)
          return tf.data.experimental.assert_cardinality(100)(ds)
        else:
          raise ValueError("Invalid split.")

    dataset_builder = ObjectsRoomBuilder()
  elif config.data.dataset_name == "clevr_with_masks":
    ds = tf.data.TFRecordDataset(
        PATH_CLEVR_WITH_MASKS,
        compression_type="GZIP", buffer_size=2*(2**20))
    ds = ds.map(_decode_clevr_with_masks,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(_preprocess_minimal,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    class CLEVRWithMasksBuilder:
      def as_dataset(self, split, *unused_args, ds=ds, **unused_kwargs):
        if split == "train":
          ds = ds.skip(100)
          return tf.data.experimental.assert_cardinality(99900)(ds)
        elif split == "validation":
          ds = ds.take(100)
          return tf.data.experimental.assert_cardinality(100)(ds)
        else:
          raise ValueError("Invalid split.")

    dataset_builder = CLEVRWithMasksBuilder()
  elif config.data.dataset_name == "waymo_open":
    train_path = os.path.join(
        PATH_WAYMO_OPEN, "training/camera_1/*tfrecords*")
    eval_path = os.path.join(
        PATH_WAYMO_OPEN, "validation/camera_1/*tfrecords*")

    train_files = tf.data.Dataset.list_files(train_path)
    eval_files = tf.data.Dataset.list_files(eval_path)

    train_data_reader = functools.partial(
        tf.data.TFRecordDataset,
        compression_type="ZLIB", buffer_size=2*(2**20))
    eval_data_reader = functools.partial(
        tf.data.TFRecordDataset,
        compression_type="ZLIB", buffer_size=2*(2**20))

    train_dataset = train_files.interleave(
        train_data_reader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    eval_dataset = eval_files.interleave(
        eval_data_reader, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = train_dataset.map(
        _decode_waymo_open, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    eval_dataset = eval_dataset.map(
        _decode_waymo_open, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # We need to set the dataset cardinality. We assume we have
    # the full dataset.
    train_dataset = train_dataset.apply(
        tf.data.experimental.assert_cardinality(158081))

    class WaymoOpenBuilder:
      def as_dataset(self, split, *unused_args, **unused_kwargs):
        if split == "train":
          return train_dataset
        elif split == "validation":
          return eval_dataset
        else:
          raise ValueError("Invalid split.")

    dataset_builder = WaymoOpenBuilder()
  elif config.data.dataset_name == "multishapenet_easy":
    dataset_builder = sunds.builder(
        name=config.get("tfds_name", "msn_easy"),
        data_dir=config.get(
            "data_dir", "gs://kubric-public/tfds"),
        try_gcs=True)
    dataset_builder.as_dataset = functools.partial(
        dataset_builder.as_dataset, task=_sunds_create_task())
  elif config.data.dataset_name == "tfds":
    dataset_builder = tfds.builder(
        config.data.tfds_name, data_dir=config.data.data_dir)
  else:
    raise ValueError("Please specify a valid dataset name.")

  batch_dims = get_batch_dims(config.batch_size)

  train_preprocess_fn = functools.partial(
      preprocess_example, preprocess_strs=config.preproc_train)
  eval_preprocess_fn = functools.partial(
      preprocess_example, preprocess_strs=config.preproc_eval)

  train_split_name = config.get("train_split", "train")
  eval_split_name = config.get("validation_split", "validation")

  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split_name,
      rng=data_rng,
      preprocess_fn=train_preprocess_fn,
      cache=False,
      shuffle_buffer_size=config.data.shuffle_buffer_size,
      batch_dims=batch_dims,
      num_epochs=None,
      shuffle=True)

  if config.data.dataset_name == "waymo_open":
    # We filter Waymo Open for empty segmentation masks.
    def filter_fn(features):
      unique_instances = tf.unique(
          tf.reshape(features[preprocessing.SEGMENTATIONS], (-1,)))[0]
      n_instances = tf.size(unique_instances, tf.int32)
      # n_instances == 1 means we only have the background.
      return 2 <= n_instances
  else:
    filter_fn = None

  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split_name,
      rng=None,
      preprocess_fn=eval_preprocess_fn,
      filter_fn=filter_fn,
      cache=False,
      batch_dims=batch_dims,
      num_epochs=1,
      shuffle=False,
      pad_up_to_batches=None)

  if config.data.dataset_name == "waymo_open":
    # We filter Waymo Open for empty segmentation masks after preprocessing.
    # For the full dataset, we know how many we will end up with.
    eval_batch_size = batch_dims[0] * batch_dims[1]
    # We don't pad the last batch => floor.
    eval_num_batches = int(
        jnp.floor(1872 / eval_batch_size / jax.host_count()))
    eval_ds = eval_ds.apply(
        tf.data.experimental.assert_cardinality(
            eval_num_batches))

  return train_ds, eval_ds
