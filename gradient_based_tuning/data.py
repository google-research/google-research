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
"""GEC datasets for training in Jax."""

import functools

import jax
import tensorflow.compat.v2 as tf
# pylint: disable=g-import-not-at-top
try:
  import input_pipeline
except ImportError:
  from gradient_based_tuning import input_pipeline
# pylint: enable=g-import-not-at-top

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_unique_examples(batch):
  """Count the unique (unpacked) examples in a maybe-batched example-dict."""
  packed = bool("targets_segmentation" in batch)
  batched = bool(len(batch["targets"].shape) > 1)
  if packed and batched:
    return sum(int(max(x)) for x in batch["targets_segmentation"])
  elif packed and not batched:
    return int(max(batch["targets_segmentation"]))
  elif batched and not packed:
    return int(batch["targets"].shape[0])
  else:  # not batched and not packed
    return 1


def cast_dataset_types(ds, cast_dict):
  """Casts the tf.data.Dataset types to those specified in the cast_dict."""

  def cast_fn(x):
    return {
        k: tf.cast(v, cast_dict[k]) if k in cast_dict else v
        for k, v in x.items()
    }

  ds = ds.map(cast_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


def get_prepacked_examples(file_pattern,
                           batch_size,
                           max_length,
                           repeat,
                           random_seed=1,
                           compression_type="",
                           deterministic=False,
                           drop_remainder=True,
                           vocab_path=None,
                           pack=None,
                           min_cutoff=None,
                           activated_vars=None,
                           guided_vars_dict=None,
                           shard_data=True,
                           additional_fields=None):
  """Constructs a dataset from a precomputed tfrecord of packed tfexamples."""
  del vocab_path, pack, guided_vars_dict, min_cutoff, activated_vars
  name_to_features = {
      "inputs": tf.io.FixedLenFeature([max_length], tf.int64),
      "inputs_segmentation": tf.io.FixedLenFeature([max_length], tf.int64),
      "inputs_position": tf.io.FixedLenFeature([max_length], tf.int64),
      "targets": tf.io.FixedLenFeature([max_length], tf.int64),
      "targets_segmentation": tf.io.FixedLenFeature([max_length], tf.int64),
      "targets_position": tf.io.FixedLenFeature([max_length], tf.int64),
  }
  input_types = {
      "inputs": tf.uint16,
      "targets": tf.uint16,
      "inputs_position": tf.uint8,
      "inputs_segmentation": tf.uint8,
      "targets_position": tf.uint8,
      "targets_segmentation": tf.uint8,
  }
  if additional_fields:
    for x in additional_fields:
      name_to_features.update(
          {x: tf.io.FixedLenFeature([max_length], tf.int64)})

  def _decode_record(record):
    """Decodes a record to a TensorFlow example, converting int64 to int32."""
    example = tf.io.parse_single_example(record, name_to_features)
    return example

  def cast_fn(x):
    return {
        k: tf.cast(v, input_types[k]) if k in input_types else v
        for k, v in x.items()
    }

  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=not deterministic)

  # This only reads the dataset files meant for this host.
  if shard_data:
    dataset = dataset.shard(jax.host_count(), jax.process_index())

  # Up to 10 files are read in parallel.
  dataset = dataset.interleave(
      map_func=lambda f: tf.data.TFRecordDataset(f, compression_type),
      cycle_length=10,
      block_length=1,
      num_parallel_calls=AUTOTUNE,
      deterministic=deterministic).prefetch(batch_size)
  dataset = dataset.map(_decode_record, num_parallel_calls=AUTOTUNE)
  dataset = dataset.map(cast_fn, num_parallel_calls=AUTOTUNE)

  # Finally, we shuffle, repeat and batch.
  if not deterministic:
    dataset = dataset.shuffle(
        buffer_size=16384 * 8, reshuffle_each_iteration=True, seed=random_seed)
  if repeat:
    dataset = dataset.repeat()
  if batch_size > 1:
    dataset = dataset.batch(
        batch_size, drop_remainder=drop_remainder, deterministic=deterministic)
  return dataset.prefetch(AUTOTUNE)


def pack_and_batch_ds(dataset,
                      batch_size,
                      max_length,
                      extend_to_fill=False,
                      drop_remainder=False,
                      pack=True):
  """Packs and batches the provided dataset.

  Args:
    dataset: the dataset to be packed / batched
    batch_size: how many (potentially packed) examples per batch
    max_length: examples will be truncated or padded to all fit max_length
    extend_to_fill: if the final batch is not full (has size batch_size), then
      repeat the dataset to fill it (NOTE: this is slow, do not use this with
      large datasets)
    drop_remainder: if the final batch is not full, drop it
    pack: if True, pack the dataset with input_pipeline.pack_dataset.

  Returns:
    a tf.data.Dataset
  """
  if pack:
    # Minimally memory-consuming types for each field.
    input_types = {
        "inputs": tf.uint16,
        "targets": tf.uint16,
        "inputs_position": tf.uint8,
        "inputs_segmentation": tf.uint8,
        "targets_position": tf.uint8,
        "targets_segmentation": tf.uint8,
    }
    dataset = input_pipeline.pack_dataset(dataset, max_length)
    dataset = cast_dataset_types(dataset, input_types)

  if batch_size > 1:
    # instead of dropping the last batch if it isn't full, extend it to fullness
    # by repeating the dataset
    if extend_to_fill:
      # count how many (not all full) batches there are
      total_batches = sum(
          1 for _ in iter(dataset.batch(batch_size, drop_remainder=False)))
      # repeat the dataset infinitely and truncate to total_batches
      dataset = dataset.repeat().batch(
          batch_size, drop_remainder=False).take(total_batches)
    else:
      dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  return dataset


def get_ds_final_batch_size(ds):
  """Count total batches and get size of last batch from a tf.data.Dataset."""
  num_batches = 0
  batch = None
  for batch in iter(ds):
    num_batches += 1
  if batch is not None:
    batch_size = int(batch["targets"].shape[0])
  else:
    batch_size = 0
  return num_batches, batch_size


def get_ds_metrics(ds):
  """Get num_examples, num_batches, final_batch_size, max_len."""
  next_batch = next(iter(ds))
  max_len = next_batch["targets"].shape[-1]
  packed = bool("targets_segmentation" in next_batch)
  batched = bool(len(next_batch["targets"].shape) > 1)

  if batched:
    batch_size = int(next_batch["targets"].shape[0])

  if packed and batched:
    # count total examples and total batches simultaneously, to avoid cost of
    # iterating through the tf.data.Dataset twice
    # pylint:disable=g-long-lambda
    num_examples, num_batches = functools.reduce(
        lambda num_unq_ex, packed_count_ex: (num_unq_ex[0] + packed_count_ex[
            0], num_unq_ex[1] + packed_count_ex[1]),
        ((get_unique_examples(b), 1) for b in iter(ds)))
    # pylint:enable=g-long-lambda

  elif not packed and not batched:
    num_batches = sum(1 for _ in iter(ds))
    num_examples = num_batches
    batch_size = 1
  else:
    raise NotImplementedError("batched XOR packed is not currently supported")
  return num_examples, num_batches, batch_size, max_len
