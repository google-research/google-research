# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Input pipeline for pat5hwax."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dataclasses
import jax
import ml_collections
import numpy as np
import t5
from t5x import train_lib
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


@dataclasses.dataclass(frozen=True)
class EvalCache:
  """Cache for all data used during evaluation."""
  # List of T5 tasks to run evaluation on.
  tasks: Sequence[t5.data.Task]
  # (unpreprocessed) examples per eval task.
  examples: Dict[str, Any]
  # (unpreprocessd) plaintext input per eval task.
  inputs: Dict[str, Any]
  # (unpreprocessd) plaintext targets per eval task.
  targets: Dict[str, Any]
  # List of preprocessed examples per eval task.
  preprocessed_examples: Mapping[str, Sequence[Any]]
  # The batch sizes for all hosts per eval task.
  preprocessed_batch_sizes: Mapping[str, np.ndarray]


# -----------------------------------------------------------------------------
# Dynamic to static shape transforms.
# -----------------------------------------------------------------------------
def bin_and_batch(dataset,
                  n_devices,
                  batch_size = 256,
                  bucket_length = 32,
                  buckets = None,
                  drop_remainder = True):
  """Dynamic batching by length-bucketing.

  Sorts data into a small number of batch x length "buckets" that have roughly
  constant token count.

  Args:
    dataset: tf.data dataset
    n_devices: int: number of local devices
    batch_size: int: target batch size
    bucket_length: int: target length for target batch size
    buckets: List[Tuple[int, int]]: pairs of bucket-length, batch-size
      boundaries to define custom length-buckets.
    drop_remainder: bool: whether or not to drop the last odd-shaped batch
      produced by bucketing a finite input data stream.

  Returns:
    tf.data dataset with dynamically batched examples.
  """
  # Create heuristic buckets is none are specified.
  if buckets is None:
    logging.info('Heuristically bucketing based on shapes of examples.')
    bucket_boundaries = [
        bucket_length // 4, bucket_length // 2, bucket_length,
        bucket_length * 2, bucket_length * 4, bucket_length * 8,
        bucket_length * 16
    ]
    bucket_batch_sizes = [
        batch_size * 4, batch_size * 2, batch_size, batch_size // 2,
        batch_size // 4, batch_size // 8, batch_size // 16
    ]
    # TF.data's bucket_by_sequence_length pads to (bucket_boundary - 1):
    # we add 1 here to pad to the correct specified length.
    bucket_boundaries = [b + 1 for b in bucket_boundaries]
    # Make batch sizes divisible by n_devices.
    bucket_batch_sizes = [
        max(b // n_devices, 1) * n_devices for b in bucket_batch_sizes
    ]
    buckets = (bucket_boundaries, bucket_batch_sizes)

  logging.info('Bucketing with buckets %s.', str(buckets))

  def example_length(example):
    """The length function used by bucket_by_sequence_length to bucket."""
    return tf.maximum(
        tf.shape(example['inputs'])[0],
        tf.shape(example['targets'])[0])

  boundaries, batch_sizes = buckets
  # bucket_by_sequence_length expects a final dummy 1 batch_size.
  batch_sizes.append(1)
  dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(
          example_length,
          boundaries,
          batch_sizes,
          pad_to_bucket_boundary=True,
          drop_remainder=drop_remainder))
  return dataset


def pack_dataset(dataset,
                 length,
                 keys = None):
  """Creates a 'packed' version of a dataset on-the-fly.

  Adapted from the mesh-tf implementation.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.
  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.
  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the inputs and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    length: an integer, or a dict from feature-key to integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
  if keys is None:
    keys = list(shapes.keys())
  for k in keys:
    if k not in shapes:
      raise ValueError('Key %s not found in dataset.  Available keys are %s' %
                       (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError('Tensors to be packed must be one-dimensional.')
  # make sure that the length dictionary contains all keys as well as the
  # keys suffixed by "_segmentation" and "_position"
  length_dict = {}
  for k in keys:
    for suffix in ['', '_segmentation', '_position']:
      length_dict[k + suffix] = (
          length[k] if isinstance(length, dict) else length)
  length = length_dict

  # trim to length
  dataset = dataset.map(
      lambda x: {k: x[k][:length[k]] for k in keys},
      num_parallel_calls=AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  dataset = _pack_with_tf_ops(dataset, keys, length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [length[k]]) for k, v in x.items()}

  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset, keys,
                      length):
  """Helper-function for packing a dataset which has already been batched.

  Helper for pack_dataset()  Uses tf.while_loop.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    length: an dict from feature-key to integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int32)
    empty_example[k + '_position'] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(
      partial, outputs
  ):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, length[k] - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.
    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[length[k]])
      outputs[k + '_position'] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[length[k]])

    def cond_fn(i, partial,
                outputs):
      del partial, outputs
      return i < dynamic_batch_size

    def body_fn(
        i, partial, outputs
    ):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray

      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), length[k]))

      def false_fn():
        return write_packed_example(partial, outputs)

      def true_fn():
        return partial, outputs

      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + '_position'] = tf.concat(
            [partial[k + '_position'],
             tf.range(new_seq_len, dtype=tf.int32)], 0)
      partial = new_partial
      return i + 1, partial, outputs

    i, partial, outputs = \
        tf.while_loop(
            cond_fn, body_fn, (i, partial, outputs),
            shape_invariants=(
                tf.TensorShape([]),
                {k: tf.TensorShape([None]) for k in keys_etc},
                {k: tf.TensorShape(None) for k in keys_etc},
            )
        )
    partial, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + '_segmentation'] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + '_position'], 0), tf.int32), axis=1) *
          tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed

  dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
  return dataset.unbatch()


# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------
def preprocess_t5_data(dataset,
                       training,
                       n_devices,
                       dynamic_batching = False,
                       pack_examples = True,
                       shuffle_buffer_size = 1024,
                       max_length = 512,
                       batch_size = 256,
                       bucket_length = 32,
                       drop_remainder = True,
                       prefetch_size = AUTOTUNE,
                       cache = False,
                       shuffle_seed = None):
  """Shuffle and batch/pack the given dataset."""
  keys = ['inputs', 'targets']
  dataset = dataset.map(lambda x: {k: tf.cast(x[k], tf.int32) for k in keys})
  if isinstance(max_length, int):
    max_length = {'inputs': max_length, 'targets': max_length}

  if training:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    dataset = dataset.repeat()

  if pack_examples and dynamic_batching:
    raise ValueError(
        "Can't use both dynamic batching and packed-examples simultaneously.")

  if pack_examples:
    dataset = pack_dataset(dataset, max_length)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  elif dynamic_batching:
    dataset = bin_and_batch(
        dataset,
        n_devices,
        batch_size=batch_size,
        bucket_length=bucket_length,
        drop_remainder=drop_remainder)
  else:  # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=max_length,
        padding_values={
            'inputs': np.int32(0),
            'targets': np.int32(0)
        },
        drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  if cache:
    dataset = dataset.cache()

  return dataset


def get_vocabulary(
    mixture_or_task_name):
  """Get the appropriate value for the utils.run.vocabulary argument.

  Args:
    mixture_or_task_name: string, an identifier for a Mixture or Task in the
      appropriate registry. Must be specified via gin.

  Returns:
    Either a single t5.data.vocabularies.Vocabulary or a tuple of
    t5.data.vocabularies.Vocabulary for inputs and targets.
  """
  provider = t5.data.get_mixture_or_task(mixture_or_task_name)
  features = provider.output_features
  feature_values = list(features.values())
  vocabulary = feature_values[0].vocabulary
  for feature in feature_values[1:]:
    if feature.vocabulary != vocabulary:
      raise ValueError('No feature_name was provided to get_vocabulary, but '
                       'output_features have different vocabularies.')
  return vocabulary


def get_datasets_and_cache(
    config, num_shards, shard_id,
    per_shard_host_id
):
  """Get train and eval datasets and some data used for evaluation.

  Args:
    config: The config dict to use.
    num_shards: How many shards to divide the datasets into.
    shard_id: Which shard to use on this host.
    per_shard_host_id: The index of this host among all hosts using that shard.

  Returns:
    ((train_ds, eval_ds), eval_cache).
  """
  num_replicas = jax.device_count() // config.num_partitions

  logging.info('Initializing dataset.')

  max_length = {
      'inputs': config.max_input_length,
      'targets': config.max_target_length
  }

  mixture_or_task = t5.data.get_mixture_or_task(config.mixture_or_task_name)

  train_data = mixture_or_task.get_dataset(
      max_length, split='train', shuffle=True,
      seed=0,
      use_cached=config.train_use_cached).shard(num_shards, shard_id)
  train_eval_data = mixture_or_task.get_dataset(
      max_length, split=config.eval_split,
      shuffle=False,
      use_cached=config.train_use_cached).shard(num_shards, shard_id)
  train_ds = preprocess_t5_data(
      train_data,
      n_devices=num_replicas // num_shards,
      batch_size=config.batch_size // num_shards,
      training=True,
      pack_examples=True,
      max_length=max_length,
      shuffle_seed=0)
  train_eval_ds = preprocess_t5_data(
      train_eval_data,
      n_devices=num_replicas // num_shards,
      batch_size=config.batch_size // num_shards,
      training=False,
      pack_examples=False,
      max_length=max_length,
  )

  # Set up per-task evaluation datasets and post-processed targets data
  # for metrics.
  max_eval_length = {
      'inputs': config.max_eval_input_length,
      'targets': config.max_eval_target_length
  }

  eval_mixture_or_task = t5.data.get_mixture_or_task(
      config.eval_mixture_or_task_name)
  if isinstance(eval_mixture_or_task, t5.data.Mixture):
    eval_tasks = eval_mixture_or_task.tasks
  elif isinstance(eval_mixture_or_task, t5.data.Task):
    eval_tasks = [eval_mixture_or_task]
  for task in eval_tasks:
    if config.eval_split not in task.splits:
      logging.info('Task %s has no "%s" split; skipping eval.', task.name,
                   config.eval_split)
  eval_tasks = [
      task for task in eval_tasks
      if config.eval_split in task.splits and task.metric_fns
  ]

  cached_examples = {}
  cached_inputs = {}
  cached_targets = {}
  preprocessed_examples = {}
  preprocessed_batch_sizes = {}
  for task in eval_tasks:
    task_ds = task.get_dataset(
        max_eval_length,
        split=config.eval_split,
        shuffle=False,
        use_cached=config.eval_use_cached).cache()
    examples = list(task_ds.as_numpy_iterator())
    if not examples:
      raise ValueError(
          f"The '{config.eval_split}' split of {task.name} is empty.")
    cached_examples[task.name] = examples
    cached_inputs[task.name] = [
        tf.compat.as_text(ex['inputs_plaintext']) for ex in examples
    ]
    cached_targets[task.name] = [
        task.postprocess_fn(  # pylint:disable=g-complex-comprehension
            tf.compat.as_text(ex['targets_plaintext']),
            example=ex,
            is_target=True) for ex in examples
    ]
    # Prepare per-task processed dataset for model inference.
    sharded_task_ds = task_ds.shard(num_shards, shard_id)
    processed_task_ds = preprocess_t5_data(
        sharded_task_ds,
        n_devices=num_replicas // num_shards,
        batch_size=config.eval_batch_size // num_shards,
        training=False,
        pack_examples=False,
        max_length=max_eval_length,
        drop_remainder=False,
    )
    preprocessed_examples[task.name] = list(
        processed_task_ds.as_numpy_iterator())
    # Gather batch numbers for each shard from across hosts.
    batch_number = len(preprocessed_examples[task.name])
    preprocessed_batch_sizes[task.name] = train_lib.host_allgather(
        np.array(batch_number, np.int32), num_shards, shard_id,
        per_shard_host_id == 0)
    logging.info(
        'Task %s with %d total examples at per-shard batch %d: '
        '%d batches in this shard, %s batches gathered from all shards',
        task.name, len(cached_examples[task.name]),
        config.eval_batch_size // num_shards, batch_number,
        preprocessed_batch_sizes[task.name])

  eval_cache = EvalCache(
      tasks=eval_tasks,
      examples=cached_examples,
      inputs=cached_inputs,
      targets=cached_targets,
      preprocessed_examples=preprocessed_examples,
      preprocessed_batch_sizes=preprocessed_batch_sizes)

  return (train_ds, train_eval_ds), eval_cache
