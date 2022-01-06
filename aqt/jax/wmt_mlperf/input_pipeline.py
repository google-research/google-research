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

"""Input pipeline for WMT datasets."""

import functools
import os
import tempfile
import time
from typing import Optional, List
from absl import logging

import jax

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as tftxt

from sentencepiece import SentencePieceTrainer

AUTOTUNE = tf.data.experimental.AUTOTUNE


# -----------------------------------------------------------------------------
# Raw TFDS dataset.
# -----------------------------------------------------------------------------
def raw_wmt_datasets(dataset_name='wmt17_translate/de-en',
                     evaluation_datasets = None,
                     reverse_translation=True,
                     shard_idx=0,
                     shard_count=1,
                     data_dir=None):
  """Load raw WMT datasets and normalize feature keys.

  Args:
    dataset_name: str: TFDS WMT dataset name.
    evaluation_datasets: Optional[List[[str]]: separate dataset names list for
      evaluation. e.g. for specifying the standard academic WMT14 test set.
    reverse_translation: bool: whether to reverse the translation direction.
      e.g. for 'de-en' this translates from english to german.
    shard_idx: int: for multihost training, index of this host.
    shard_count: int: for mulithost training, number of total hosts.
    data_dir: str: location of TFDS data directory.

  Returns:
    training tf.dataset, evaluation tf.dataset, and training features_info
    source and target language features are mapped to 'inputs' and 'targets'
    keys.
  """
  builder = tfds.builder(dataset_name, data_dir=data_dir)
  shard_spec = (f'[{int(100 * shard_idx / shard_count)}%'
                f':{int(100 * (shard_idx + 1) / shard_count)}%]')
  logging.info('Training on TFDS dataset %s with split %s', dataset_name,
               'train' + shard_spec)
  train_data = builder.as_dataset(
      split='train' + shard_spec, shuffle_files=True)

  eval_data = {}
  if evaluation_datasets is None:
    logging.info('Evaluating on TFDS dataset %s with split %s', dataset_name,
                 'validation' + shard_spec)
    # Using validation split of training dataset.
    dataset = builder.as_dataset(
        split='validation' + shard_spec, shuffle_files=False)
    eval_data[dataset_name + ':validation'] = dataset
  else:
    for name in evaluation_datasets:
      eval_dataset_name, *eval_split = name.split(':')
      if not eval_split:
        eval_split = 'validation'
        dataset_name = name + ':validation'
      else:
        eval_split = eval_split[0]
        dataset_name = name
      logging.info('Evaluating on TFDS dataset %s with split %s',
                   eval_dataset_name, eval_split + shard_spec)
      eval_builder = tfds.builder(eval_dataset_name, data_dir=data_dir)
      dataset = eval_builder.as_dataset(
          split=eval_split + shard_spec, shuffle_files=False)
      eval_data[dataset_name] = dataset

  # Train and eval set should have the same features
  features_info = builder.info

  # standardize on 'inputs' and 'targets' features.
  input_lang = features_info.supervised_keys[0]
  target_lang = features_info.supervised_keys[1]
  if reverse_translation:
    input_lang, target_lang = target_lang, input_lang

  def to_features_dict(x):
    return {'inputs': x[input_lang], 'targets': x[target_lang]}

  train_data = train_data.map(to_features_dict, num_parallel_calls=AUTOTUNE)
  standardized_eval_data = {
      name: dataset.map(to_features_dict, num_parallel_calls=AUTOTUNE)
      for name, dataset in eval_data.items()
  }

  return train_data, standardized_eval_data, features_info


# -----------------------------------------------------------------------------
# Tokenization.
# -----------------------------------------------------------------------------
def dump_chars_to_textfile(dataset,
                           maxchars=1e7,
                           data_keys=('inputs', 'targets')):
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: tf.dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  ds_iter = dataset.as_numpy_iterator()
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/ds_chars') as outfp:
    while char_count < maxchars:
      example = next(ds_iter)
      for k in data_keys:
        line = example[k] + b'\n'
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def train_sentencepiece(dataset,
                        vocab_size,
                        maxchars=1e7,
                        character_coverage=1.0,
                        model_path='wmt_model.model',
                        model_type='unigram',
                        data_keys=('inputs', 'targets')):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    dataset: tf.dataset
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    character_coverage: amount of characters covered by the model, good
      defaults are 0.9995 for languages with rich character set like Japanese
      or Chinese and 1.0 for other languages with small character set.
    model_path: str: path of model file to save vocab model to.
    model_type: str: type of sentencepiece vocab to train.
    data_keys: Tuple[str]: keys of dataset to use for training.

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  fname, _ = dump_chars_to_textfile(
      dataset, maxchars=maxchars, data_keys=data_keys)
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/sp_tmp') as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = ' '.join([
      f'--input={fname}', f'--vocab_size={vocab_size}',
      f'--character_coverage={character_coverage}',
      f'--model_prefix={model_fp.name}', f'--model_type={model_type}'
  ])
  SentencePieceTrainer.Train(argstr)
  # Only write to CNS if host id is 0 to prevent race conditions during
  # multihost training, otherwise wait until host 0 has written the file.
  if jax.host_id() == 0:
    # Use an intermediate filename that is renamed to the target name to address
    # create and fill delays.  Using finalization (CNS) as a indicator is not
    # portable.
    copy_rename_path = abs_model_path + '.rntmp'
    tf.io.gfile.copy(model_fp.name + '.model', copy_rename_path, overwrite=True)
    tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)
    logging.info('copied %s to %s', model_fp.name + '.model', abs_model_path)
  else:
    while not tf.io.gfile.exists(abs_model_path):
      time.sleep(1)
    time.sleep(1)
  return abs_model_path


def load_sentencepiece_tokenizer(model_path,
                                 add_bos=False,
                                 add_eos=True,
                                 reverse=False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with tf.io.gfile.GFile(model_path, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  logging.info('Loaded SentencePiece Tokenizer from file.')
  return sp_tokenizer


# -----------------------------------------------------------------------------
# Dynamic to static shape transforms.
# -----------------------------------------------------------------------------
def bin_and_batch(dataset,
                  n_devices,
                  batch_size=256,
                  bucket_length=32,
                  buckets=None,
                  drop_remainder=True):
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


def pack_dataset(dataset, length, keys=None):
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
      length_dict[k + suffix] = length if isinstance(length, int) else length[k]
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


def _pack_with_tf_ops(dataset, keys, length):
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

  def write_packed_example(partial, outputs):
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

    def cond_fn(i, partial, outputs):
      del partial, outputs
      return i < dynamic_batch_size

    def body_fn(i, partial, outputs):
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

    i, partial, outputs = tf.while_loop(
        cond_fn,
        body_fn, (i, partial, outputs),
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
        ))
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
def preprocess_wmt_data(dataset,
                        training,
                        n_devices,
                        dynamic_batching=False,
                        pack_examples=True,
                        shuffle_buffer_size=1024,
                        max_length=512,
                        batch_size=256,
                        bucket_length=32,
                        drop_remainder=True,
                        prefetch_size=AUTOTUNE,
                        seed = None):
  """Shuffle and batch/pack the given dataset."""

  def length_filter(max_len):

    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)

    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if training:
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
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
        padded_shapes={
            'inputs': max_length,
            'targets': max_length
        },
        padding_values={
            'inputs': 0,
            'targets': 0
        },
        drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_wmt_datasets(
    n_devices,
    dataset_name='wmt17_translate/de-en',
    eval_dataset_list=None,
    reverse_translation=True,
    shard_idx=0,
    shard_count=1,
    data_dir=None,
    vocab_path=None,
    target_vocab_size=2**15,  # 32000
    max_corpus_chars=10**7,
    batch_size=256,
    bucket_length=32,
    dynamic_batching=False,
    pack_examples=True,
    max_length=256,
    max_eval_length=256,
    seed = None):
  """Load and return dataset of batched examples for use during training."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/wmt_sentencepiece_model')

  train_data, eval_data_dict, _ = raw_wmt_datasets(
      dataset_name=dataset_name,
      evaluation_datasets=eval_dataset_list,
      reverse_translation=reverse_translation,
      shard_idx=shard_idx,
      shard_count=shard_count,
      data_dir=data_dir)

  try:
    sp_tokenizer = load_sentencepiece_tokenizer(vocab_path, add_eos=True)
  except tf.errors.NotFoundError:
    logging.info('SentencePiece vocab not found, building one from data.')
    vocab_dir = os.path.split(vocab_path)[0]
    tf.io.gfile.makedirs(vocab_dir)
    abs_vocab_path = train_sentencepiece(
        train_data,
        target_vocab_size,
        maxchars=max_corpus_chars,
        character_coverage=1.0,
        model_path=vocab_path,
        data_keys=('inputs', 'targets'))
    sp_tokenizer = load_sentencepiece_tokenizer(abs_vocab_path, add_eos=True)

  # Encode strings with sentencepiece tokenizer.
  def tokenize(data):
    return {
        'inputs': sp_tokenizer.tokenize(data['inputs']),
        'targets': sp_tokenizer.tokenize(data['targets'])
    }

  train_data = train_data.map(tokenize, num_parallel_calls=AUTOTUNE)
  eval_data_dict = {
      eval_dataset_name: eval_data.map(tokenize, num_parallel_calls=AUTOTUNE)
      for eval_dataset_name, eval_data in eval_data_dict.items()
  }

  train_batches = preprocess_wmt_data(
      train_data,
      training=True,
      dynamic_batching=dynamic_batching,
      pack_examples=pack_examples,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_length,
      seed=seed)

  preprocess_eval_data = functools.partial(
      preprocess_wmt_data,
      training=False,
      dynamic_batching=dynamic_batching,
      pack_examples=False,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_eval_length,
      seed=seed)

  eval_batches_dict = {
      eval_dataset_name: preprocess_eval_data(eval_data)
      for eval_dataset_name, eval_data in eval_data_dict.items()
  }

  # TODO(b/155933172): Would be better to refactor raw_wmt_datasets to get a
  # fixed train-eval split of training data. Currently it is changing on every
  # step.
  train_eval_batches = preprocess_wmt_data(
      train_data,
      training=False,
      dynamic_batching=dynamic_batching,
      pack_examples=False,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_length,
      seed=seed)

  preprocess_predict_data = functools.partial(
      preprocess_wmt_data,
      training=False,
      dynamic_batching=dynamic_batching,
      pack_examples=False,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_eval_length,
      drop_remainder=False,
      seed=seed)

  predict_batches_dict = {
      eval_dataset_name: preprocess_predict_data(eval_data)
      for eval_dataset_name, eval_data in eval_data_dict.items()
  }

  return train_batches, eval_batches_dict, train_eval_batches, predict_batches_dict, sp_tokenizer
