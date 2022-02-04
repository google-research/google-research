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

"""Input pipeline for a WMT dataset.

This file was branched from flax/examples/wmt/input_pipeline.py.
"""

import collections
import csv
import os
import random
from typing import Dict, List, Optional, Union

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow.compat.v2 as tf

from data_selection.wmt import dataset_utils
from data_selection.wmt import tokenizer

AUTOTUNE = tf.data.AUTOTUNE
Features = Dict[str, tf.Tensor]


# -----------------------------------------------------------------------------
# Raw TFDS dataset.
# -----------------------------------------------------------------------------
def raw_wmt_datasets(dataset_name='wmt17_translate/de-en',
                     eval_dataset_name=None,
                     reverse_translation=False,
                     shard_idx=0,
                     shard_count=1,
                     data_dir=None,
                     paracrawl_size=0,
                     shuffle_train_files=True,
                     pseudo_path=None,
                     newscommentary_size=None,
                     newscomment_sample_ratio=1.0):
  """Load raw WMT datasets and normalize feature keys.

  Args:
    dataset_name: str: TFDS WMT dataset name.
    eval_dataset_name: Optional[str]: separate dataset name for evaluation.
      e.g. for specifying the standard academic WMT14 test set.
    reverse_translation: bool: whether to reverse the translation direction.
      e.g. for 'de-en' this translates from english to german.
    shard_idx: int: for multihost training, index of this host.
    shard_count: int: for mulithost training, number of total hosts.
    data_dir: str: location of TFDS data directory.
    paracrawl_size: if paracrawl is used, we will sample this many examples.
    shuffle_train_files: whether to shuffle the input data files
    pseudo_path: path to pseudo references
    newscommentary_size: Size of news commentary ft set
    newscomment_sample_ratio: how much to downsample newscommentary data

  Returns:
    training tf.dataset, evaluation tf.dataset, and training features_info
    source and target language features are mapped to 'inputs' and 'targets'
    keys.
  """
  wmt_dataset_builder = dataset_utils.WmtDatasetBuilder(shard_idx,
                                                        shard_count,
                                                        data_dir,
                                                        shuffle_train_files,
                                                        pseudo_path)
  train_data, eval_data = wmt_dataset_builder.build_train_and_eval_datasets(
      dataset_name, eval_dataset_name, paracrawl_size, newscommentary_size,
      newscomment_sample_ratio)

  builder = wmt_dataset_builder.retrieve_builder()

  if builder is not None:
    features_info = builder.info

    # standardize on 'inputs' and 'targets' features.
    input_lang = features_info.supervised_keys[0]
    target_lang = features_info.supervised_keys[1]
    if reverse_translation:
      input_lang, target_lang = target_lang, input_lang
    def to_features_dict(x):
      return {'inputs': x[input_lang], 'targets': x[target_lang]}
    if 'pseudo' not in dataset_name:  # Perhaps remove this code path.
      train_data = train_data.map(to_features_dict, num_parallel_calls=AUTOTUNE)
    eval_data = eval_data.map(to_features_dict, num_parallel_calls=AUTOTUNE)
  else:
    features_info = None

  return train_data, eval_data, features_info


def pack_dataset(dataset,
                 key2length,
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
    key2length: an integer, or a dict from feature-key to integer
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
  if isinstance(key2length, int):
    key2length = {k: key2length for k in keys}
  for k in keys:
    for suffix in ['_segmentation', '_position']:
      key2length[k + suffix] = key2length[k]

  # trim to length
  dataset = dataset.map(
      lambda x: {k: x[k][:key2length[k]] for k in keys},
      num_parallel_calls=AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(key2length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  dataset = _pack_with_tf_ops(dataset, keys, key2length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset, keys,
                      key2length):
  """Helper-function for packing a dataset which has already been batched.

  Helper for pack_dataset()  Uses tf.while_loop.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    key2length: an dict from feature-key to integer

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
          tf.pad(partial[k], [[0, key2length[k] - tf.size(partial[k])]]))
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
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
      outputs[k + '_position'] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])

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
                tf.size(partial[k]) + tf.size(one_example[k]), key2length[k]))

      def false_fn():
        return write_packed_example(partial, outputs)

      def true_fn():
        return partial, outputs

      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:key2length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + '_position'] = tf.concat(
            [partial[k + '_position'],
             tf.range(new_seq_len)], 0)
      partial = new_partial
      return i + 1, partial, outputs

    # For loop over all examples in the batch.
    i, partial, outputs = tf.while_loop(
        cond=lambda *_: True,
        body=body_fn,
        loop_vars=(i, partial, outputs),
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
        ),
        maximum_iterations=dynamic_batch_size)
    _, outputs = write_packed_example(partial, outputs)
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
                        shuffle,
                        num_epochs = 1,
                        pack_examples = True,
                        shuffle_buffer_size = 1024000,
                        max_length = 512,
                        batch_size = 256,
                        drop_remainder = True,
                        prefetch_size = AUTOTUNE,
                        is_scores_path=None,
                        num_to_keep=0,
                        truncate=False,
                        sample_size=-1):
  """Shuffle and batch/pack the given dataset."""

  def length_filter(max_len):

    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)

    return filter_fn

  if truncate:
    dataset = dataset.map(
        lambda x: {k: v[:max_length] for k, v in x.items()},
        num_parallel_calls=AUTOTUNE)
  elif max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if is_scores_path is not None:
    logging.info('Doing data selection!')
    logging.info('Num to keep = %d', num_to_keep)
    dataset = data_selection(dataset, is_scores_path, num_to_keep)

  if sample_size > 0:
    logging.info('Downsampling: %d', sample_size)
    shuff_buff = 200000  # num_to_keep if num_to_keep > 0 else 200000
    dataset = dataset.shuffle(shuff_buff).take(sample_size)

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat(num_epochs)

  if pack_examples:
    dataset = pack_dataset(dataset, max_length)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
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


def data_selection(train_data, is_scores_path, num_to_keep=-1):
  """Select data based on intelligent selection scores."""
  if num_to_keep < 0:
    return train_data

  scores = []
  with tf.io.gfile.GFile(is_scores_path, 'r') as f:
    reader = csv.reader(f)
    for val in reader:
      scores.extend(val)
  scores = [float(s) for s in scores]

  lengths = []
  with tf.io.gfile.GFile(is_scores_path.replace('.csv', '_length.csv'),
                         'r') as f:
    reader = csv.reader(f)
    for val in reader:
      lengths.extend(val)
  lengths = [int(s) for s in lengths]

  if num_to_keep >= len(scores):
    return train_data

  threshold = np.sort(scores)[num_to_keep]

  tf_is_scores = tf.data.Dataset.from_tensor_slices(scores)
  tf_lengths = tf.data.Dataset.from_tensor_slices(lengths)

  scored_data = tf.data.Dataset.zip((tf_is_scores, tf_lengths, train_data))
  def filter_fn(score, _, __):  #  # pylint: disable=invalid-name
    return tf.math.less_equal(score, threshold)

  def remove_enum(_, length, el):
    targ_size = tf.math.count_nonzero(el['targets'], dtype=tf.dtypes.int32)
    assert_op = tf.debugging.assert_equal(
        length, targ_size, message='Lengths not alligned')
    with tf.control_dependencies([assert_op]):
      return el

  train_data = scored_data.filter(filter_fn).map(remove_enum)
  train_data = train_data.cache()
  return train_data


def get_wmt_datasets(dataset_name='wmt17_translate/de-en',
                     eval_dataset_name=None,
                     reverse_translation=True,
                     shard_idx=0,
                     shard_count=1,
                     data_dir=None,
                     vocab_path=None,
                     target_vocab_size=2**15,  # 32000
                     max_corpus_chars=10**7,
                     batch_size=256,
                     pack_examples=True,
                     max_length=256,
                     max_eval_length=256,
                     paracrawl_size=0,
                     is_scores_path=None,
                     num_to_keep=-1,
                     pseudo_path=None,
                     shuffle_repeat_train=True,
                     repeat_count=-1,
                     newscommentary_size=None,
                     split_tokenizer=False,
                     sample_size=-1,
                     newscomment_sample_ratio=1.0):
  """Load and return dataset of batched examples for use during training."""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/wmt_sentencepiece_model')

  train_data, eval_data, _ = raw_wmt_datasets(
      dataset_name=dataset_name,
      eval_dataset_name=eval_dataset_name,
      reverse_translation=reverse_translation,
      shard_idx=shard_idx,
      shard_count=shard_count,
      data_dir=data_dir,
      paracrawl_size=paracrawl_size,
      shuffle_train_files=(is_scores_path is None) and shuffle_repeat_train,
      pseudo_path=pseudo_path,
      newscommentary_size=newscommentary_size,
      newscomment_sample_ratio=newscomment_sample_ratio)
  # If is_score_path is None, there is no data selection so we can shuffle.
  # If it is not None, then we cannot shuffle the input files.

  # Tokenize data.
  if split_tokenizer:
    sp_tokenizer_input = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path + '_input',
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars,
        data_keys=('inputs',))
    sp_tokenizer_target = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path + '_target',
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars,
        data_keys=('targets',))
    train_data = train_data.map(
        tokenizer.DoubleTokenizeOp(sp_tokenizer_input=sp_tokenizer_input,
                                   sp_tokenizer_target=sp_tokenizer_target),
        num_parallel_calls=AUTOTUNE)
    eval_data = eval_data.map(
        tokenizer.DoubleTokenizeOp(sp_tokenizer_input=sp_tokenizer_input,
                                   sp_tokenizer_target=sp_tokenizer_target),
        num_parallel_calls=AUTOTUNE)
    sp_tokenizer = sp_tokenizer_target
  else:
    sp_tokenizer = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path,
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars)

    # Currently the pseudorefs are stored in pickle files and are pre-tokenized
    # so we would not tokenize them here. Instead we should write the
    # pseudo references to a tfrecord in the future.
    if 'pseudo' not in dataset_name:
      train_data = train_data.map(
          tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
    eval_data = eval_data.map(
        tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  train_ds = preprocess_wmt_data(
      train_data,
      shuffle=shuffle_repeat_train,
      num_epochs=repeat_count,
      pack_examples=pack_examples,
      batch_size=batch_size,
      max_length=max_length,
      is_scores_path=is_scores_path,
      num_to_keep=num_to_keep,
      sample_size=sample_size)

  eval_ds = preprocess_wmt_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=batch_size,
      max_length=max_eval_length)

  predict_ds = preprocess_wmt_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=batch_size,
      max_length=max_eval_length,
      drop_remainder=False)

  return train_ds, eval_ds, predict_ds, sp_tokenizer


def get_wmt_is_datasets(n_devices,
                        dataset_name='wmt17_translate/de-en',
                        reverse_translation=True,
                        shard_idx=0,
                        shard_count=1,
                        data_dir=None,
                        vocab_path=None,
                        target_vocab_size=2**15,  # 32000
                        max_corpus_chars=10**7,
                        batch_size=256,
                        max_length=256,
                        paracrawl_size=0,
                        split_tokenizer=False,
                        use_eval_data=False,
                        truncate=False):
  """Load and return dataset of batched examples for use during training."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/wmt_sentencepiece_model')

  train_data, eval_data, _ = raw_wmt_datasets(
      dataset_name=dataset_name,
      eval_dataset_name=None,
      reverse_translation=reverse_translation,
      shard_idx=shard_idx,
      shard_count=shard_count,
      data_dir=data_dir,
      paracrawl_size=paracrawl_size,
      shuffle_train_files=False)

  if use_eval_data:
    # Unfortunate use of names but easiest for refactor w/o errors.
    train_data = eval_data

  # Tokenize data.
  if split_tokenizer:
    sp_tokenizer_input = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path + '_input',
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars,
        data_keys=('inputs',))
    sp_tokenizer_target = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path + '_target',
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars,
        data_keys=('targets',))
    train_data = train_data.map(
        tokenizer.DoubleTokenizeOp(sp_tokenizer_input=sp_tokenizer_input,
                                   sp_tokenizer_target=sp_tokenizer_target),
        num_parallel_calls=AUTOTUNE)
    sp_tokenizer = sp_tokenizer_target
  else:
    sp_tokenizer = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path,
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars)

    # Encode strings with sentencepiece tokenizer.
    train_data = train_data.map(
        tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  train_batches = preprocess_wmt_data(
      train_data,
      shuffle=False,
      num_epochs=1,
      pack_examples=False,
      batch_size=batch_size,
      max_length=max_length,
      drop_remainder=False,
      truncate=truncate)
  # Note: we drop remainder which will truncate the training data but the
  # effect is 0.017% of the dataset so shouldn't effect model

  if split_tokenizer:
    return train_batches, (sp_tokenizer_input, sp_tokenizer_target)
  return train_batches, (sp_tokenizer, sp_tokenizer)


def get_dynamic_datasets(dataset_name='wmt17_translate/de-en',
                         eval_dataset_name=None,
                         reverse_translation=True,
                         shard_idx=0,
                         shard_count=1,
                         data_dir=None,
                         vocab_path=None,
                         target_vocab_size=2**15,  # 32000
                         max_corpus_chars=10**7,
                         batch_size=256,
                         max_length=256,
                         max_eval_length=256,
                         paracrawl_size=0,
                         is_scores_path=None,
                         num_buckets=100,
                         split_tokenizer=False):
  """Load and return dataset of batched examples for use during training."""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/wmt_sentencepiece_model')

  train_data, eval_data, _ = raw_wmt_datasets(
      dataset_name=dataset_name,
      eval_dataset_name=eval_dataset_name,
      reverse_translation=reverse_translation,
      shard_idx=shard_idx,
      shard_count=shard_count,
      data_dir=data_dir,
      paracrawl_size=paracrawl_size,
      shuffle_train_files=False)

  if split_tokenizer:
    sp_tokenizer_input = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path + '_input',
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars,
        data_keys=('inputs',))
    sp_tokenizer_target = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path + '_target',
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars,
        data_keys=('targets',))
    train_data = train_data.map(
        tokenizer.DoubleTokenizeOp(sp_tokenizer_input=sp_tokenizer_input,
                                   sp_tokenizer_target=sp_tokenizer_target),
        num_parallel_calls=AUTOTUNE)
    eval_data = eval_data.map(
        tokenizer.DoubleTokenizeOp(sp_tokenizer_input=sp_tokenizer_input,
                                   sp_tokenizer_target=sp_tokenizer_target),
        num_parallel_calls=AUTOTUNE)
    sp_tokenizer = sp_tokenizer_target
  else:
    sp_tokenizer = tokenizer.load_or_train_tokenizer(
        train_data,
        vocab_path=vocab_path,
        vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars)
    train_data = train_data.map(
        tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
    eval_data = eval_data.map(
        tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  train_data_manager = build_dynamic_data(
      train_data,
      batch_size=batch_size,
      max_length=max_length,
      is_scores_path=is_scores_path,
      num_buckets=num_buckets)

  eval_batches = preprocess_wmt_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=batch_size,
      max_length=max_eval_length)

  predict_batches = preprocess_wmt_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=batch_size,
      max_length=max_eval_length,
      drop_remainder=False)

  return train_data_manager, eval_batches, predict_batches, sp_tokenizer


def build_dynamic_data(dataset,
                       shuffle_buffer_size=1000,
                       max_length=512,
                       batch_size=256,
                       is_scores_path=None,
                       num_buckets=100):
  """Shuffle and batch/pack the given dataset."""
  def length_filter(max_len):
    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)
    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  assert is_scores_path is not None
  # Break into buckets
  buckets = create_buckets(dataset, is_scores_path, num_buckets)
  # Create DatasetBucketManager
  bucket_manager = DatasetBucketManager(buckets, shuffle_buffer_size,
                                        max_length, batch_size)
  return bucket_manager


def create_buckets(dataset, is_scores_path, num_buckets):
  """Split dataset into buckets."""

  scores = []
  with tf.io.gfile.GFile(is_scores_path, 'r') as f:
    reader = csv.reader(f)
    for val in reader:
      scores.extend(val)
  scores = [float(s) for s in scores]

  lengths = []
  with tf.io.gfile.GFile(is_scores_path.replace('.csv', '_length.csv'),
                         'r') as f:
    reader = csv.reader(f)
    for val in reader:
      lengths.extend(val)
  lengths = [int(s) for s in lengths]

  # compute bucket thresholds
  sorted_scores = np.sort(scores)
  logging.info('len scores %d', len(scores))
  bucket_size = int(len(scores) / num_buckets)
  ends = sorted_scores[bucket_size-1::bucket_size]

  # Iterate through dataset and write to memory
  bin_assignments = np.digitize(scores, ends)
  tf_is_bins = tf.data.Dataset.from_tensor_slices(bin_assignments)
  tf_lengths = tf.data.Dataset.from_tensor_slices(lengths)
  scored_data = tf.data.Dataset.zip((tf_is_bins, tf_lengths, dataset))
  bucket_examples = collections.defaultdict(list)
  iter_index = 0
  for ex_bin, ex_len, data in iter(scored_data):
    assert ex_len.numpy() == np.count_nonzero(
        data['targets'].numpy()), (ex_len, data, iter_index)
    iter_index += 1
    bucket_examples[ex_bin.numpy()].append(data)

  bucket_datasets = []
  index_memory = [0]* num_buckets
  for i in range(num_buckets):
    logging.info('Bin %d num el: %d', i, len(bucket_examples[i]))
    def gen_creator(bin_i):
      def gen():
        for ex_i in range(index_memory[bin_i], len(bucket_examples[bin_i])):
          index_memory[bin_i] = ex_i + 1
          yield bucket_examples[bin_i][ex_i]
          if ex_i == len(bucket_examples[bin_i]) - 1:
            logging.info('SHUFFLING BIN!! %d', bin_i)
            index_memory[bin_i] = 0
            random.shuffle(bucket_examples[bin_i])
      return gen

    gen_ds = tf.data.Dataset.from_generator(
        gen_creator(i), output_types={
            'inputs': tf.int32,
            'targets': tf.int32
        })
    gen_ds = gen_ds.repeat()
    bucket_datasets.append(gen_ds)

  # Sanity check that we are not creating the same dataset on each loop
  assert bucket_datasets[0] != bucket_datasets[1]
  return bucket_datasets


class DatasetBucketManager():
  """For dynamic data selection, sample or draw from buckets."""

  def __init__(self, datasets, shuffle_buffer_size=1000,
               max_length=256, batch_size=256):
    self.shuffle_buffer_size = shuffle_buffer_size
    self.max_length = max_length
    self.batch_size = batch_size
    self.unproccessed_buckets = datasets
    self.buckets = self._proc_buckets(self.unproccessed_buckets)

  def _proc_buckets(self, buckets):
    return list(map(iter, map(self._proc_dataset, buckets)))

  def _proc_dataset(self, dataset):
    dataset = dataset.repeat()
    dataset = dataset.shuffle(self.shuffle_buffer_size)
    dataset = pack_dataset(dataset, self.max_length)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

  def sampled_dataset(self, distribution):
    """Return a dataset that samples from the buckets."""
    sampled_ds = tf.data.experimental.sample_from_datasets(
        self.unproccessed_buckets,
        weights=distribution)
    # Optionally you can add a seed for better reproducibility
    # seed=dataset_utils.RANDOM_SAMPLE_SEED)
    # You shouldn't cache this dataset because it might not properly resample
    return self._proc_dataset(sampled_ds)

  def get_bucket(self, index):
    return self.buckets[index]
