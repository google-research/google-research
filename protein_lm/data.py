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

# Lint as: python3
"""Dataset preprocessing and pipeline.

Built for Trembl dataset.
"""
import os
import types
from absl import logging
import gin
import numpy as np
import tensorflow.compat.v1 as tf

from protein_lm import domains

protein_domain = domains.VariableLengthDiscreteDomain(
    vocab=domains.ProteinVocab(
        include_anomalous_amino_acids=True,
        include_bos=True,
        include_eos=True,
        include_pad=True,
        include_mask=True),
    length=1024)  # TODO(ddohan): Make a `make_protein_domain` fn.


def dataset_from_tensors(tensors):
  """Converts nested tf.Tensors or np.ndarrays to a tf.Data.Dataset."""
  if isinstance(tensors, types.GeneratorType) or isinstance(tensors, list):
    tensors = tuple(tensors)
  return tf.data.Dataset.from_tensor_slices(tensors)


def _parse_example(value):
  parsed = tf.parse_single_example(
      value, features={'sequence': tf.io.VarLenFeature(tf.int64)})
  sequence = tf.sparse.to_dense(parsed['sequence'])
  return sequence


@gin.configurable
def get_train_valid_files(directory, num_test_files=10, num_valid_files=1):
  """Given a directory, list files and split into train/test files.

  Args:
    directory: Directory containing data.
    num_test_files: Number of files to set aside for testing.
    num_valid_files: Number of files to use for validation.

  Returns:
    Tuple of lists of (train files, test files).
  """
  files = tf.gfile.ListDirectory(directory)
  files = [os.path.join(directory, f) for f in files if 'tmp' not in f]
  files = sorted(files)
  # Set aside the first num_test_files files for testing.
  valid_files = files[num_test_files:num_test_files + num_valid_files]
  train_files = files[num_test_files + num_valid_files:]
  return train_files, valid_files


def _add_eos(seq):
  """Add end of sequence markers."""
  # TODO(ddohan): Support non-protein domains.
  return tf.concat([seq, [protein_domain.vocab.eos]], axis=-1)


def load_dataset(train_files,
                 test_files,
                 shuffle_buffer=8192,
                 batch_size=32,
                 max_train_length=512,
                 max_eval_length=None):
  """Load data from directory.

  Takes first shard as test split.

  Args:
    train_files: Files to load training data from.
    test_files: Files to load test data from.
    shuffle_buffer: Shuffle buffer size for training.
    batch_size: Batch size.
    max_train_length: Length to crop train sequences to.
    max_eval_length: Length to crop eval sequences to.

  Returns:
    Tuple of (train dataset, test dataset)
  """
  max_eval_length = max_eval_length or max_train_length
  logging.info('Training on %s shards', len(train_files))
  print('Training on %s shards' % len(train_files))
  print('Test on %s shards' % str(test_files))

  test_ds = tf.data.TFRecordDataset(test_files)

  # Read training data from many files in parallel
  filenames_dataset = tf.data.Dataset.from_tensor_slices(train_files).shuffle(
      2048)
  train_ds = filenames_dataset.interleave(
      tf.data.TFRecordDataset,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=False)

  train_ds = train_ds.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_ds = test_ds.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  train_ds = batch_ds(
      train_ds,
      batch_size=batch_size,
      shuffle_buffer=shuffle_buffer,
      length=max_train_length)
  test_ds = batch_ds(
      test_ds,
      batch_size=batch_size,
      shuffle_buffer=None,
      length=max_eval_length)

  train_ds.prefetch(tf.data.experimental.AUTOTUNE)
  test_ds.prefetch(tf.data.experimental.AUTOTUNE)
  return train_ds, test_ds


@gin.configurable
def batch_ds(ds,
             length=512,
             batch_size=32,
             shuffle_buffer=8192,
             pack_length=None):
  """Crop, shuffle, and batch a dataset of sequences."""

  def _crop(x):
    return x[:length]

  if length:
    ds = ds.map(_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if shuffle_buffer:
    ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

  if pack_length:
    logging.info('Packing sequences to length %s', pack_length)
    # Add EOS tokens.
    ds = ds.map(_add_eos, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Pack sequences together by concatenating.
    ds = ds.unbatch()
    ds = ds.batch(pack_length)  # Pack length
    ds = ds.batch(batch_size, drop_remainder=True)  # Add batch dimension.
  else:
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=length,
        padding_values=np.array(protein_domain.vocab.pad, dtype=np.int64),
        drop_remainder=True)
  return ds


def _encode_protein(protein_string):
  array = protein_domain.encode([protein_string], pad=False)
  array = np.array(array)
  return array


def _sequence_to_tf_example(sequence):
  sequence = np.array(sequence)
  features = {
      'sequence':
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=sequence.reshape(-1))),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


def _write_tfrecord(sequences, outdir, idx, total):
  """Write iterable of sequences to sstable shard idx/total in outdir."""
  idx = '%0.5d' % idx
  total = '%0.5d' % total
  name = 'data-%s-of-%s' % (idx, total)
  path = os.path.join(outdir, name)
  with tf.io.TFRecordWriter(path) as writer:
    for seq in sequences:
      proto = _sequence_to_tf_example(seq)
      writer.write(proto.SerializeToString())


def csv_to_tfrecord(csv_path, outdir, idx, total):
  """Process csv at `csv_path` to shard idx/total in outdir."""
  with tf.gfile.GFile(csv_path) as f:

    def iterator():
      for line in f:
        _, seq = line.strip().split(',')
        yield _encode_protein(seq)

    it = iterator()
    _write_tfrecord(it, outdir, idx, total)
