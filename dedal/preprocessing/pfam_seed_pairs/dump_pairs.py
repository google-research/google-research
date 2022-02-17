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

"""Brute-force script to precompute Pfam-A seed sequence pairs.

The aim of this script is to relieve the input data pipeline from executing
the pairing transform, which appears to be memory-heavy in its current version.

The current version of this script is fairly slow, with a processing time of
about 2 min per batch. It is strongly recommended to run multiple instances in
parallel to write N shards simultaneously.
"""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

from dedal.data import align_transforms
from dedal.data import specs
from dedal.data import transforms


flags.DEFINE_string(
    'out_dir', None,
    'Parent directory in which output TFRecords will be saved.')
flags.DEFINE_string(
    'data_dir', None,
    '<data_dir>/<task> contains output of create_splits.py.')
flags.DEFINE_enum(
    'loader_cls', 'Pfam34Loader', ['Pfam34Loader'],
    'The loader class to fetch the unpaired sequences with.')
flags.DEFINE_string(
    'task', 'iid_ood_clans',
    'Task for which to generate TFRecords.')
flags.DEFINE_string(
    'split', 'train',
    'Data split for which to generate TFRecords.')
flags.DEFINE_integer(
    'max_len', 512,
    'Maximum sequence length, including any special tokens.')
flags.DEFINE_multi_string(
    'index_keys', ['fam_key', 'ci_100'],
    'Indexing keys for stratified sampling of sequence pairs.')
flags.DEFINE_multi_float(
    'smoothing', [1.0, 1.0],
    'Smoothing coefficients for stratified sampling of sequence pairs.')
flags.DEFINE_string(
    'branch_key', 'ci_100',
    'Branching key for stratified sampling of sequence pairs.')
flags.DEFINE_integer(
    'seed', 0,
    'PRNG seed to generate the shard with.')
flags.DEFINE_integer(
    'n_pairs', 102400,
    'Number of sequence pairs per shard.')
FLAGS = flags.FLAGS


LOADERS = {
    'Pfam34Loader': specs.make_pfam34_loader,
}


def make_serialize_fn():
  """Creates a serialization function for paired examples."""
  has_context = FLAGS.task.endswith('with_ctx')

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _serialize_fn(seq_1, seq_2, cla_key_1, cla_key_2, fam_key_1, fam_key_2,
                    ci_key_1, ci_key_2, seq_key_1, seq_key_2):
    """Serializes the example for TFRecord storage."""
    feature = {
        'seq_1': _int64_feature(seq_1),
        'seq_2': _int64_feature(seq_2),
        'cla_key_1': _int64_feature([cla_key_1]),
        'cla_key_2': _int64_feature([cla_key_2]),
        'fam_key_1': _int64_feature([fam_key_1]),
        'fam_key_2': _int64_feature([fam_key_2]),
        f'{FLAGS.branch_key}_1': _int64_feature([ci_key_1]),
        f'{FLAGS.branch_key}_2': _int64_feature([ci_key_2]),
        'seq_key_1': _int64_feature([seq_key_1]),
        'seq_key_2': _int64_feature([seq_key_2]),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

  def _serialize_fn_with_ctx(seq_1, seq_2, cla_key_1, cla_key_2, fam_key_1,
                             fam_key_2, ci_key_1, ci_key_2, seq_key_1,
                             seq_key_2, full_seq_1, full_seq_2, start_1,
                             start_2, end_1, end_2):
    """Serializes an example with UniprotKB ctx for TFRecord storage."""
    feature = {
        'seq_1': _int64_feature(seq_1),
        'seq_2': _int64_feature(seq_2),
        'full_seq_1': _int64_feature(full_seq_1),
        'full_seq_2': _int64_feature(full_seq_2),
        'cla_key_1': _int64_feature([cla_key_1]),
        'cla_key_2': _int64_feature([cla_key_2]),
        'fam_key_1': _int64_feature([fam_key_1]),
        'fam_key_2': _int64_feature([fam_key_2]),
        f'{FLAGS.branch_key}_1': _int64_feature([ci_key_1]),
        f'{FLAGS.branch_key}_2': _int64_feature([ci_key_2]),
        'seq_key_1': _int64_feature([seq_key_1]),
        'seq_key_2': _int64_feature([seq_key_2]),
        'start_1': _int64_feature([start_1]),
        'start_2': _int64_feature([start_2]),
        'end_1': _int64_feature([end_1]),
        'end_2': _int64_feature([end_2]),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

  def serialize_fn(example):
    """Wraps serialize_fn on tf.py_function for graph-mode compatibility."""
    inputs = [
        example['sequence_1'], example['sequence_2'],
        example['cla_key_1'], example['cla_key_2'],
        example['fam_key_1'], example['fam_key_2'],
        example[f'{FLAGS.branch_key}_1'], example[f'{FLAGS.branch_key}_2'],
        example['seq_key_1'], example['seq_key_2'],
    ]

    py_fn = _serialize_fn_with_ctx if has_context else _serialize_fn
    if has_context:
      inputs.extend([example['full_sequence_1'], example['full_sequence_2'],
                     example['start_1'], example['start_2'],
                     example['end_1'], example['end_2']])

    serialized_example = tf.py_function(py_fn, inputs, tf.string)
    return tf.reshape(serialized_example, ())

  return serialize_fn


def main(_):
  has_context = FLAGS.task.endswith('with_ctx')

  logging.info('seed: %d', FLAGS.seed)
  logging.info('task: %s', FLAGS.task)
  logging.info('split: %s', FLAGS.split)
  logging.info('max_len: %d', FLAGS.max_len)
  logging.info('index_keys: %s', FLAGS.index_keys)
  logging.info('smoothing: %s', FLAGS.smoothing)
  logging.info('branch_key: %s', FLAGS.branch_key)
  logging.info('n_pairs: %d', FLAGS.n_pairs)
  logging.info('has_context: %s', has_context)

  tf.random.set_seed(FLAGS.seed)

  extra_keys = ['cla_key', 'seq_key', 'seq_len'] + FLAGS.index_keys
  if has_context:
    extra_keys.extend(['start', 'end'])
  ds_loader = LOADERS[FLAGS.loader_cls](root_dir=FLAGS.data_dir,
                                        sub_dir='',
                                        extra_keys=extra_keys,
                                        task=FLAGS.task)
  ds = ds_loader.load(FLAGS.split)

  filter_fn = transforms.FilterByLength(max_len=FLAGS.max_len - 1)
  pair_fn = align_transforms.StratifiedSamplingPairing(
      index_keys=FLAGS.index_keys,
      branch_key=FLAGS.branch_key,
      smoothing=FLAGS.smoothing)
  ds = ds.apply(filter_fn).apply(pair_fn).take(FLAGS.n_pairs)
  ds = ds.map(make_serialize_fn(), num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  out_dir = os.path.join(
      FLAGS.out_dir, FLAGS.task, FLAGS.branch_key, FLAGS.split)
  filename = f'{FLAGS.seed}.tfrecords'
  tf.io.gfile.makedirs(out_dir)
  path = os.path.join(out_dir, filename)
  logging.info('Writing TFRecords to %s...', path)
  with tf.io.TFRecordWriter(path) as writer:
    for i, serialized_example in enumerate(ds):
      if (i % 1024) == 0:
        logging.info('Seed: %d, progress: %d/%d.', FLAGS.seed, i, FLAGS.n_pairs)
      writer.write(serialized_example.numpy())


if __name__ == '__main__':
  app.run(main)
