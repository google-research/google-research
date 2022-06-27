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

r"""Creates train/validation/test splits for Pfam-A seed 34.0.

For now, two types of splits have been implemented:
+ 'iid': each Pfam-A seed family is split into train/validation/test sets
  uniformly at random.
+ 'ood_families': entire Pfam-A seed families are allocated to either train,
  validation or test uniformly at random.
+ 'ood_clans': entire Pfam-A seed clans are allocated to either train,
  validation or test uniformly at random.
+ iid_ood_families: first, an 'ood_families'-type split is performed, reserving
  p_ood * p_tst and p_ood * p_val for ood_test and ood_validation, respectively.
  The remainder of the data is finally split again into train, iid_test and
  iid_validation, respectively.
+ iid_ood_clans: first, an 'ood_clans'-type split is performed, reserving
  p_ood * p_tst and p_ood * p_val for ood_test and ood_validation, respectively.
  The remainder of the data is finally split again into train, iid_test and
  iid_validation, respectively.

Future split types -- not yet implemented -- will include:
+ 'ood_clusters': entire PID-based clusters within each Pfam-A seed family are
  allocated to either train, validation or test uniformly at random.
"""

import collections
import functools
import itertools
import json
import os
from typing import Mapping, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from dedal.data import serialization
from dedal.data import specs


flags.DEFINE_string(
    'data_dir', None,
    'Directory containing the output of save_to_disk.py.')
flags.DEFINE_string(
    'out_dir', None,
    'Directory that will contain TFRecords for each split.')
flags.DEFINE_enum(
    'type', 'iid_ood_clans',
    ['iid', 'ood_families', 'ood_clans', 'iid_ood_families', 'iid_ood_clans'],
    'Type of train/validation/test splits.')
flags.DEFINE_float(
    'p_tr', 0.8,
    'Proportion of sequences allocated to training set.')
flags.DEFINE_float(
    'p_tst', 0.15,
    'Proportion of sequences allocated to test set.')
flags.DEFINE_float(
    'p_ood', 0.5,
    'Proportion of validation and test made out of OOD data. Only used by split'
    'types iid_ood_families and iid_ood_clans.')
flags.DEFINE_integer(
    'num_shards_train', 30,
    'Number of shards for training set TFRecords.')
flags.DEFINE_integer(
    'num_shards_eval', 5,
    'Number of shards for validation and test set TFRecords.')
flags.DEFINE_integer(
    'seed', 1,
    'Seed for the PRNG of NumPy.')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('out_dir')

FLAGS = flags.FLAGS


Example = Mapping[str, tf.Tensor]


def serialize(example, pid_ths):
  int_keys = ['seq_key', 'start', 'end', 'cla_key', 'fam_key', 'seq_len']
  int_keys.extend([f'ci_{pid_th}' for pid_th in pid_ths])
  str_keys = ['id', 'ac', 'ss']
  keys = int_keys + str_keys + ['sequence']
  all_specs = {k: v.dtype for k, v in example.items() if k in keys}
  coder = serialization.FlatCoder(specs=all_specs)
  for k in int_keys:
    example[k] = [example[k]]
  return coder.encode(example)


def iid_splits(
    examples,
    p_tr,
    p_tst,
):
  """Generates train/validation/test splits per family uniformly at random."""
  examples_tr, examples_tst, examples_val = [], [], []

  for family in examples:
    n = len(family)
    n_eval = int(np.round((1 - p_tr) * n))
    n_tr = n - n_eval
    n_tst = int(np.round((p_tst / (1.0 - p_tr)) * n_eval))
    n_val = n_eval - n_tst

    # If we don't have enough sequences to form a pair, the data is "wasted".
    # When this happens, we first "sacrifice" the validation set and, next, the
    # test set.
    if n_val < 2:
      n_tst += n_val
      n_val = 0
    if n_tst < 2:
      n_tr += n_tst
      n_tst = 0
    assert (n_tr + n_tst + n_val) == n

    indices = np.random.permutation(n)
    family_tr = [family[i] for i in indices[:n_tr]]
    family_tst = [family[i] for i in indices[n_tr:n_tr + n_tst]]
    family_val = [family[i] for i in indices[n_tr + n_tst:]]
    assert (len(family_tr) + len(family_tst) + len(family_val)) == n

    examples_tr.extend(family_tr)
    examples_tst.extend(family_tst)
    examples_val.extend(family_val)

  return examples_tr, examples_tst, examples_val


def ood_splits(
    examples,
    p_tr,
    p_tst,
):
  """Generates train/validation/test splits at the family/clan level."""
  n = sum(len(x) for x in examples)  # Number of examples.
  n_eval = int(np.round((1 - p_tr) * n))
  n_tr = n - n_eval
  n_tst = int(np.round((p_tst / (1.0 - p_tr)) * n_eval))

  n_groups = len(examples)
  indices = np.random.permutation(n_groups)

  i = 0
  examples_tr, examples_tst, examples_val = [], [], []
  while len(examples_tr) < n_tr:
    examples_tr.extend(examples[indices[i]])
    i += 1
  while len(examples_tst) < n_tst:
    examples_tst.extend(examples[indices[i]])
    i += 1
  while i < n_groups:
    examples_val.extend(examples[indices[i]])
    i += 1
  return examples_tr, examples_tst, examples_val


def records_to_disk(split,
                    examples,
                    pid_ths,
                    num_shards):
  """Writes all examples in split to disk as sharded TFRecords."""
  # TODO(fllinares): avoid duplication w.r.t. `save_to_disk.py`.
  out_dir = os.path.join(FLAGS.out_dir, split)
  _ = tf.io.gfile.makedirs(out_dir)

  def gen():
    for example in examples:
      yield serialize(example, pid_ths)

  def reduce_func(key, ds):
    filename = tf.strings.join(
        [out_dir, '/', tf.strings.as_string(key), '.tfrecords'])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(ds.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  ds = tf.data.Dataset.from_generator(
      gen, output_signature=tf.TensorSpec(shape=(), dtype=tf.string))
  ds = ds.enumerate()
  ds = ds.group_by_window(
      lambda i, _: i % num_shards, reduce_func, tf.int64.max)
  _ = list(ds)


def main(_):
  np.random.seed(FLAGS.seed)

  path = os.path.join(FLAGS.data_dir, 'metadata.json')
  logging.info('Reading metadata from %s...', path)
  with tf.io.gfile.GFile(path, 'r') as f:
    metadata = json.load(f)

  logging.info('Reading all data from disk...')
  extra_keys = ['seq_key', 'fam_key', 'cla_key', 'seq_len', 'id', 'ac', 'start',
                'end', 'ss']
  for pid_th in metadata['pid_ths']:
    extra_keys.append(f'ci_{pid_th}')
  loader = specs.make_pfam34_loader(
      root_dir=FLAGS.data_dir, sub_dir='', extra_keys=extra_keys, task='')
  ds = loader.load('all').prefetch(tf.data.AUTOTUNE)

  if FLAGS.type in ('iid', 'ood_families', 'iid_ood_families'):
    key, group = 'fam_key', 'families'
  elif FLAGS.type in ('ood_clans', 'iid_ood_clans'):
    key, group = 'cla_key', 'clans'

  examples = collections.defaultdict(list)
  for ex in ds:
    examples[int(ex[key])].append(ex)
  examples = [examples[k] for k in sorted(examples)]
  n_seqs = np.fromiter((len(v) for v in examples), np.int32)
  logging.info('Found %s sequences in %s %s.', sum(n_seqs), len(n_seqs), group)

  if FLAGS.type in('iid_ood_families', 'iid_ood_clans'):
    p_tr, p_tst = 1 - FLAGS.p_ood * (1 - FLAGS.p_tr), FLAGS.p_ood * FLAGS.p_tst
    examples, examples_ood_tst, examples_ood_val = ood_splits(
        examples, p_tr, p_tst)

    examples_iid = collections.defaultdict(list)
    for ex in examples:
      examples_iid[int(ex[key])].append(ex)
    examples_iid = [examples_iid[k] for k in sorted(examples_iid)]
    assert sum(len(x) for x in examples_iid) == len(examples)

    p_tr, p_tst = FLAGS.p_tr / p_tr, (1 - FLAGS.p_ood) * FLAGS.p_tst / p_tr
    examples_tr, examples_iid_tst, examples_iid_val = iid_splits(
        examples_iid, p_tr, p_tst)
    examples = {
        'train': examples_tr,
        'iid_test': examples_iid_tst,
        'ood_test': examples_ood_tst,
        'iid_validation': examples_iid_val,
        'ood_validation': examples_ood_val,
    }
  elif FLAGS.type in ('iid',):
    examples_tr, examples_tst, examples_val = iid_splits(
        examples, FLAGS.p_tr, FLAGS.p_tst)
    examples = {
        'train': examples_tr,
        'test': examples_tst,
        'validation': examples_val,
    }
  elif FLAGS.type in ('ood_families', 'ood_clans'):
    examples_tr, examples_tst, examples_val = ood_splits(
        examples, FLAGS.p_tr, FLAGS.p_tst)
    examples = {
        'train': examples_tr,
        'test': examples_tst,
        'validation': examples_val,
    }

  keys = {k: set(int(ex['seq_key']) for ex in v) for k, v in examples.items()}
  for keys_1, keys_2 in itertools.combinations(keys.values(), r=2):
    assert not keys_1 & keys_2
  assert len(functools.reduce(lambda x, y: x | y, keys.values())) == sum(n_seqs)
  logging.info('Split dataset into %s seqs.',
               ', '.join(f'{len(v)} {k}' for k, v in examples.items()))

  for k, v in examples.items():
    logging.info('Writing %s TFRecords to disk...', k)
    num_shards = (FLAGS.num_shards_train if k == 'train'
                  else FLAGS.num_shards_eval)
    records_to_disk(
        k, v, metadata['pid_ths'], num_shards)


if __name__ == '__main__':
  app.run(main)
