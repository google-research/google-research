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

r"""Saves updated Pfam-A seed 34.0 TFRecords with UniprotKB context sequences.

The new TFRecords will be identical to the original, except for the following:
+ A new 'full_sequence' field will be included, containing the context sequence
  from UniprotKB.
+ The fields 'start' and 'end' will be overwritten whenever the values reported
  by Pfam-A seed do not match with those computed here (occurs in 11 entries).
+ Pfam-A seed entries that occur multiple times within their context (repeats)
  are no longer included (549 entries).
+ Pfam-A seed entries not found within their context (sequence mismatches) are
  no longer included (1 entry).
"""

import collections
import json
import os
import time
from typing import Any, Callable, List, Mapping, MutableMapping, Sequence, Union

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from dedal import vocabulary
from dedal.data import specs


flags.DEFINE_string(
    'out_dir',
    None,
    'TFrecords for <split> to be stored in <out_dir>/<task>_with_ctx/<split>.')
flags.DEFINE_string(
    'data_dir', None,
    '<data_dir>/<task> contains output of create_splits.py.')
flags.DEFINE_string(
    'full_sequences_file',
    None,
    'Path to input csv file produced by the script `save_full_sequences.py`.')
flags.DEFINE_string(
    'pfam34_metadata',
    None,
    'Path to input json file produced by the script `save_to_disk.py`.')
flags.DEFINE_string(
    'task',
    'iid_ood_clans',
    'Task for which to generate TFRecords.')
flags.DEFINE_list(
    'splits',
    ['train', 'iid_validation', 'ood_validation', 'iid_test', 'ood_test'],
    'Data split for which to generate TFRecords.')
flags.DEFINE_integer(
    'num_shards_train', 30,
    'Number of shards for TFRecords in the train split.')
flags.DEFINE_integer(
    'num_shards_eval', 5,
    'Number of shards for TFRecords in eval splits, i.e. anything not train.')
flags.mark_flag_as_required('out_dir')
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('full_sequences_file')
flags.mark_flag_as_required('pfam34_metadata')

FLAGS = flags.FLAGS


# Preprocessed Pfam-A seed 34.0 config.
INT_KEYS = ('seq_key', 'fam_key', 'cla_key', 'seq_len', 'start', 'end')
CI_KEY = 'ci'
STR_KEYS = ('id', 'ac', 'ss')


# Type aliases
PfamExample = MutableMapping[str, Union[np.ndarray, int, str]]
PfamSplit = List[PfamExample]
UniprotKBContext = Mapping[str, Union[Sequence[int], bool, str]]


def load_pfam_metadata():
  """Loads Pfam-A seed metadata with vocabulary and additional key info."""
  with tf.io.gfile.GFile(FLAGS.pfam34_metadata, 'r') as f:
    metadata = json.load(f)
  metadata['vocab'] = vocabulary.Vocabulary(**metadata['vocab'])
  return metadata


def load_pfam_examples(pid_ths):
  """Loads Pfam-A seed TFRecords w/o UniprotKB context for each split."""
  logging.info('Loading original Pfam-A seed examples...')

  examples_from_split = collections.defaultdict(list)
  ci_keys = tuple(f'{CI_KEY}_{pid}' for pid in pid_ths)
  ds_loader = specs.make_pfam34_loader(root_dir=FLAGS.data_dir,
                                       sub_dir='',
                                       extra_keys=INT_KEYS + ci_keys+ STR_KEYS,
                                       task=FLAGS.task)
  for split in FLAGS.splits:
    for ex in ds_loader.load(split).prefetch(tf.data.AUTOTUNE):
      ex = tf.nest.map_structure(lambda t: t.numpy(), ex)
      for key in STR_KEYS:
        ex[key] = ex[key].decode('utf-8')
      # Multiple Pfam entries can share the same UniprotKB ID, but are different
      # subsequences of that protein.
      ex['id'] = f"{ex['id']}/{ex['start']}-{ex['end']}"
      examples_from_split[split].append(ex)
    logging.info('Found %d Pfam-A seed examples for split %s.',
                 len(examples_from_split[split]), split)
  return examples_from_split


def load_uniprotkb_context():
  """Loads UniprotKB sequences containing Pfam-A seed (sub)sequences."""
  logging.info('Loading UniprotKB context from %s...',
               FLAGS.full_sequences_file)

  context_from_id = {}
  with tf.io.gfile.GFile(FLAGS.full_sequences_file, 'r') as f:
    _ = f.readline()  # Discards CSV header.
    for line in f:
      pfam_id, uniprot_starts_str, full_sequence = line.strip().split(',')
      uniprot_starts = [int(s) if s else -1
                        for s in uniprot_starts_str.split(';')]
      context_from_id[pfam_id] = {'starts': uniprot_starts,
                                  'has_repeats': len(uniprot_starts) > 1,
                                  'mismatched': -1 in uniprot_starts,
                                  'full_sequence': full_sequence}
  logging.info('Found %d UniprotKB context entries.', len(context_from_id))
  return context_from_id


def add_uniprotkb_context_to_pfam_examples(
    pfam_examples_from_split,
    uniprotkb_context_from_id,
    vocab):
  """Cross-references UniprotKB context sequences with Pfam-A seed examples."""

  updated_pfam_examples_from_split = collections.defaultdict(list)
  for split, pfam_split in pfam_examples_from_split.items():
    removed_pfam_ids = []
    for ex in pfam_split:
      pfam_id = ex['id']

      # Ensures all Pfam-A seed entries were cross-referenced against UniprotKB.
      if pfam_id not in uniprotkb_context_from_id:
        raise ValueError(
            f'Pfam entry {pfam_id} not present in {FLAGS.full_sequences_file}.')
      uniprotkb_context = uniprotkb_context_from_id[pfam_id]

      # Skips any Pfam entries that occur more than once in their UniprotKB
      # context (ambiguous ground-truth alignment) or whose sequences did not
      # perfectly match a subsequence in the UniprotKB context.
      if uniprotkb_context['has_repeats'] or uniprotkb_context['mismatched']:
        removed_pfam_ids.append(pfam_id)
        continue
      updated_pfam_examples_from_split[split].append(ex)

      # Encodes UniprotKB context sequence using the same vocabulary as
      # `ex['sequence']`.
      ex['full_sequence'] = vocab.encode(uniprotkb_context['full_sequence'])

      # Overrides start / end positions for rare cases in which Pfam-A seed
      # reports values inconsistent with UniprotKB context.
      if ex['start'] != uniprotkb_context['starts'][0]:
        logging.info('Overriding start / end for Pfam entry %s.', pfam_id)
        ex['start'] = uniprotkb_context['starts'][0]
        ex['end'] = ex['start'] + ex['seq_len'] - 1

    logging.info(
        '%d Pfam entries in %s split removed for repeats / mismatches: %s.',
        len(removed_pfam_ids), split, removed_pfam_ids)

  return updated_pfam_examples_from_split


def make_serialize_example_fn(pid_ths):
  """Returns fn for serialization of Pfam-A seed examples."""
  ci_keys = tuple(f'{CI_KEY}_{pid}' for pid in pid_ths)

  def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def str_feature(value, encoding='ascii'):
    value = bytes(value, encoding=encoding)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def serialize_example_fn(ex):
    """Serializes the Pfam-A seed example for TFRecord storage."""
    feature = {}
    for key in INT_KEYS + ci_keys:
      feature[key] = int64_feature([ex[key]])
    for key in STR_KEYS:
      feature[key] = str_feature(ex[key])
    feature['seq'] = int64_feature(ex['sequence'])
    feature['full_seq'] = int64_feature(ex['full_sequence'])
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()

  return serialize_example_fn


def split_to_disk(
    out_dir,
    num_shards,
    pfam_split,
    serialize_example_fn):
  """Saves new Pfam-A seed 34.0 TFRecords with extra 'full_sequence' field."""
  def gen():
    for ex in pfam_split:
      yield serialize_example_fn(ex)

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
  start = time.time()

  metadata = load_pfam_metadata()
  pfam_examples_from_split = load_pfam_examples(metadata['pid_ths'])
  uniprotkb_context_from_id = load_uniprotkb_context()

  updated_pfam_examples_from_split = add_uniprotkb_context_to_pfam_examples(
      pfam_examples_from_split,
      uniprotkb_context_from_id,
      metadata['vocab'])

  serialize_example_fn = make_serialize_example_fn(metadata['pid_ths'])
  for split, pfam_split in updated_pfam_examples_from_split.items():
    logging.info('Saving updated TFRecords for split %s...', split)
    out_dir = os.path.join(FLAGS.out_dir, f'{FLAGS.task}_with_ctx', split)
    tf.io.gfile.makedirs(out_dir)
    num_shards = (FLAGS.num_shards_train if split == 'train'
                  else FLAGS.num_shards_eval)
    split_to_disk(out_dir, num_shards, pfam_split, serialize_example_fn)

  runtime = time.time() - start
  logging.info('Total time elapsed: %.3f seconds.', runtime)


if __name__ == '__main__':
  app.run(main)
