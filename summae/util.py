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

"""Utility functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import tempfile
import time

import six
from six.moves import range
import tensorflow.compat.v1 as tf  # tf

from rouge import rouge_scorer
from summae import text_encoder

_ROCSTORIES_BASE = 'rocstories_springwintertrain.all'

PAD_ID = 0
EOS_ID = 1


def file_list(data_dir, subset, task='wiki103'):
  """Get list of example files."""
  if task == 'wiki103':
    pattern = os.path.join(data_dir, 'encoded.%s.*.tfrecord' % subset)
    return tf.gfile.Glob(pattern)
  elif task == 'rocstories':
    if subset == 'train':
      flist = tf.gfile.Glob(os.path.join(
          data_dir, _ROCSTORIES_BASE + '.000[0-9].tfrecord'))
      flist.extend(tf.gfile.Glob(os.path.join(
          data_dir, _ROCSTORIES_BASE + '.001[0-7].tfrecord')))
      assert len(flist) == 18
    elif subset == 'valid':
      flist = [os.path.join(data_dir, _ROCSTORIES_BASE + '.0018.tfrecord')]
    elif subset == 'valid_gt':
      flist = [os.path.join(data_dir, 'rocstories_gt.valid.tfrecord')]
    elif subset == 'test_gt':
      flist = [os.path.join(data_dir, 'rocstories_gt.test.tfrecord')]
    else:
      # Test
      flist = [os.path.join(data_dir, _ROCSTORIES_BASE + '.0019.tfrecord')]
    tf.logging.info('File list for %s: %s', subset, flist)
    return flist
  else:
    tf.logging.fatal('Unsupported task %s', task)


def get_tokenizer(path):
  tf.logging.info('Loaded vocabulary from: %s', path)
  return text_encoder.SubwordTextEncoder(path)


def strip_after_eos(token_ids):
  for i, tid in enumerate(token_ids):
    if tid == text_encoder.EOS_ID:
      return token_ids[:i]
  # No EOS
  return token_ids


def add_summary_if_exists(name, tensor):
  try:
    tf.summary.scalar(name, tensor)
  except NameError:
    tf.logging.info('skipping %s tf.summary', name)


def compute_avg_rouge(decodes, targets):
  """Computes ROUGE between two lists of strings.

  Args:
    decodes: list of strings for candidate
    targets: list of reference strings

  Returns:
    3-tuple of rouge-1, rouge-2, rouge-L
  """
  # TODO(peterjliu): Use BootstrapAggregator.
  assert decodes
  assert len(decodes) == len(targets)
  rtypes = ['rouge1', 'rouge2', 'rougeL']
  rs = rouge_scorer.RougeScorer(rtypes,
                                use_stemmer=True)

  rouge_f = {}
  for i in range(len(decodes)):
    score_dict = rs.score(targets[i], decodes[i])
    for rtype in rtypes:
      if rtype not in rouge_f:
        rouge_f[rtype] = []
      rouge_f[rtype].append(score_dict[rtype].fmeasure)

  def mean(lst):
    return sum(lst) / len(lst)

  return (mean(rouge_f['rouge1']),
          mean(rouge_f['rouge2']),
          mean(rouge_f['rougeL']))


def get_tokenizer_with_special(init_vocab_file_path, special_tokens):
  """Returns a text_encoder.SubwordTokenizer, but with extra special tokens.

  New tokens are added to end of the vocabulary so that tokenized doesn't need
  to be re-generated with new vocab.

  Args:
    init_vocab_file_path: path to initial vocab file
    special_tokens: string list of special tokens to add

  Returns:
    A tuple of (tokenizer with special tokens, dict of special_token->id).
  """
  # Add extra reserved tokens to end of vocab a temporary file.
  with tf.gfile.Open(init_vocab_file_path, 'rb') as f:
    contents = f.read()
  with tempfile.NamedTemporaryFile(delete=True) as f:
    f.write(contents)
    for s in special_tokens:
      f.write(six.ensure_binary('%s_\n' % s, 'utf-8'))
    f.flush()
    tk = text_encoder.SubwordTextEncoder(f.name)
    ids = {}
    # Each line of vocab ends with '\n', so there is vocab+1 elements
    # in the result of the below split.
    o_size = len(six.ensure_str(contents, 'utf-8').split('\n')) - 1
    for i, s in enumerate(special_tokens):
      ids[s] = o_size + i
    return (tk, ids)


def read_records(filename):
  reader = tf.python_io.tf_record_iterator(filename)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      tf.logging.info('read: %d', len(records))
  return records


def get_seq_exs(filename):
  def parse(r):
    s = tf.train.SequenceExample()
    s.ParseFromString(r)
    return s

  return [parse(r) for r in read_records(filename)]


def write_records(records, out_filename):
  writer = tf.python_io.TFRecordWriter(out_filename)
  for count, record in enumerate(records):
    writer.write(record)
    if count > 0 and count % 100000 == 0:
      tf.logging.info('write: %d', count)
  writer.close()


def get_story(s):
  return b' '.join(
      [f.bytes_list.value[0]
       for f in s.feature_lists.feature_list['untokenized_sentences'].feature])


def get_id(s):
  return s.context.feature['storyid'].bytes_list.value[0]


def get_mturk_ground_truth(file_path):
  """Returns dict of story->summary_list."""
  story2summaries = collections.defaultdict(list)
  with tf.gfile.Open(file_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
      story = row['Input.story']
      story2summaries[story].append(row['Answer.summary'])

  return story2summaries


def checkpoint_file_gen(estimator, step_list_csv, sleep_secs,
                        max_sleep_secs=1800):
  """Yields model checkpoints for an estimator.

  Args:
    estimator: tf.Estimator object
    step_list_csv: csv string of checkpoints
    sleep_secs: how many seconds to wait for next checkpoint
    max_sleep_secs: maximum cumulative sleep seconds

  Yields:
    Checkpoint file name.
  """
  if step_list_csv:
    for s in six.ensure_str(step_list_csv, 'utf-8').split(','):
      yield os.path.join(estimator.model_dir, 'model.ckpt-%s' % s)
  else:
    # Continuous eval
    prev_ckpt_file = ''
    total_sleep_secs = 0.0
    while True:
      ckpt_file = estimator.latest_checkpoint()
      if ckpt_file is None or ckpt_file == prev_ckpt_file:
        # First checkpoint may cause issues with eval
        if total_sleep_secs > max_sleep_secs:
          tf.logging.info(
              'Slept for %g s. Probably training is done. Exiting.',
              total_sleep_secs)
          break
        tf.logging.info('sleep for a %g s', sleep_secs)
        time.sleep(sleep_secs)
        total_sleep_secs += sleep_secs
        continue
      else:
        total_sleep_secs = 0
        prev_ckpt_file = ckpt_file
      yield ckpt_file
