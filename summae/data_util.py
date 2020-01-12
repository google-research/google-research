# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Data wrangling utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib

from absl import logging
import six
from six.moves import range
import tensorflow.compat.v1 as tf  # tf
from summae import util


def get_filewriters(basename, subset, num_shards):
  """Opens and returns list of tfrecord writers."""
  fh = []
  logging.info('Opening %s.%s.*.tfrecord for writing.', basename, subset)
  for i in range(num_shards):
    fh.append(tf.python_io.TFRecordWriter(
        basename + '.%s.%s.tfrecord' % (subset, str(i).zfill(4))))
  return fh


def get_text_sharder(num_shards):
  """Returns a deterministic string-sharding function."""
  def sharder(text):
    return int(hashlib.md5(six.ensure_binary(text, 'utf-8')).hexdigest(),
               16) % num_shards

  return sharder


def close_writers(ws):
  for x in ws:
    x.close()
  logging.info('Closed tfrecord writers')


def sents2seqex(sentences_list, tokenizer,
                context_features=None, add_eos=False, add_untokenized=False):
  """Convert list of sentences into a SequenceExample."""
  assert tokenizer
  def encode(s):
    e = tokenizer.encode(s)
    if add_eos:
      e.append(util.EOS_ID)
    return e

  sentences = tf.train.FeatureList(
      feature=[tf.train.Feature(
          int64_list=tf.train.Int64List(
              value=encode(s))) for s in sentences_list])
  if add_untokenized:
    untokenized_sentences = tf.train.FeatureList(
        feature=[tf_bytes_feature(s) for s in sentences_list])
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={'sentences': sentences,
                          'untokenized_sentences': untokenized_sentences}),
        context=context_features)
  else:
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={'sentences': sentences},
            context=context_features))


def tf_bytes_feature(s):
  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[six.ensure_binary(s)]))


