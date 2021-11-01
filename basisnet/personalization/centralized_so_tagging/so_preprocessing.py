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

"""Useful functions for preprocessing Stackoverflow tagging task."""
from typing import Callable, List

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff


def create_word_vocab(vocab_size):
  """Creates a vocab from the `vocab_size` most common words in Stack Overflow."""
  vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts()
  return list(vocab_dict.keys())[:vocab_size]


def create_tag_vocab(vocab_size):
  """Creates a vocab from the `vocab_size` most common tags in Stack Overflow."""
  tag_dict = tff.simulation.datasets.stackoverflow.load_tag_counts()
  return list(tag_dict.keys())[:vocab_size]


def build_to_ids_fn(word_vocab,
                    tag_vocab):
  """Constructs a function mapping examples to sequences of token indices."""
  word_vocab_size = len(word_vocab)
  word_table_values = np.arange(word_vocab_size, dtype=np.int64)
  word_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(word_vocab, word_table_values),
      num_oov_buckets=1)

  tag_vocab_size = len(tag_vocab)
  tag_table_values = np.arange(tag_vocab_size, dtype=np.int64)
  tag_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(tag_vocab, tag_table_values),
      num_oov_buckets=1)

  def to_ids(example):
    """Converts a Stack Overflow example to a bag-of-words/tags format."""
    sentence = tf.strings.join([example['tokens'], example['title']],
                               separator=' ')
    words = tf.strings.split(sentence)
    tokens = word_table.lookup(words)
    tokens = tf.one_hot(tokens, word_vocab_size + 1)
    tokens = tf.reduce_mean(tokens, axis=0)[:word_vocab_size]

    tags = example['tags']
    tags = tf.strings.split(tags, sep='|')
    tags = tag_table.lookup(tags)
    tags = tf.one_hot(tags, tag_vocab_size + 1)
    tags = tf.reduce_sum(tags, axis=0)[:tag_vocab_size]

    return (tokens, tags)

  return to_ids


def create_preprocess_fn(word_vocab, tag_vocab, shuffle_buffer_size=1000):
  """Constructs a function for Stackoverflow tagging preprocessing."""
  def preprocess_fn(org_dataset):
    to_ids = build_to_ids_fn(word_vocab, tag_vocab)
    dataset = (
        org_dataset.shuffle(shuffle_buffer_size)
        # Map sentences to tokenized vectors
        .map(to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE))

    id_dataset = org_dataset.map(
        lambda x: x['client_id'],
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((dataset, id_dataset))

    def _reorder_id(x, idx):
      return (x[0], idx), x[1]

    dataset = dataset.map(_reorder_id)

    return dataset

  return preprocess_fn
