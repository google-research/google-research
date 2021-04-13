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

# Lint as: python3
"""Utility functions for probing experiments."""
import collections
import random
from typing import Dict, Generator, Iterator, List, Set, Text, Tuple

from absl import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_extraction
from sklearn import neighbors



def split_by_length_threshold(
    df, test_set_size):
  """Splits off a test set based on text length.

  Finds a length threshold such that all texts longer than this threshold make
  up a test set that is at max `test_set_size` in size. Note that the exact
  size cannot be guaranteed.

  Args:
    df: DataFrame with 'text_len' column containing the length of every text.
    test_set_size: Number of elements the test set must contain.

  Returns:
    1. Length threshold.
    2. Lengths that exist in the dataset and that are larger than the threshold.
    3. Binary mask in the size of the original data, where True indicates that
      the example must be part of the test set.

  Raises:
    ValueError: In case no threshold can be found such that `test_set_size`
      elements end up in the test set.
  """
  current_count = 0

  # Start from the longest texts.
  for i in range(max(df['text_len']), 0, -1):
    current_count += len(df.loc[df['text_len'] == i, :])

    if current_count > test_set_size:
      return (i,
              set(df['text_len']) & set(range(i + 1,
                                              max(df['text_len']) + 1)),
              df.loc[:, 'text_len'] > i)
  raise ValueError(
      'No length threshold found to create a test set with size {}.'.format(
          test_set_size))


def split_by_random_length(
    df,
    test_set_size,
    size_tolerance = 0.01,
    max_attempts = 100):
  """Creates new train/dev/test split for SentEval data based on text length.

  Finds a random subset of text lengths such that the number of examples having
  these lengths is within an accepted window around `test_set_size`. The
  accepted window is:
    test_set_size * 1 - size_tolerance < window_size < test_set_size * 1 +
    size_tolerance$.

  Since this is a random process, the script can generate multiple such samples.
  Note that the exact size cannot be guaranteed.

  Args:
    df: DataFrame with 'text_len' column containing the length of every text.
    test_set_size: Number of elements the test set must contain. This is the
      middle of the window in which the number of test set examples must fall.
    size_tolerance: Indicates the width of the window according to the above
      formula. This should likely be in [0, 1).
    max_attempts: Number of attempts to try to sample a test set that's size is
      accepted.

  Returns:
    1. Lengths that are reserved for the test set.
    2. Binary mask in the size of the original data, where True indicates that
      the example must be part of the test set.

  Raises:
    RuntimeError: In case no proper length subset can be found such that the
      number of test set examples falls into the window.
  """
  all_lengths = set(df['text_len'])

  for _ in range(max_attempts):
    test_mask = pd.Series([False] * len(df), dtype=bool)
    remaining_lengths = all_lengths.copy()

    while remaining_lengths:
      selected_length = random.choice(tuple(remaining_lengths))
      remaining_lengths.remove(selected_length)
      test_mask.loc[df['text_len'] == selected_length] = True

      # Keep adding lengths until we have a minimum of test examples.
      if test_mask.sum() < test_set_size * (1 - size_tolerance):
        continue
      # If we are still within the tolerance we take this configuration.
      if test_mask.sum() < test_set_size * (1 + size_tolerance):
        return all_lengths - remaining_lengths, test_mask
      # Otherwise we need to start over again.
      break

  raise RuntimeError(
      'No proper split found. Consider increasing the number of attempts or '
      'the size tolerance.')


def split_with_wasserstein(texts, test_set_size,
                           no_of_trials, min_df,
                           leaf_size):
  """Finds test sets by maximizing Wasserstein distances among the given texts.

  This is separating the given texts into training/dev and test sets based on an
  approximate Wasserstein method. First all texts are indexed in a nearest
  neighbors structure. Then a new test centroid is sampled randomly, from which
  the nearest neighbors in Wasserstein space are extracted. Those constitute
  the new test set.
  Similarity is computed based on document-term counts.

  Args:
    texts: Texts to split into training/dev and test sets.
    test_set_size: Number of elements the new test set should contain.
    no_of_trials: Number of test sets requested.
    min_df: Mainly for speed-up and memory efficiency. All tokens must occur at
      least this many times to be considered in the Wasserstein computation.
    leaf_size: Leaf size parameter of the nearest neighbor search. Set high
      values for slower, but less memory-heavy computation.

  Returns:
    Returns a List of test set indices, one for each trial. The indices
    correspond to the items in `texts` that should be part of the test set.
  """
  vectorizer = feature_extraction.text.CountVectorizer(
      dtype=np.int8, min_df=min_df)
  logging.info('Creating count vectors.')
  text_counts = vectorizer.fit_transform(texts)
  text_counts = text_counts.todense()
  logging.info('Count vector shape %s.', text_counts.shape)
  logging.info('Creating tree structure.')
  nn_tree = neighbors.NearestNeighbors(
      n_neighbors=test_set_size,
      algorithm='ball_tree',
      leaf_size=leaf_size,
      metric=stats.wasserstein_distance)
  nn_tree.fit(text_counts)
  logging.info('Sampling test sets.')
  test_set_indices = []

  for trial in range(no_of_trials):
    logging.info('Trial set: %d.', trial)
    # Sample random test centroid.
    sampled_poind = np.random.randint(
        text_counts.max().max() + 1, size=(1, text_counts.shape[1]))
    nearest_neighbors = nn_tree.kneighbors(sampled_poind, return_distance=False)
    # We queried for only one datapoint.
    nearest_neighbors = nearest_neighbors[0]
    logging.info(nearest_neighbors[:10])
    test_set_indices.append(nearest_neighbors)

  return test_set_indices




def get_target_word_to_sentence_mapping(
    target_words, ignore_sentences,
    sentence_iter):
  """Finds target words in sentences and groups sentences together.

  Maps all target words to sentences that contain only this target word but no
  other.

  Args:
    target_words: Tokens to find within sentences. Only one of them is allowed
      to occur in the sentences.
    ignore_sentences: Sentences to ignore during iteration. These are likely the
      sentences from the original SentEval dataset, because we want to create
      entirely new test sets.
    sentence_iter: Provider for sentences to filter.

  Returns:
    Mapping from target word to all the sentences from `sentence_iter` that
    contain this target word but no other target words.
  """
  target_word_to_sentences = collections.defaultdict(list)
  duplicate_count = 0

  for i, sentence in enumerate(sentence_iter):
    logging.log_every_n(logging.INFO, f'Sentences analyzed: {i}.', 100000)

    if sentence in ignore_sentences:
      logging.warning('Sentence already exists and will be ignored: "%s".',
                      sentence)
      duplicate_count += 1
      continue
    tokens = set(sentence.split())

    intersection = tokens.intersection(target_words)

    # The sentence can only be used if exactly one of the target words occurs in
    # it.
    if len(intersection) != 1:
      continue

    target_word_to_sentences[next(iter(intersection))].append(sentence)

  logging.info('Duplicate sentences found: %d.', duplicate_count)
  return dict(target_word_to_sentences)


def read_senteval_data(senteval_path, task_name):
  """Loads one official SentEval data from given a path into a pandas DataFrame.

  Args:
    senteval_path: base directory of the original SentEval data. Most likely
      this is the "probing" directory of the original directory structure.
    task_name: name of the task whose data file exist in the `senteval_path`.

  Returns:
    DataFrame with all content from the input file.
  """
  filename = '{}/{}'.format(senteval_path, task_name)
  with open(filename) as handle:
    # Setting quotechar to the delimiter prevents pandas from mistreating
    # quotes. The original data isn't properly escaped csv in this sense.
    df = pd.read_csv(
        handle,
        sep='\t',
        header=None,
        names=['set', 'target', 'text'],
        encoding='utf-8',
        quotechar='\t')
    return df
  raise ValueError('Error reading SentEval data file: {}.'.format(filename))
