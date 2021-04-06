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

"""Utility function to run glove_model and glove_model_tf.
"""

from __future__ import print_function

import collections
import copy
import random


def count_cooccurrences(walks, window_size, vocab_index_lookup=None):
  """Counts co-occurrences from an open file f, line-by-line.

  Args:
    walks: a list of random walks
    window_size: window size for co-occurrence counting
    vocab_index_lookup: dictionary mapping tokens matrix indices. NB: these will
      be 1-indexed. Tokens must be the same type as the walks list items. If
      None (default), tokens are indexed according to order seen in walks.
  Returns:
    cooccurrence_list: list of shuffled (id, id, score) co-occurrence score
      triples
    index_vocab_list: list of tokens ordered by the vocab index lookup
    vocab_index_list: dict mapping tokens to integers
  """
  cooccurrence_dict = {}

  # Pre-load cooccurrence dictionary
  if vocab_index_lookup:
    for w in vocab_index_lookup:
      cooccurrence_dict[w] = collections.defaultdict(float)
  else:
    vocab_index_lookup = {}
    n = 0

  # Extract co-occurrences
  print('Extracting co-occurrences...')
  for walk in walks:
    for j in range(len(walk)):
      w1 = walk[j]
      if w1 not in vocab_index_lookup:
        vocab_index_lookup[w1] = n
        cooccurrence_dict[w1] = collections.defaultdict(float)
        n += 1
      for k in range(j - window_size, j):
        if k >= 0:
          w2 = walk[k]
          cooccurrence_dict[w1][w2] += 1.0 / (j - k)
          cooccurrence_dict[w2][w1] += 1.0 / (j - k)

  # Store original cooccurrence dict
  tokenized_cooccurrences = copy.deepcopy(cooccurrence_dict)

  # Remove empty cooccurrence dicts
  for node in list(cooccurrence_dict.keys()):  # pylint: disable=g-builtin-op
    if not cooccurrence_dict[node]:
      del cooccurrence_dict[node]

  # Shuffle cooccurrences into a list
  print('Computing co-occurrences...')
  for node in cooccurrence_dict:
    cooccurrence_dict[node] = [
        (vocab_index_lookup[node] + 1,
         vocab_index_lookup[target] + 1, score) for
        (target, score) in cooccurrence_dict[node].items()]
  print('Flattening co-occurrences...')
  cooccurrence_list = []
  for cooccurrence in cooccurrence_dict.values():
    for c in cooccurrence:
      cooccurrence_list.append(c)
  print('Shuffling co-occurrences...')
  random.shuffle(cooccurrence_list)

  # Extract vocab into ordered tokens
  index_vocab_list = [0] * len(vocab_index_lookup)
  for token, index in vocab_index_lookup.items():
    index_vocab_list[index] = token
  return (cooccurrence_list, index_vocab_list, vocab_index_lookup,
          tokenized_cooccurrences)


def _get_keyed_vector(line):
  numbers = line.strip().split()
  return {str(int(numbers[0])): [float(x) for x in numbers[1:]]}


def _get_weight_string(key, weights):
  return '%s %s\n' % (key, ' '.join(['{0:.17g}'.format(v) for v in weights]))


class KeyedVectors(object):
  """A lightweight KeyedVector object that mimics gensim.models.KeyedVector.

     For loading and saving only (does not mimic all functionality for those).
     Written to avoid building gensim when using GloVe in colab.

  Attributes:
    dim: column dimension of the weights
    weights: dict of (str, numpy vector)
  """

  def __init__(self, dim):
    self._dim = dim
    self._weights = {}

  def add(self, tokens, weights):
    """Add weights with give tokens to model.

    Args:
      tokens: a list of tokens
      weights: a list of weight lists or numpy arrays
    Returns: (nothing)
    """
    if len(tokens) != len(weights):
      print('Error: number of tokens does not match number of weight vecs')
      return
    for i, key in enumerate(tokens):
      if len(weights[i]) != self._dim:
        print('Error: weights from key %s have dim %d but %d is needed' % (
            key, len(weights[i]), self._dim))
        return
    self._weights.update(dict(zip(tokens, weights)))

  def save_word2vec_format(self, filename):
    """Saves vectors in word2vec format.

    Args:
      filename: a filename
    Returns: (nothing)
    """
    with open(filename, 'w') as f:
      f.write('%d %d\n' % (len(self._weights), self._dim))
      for key in self._weights:
        f.write(_get_weight_string(key, self._weights[key]))

  @staticmethod
  def load_word2vec_format(filename, binary=False):
    """Loads vectors in word2vec format.

    Args:
      filename: a filename
      binary: whether a filename points to a binary file
    Returns:
      model: a {key, word vector} dict
    """
    with open(filename, 'rb' if binary else 'r') as f:
      _ = f.readline()
      model = {}
      for line in f:
        keyed_weights = _get_keyed_vector(line)
        model.update(keyed_weights)
    return model
