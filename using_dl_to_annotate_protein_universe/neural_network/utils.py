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

"""Utility functions for protein models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from typing import (Callable, List, Text)

import numpy as np
import tensorflow.compat.v1 as tf


AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]

_PFAM_GAP_CHARACTER = '.'

# Other characters representing amino-acids not in AMINO_ACID_VOCABULARY.
_ADDITIONAL_AA_VOCABULARY = [
    # Substitutions
    'U',
    'O',
    # Ambiguous Characters
    'B',
    'Z',
    'X',
    # Gap Character
    _PFAM_GAP_CHARACTER
]

# Vocab of all possible tokens in a valid input sequence
FULL_RESIDUE_VOCAB = AMINO_ACID_VOCABULARY + _ADDITIONAL_AA_VOCABULARY

# Map AA characters to their index in FULL_RESIDUE_VOCAB.
_RESIDUE_TO_INT = {aa: idx for idx, aa in enumerate(FULL_RESIDUE_VOCAB)}


def residues_to_indices(amino_acid_residues):
  return [_RESIDUE_TO_INT[c] for c in amino_acid_residues]


def normalize_sequence_to_blosum_characters(seq):
  """Make substitutions, since blosum62 doesn't include amino acids U and O.

  We take the advice from here for the appropriate substitutions:
  https://www.cgl.ucsf.edu/chimera/docs/ContributedSoftware/multalignviewer/multalignviewer.html

  Args:
    seq: amino acid sequence. A string.

  Returns:
    An amino acid sequence string that's compatible with the blosum substitution
    matrix.
  """
  return seq.replace('U', 'C').replace('O', 'X')


@functools.lru_cache(maxsize=1)
def _build_one_hot_encodings():
  """Create array of one-hot embeddings.

  Row `i` of the returned array corresponds to the one-hot embedding of amino
    acid FULL_RESIDUE_VOCAB[i].

  Returns:
    np.array of shape `[len(FULL_RESIDUE_VOCAB), 20]`.
  """
  base_encodings = np.eye(len(AMINO_ACID_VOCABULARY))
  to_aa_index = AMINO_ACID_VOCABULARY.index

  special_mappings = {
      'B':
          .5 *
          (base_encodings[to_aa_index('D')] + base_encodings[to_aa_index('N')]),
      'Z':
          .5 *
          (base_encodings[to_aa_index('E')] + base_encodings[to_aa_index('Q')]),
      'X':
          np.ones(len(AMINO_ACID_VOCABULARY)) / len(AMINO_ACID_VOCABULARY),
      _PFAM_GAP_CHARACTER:
          np.zeros(len(AMINO_ACID_VOCABULARY)),
  }
  special_mappings['U'] = base_encodings[to_aa_index('C')]
  special_mappings['O'] = special_mappings['X']
  special_encodings = np.array(
      [special_mappings[c] for c in _ADDITIONAL_AA_VOCABULARY])
  return np.concatenate((base_encodings, special_encodings), axis=0)


def residues_to_one_hot(amino_acid_residues):
  """Given a sequence of amino acids, return one hot array.

  Supports ambiguous amino acid characters B, Z, and X by distributing evenly
  over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].

  Supports rare amino acids by appropriately substituting. See
  normalize_sequence_to_blosum_characters for more information.

  Supports gaps and pads with the '.' and '-' characters; which are mapped to
  the zero vector.

  Args:
    amino_acid_residues: string. consisting of characters from
      AMINO_ACID_VOCABULARY

  Returns:
    A numpy array of shape (len(amino_acid_residues),
     len(AMINO_ACID_VOCABULARY)).

  Raises:
    KeyError: if amino_acid_residues has a character not in FULL_RESIDUE_VOCAB.
  """
  residue_encodings = _build_one_hot_encodings()
  int_sequence = residues_to_indices(amino_acid_residues)
  return residue_encodings[int_sequence]


def fasta_indexer():
  """Get a function for converting tokenized protein strings to indices."""
  mapping = tf.constant(FULL_RESIDUE_VOCAB)
  table = tf.lookup.index_table_from_tensor(mapping)

  def mapper(residues):
    return tf.ragged.map_flat_values(table.lookup, residues)

  return mapper


def fasta_encoder():
  """Get a function for converting indexed amino acids to one-hot encodings."""
  encoded = residues_to_one_hot(''.join(FULL_RESIDUE_VOCAB))
  one_hot_embeddings = tf.constant(encoded, dtype=tf.float32)

  def mapper(residues):
    return tf.ragged.map_flat_values(
        tf.gather, indices=residues, params=one_hot_embeddings)

  return mapper


def in_graph_residues_to_onehot(residues):
  """Performs mapping in `residues_to_one_hot` in-graph.

  Args:
    residues: A tf.RaggedTensor with tokenized residues.

  Returns:
    A tuple of tensors (one_hots, row_lengths):
      `one_hots` is a Tensor<shape=[None, None, len(AMINO_ACID_VOCABULARY)],
                             dtype=tf.float32>
       that contains a one_hot encoding of the residues and pads out all the
       residues to the max sequence length in the batch by 0s.
       `row_lengths` is a Tensor<shape=[None], dtype=tf.int32> with the length
       of the unpadded sequences from residues.

  Raises:
    tf.errors.InvalidArgumentError: if `residues` contains a token not in
    `FULL_RESIDUE_VOCAB`.
  """
  ragged_one_hots = fasta_encoder()(fasta_indexer()(residues))
  return (ragged_one_hots.to_tensor(default_value=0),
          tf.cast(ragged_one_hots.row_lengths(), dtype=tf.int32))


def batch_iterable(iterable, batch_size):
  """Yields batches from an iterable.

  If the number of elements in the iterator is not a multiple of batch size,
  the last batch will have fewer elements.

  Args:
    iterable: a potentially infinite iterable.
    batch_size: the size of batches to return.

  Yields:
    array of length batch_size, containing elements, in order, from iterable.

  Raises:
    ValueError: if batch_size < 1.
  """
  if batch_size < 1:
    raise ValueError(
        'Cannot have a batch size of less than 1. Received: {}'.format(
            batch_size))

  current = []
  for item in iterable:
    if len(current) == batch_size:
      yield current
      current = []
    current.append(item)

  # Prevent yielding an empty batch. Instead, prefer to end the generation.
  if current:
    yield current


def pad_one_hot(one_hot, length):
  if length < one_hot.shape[0]:
    raise ValueError("The padding value must be longer than the one-hot's 0th "
                     'dimension. Padding value is ' + str(length) + ' '
                     'and one-hot shape is ' + str(one_hot.shape))
  padding = np.zeros((length - one_hot.shape[0], len(AMINO_ACID_VOCABULARY)))
  return np.append(one_hot, padding, axis=0)


def make_padded_np_array(ragged_arrays):
  """Converts ragged array of one-hot amino acids to constant-length np.array.

  Args:
    ragged_arrays: list of list of int. Each entry in the list is a one-hot
      encoded protein, where each entry corresponds to an amino acid.

  Returns:
    np.array of int, shape (len(ragged_arrays),
      len(longest_array_in_ragged_arrays), len(AMINO_ACID_VOCABULARY)).
  """
  max_array_length = max(len(a) for a in ragged_arrays)
  return np.array([
      pad_one_hot(ragged_array, max_array_length)
      for ragged_array in ragged_arrays
  ])


def absolute_paths_of_files_in_dir(dir_path):
  files = os.listdir(dir_path)
  return sorted([os.path.join(dir_path, f) for f in files])
