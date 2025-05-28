# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
import gzip
import itertools
import json
import os
import tarfile
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
import urllib

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf  # tf
import tqdm

from tensorflow.contrib import lookup as contrib_lookup


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

# This structure is 1-indexed by residue (like all tools in the bioinformatics
# world), not 0-indexed, and is left-inclusive, right-inclusive.
# The reason to be 1-indexed is for better interoperability with tools like
# HMMER and InterProScan.
# See `utils.programmer_range_to_biologist_range`.
SingleDomainCall = Tuple[str, Tuple[int, int]]

MAX_NUM_ENSEMBLE_ELS_FOR_INFERENCE = 31

PFAM_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS = [
    31382545,
    31382748,
    31383322,
    31383424,
    31383478,
    31383654,
    31389180,
    31389209,
    31389438,
    31389472,
    31389671,
    31389736,
    31390585,
    31390960,
    31391286,
    31391322,
    31391598,
    31413768,
    31413880,
    31414257,
    31414595,
    31414687,
    31414709,
    31415235,
    31415474,
    31415675,
    31415839,
    31416206,
    31416435,
    31416516,
    31416809,
]
OSS_ZIPPED_MODELS_ROOT_URL = 'https://storage.googleapis.com/brain-genomics-public/research/proteins/protenn2/trained_zipped_models/pfam_35'
OSS_PFAM_ZIPPED_MODELS_URLS = [
    '{}/{}.tar.gz'.format(OSS_ZIPPED_MODELS_ROOT_URL, p)
    for p in PFAM_RANDOM_ENSEMBLE_ELEMENT_EXPERIMENT_IDS
]


def residues_to_indices(amino_acid_residues):
  return [_RESIDUE_TO_INT[c] for c in amino_acid_residues]


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
  table = contrib_lookup.index_table_from_tensor(mapping)

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


def fetch_oss_pretrained_models(
    output_dir_path, num_ensemble_elements = None
):
  """Fetch, unzip, and untar a number of models to output_dir_path.

  Does not store the tar.gz versions, just the unzipped ones.

  Args:
    output_dir_path: output directory to which ensemble elements should be
      written.
    num_ensemble_elements: number of elements to fetch. If None, fetch all
      available.

  Raises:
    ValueError if model_type is invalid, or num_ensemble_elements is too large.
  """
  num_ensemble_elements = (
      num_ensemble_elements
      if num_ensemble_elements is not None
      else len(OSS_PFAM_ZIPPED_MODELS_URLS)
  )

  if num_ensemble_elements > len(OSS_PFAM_ZIPPED_MODELS_URLS):
    raise ValueError(
        'Requested {} ensemble elements, but only {} were available.'.format(
            num_ensemble_elements, len(OSS_PFAM_ZIPPED_MODELS_URLS)
        )
    )

  absolute_model_urls = OSS_PFAM_ZIPPED_MODELS_URLS[:num_ensemble_elements]
  for absolute_url in tqdm.tqdm(
      absolute_model_urls,
      desc='Downloading and unzipping models to {}'.format(output_dir_path),
      position=0,
      leave=True,
  ):
    # TODO(mlbileschi): consider parallelizing to make faster.
    with urllib.request.urlopen(absolute_url) as url_contents:
      with tarfile.open(fileobj=url_contents, mode='r|gz') as tar:
        tar.extractall(output_dir_path)


def read_pfam_clan_file(path):
  """Parses pfam clan tsv file.

  Args:
    path: str. path to clan file. The current release of this file can be
      downloaded at
      ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz

  Returns:
    pd.DataFrame with columns family_accession (str), clan_accession (str),
    clan_description (str), family_name (str), family_description (str).
  """
  with tf.io.gfile.GFile(path, 'r') as f:
    return pd.read_csv(
        f,
        names=[
            'family_accession',
            'clan_accession',
            'clan_description',
            'family_name',
            'family_description',
        ],
        sep='\t',
        # Some fields are missing, and we want to keep those
        # as empty strings instead of the default behavior,
        # which is to convert them to NaNs.
        keep_default_na=False,
    )


def family_to_clan_mapping(
    *,
    model_cache_path,
    clans_filename = 'clans_pfam35.tsv',
    use_lifted_clan_semantics = False,
):
  """Parse tsv contents, returning dict from pfam family to clan accession.

  Args:
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    clans_filename: relative filename of clans tsv within directory The current
      release of this file can be downloaded at
      ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz
    use_lifted_clan_semantics: Describes how to treat families without a clan.
      If False, families without a clan will not have values in the returned
      dictionary. If True, families without a clan will get their own clan in
      the returned dictionary, with clan name == to the accession (e.g. PF12345
      -> PF12345).

  Returns:
    dict from string to string, e.g. {'PF12345': 'CL9999'}.
  """
  dataframe = read_pfam_clan_file(
      os.path.join(model_cache_path, clans_filename)
  )

  if use_lifted_clan_semantics:
    dataframe['clan_accession'] = dataframe.apply(
        axis='columns',
        func=lambda row: row.clan_accession  # pylint: disable=g-long-lambda
        if row.clan_accession
        else row.family_accession,
    )

  # Filter family names without clans (they are are stored in the csv
  # as empty strings). If we're using lifted clan semantics, every family will
  # have a clan (see docstring).
  return dict(
      (family_id, clan_id)  # pylint: disable=g-complex-comprehension
      for family_id, clan_id in zip(
          dataframe['family_accession'].values,
          dataframe['clan_accession'].values,
      )
      if clan_id
  )


def programmer_range_to_biologist_range(
    start, end
):
  """Converts 0-indexed, right-exclusive index to 1-indexed, right-inclusive.

  All tools in the bioinformatics world, including HMMER and InterProScan
  denote a range of indexes in a protein (a domain) as 1-indexed, left- and
  right- inclusive, whereas (almost) all programming languages use 0-indexed,
  left-inclusive right-exclusive ranges (e.g. `builtins.slice`).

  Args:
    start: 0-based, inclusive.
    end: 0-based, exclusive.

  Returns:
    start: 1-based, inclusive.
    end: 1-based, inclusive.
  """
  return start + 1, end


def biologist_range_to_programmer_range(
    start, end
):
  """Converts 1-indexed, right-inclusive index to 0-indexed, right-exclusive.

  All tools in the bioinformatics world, including HMMER and InterProScan
  denote a range of indexes in a protein (a domain) as 1-indexed, left- and
  right- inclusive, whereas (almost) all programming languages use 0-indexed,
  left-inclusive right-exclusive ranges (e.g. `builtins.slice`).

  Args:
    start: 0-based, inclusive.
    end: 0-based, exclusive.

  Returns:
    start: 1-based, inclusive.
    end: 1-based, inclusive.
  """
  return start - 1, end


def midpoint_in_either_range(
    seq1_start, seq1_end, seq2_start, seq2_end
):
  """Computes whether the midpiont of seq1 is within seq2 or vice-versa.

  This function is useful for computing metrics like precision and recall
  of domain calls.

  Args:
    seq1_start: int. 0-indexed inclusive.
    seq1_end: int. 0-indexed, exclusive.
    seq2_start: int. 0-indexed inclusive.
    seq2_end: int. 0-indexed, exclusive.

  Returns:
    bool.
  """
  seq1_midpoint = (seq1_start + seq1_end) / 2
  seq2_midpoint = (seq2_start + seq2_end) / 2

  return (seq1_midpoint >= seq2_start and seq1_midpoint <= seq2_end) or (
      seq2_midpoint >= seq1_start and seq2_midpoint <= seq1_end
  )


def get_known_nested_domains(
    *,
    model_cache_path,
    family_to_clan = None,
    nested_domains_filename='nested_domains_pfam35.txt',
):
  """Gets set of known domains that are nested in one another.

  Expects gunzipped file e.g. from
  http://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam32.0/database_files/nested_domains.txt.gz

  Computes both orders of this file, and includes clan mappings for each.

  Args:
    model_cache_path: path that contains downloaded savedmodels and associated
      metadata. same path that was used when installing the models via
      install_models.
    family_to_clan: Mapping from label to clan. Lifted clan semantics assumed.
    nested_domains_filename: str, pointing to txt file like the one above on
      Pfam's FTP servers.

  Returns:
    Set of pfamA_accession1, pfamA_accession2
  """
  if family_to_clan is None:
    family_to_clan = family_to_clan_mapping(
        model_cache_path=model_cache_path, use_lifted_clan_semantics=True
    )
  nested_domains_df = pd.read_csv(
      os.path.join(model_cache_path, nested_domains_filename),
      sep='\t',
      names=['class_1', 'class_2'],
  )
  nested_domains = [tuple(sorted(x)) for x in nested_domains_df.values]
  for inner, outer in list(nested_domains).copy():
    if inner == outer:
      # (Clan_0, Clan_0) doesn't presently occur in the data, but
      # if it did, it'd be hard to figure out and it would mess up downstream
      # analysis. So, we just continue.
      continue
    nested_domains.append((family_to_clan[inner], family_to_clan[outer]))

  for lhs, rhs in list(nested_domains).copy():
    nested_domains.append((rhs, lhs))

  return set(nested_domains)


def _labels_should_be_competed(
    first_label,
    second_label,
    known_nested_domains,
):
  """Decides whether one of two labels should be removed from a call.

  See `compete_clan_labels`. This function decides whether two labels overlap
  and are *not* known nested domains.

  Args:
    first_label: one of two labels that overlap.
    second_label: one of two labels that overlap.
    known_nested_domains: Set (label1, label2). Assumed that both orders of
      label1, label2 are members of this set.

  Returns:
    bool. Whether we should express a preference for one of the two labels.
  """
  family1, (start1, end1) = first_label
  family2, (start2, end2) = second_label

  if (family1, family2) in known_nested_domains:
    return False

  start1, end1 = biologist_range_to_programmer_range(start1, end1)
  start2, end2 = biologist_range_to_programmer_range(start2, end2)

  return midpoint_in_either_range(start1, end1, start2, end2)


def _choose_label_to_keep_by_length(
    first_label, second_label
):
  """Given two labels that should be competed, choose the one to keep.

  See `compete_clan_labels`, condition 2B (from docstring).
  Assumes that the two labels are not members of the same clan.

  Args:
    first_label: one of two labels that overlap.
    second_label: one of two labels that overlap.

  Returns:
    the more-preferred label of first_label, second_label.
  """
  first_label_accession, (first_label_start, first_label_end) = first_label
  second_label_accession, (second_label_start, second_label_end) = second_label

  ###### Compute various attributes that may lead to different preferences. ####
  first_label_longer = (
      first_label_end - first_label_start
      > second_label_end - second_label_start
  )
  first_label_equal = (
      first_label_end - first_label_start
      == second_label_end - second_label_start
  )
  # Whether label 1 comes from a lower pfamA family accession number.
  # (Less-than comparison is done over strings.)
  first_label_lower = first_label_accession < second_label_accession

  ###### Compute preferences based on various attributes. ######################
  prefer_first_label_due_to_length = first_label_longer
  prefer_second_label_due_to_length = (
      not first_label_longer and not first_label_equal
  )
  prefer_first_label_due_to_pfam_accession = first_label_lower
  prefer_second_label_due_to_pfam_accession = not first_label_lower

  ###### Apply preferences, in order. ##########################################
  # First, check for length.
  if prefer_first_label_due_to_length:
    return first_label
  elif prefer_second_label_due_to_length:
    return second_label
  # Then, check for an earlier Pfam family.
  elif prefer_first_label_due_to_pfam_accession:
    return first_label
  elif prefer_second_label_due_to_pfam_accession:
    return second_label
  else:
    raise ValueError(
        'Unexpected combination of preferences for competing '
        'labels: '
        f'first_label: {first_label}; '
        f'second_label: {second_label}; '
        f'first_label_longer: {first_label_longer}; '
        f'first_label_equal: {first_label_equal}; '
        f'first_label_lower: {first_label_lower}.'
    )


def _choose_label_to_keep_by_competing(
    first_label,
    second_label,
    family_to_clan,
):
  """Given two labels that should be competed, choose the one to keep.

  See `compete_clan_labels`, condition 2 (from docstring).

  Args:
    first_label: one of two labels that overlap.
    second_label: one of two labels that overlap.
    family_to_clan: Mapping from label to clan. Lifted clan semantics assumed.

  Returns:
    the more-preferred label of first_label, second_label.
  """
  # Use family_to_clan.get instead of family_to_clan.__getitem__ because
  # for clans, we want to return that same clan accession.
  first_label_clan_accession = family_to_clan.get(
      first_label[0], first_label[0]
  )
  second_label_clan_accession = family_to_clan.get(
      second_label[0], second_label[0]
  )

  labels_in_same_clan = (
      first_label_clan_accession == second_label_clan_accession
  )

  if labels_in_same_clan:
    first_label_is_clan = 'CL' in first_label[0]
    second_label_is_clan = 'CL' in second_label[0]
    if first_label_is_clan and not second_label_is_clan:
      return second_label
    elif second_label_is_clan and not first_label_is_clan:
      return first_label

  return _choose_label_to_keep_by_length(first_label, second_label)


def compete_clan_labels(
    flat_labels,
    known_nested_domains,
    family_to_clan,
):
  """Given a list of labels, chooses the best label for each range.

  Whenever a set of labels corresponds to the same region of a sequence, we have
  a choice:
  1. keep these labels (because they each provide valuable information)
  2. pick one label that's most likely to be correct

  If the labels are known to be part of nested domains, we keep them.

  Else,
  A. if two labels are part of the same clan, solve the following optimization
     problem in the set of labels within each clan, such that:
     - no pair has overlaps >=50% of residues (midpoint overlap)
     - we prefer Pfam family accessions to clan accessions.
     - the sum of the lengths is the longest.

  B. if two labels are not part of the same clan, solve the same optimization
     problem, except we no longer prefer family labels to clan labels.

  This implementation is a greedy, approximate solution to the optimization
  problem.

  Args:
    flat_labels: List (label, (start, end)).
    known_nested_domains: Set (label1, label2). Assumed that both orders of
      label1, label2 are members of this set.
    family_to_clan: Mapping from label to clan. Lifted clan semantics assumed.

  Returns:
    List (label, (start, end)).

    Whenever there is midpoint overlap between two labels, choose the longer
    one. If they are the same length, prefer the one with the smaller Pfam
    family accession.
  """

  def _sort_label_by_len(t):
    return t[1][1] - t[1][0]

  # Sort labels by length so that the function is invariant of input list order.
  old_competed_labels = sorted(
      flat_labels, key=_sort_label_by_len, reverse=True
  )
  marked_for_deletion = set()

  for first_label, second_label in itertools.combinations(
      old_competed_labels, 2
  ):
    if not _labels_should_be_competed(
        first_label, second_label, known_nested_domains
    ):
      continue

    to_keep = _choose_label_to_keep_by_competing(
        first_label, second_label, family_to_clan
    )
    if to_keep == first_label:
      marked_for_deletion.add(second_label)
    else:
      marked_for_deletion.add(first_label)

  return list(set(old_competed_labels) - marked_for_deletion)


def load_gz_json(path):
  with open(path, 'rb') as f:
    with gzip.GzipFile(fileobj=f, mode='rb') as gzip_file:
      return json.load(gzip_file)


@functools.lru_cache()
def get_pfam_vocab_with_clans_pfam_35(model_cache_path):
  with tf.io.gfile.GFile(
      os.path.join(model_cache_path, 'vocab_pfam35.tsv')
  ) as f:
    return np.array([x.strip() for x in f.readlines()])
