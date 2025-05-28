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

"""Utilities for working with sparse arrays for per-residue models."""

import collections
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse
from protenn import utils


# list of triples (i index, j index, values).
# Compatible with scipy.sparse.coo format.
# The i index is the sequence position index, and the j index is the label
# index.
# This structure is 0-indexed by residue, unlike most tools in the
# bioinformatics world.
COO_ijv_list = List[Tuple[int, int, float]]  # pylint: disable=invalid-name


# label -> list of (start index, end index).
# This structure is 1-indexed by residue (like all tools in the bioinformatics
# world), not 0-indexed, and is left-inclusive, right-inclusive.
# The reason to be 1-indexed is for better interoperability with tools like
# HMMER and InterProScan.
# See `programmer_range_to_biologist_range`.
DenseLabelDict = Dict[str, List[Tuple[int, int]]]


DEFAULT_DOMAIN_CALL_MIN_LENGTH = 20


def true_label_to_coo(true_label_tuples):
  """Converts tuples (seq_idx, class_idx) into ijv COO with "v" value 1."""
  return [(x[0], x[1], 1.) for x in true_label_tuples]


def dense_to_sparse_coo_list_of_tuples(
    twod_nparray):
  """Converts dense array to list of triples (i index, j index, values).

  Compatible with scipy.sparse.coo format.

  Args:
    twod_nparray: array.

  Returns:
    List of triples i, j, v.
  """
  to_return = []
  for nonzero_i, nonzero_j in np.array(twod_nparray.nonzero()).T:  # pylint: disable=not-an-iterable
    to_return.append((nonzero_i, nonzero_j, twod_nparray[nonzero_i, nonzero_j]))
  return to_return


def np_matrix_to_array(a):
  """Converts scipy.sparse.coo_matrix.todense() to array."""
  return np.squeeze(np.asarray(a))


def ijv_tuples_to_sparse_coo(ijv_list, sequence_length,
                             num_classes):
  """Converts list of triples (i index, j index, values) to coo_matrix.

  Args:
    ijv_list: see COO_ijv_list above.
    sequence_length: int.
    num_classes: int.

  Returns:
    coo_matrix of shape (sequence_length, num_classes)
  """
  if len(ijv_list) == 0:  # pylint: disable=g-explicit-length-test
    return scipy.sparse.coo_matrix((sequence_length, num_classes), np.float64)

  ijv_np = np.array(ijv_list)

  try:
    i = ijv_np[:, 0]
    j = ijv_np[:, 1]
    v = ijv_np[:, 2]
  except IndexError as e:
    # If there is an error, reraise it and include contents of ijv_np in the
    # stack trace to aid debugging.
    raise ValueError(ijv_np) from e
  return scipy.sparse.coo_matrix((v, (i, j)),
                                 shape=(sequence_length, num_classes))


def ijv_tuples_to_dense(ijv_list, sequence_length,
                        num_classes):
  """Converts list of triples (i index, j index, values) to dense np.array.

  Args:
    ijv_list: see COO_ijv_list above.
    sequence_length: int.
    num_classes: int.

  Returns:
    np.ndarray of shape (sequence_length, num_classes)
  """
  coo = ijv_tuples_to_sparse_coo(
      ijv_list, sequence_length=sequence_length, num_classes=num_classes)
  return np_matrix_to_array(coo.todense())


# https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
def contiguous_regions_1d(boolean_condition):
  """Finds contiguous True regions of the boolean array "boolean_condition".

  Args:
    boolean_condition: boolean array of shape (sequence_length,).

  Returns:
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    The output is 0-indexed, both by family and residue, and is left-inclusive,
    right exclusive.
  """

  # Find the indices of changes in "boolean_condition".
  d = np.diff(boolean_condition)
  (idx,) = d.nonzero()

  # We need to start things after the change in "boolean_condition". Therefore,
  # we'll shift the index by 1 to the right.
  idx += 1

  if boolean_condition[0]:
    # If the start of boolean_condition is True prepend a 0.
    idx = np.r_[0, idx]

  if boolean_condition[-1]:
    # If the end of boolean_condition is True, append the length of the array.
    idx = np.r_[idx, boolean_condition.size]

  # Reshape the result into two columns.
  idx.shape = (-1, 2)
  return idx


def normalize_ijv_tuples(
    ijv_list,
    vocab,
    applicable_label_dict,
    label_to_idx = None,
):
  """Gives, for example, clan labels for each family label.

  For each ijv, if there is an associated label that is implied by that label,
  then also return that ijv. If a clan label is implied by
  more than one other label, ties are broken by taking the max.

  Args:
    ijv_list: see COO_ijv_list above.
    vocab: 1d array of string values corresponding to label indexes.
    applicable_label_dict: Mapping from labels to their parents (including
      indirect parents). E.g. utils.family_to_clan_mapping. Note that this is
      different from proteinfer-style applicable label dicts, where more than
      one label may be implied.
    label_to_idx: optional inverted lookup of vocab. Often, this function is
      called many times, and inverting the vocabulary for each call can cause
      performance problems. In this case, one can provide a precomputed lookup.
      If not provided, vocab will be manually inverted.

  Returns:
    ijv list as described above.
  """
  if label_to_idx is None:
    label_to_idx = {v: i for i, v in enumerate(vocab)}

  seq_and_label_to_v = {}

  for ijv in ijv_list:
    seq_idx, label_idx, activation_confidence = ijv

    value_key = (seq_idx, label_idx)
    if value_key not in seq_and_label_to_v:
      seq_and_label_to_v[value_key] = activation_confidence
    elif seq_and_label_to_v[value_key] < activation_confidence:
      seq_and_label_to_v[value_key] = activation_confidence

    label = vocab[label_idx]
    if label in applicable_label_dict:
      implied_label = applicable_label_dict[label]
      implied_label_idx = label_to_idx[implied_label]
      value_key = (seq_idx, implied_label_idx)
      if value_key not in seq_and_label_to_v:
        seq_and_label_to_v[value_key] = activation_confidence
      elif seq_and_label_to_v[value_key] < activation_confidence:
        seq_and_label_to_v[value_key] = activation_confidence

  return [(i, j, v) for (i, j), v in seq_and_label_to_v.items()]


def contiguous_regions_2d(activations,
                          sequence_length,
                          vocab,
                          reporting_threshold = .5):
  """For a list of tuple activations ijv, compute contiguous domain calls.

  For each entry, consider it a call if the v in ijv > reporting_threshold.
  Then, coalesce contiguous entries (along the sequence dimension, i.e.
  fixing a particular label) for each label.

  No handling of label propagation is done in this function.

  Args:
    activations: see COO_ijv_list above.
    sequence_length: int.
    vocab: 1d array of string values corresponding to label indexes.
    reporting_threshold: float.

  Returns:
    label -> list of (start index, end index).
  """
  calls = [(x[0], x[1]) for x in activations if x[2] > reporting_threshold]
  residue_calls_by_family = collections.defaultdict(list)
  for residue_idx, label_idx in calls:
    residue_calls_by_family[vocab[label_idx]].append(residue_idx)

  domain_calls_by_family = {}
  for family, residues in residue_calls_by_family.items():
    dense = np.zeros((sequence_length,), dtype=np.bool_)
    dense[residues] = True

    # DenseLabelDict is a biologist-index-scheme based data structure.
    domain_calls_by_family[family] = [
        utils.programmer_range_to_biologist_range(x[0], x[1])
        for x in contiguous_regions_1d(dense)
    ]

  return domain_calls_by_family


def filter_domain_calls_by_length(
    calls_dict,
    min_length = DEFAULT_DOMAIN_CALL_MIN_LENGTH,
):
  """Filters out short calls from calls_dict."""
  calls_filtered = collections.defaultdict(list)
  for label, domain_ranges in calls_dict.items():
    for domain_start, domain_end in domain_ranges:

      # Add 1 because the input to this function is a DenseLabelDict,
      # which is 1-indexed, right inclusive.
      if domain_end - domain_start + 1 >= min_length:
        calls_filtered[label].append((domain_start, domain_end))

  # Convert from a defaultdict to a dict so as not to confuse
  # downstream users with weird behavior for new keys.
  calls_filtered = dict(calls_filtered.items())

  return calls_filtered


def activations_to_domain_calls(
    activations,
    sequence_length,
    vocab,
    reporting_threshold = 0.025,
    min_domain_call_length = DEFAULT_DOMAIN_CALL_MIN_LENGTH,
):
  """Convert activations to dict of domain calls."""
  domain_calls = contiguous_regions_2d(activations, sequence_length, vocab,
                                       reporting_threshold)
  return filter_domain_calls_by_length(domain_calls, min_domain_call_length)


def num_labels_in_dense_label_dict(d):
  count = 0
  for ranges in d.values():
    count += len(ranges)
  return count


def flatten_dict_of_domain_calls(
    calls_dict):
  """Flattens label -> list[(start, end)] to list[(label, (start, end))]."""
  to_return = []
  for family, ranges in calls_dict.items():
    for r in ranges:
      to_return.append((family, tuple(r)))
  return to_return
