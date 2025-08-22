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

"""Compute activations for trained model from input sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse
import six
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tqdm

from protenn import per_residue_sparse
from protenn import utils


def call_module(module, one_hots, row_lengths, signature):
  """Call a tf_hub.Module using the standard blundell signature.

  This expects that `module` has a signature named `signature` which conforms to
      ('sequence',
       'sequence_length') -> output

  To use an existing SavedModel
  file you may want to create a module_spec with
  `tensorflow_hub.saved_model_module.create_module_spec_from_saved_model`.

  Args:
    module: a tf_hub.Module to call.
    one_hots: a rank 3 tensor with one-hot encoded sequences of residues.
    row_lengths: a rank 1 tensor with sequence lengths.
    signature: the graph signature to validate and call.

  Returns:
    The output tensor of `module`.
  """
  if signature not in module.get_signature_names():
    raise ValueError('signature not in ' +
                     six.ensure_str(str(module.get_signature_names())) +
                     '. Was ' + six.ensure_str(signature) + '.')
  inputs = module.get_input_info_dict(signature=signature)
  expected_inputs = [
      'sequence',
      'sequence_length',
  ]
  if set(inputs.keys()) != set(expected_inputs):
    raise ValueError(
        'The signature_def does not have the expected inputs. Please '
        'reconfigure your saved model to only export signatures '
        'with sequence and length inputs. (Inputs were %s, expected %s)' %
        (str(inputs), str(expected_inputs)))

  outputs = module.get_output_info_dict(signature=signature)
  if len(outputs) > 1:
    raise ValueError('The signature_def given has more than one output. Please '
                     'reconfigure your saved model to only export signatures '
                     'with one output. (Outputs were %s)' % str(outputs))

  return list(
      module({
          'sequence': one_hots,
          'sequence_length': row_lengths,
      },
             signature=signature,
             as_dict=True).values())[0]


def in_graph_inferrer(sequences,
                      savedmodel_dir_path,
                      signature,
                      name_scope='inferrer'):
  """Add an in-graph inferrer to the active default graph.

  Additionally performs in-graph preprocessing, splitting strings, and encoding
  residues.

  Args:
    sequences: A tf.string Tensor representing a batch of sequences with shape
      [None].
    savedmodel_dir_path: Path to the directory with the SavedModel binary.
    signature: Name of the signature to use in `savedmodel_dir_path`. e.g.
      'pooled_representation'
    name_scope: Name scope to use for the loaded saved model.

  Returns:
    Output Tensor
  Raises:
    ValueError if signature does not conform to
      ('sequence',
       'sequence_length') -> output
    or if the specified signature is not present.
  """
  # Add variable to make it easier to refactor with multiple tags in future.
  tags = [tf.saved_model.tag_constants.SERVING]

  # Tokenization
  residues = tf.strings.unicode_split(sequences, 'UTF-8')
  # Convert to one-hots and pad.
  one_hots, row_lengths = utils.in_graph_residues_to_onehot(residues)
  module_spec = hub.saved_model_module.create_module_spec_from_saved_model(
      savedmodel_dir_path)
  module = hub.Module(module_spec, trainable=False, tags=tags, name=name_scope)
  return call_module(module, one_hots, row_lengths, signature)


@functools.lru_cache(maxsize=None)
def memoized_inferrer(
    savedmodel_dir_path,
    activation_type='label',
    use_tqdm=False,
    session_config=None,
    memoize_inference_results=False,
    use_latest_savedmodel=False,
):
  """Alternative constructor for Inferrer that is memoized."""
  return Inferrer(
      savedmodel_dir_path=savedmodel_dir_path,
      activation_type=activation_type,
      use_tqdm=use_tqdm,
      session_config=session_config,
      memoize_inference_results=memoize_inference_results,
      use_latest_savedmodel=use_latest_savedmodel,
  )


class Inferrer(object):
  """Uses a SavedModel to provide inference."""

  def __init__(
      self,
      savedmodel_dir_path,
      activation_type='label',
      use_tqdm=False,
      session_config=None,
      memoize_inference_results=False,
      use_latest_savedmodel=False,
  ):
    """Construct Inferrer.

    Args:
      savedmodel_dir_path: path to directory where a SavedModel pb or pbtxt is
        stored. The SavedModel must only have one input per signature and only
        one output per signature.
      activation_type: one of the keys in saved_model.signature_def.keys().
      use_tqdm: Whether to print progress using tqdm.
      session_config: tf.ConfigProto for tf.Session creation.
      memoize_inference_results: if True, calls to inference.get_activations
        will be memoized.
      use_latest_savedmodel: If True, the model will be loaded from
        latest_savedmodel_path_from_base_path(savedmodel_dir_path).

    Raises:
      ValueError: if activation_type is not the name of a signature_def in the
        SavedModel.
      ValueError: if SavedModel.signature_def[activation_type] has an input
        other than 'sequence'.
      ValueError: if SavedModel.signature_def[activation_type] has more than
        one output.
    """
    if use_latest_savedmodel:
      savedmodel_dir_path = latest_savedmodel_path_from_base_path(
          savedmodel_dir_path
      )
    self._graph = tf.Graph()
    self._model_name_scope = 'inferrer'
    with self._graph.as_default():
      self._sequences = tf.placeholder(
          shape=[None], dtype=tf.string, name='sequences')
      self._fetch = in_graph_inferrer(
          self._sequences,
          savedmodel_dir_path,
          activation_type,
          name_scope=self._model_name_scope)
      self._sess = tf.Session(
          config=session_config if session_config else tf.ConfigProto())
      self._sess.run([
          tf.initializers.global_variables(),
          tf.initializers.local_variables(),
          tf.initializers.tables_initializer(),
      ])

    self._savedmodel_dir_path = savedmodel_dir_path
    self.activation_type = activation_type
    self._use_tqdm = use_tqdm
    if memoize_inference_results:
      self._get_activations_for_batch = self._get_activations_for_batch_memoized
    else:
      self._get_activations_for_batch = (
          self._get_activations_for_batch_unmemoized
      )

  def __repr__(self):
    return ('{} with feed tensors savedmodel_dir_path {} and '
            'activation_type {}').format(
                type(self).__name__, self._savedmodel_dir_path,
                self.activation_type)

  def _get_tensor_by_name(self, name):
    return self._graph.get_tensor_by_name('{}/{}'.format(
        self._model_name_scope, name))

  def _get_activations_for_batch_unmemoized(
      self, seq, custom_tensor_to_retrieve=None
  ):
    """Gets activations for sequence.

      [activation_1, activation_2, ...]

    In the case that the activations are the normalized probabilities that a
    sequence belongs to a class, entry `i, j` of
    `inferrer.get_activations(sequence)` contains the probability that
    sequence `i` is in family `j`.

    Args:
      seq: string with characters that are amino acids.
      custom_tensor_to_retrieve: string name for a tensor to retrieve, if unset
        uses default for signature.

    Returns:
      np.array of floats containing the value from fetch_op.
    """
    if custom_tensor_to_retrieve:
      fetch = self._get_tensor_by_name(custom_tensor_to_retrieve)
    else:
      fetch = self._fetch
    with self._graph.as_default():
      return self._sess.run(fetch, {self._sequences: np.array([seq])})[0]

  @functools.lru_cache(maxsize=1_000_000)
  def _get_activations_for_batch_memoized(
      self, seq, custom_tensor_to_retrieve=None
  ):
    return self._get_activations_for_batch_unmemoized(
        seq, custom_tensor_to_retrieve
    )

  def get_activations(self, list_of_seqs, custom_tensor_to_retrieve=None):
    """Gets activations where batching may be needed to avoid OOM.

    Inputs are strings of amino acids, outputs are activations from the network.

    Args:
      list_of_seqs: iterable of strings as input for inference.
      custom_tensor_to_retrieve: string name for a tensor to retrieve, if unset
        uses default for signature.

    Returns:
      ragged list of activations, one for each sequence.
      If the Inferrer is computing Pfam labels, the shape of each output (for
      each sequence) is num_pfam_classes x num_residues.
    """
    np_seqs = np.array(list_of_seqs, dtype=np.str_)
    if np_seqs.size == 0:
      return np.array([], dtype=float)

    if len(np_seqs.shape) != 1:
      raise ValueError('`list_of_seqs` should be convertible to a numpy vector '
                       'of strings. Got {}'.format(np_seqs))

    lengths = np.array([len(seq) for seq in np_seqs])
    # Sort by reverse length, so that the longest element is first.
    # This is because the longest element can cause memory issues, and we'd like
    # to fail-fast in this case.
    sorter = np.argsort(lengths)[::-1]
    # The inverse of a permutation A is the permutation B such that B(A) is the
    # the identity permutation (a sorted list).
    reverser = np.argsort(sorter)

    activation_list = []
    if self._use_tqdm:
      list_of_seqs = tqdm.tqdm(
          list_of_seqs,
          position=0,
          desc='Annotating batches of sequences',
          leave=True,
          dynamic_ncols=True,
      )
    for sequence in list_of_seqs:
      batch_activations = self._get_activations_for_batch(
          sequence, custom_tensor_to_retrieve=custom_tensor_to_retrieve
      )

      activation_list.append(batch_activations)

    activations_list = [activation_list[i] for i in reverser]

    return activations_list

  def get_variable(self, variable_name):
    """Gets the value of a variable from the graph.

    Args:
      variable_name: string name for retrieval. E.g. "vocab_name:0"

    Returns:
      output from TensorFlow from attempt to retrieve this value.
    """
    with self._graph.as_default():
      return self._sess.run(self._get_tensor_by_name(variable_name))


def latest_savedmodel_path_from_base_path(base_path):
  """Get the most recent savedmodel from a base directory path."""

  protein_export_base_path = os.path.join(base_path, 'export/protein_exporter')

  suffixes = [
      x for x in tf.io.gfile.listdir(protein_export_base_path)
      if 'temp-' not in x
  ]

  if not suffixes:
    raise ValueError('No SavedModels found in %s' % protein_export_base_path)

  # Sort by suffix to take the model corresponding the most
  # recent training step.
  return os.path.join(protein_export_base_path, sorted(suffixes)[-1])


def predictions_for_df(df, inferrer):
  """Returns df with column that's the activations for each sequence.

  Args:
    df: DataFrame with columns 'sequence' and 'sequence_name'.
    inferrer: inferrer.

  Returns:
    pd.DataFrame with columns 'sequence_name', 'predicted_label', and
    'predictions'. 'predictions' has type np.ndarray, whose shape depends on
    inferrer.activation_type.
  """
  working_df = df.copy()
  working_df['predictions'] = inferrer.get_activations(
      working_df.sequence.values).tolist()  # pytype: disable=attribute-error
  return working_df


def mean_sparse_acts(
    sequence_length,
    acts,
    num_output_classes,
):
  """Averages a list of sparse activations.

  To save memory, the outputted activations are rounded to 2 decimal places.
  If the outputs are zero when rounded, these are omitted when writing to disk,
  as they are implied.

  Args:
    sequence_length: int.
    acts: list of ijv activations for the sequence from different inferrers.
    num_output_classes: int.

  Returns:
    meaned activations.
  """
  coo_all_ens_els = [
      per_residue_sparse.ijv_tuples_to_sparse_coo(
          act, sequence_length, num_output_classes
      )
      for act in acts
  ]

  mean_acts = scipy.sparse.coo_matrix.mean(np.array(coo_all_ens_els), axis=0)
  mean_acts_rounded = np.around(mean_acts, 4).tocoo()
  mean_acts_rounded.eliminate_zeros()
  if mean_acts_rounded.nnz == 0:
    return []
  mean_acts_sparse = list(
      zip(mean_acts_rounded.row, mean_acts_rounded.col, mean_acts_rounded.data)
  )

  return mean_acts_sparse


def get_sparse_calls_by_inferrer(
    sequences, inferrer_list
):
  """For each inferrer, get list of sparse calls for each sequence.

  Args:
    sequences: amino acid sequences.
    inferrer_list: list of inferrers.

  Returns:
    Outer dim corresponds to inferrer_list.
    Inner dim corresponds to sequences.
  """
  sparse_calls_by_inferrer = []
  for inferrer_idx, inferrer in enumerate(inferrer_list):
    inferrer_calls = []
    for seq in tqdm.tqdm(
        sequences,
        position=0,
        desc=f'Predicting for model {inferrer_idx+1} of {len(inferrer_list)}',
    ):
      class_probabilities = inferrer.get_activations([seq])[0]

      # Rounding to a few decimal places substantially decreases the size of the
      # output of this step, preventing some OOMs.
      sparse_class_probabilities = np.around(class_probabilities, 4)

      inferrer_calls.append(
          per_residue_sparse.dense_to_sparse_coo_list_of_tuples(
              sparse_class_probabilities
          )
      )
    sparse_calls_by_inferrer.append(inferrer_calls)
  return sparse_calls_by_inferrer


def get_competed_labels(
    *,
    sparse_act,
    sequence_length,
    family_to_clan,
    label_to_idx,
    vocab,
    known_nested_domains,
    reporting_threshold,
    min_domain_call_length,
):
  """Normalizes, generates domain calls, and competes them for a single seq.

  Args:
    sparse_act: e.g. mean activations of a bunch of ensemble elements.
    sequence_length: int.
    family_to_clan: lookup from family to clan, lifted clan semantics assumed.
    label_to_idx: lookup of label to index of label within vocab.
    vocab: 1d array of pfam family and clan labels.
    known_nested_domains: See utils.get_known_nested_domains.
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
    min_domain_call_length: don't consider as valid any domain calls shorter
      than this length.

  Returns:
    List of domain calls.
  """
  normalized_activations = per_residue_sparse.normalize_ijv_tuples(
      sparse_act, vocab, family_to_clan, label_to_idx
  )

  predicted_labels = per_residue_sparse.activations_to_domain_calls(
      normalized_activations,
      sequence_length=sequence_length,
      vocab=vocab,
      reporting_threshold=reporting_threshold,
      min_domain_call_length=min_domain_call_length,
  )

  flattened_labels = per_residue_sparse.flatten_dict_of_domain_calls(
      predicted_labels
  )
  competed_labels = utils.compete_clan_labels(
      flattened_labels, known_nested_domains, family_to_clan
  )
  return competed_labels


def get_preds_at_or_above_threshold(
    *,
    input_df,
    inferrer_list,
    model_cache_path,
    reporting_threshold,
    min_domain_call_length = 20,
):
  """For each input sequence, returns list of domain calls for that sequence.

  Args:
    input_df: DataFrame with columns 'sequence' and 'sequence_name'.
    inferrer_list: list of Pfam inferrers.
    model_cache_path: path that contains downloaded SavedModels and associated
      metadata. Same path that was used when installing the models via
      install_models.
    reporting_threshold: report labels with mean confidence across ensemble
      elements that exceeds this threshold.
    min_domain_call_length: don't consider as valid any domain calls shorter
      than this length.

  Returns:
    List (ordered same as input_df) of domain calls for each input sequence.
  """
  vocab = utils.get_pfam_vocab_with_clans_pfam_35(model_cache_path)
  family_to_clan = utils.family_to_clan_mapping(
      model_cache_path=model_cache_path,
      use_lifted_clan_semantics=True,
  )
  known_nested_domains = utils.get_known_nested_domains(
      model_cache_path=model_cache_path
  )
  label_to_idx = {v: i for i, v in enumerate(vocab)}

  sparse_calls_by_inferrer = get_sparse_calls_by_inferrer(
      input_df.sequence, inferrer_list
  )

  # Transpose list-of-lists so that we can examine all calls for a particular
  # sequence at the same time.
  sparse_calls_by_sequence = list(zip(*sparse_calls_by_inferrer))

  num_output_classes = len(vocab)
  meaned_sparse_acts = [
      mean_sparse_acts(len(sequence), acts, num_output_classes)
      for sequence, acts in zip(input_df.sequence, sparse_calls_by_sequence)
  ]

  to_return = []
  for sequence, sparse_act in zip(input_df.sequence, meaned_sparse_acts):
    to_return.append(
        get_competed_labels(
            sparse_act=sparse_act,
            sequence_length=len(sequence),
            family_to_clan=family_to_clan,
            label_to_idx=label_to_idx,
            vocab=vocab,
            known_nested_domains=known_nested_domains,
            reporting_threshold=reporting_threshold,
            min_domain_call_length=min_domain_call_length,
        )
    )

  return to_return
