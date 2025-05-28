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

"""Custom `seqio`-based `FeatureConverter` implementations."""

import enum
from typing import Mapping

import seqio
import tensorflow as tf


# Type aliases.
FeatureConverter = seqio.FeatureConverter
FeatureSpec = FeatureConverter.FeatureSpec


class DEDALMaskedLMFeatureConverter(FeatureConverter):
  """Provides examples for DEDAL's masked language modelling task."""

  TASK_FEATURES = {
      'inputs': FeatureSpec(dtype=tf.int32),
      'targets': FeatureSpec(dtype=tf.int32),
      'noise_mask': FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      'encoder_input_tokens': FeatureSpec(dtype=tf.int32),
      'encoder_target_tokens': FeatureSpec(dtype=tf.int32),
      'encoder_loss_weights': FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      'encoder_segment_ids': tf.int32,
      'encoder_positions': tf.int32,
  }

  def __init__(
      self,
      pack = True,
      use_custom_packing_ops = False,
  ):
    super().__init__(pack=pack, use_custom_packing_ops=use_custom_packing_ops)

  def _convert_features(
      self,
      ds,
      input_lengths,
  ):
    """Convert the input dataset to an output dataset to be fed to the model.

    The conversion process involves three steps
    1. Each feature in the `input_lengths` is packed.
    2. 'inputs' fields are mapped to the encoder input and 'targets' are mapped
        to encoder target. Loss is taken only on the masked positions just as in
        Masked Language Modeling objective.

    Args:
      ds: an input tf.data.Dataset to be converted.
      input_lengths: a mapping from a feature to its length.

    Returns:
      ds: the converted dataset.
    """

    @seqio.map_over_dataset
    def convert_example(
        example,
    ):
      output = {
          'encoder_input_tokens': example['inputs'],
          'encoder_target_tokens': example['targets'],
          'encoder_loss_weights': tf.where(example['noise_mask'] == 1, 1, 0),
      }

      if self.pack:
        output['encoder_segment_ids'] = example['inputs_segment_ids']
        output['encoder_positions'] = example['inputs_positions']

      return output

    ds = self._pack_or_pad(ds, input_lengths)
    return convert_example(ds)

  def get_model_feature_lengths(
      self,
      task_feature_lengths,
  ):
    """Define the length relationship between input and output features."""
    inputs_length = task_feature_lengths['inputs']
    model_feature_lengths = {
        'encoder_input_tokens': inputs_length,
        'encoder_target_tokens': task_feature_lengths['targets'],
        'encoder_loss_weights': task_feature_lengths['noise_mask'],
    }
    if self.pack:
      model_feature_lengths['encoder_segment_ids'] = inputs_length
      model_feature_lengths['encoder_positions'] = inputs_length

    return model_feature_lengths


class State(enum.IntEnum):
  MATCH = 0
  GAP_OPEN = 1
  GAP_EXTEND = 2


class AlignmentFeatureConverter(FeatureConverter):
  """Provides examples for DEDAL's pairwise sequence alignment task."""

  TASK_FEATURES = {
      'sequence_x': FeatureSpec(dtype=tf.int32),
      'sequence_y': FeatureSpec(dtype=tf.int32),
      'states': FeatureSpec(dtype=tf.int32),
  }

  MODEL_FEATURES = {
      'sequence_x': FeatureSpec(dtype=tf.int32),
      'sequence_y': FeatureSpec(dtype=tf.int32),
      'align_path_x': FeatureSpec(dtype=tf.int32),
      'align_path_y': FeatureSpec(dtype=tf.int32),
      'align_path_s': FeatureSpec(dtype=tf.int32),
  }

  METADATA_FEATURES = {
      'key_x': FeatureSpec(dtype=tf.string),
      'key_y': FeatureSpec(dtype=tf.string),
      'extended_key_x': FeatureSpec(dtype=tf.string),
      'extended_key_y': FeatureSpec(dtype=tf.string),
      'pfam_acc_x': FeatureSpec(dtype=tf.string),
      'pfam_acc_y': FeatureSpec(dtype=tf.string),
      'clan_acc_x': FeatureSpec(dtype=tf.string),
      'clan_acc_y': FeatureSpec(dtype=tf.string),
      'percent_identity': FeatureSpec(dtype=tf.float32),
      'maybe_confounded': FeatureSpec(dtype=tf.bool),
      'fallback': FeatureSpec(dtype=tf.bool),
      'shares_clans_in_flanks': FeatureSpec(dtype=tf.bool),
      'shares_coiled_coil_in_flanks': FeatureSpec(dtype=tf.bool),
      'shares_disorder_in_flanks': FeatureSpec(dtype=tf.bool),
      'shares_low_complexity_in_flanks': FeatureSpec(dtype=tf.bool),
      'shares_sig_p_in_flanks': FeatureSpec(dtype=tf.bool),
      'shares_transmembrane_in_flanks': FeatureSpec(dtype=tf.bool),
  }

  def __init__(
      self,
      vocab,
      dtype = tf.int32,
      encode_by_transition_type = False,
      include_metadata_features = True,
  ):
    super().__init__(pack=False, use_custom_packing_ops=False)
    self.vocab = vocab
    self.dtype = dtype
    self.encode_by_transition_type = encode_by_transition_type
    self.include_metadata_features = include_metadata_features
    if self.include_metadata_features:
      self.MODEL_FEATURES.update(self.METADATA_FEATURES)

    self._start = self.vocab.encode('S')[0]
    self._match = self.vocab.encode('M')[0]
    self._gap_in_x = self.vocab.encode('X')[0]
    self._gap_in_y = self.vocab.encode('Y')[0]
    self._pad_id = self.vocab.pad_id

    look_up = {
        (self._start, self._match): State.MATCH,
        (self._match, self._match): State.MATCH,
        (self._gap_in_x, self._match): State.MATCH,
        (self._gap_in_y, self._match): State.MATCH,
        (self._match, self._gap_in_x): State.GAP_OPEN,
        (self._gap_in_x, self._gap_in_x): State.GAP_EXTEND,
        (self._match, self._gap_in_y): State.GAP_OPEN,
        (self._gap_in_x, self._gap_in_y): State.GAP_OPEN,
        (self._gap_in_y, self._gap_in_y): State.GAP_EXTEND,
        (self._gap_in_y, self._gap_in_x): State.GAP_OPEN,  # "forbidden" case.
    }

    if self.encode_by_transition_type:
      codes = list(range(len(look_up)))  # Treats each transition differently.
    else:
      codes = list(look_up.values())  # Merges transitions by state type.

    self._hash_fn = lambda d0, d1: d0 + d1 * self.vocab.vocab_size

    self._trans_encoder = tf.scatter_nd(
        indices=[[self._hash_fn(d0, d1)] for d0, d1 in look_up],
        updates=tf.convert_to_tensor(codes, self.dtype),
        shape=[self.vocab.vocab_size ** 2])

  def _convert_features(
      self,
      ds,
      task_feature_lengths,
  ):
    """Convert the input dataset to an output dataset to be fed to the model.

    The conversion process involves three steps
    1. The pair of sequences and any of the requested metadata are fetched from
       `example`.
    2. The state sequence is encoded as an alignment path.
    3. The sequences and alignment path are padded. Packing is not yet
       supported.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from a feature to its length.

    Returns:
      ds: the converted dataset.
    """

    @seqio.map_over_dataset
    def convert_example(
        example,
    ):
      out = {'sequence_x': example['sequence_x'],
             'sequence_y': example['sequence_y']}
      if self.include_metadata_features:
        for k in self.METADATA_FEATURES:
          out[k] = tf.reshape(example[k], [-1])

      states = example['states']
      ali_start_x, ali_start_y = example['ali_start_x'], example['ali_start_y']

      matches = tf.cast(states[1:] == self._match, self.dtype)
      gaps_in_x = tf.cast(states[1:] == self._gap_in_x, self.dtype)
      gaps_in_y = tf.cast(states[1:] == self._gap_in_y, self.dtype)
      padding_mask = states[1:] != self._pad_id

      out['align_path_x'] = tf.where(
          padding_mask, ali_start_x - 1 + tf.cumsum(matches + gaps_in_y), 0)
      out['align_path_y'] = tf.where(
          padding_mask, ali_start_y - 1 + tf.cumsum(matches + gaps_in_x), 0)
      out['align_path_s'] = tf.where(
          padding_mask,
          tf.gather(
              params=self._trans_encoder,
              indices=self._hash_fn(states[:-1], states[1:])),
          0)

      return out

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return convert_example(ds)

  def get_model_feature_lengths(
      self,
      task_feature_lengths,
  ):
    align_path_length = task_feature_lengths['states'] - 1
    model_feature_lengths = {
        'sequence_x': task_feature_lengths['sequence_x'],
        'sequence_y': task_feature_lengths['sequence_y'],
        'align_path_x': align_path_length,
        'align_path_y': align_path_length,
        'align_path_s': align_path_length,
    }
    if self.include_metadata_features:
      for k in self.METADATA_FEATURES:
        model_feature_lengths[k] = 1
    return model_feature_lengths


class HomologyFeatureConverter(FeatureConverter):
  """Provides examples for DEDAL's pairwise homology detection task."""

  TASK_FEATURES = {
      'sequence_x': FeatureSpec(dtype=tf.int32),
      'sequence_y': FeatureSpec(dtype=tf.int32),
  }

  MODEL_FEATURES = {
      'sequence_x': FeatureSpec(dtype=tf.int32),
      'sequence_y': FeatureSpec(dtype=tf.int32),
      'homology_label': FeatureSpec(dtype=tf.int32),
  }

  METADATA_FEATURES = {
      'key_x': FeatureSpec(dtype=tf.string),
      'key_y': FeatureSpec(dtype=tf.string),
      'extended_key_x': FeatureSpec(dtype=tf.string),
      'extended_key_y': FeatureSpec(dtype=tf.string),
      'pfam_acc_x': FeatureSpec(dtype=tf.string),
      'pfam_acc_y': FeatureSpec(dtype=tf.string),
      'clan_acc_x': FeatureSpec(dtype=tf.string),
      'clan_acc_y': FeatureSpec(dtype=tf.string),
      'percent_identity': FeatureSpec(dtype=tf.float32),
      'maybe_confounded': FeatureSpec(dtype=tf.bool),
      'shares_clans': FeatureSpec(dtype=tf.bool),
      'shares_coiled_coil': FeatureSpec(dtype=tf.bool),
      'shares_disorder': FeatureSpec(dtype=tf.bool),
      'shares_low_complexity': FeatureSpec(dtype=tf.bool),
      'shares_sig_p': FeatureSpec(dtype=tf.bool),
      'shares_transmembrane': FeatureSpec(dtype=tf.bool),
  }

  def __init__(
      self,
      fine_grained_labels = True,
      include_metadata_features = True,
  ):
    super().__init__(pack=False, use_custom_packing_ops=False)
    self.fine_grained_labels = fine_grained_labels
    self.include_metadata_features = include_metadata_features
    if self.include_metadata_features:
      self.MODEL_FEATURES.update(self.METADATA_FEATURES)

  def _convert_features(
      self,
      ds,
      task_feature_lengths,
  ):
    """Convert the input dataset to an output dataset to be fed to the model.

    The conversion process involves three steps
    1. The pair of sequences and any of the requested metadata are fetched from
       `example`.
    2. The homology label is fetched from `example` and, if binarized if
       `self.fine_grained_labels` is `True`.
    3. The sequences are padded. Packing is not yet supported.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from a feature to its length.

    Returns:
      ds: the converted dataset.
    """

    @seqio.map_over_dataset
    def convert_example(
        example,
    ):
      homology_label = tf.reshape(example['homology_label'], [-1])
      if not self.fine_grained_labels:
        homology_label = tf.cast(homology_label > 0, homology_label.dtype)
      out = {'sequence_x': example['sequence_x'],
             'sequence_y': example['sequence_y'],
             'homology_label': homology_label}
      if self.include_metadata_features:
        for k in self.METADATA_FEATURES:
          out[k] = tf.reshape(example[k], [-1])

      return out

    ds = self._pack_or_pad(ds, task_feature_lengths)
    return convert_example(ds)

  def get_model_feature_lengths(
      self,
      task_feature_lengths,
  ):
    model_feature_lengths = {
        'sequence_x': task_feature_lengths['sequence_x'],
        'sequence_y': task_feature_lengths['sequence_y'],
        'homology_label': 1,
    }
    if self.include_metadata_features:
      for k in self.METADATA_FEATURES:
        model_feature_lengths[k] = 1
    return model_feature_lengths
