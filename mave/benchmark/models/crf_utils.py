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

# This file is branched from 'tensorflow_addons/text/crf.py'
# As of 2022-11-15, the usage of `tf.cond` in this file does not support JIT
# compile when using TPU. So they are removed, as for our case, the sequencce
# length is always > 1.
"""Utils for Conditional Random Field layer."""
from typing import Any, Optional, Tuple
import warnings

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


def crf_filtered_inputs(inputs,
                        tag_bitmap):
  """Constrains the inputs to filter out certain tags at each time step.

  tag_bitmap limits the allowed tags at each input time step.
  This is useful when an observed output at a given time step needs to be
  constrained to a selected set of tags.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
    tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
      representing all active tags at each index for which to calculate the
      unnormalized score.

  Returns:
    filtered_inputs: A [batch_size] vector of unnormalized sequence scores.
  """

  # set scores of filtered out inputs to be -inf.
  filtered_inputs = tf.where(
      tag_bitmap,
      inputs,
      tf.fill(tf.shape(inputs), tf.cast(float("-inf"), inputs.dtype)),
  )
  return filtered_inputs


def crf_sequence_score(
    inputs,
    tag_indices,
    sequence_lengths,
    transition_params,
):
  """Computes the unnormalized score for a tag sequence.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
      we compute the unnormalized score.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.

  Returns:
    sequence_scores: A [batch_size] vector of unnormalized sequence scores.
  """
  tag_indices = tf.cast(tag_indices, dtype=tf.int32)
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

  def _multi_seq_fn():
    # Compute the scores of the given tag sequence.
    unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
    binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                     transition_params)
    sequence_scores = unary_scores + binary_scores
    return sequence_scores

  return _multi_seq_fn()


def crf_multitag_sequence_score(
    inputs,
    tag_bitmap,
    sequence_lengths,
    transition_params,
):
  """Computes the unnormalized score of all tag sequences matching tag_bitmap.

  tag_bitmap enables more than one tag to be considered correct at each time
  step. This is useful when an observed output at a given time step is
  consistent with more than one tag, and thus the log likelihood of that
  observation must take into account all possible consistent tags.

  Using one-hot vectors in tag_bitmap gives results identical to
  crf_sequence_score.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
    tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
      representing all active tags at each index for which to calculate the
      unnormalized score.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.

  Returns:
    sequence_scores: A [batch_size] vector of unnormalized sequence scores.
  """
  tag_bitmap = tf.cast(tag_bitmap, dtype=tf.bool)
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
  filtered_inputs = crf_filtered_inputs(inputs, tag_bitmap)

  # If max_seq_len is 1, we skip the score calculation and simply gather the
  # unary potentials of all active tags.
  def _single_seq_fn():
    return tf.reduce_logsumexp(filtered_inputs, axis=[1, 2], keepdims=False)

  def _multi_seq_fn():
    # Compute the logsumexp of all scores of sequences
    # matching the given tags.
    return crf_log_norm(
        inputs=filtered_inputs,
        sequence_lengths=sequence_lengths,
        transition_params=transition_params,
    )

  return tf.cond(
      tf.equal(tf.shape(inputs)[1], 1), _single_seq_fn, _multi_seq_fn)


def crf_log_norm(inputs, sequence_lengths,
                 transition_params):
  """Computes the normalization for a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix.

  Returns:
    log_norm: A [batch_size] vector of normalizers for a CRF.
  """
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
  # Split up the first and rest of the inputs in preparation for the forward
  # algorithm.
  first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
  first_input = tf.squeeze(first_input, [1])

  def _multi_seq_fn():
    """Forward computation of alpha values."""
    rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
    # Compute the alpha values in the forward algorithm in order to get the
    # partition function.

    alphas = crf_forward(rest_of_input, first_input, transition_params,
                         sequence_lengths)
    log_norm = tf.reduce_logsumexp(alphas, [1])
    # Mask `log_norm` of the sequences with length <= zero.
    log_norm = tf.where(
        tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm), log_norm)
    return log_norm

  return _multi_seq_fn()


def crf_log_likelihood(
    inputs,
    tag_indices,
    sequence_lengths,
    transition_params = None,
):
  """Computes the log-likelihood of tag sequences in a CRF.

  Args:
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
      we compute the log-likelihood.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] transition matrix, if available.

  Returns:
    log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
      each example, given the sequence of tag indices.
    transition_params: A [num_tags, num_tags] transition matrix. This is
        either provided by the caller or created in this function.
  """
  inputs = tf.convert_to_tensor(inputs)

  num_tags = inputs.shape[2]

  # cast type to handle different types
  tag_indices = tf.cast(tag_indices, dtype=tf.int32)
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

  if transition_params is None:
    initializer = tf.keras.initializers.GlorotUniform()
    transition_params = tf.Variable(
        initializer([num_tags, num_tags]), "transitions")
  transition_params = tf.cast(transition_params, inputs.dtype)
  sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                       transition_params)
  log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

  # Normalize the scores to get the log-likelihood per example.
  log_likelihood = sequence_scores - log_norm
  return log_likelihood, transition_params


def crf_unary_score(tag_indices, sequence_lengths,
                    inputs):
  """Computes the unary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.

  Returns:
    unary_scores: A [batch_size] vector of unary scores.
  """
  tag_indices = tf.cast(tag_indices, dtype=tf.int32)
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

  batch_size = tf.shape(inputs)[0]
  max_seq_len = tf.shape(inputs)[1]
  num_tags = tf.shape(inputs)[2]

  flattened_inputs = tf.reshape(inputs, [-1])

  offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
  offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
  # Use int32 or int64 based on tag_indices' dtype.
  if tag_indices.dtype == tf.int64:
    offsets = tf.cast(offsets, tf.int64)
  flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

  unary_scores = tf.reshape(
      tf.gather(flattened_inputs, flattened_tag_indices),
      [batch_size, max_seq_len])

  masks = tf.sequence_mask(
      sequence_lengths,
      maxlen=tf.shape(tag_indices)[1],
      dtype=unary_scores.dtype)

  unary_scores = tf.reduce_sum(unary_scores * masks, 1)
  return unary_scores


def crf_binary_score(tag_indices, sequence_lengths,
                     transition_params):
  """Computes the binary scores of tag sequences.

  Args:
    tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
    sequence_lengths: A [batch_size] vector of true sequence lengths.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    binary_scores: A [batch_size] vector of binary scores.
  """
  tag_indices = tf.cast(tag_indices, dtype=tf.int32)
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

  num_tags = tf.shape(transition_params)[0]
  num_transitions = tf.shape(tag_indices)[1] - 1

  # Truncate by one on each side of the sequence to get the start and end
  # indices of each transition.
  start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
  end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

  # Encode the indices in a flattened representation.
  flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
  flattened_transition_params = tf.reshape(transition_params, [-1])

  # Get the binary scores based on the flattened representation.
  binary_scores = tf.gather(flattened_transition_params,
                            flattened_transition_indices)

  masks = tf.sequence_mask(
      sequence_lengths,
      maxlen=tf.shape(tag_indices)[1],
      dtype=binary_scores.dtype)
  truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
  binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
  return binary_scores


def crf_forward(
    inputs,
    state,
    transition_params,
    sequence_lengths,
):
  """Computes the alpha values in a linear-chain CRF.

  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

  Args:
    inputs: A [batch_size, num_tags] matrix of unary potentials.
    state: A [batch_size, num_tags] matrix containing the previous alpha
      values.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
      This matrix is expanded into a [1, num_tags, num_tags] in preparation
      for the broadcast summation occurring within the cell.
    sequence_lengths: A [batch_size] vector of true sequence lengths.

  Returns:
    new_alphas: A [batch_size, num_tags] matrix containing the
        new alpha values.
  """
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

  last_index = tf.maximum(
      tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)
  inputs = tf.transpose(inputs, [1, 0, 2])
  transition_params = tf.expand_dims(transition_params, 0)

  def _scan_fn(scan_state, scan_inputs):
    scan_state = tf.expand_dims(scan_state, 2)
    transition_scores = scan_state + transition_params
    new_alphas = scan_inputs + tf.reduce_logsumexp(transition_scores, [1])
    return new_alphas

  all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
  # add first state for sequences of length 1
  all_alphas = tf.concat([tf.expand_dims(state, 1), all_alphas], 1)

  idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
  return tf.gather_nd(all_alphas, idxs)


def viterbi_decode(
    score, transition_params
):
  """Decode the highest scoring sequence of tags outside of TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
  """Computes the forward decoding in a linear-chain CRF."""

  def __init__(self, transition_params, **kwargs):
    """Initialize the CrfDecodeForwardRnnCell.

    Args:
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
        This matrix is expanded into a [1, num_tags, num_tags] in
        preparation for the broadcast summation occurring within the cell.
      **kwargs: Key word arguments.
    """
    super().__init__(**kwargs)
    self._transition_params = tf.expand_dims(transition_params, 0)
    self._num_tags = transition_params.shape[0]

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def call(self, inputs, state):
    """Build the CrfDecodeForwardRnnCell.

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
        score values.

    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    state = tf.expand_dims(state[0], 2)
    transition_scores = state + tf.cast(self._transition_params,
                                        self._compute_dtype)
    new_state = inputs + tf.reduce_max(transition_scores, [1])
    backpointers = tf.argmax(transition_scores, 1)
    backpointers = tf.cast(backpointers, dtype=tf.int32)
    return backpointers, new_state

  def get_config(self):
    config = {
        "transition_params":
            tf.squeeze(self._transition_params, 0).numpy().tolist()
    }
    base_config = super(CrfDecodeForwardRnnCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config["transition_params"] = np.array(
        config["transition_params"], dtype=np.float32)
    return cls(**config)


def crf_decode_forward(
    inputs,
    state,
    transition_params,
    sequence_lengths,
):
  """Computes forward decoding in a linear-chain CRF.

  Args:
    inputs: A [batch_size, num_tags] matrix of unary potentials.
    state: A [batch_size, num_tags] matrix containing the previous step's
      score values.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    sequence_lengths: A [batch_size] vector of true sequence lengths.

  Returns:
    backpointers: A [batch_size, num_tags] matrix of backpointers.
    new_state: A [batch_size, num_tags] matrix of new score values.
  """
  sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
  mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
  crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params, dtype=inputs.dtype)
  crf_fwd_layer = tf.keras.layers.RNN(
      crf_fwd_cell,
      return_sequences=True,
      return_state=True,
      dtype=inputs.dtype)
  return crf_fwd_layer(inputs, state, mask=mask)


def crf_decode_backward(inputs, state):
  """Computes backward decoding in a linear-chain CRF.

  Args:
    inputs: A [batch_size, num_tags] matrix of backpointer of next step (in
      time order).
    state: A [batch_size, 1] matrix of tag index of next step.

  Returns:
    new_tags: A [batch_size, num_tags]
      tensor containing the new tag indices.
  """
  inputs = tf.transpose(inputs, [1, 0, 2])

  def _scan_fn(state, inputs):
    state = tf.squeeze(state, axis=[1])
    idxs = tf.stack([tf.range(tf.shape(inputs)[0]), state], axis=1)
    new_tags = tf.expand_dims(tf.gather_nd(inputs, idxs), axis=-1)
    return new_tags

  return tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])


def crf_decode(
    potentials,
    transition_params,
    sequence_length,
):
  """Decode the highest scoring sequence of tags.

  Args:
    potentials: A [batch_size, max_seq_len, num_tags] tensor of unary
      potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    sequence_length: A [batch_size] vector of true sequence lengths.

  Returns:
    decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                Contains the highest scoring tag indices.
    best_score: A [batch_size] vector, containing the score of `decode_tags`.
  """
  warnings.warn(
      "CRF Decoding does not work with KerasTensors in TF2.4. The bug has since been fixed in tensorflow/tensorflow##45534"
  )
  sequence_length = tf.cast(sequence_length, dtype=tf.int32)

  def _multi_seq_fn():
    # Computes forward decoding. Get last score and backpointers.
    initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
    initial_state = tf.squeeze(initial_state, axis=[1])
    inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

    sequence_length_less_one = tf.maximum(
        tf.constant(0, dtype=tf.int32), sequence_length - 1)

    backpointers, last_score = crf_decode_forward(inputs, initial_state,
                                                  transition_params,
                                                  sequence_length_less_one)

    backpointers = tf.reverse_sequence(
        backpointers, sequence_length_less_one, seq_axis=1)

    initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
    initial_state = tf.expand_dims(initial_state, axis=-1)

    decode_tags = crf_decode_backward(backpointers, initial_state)
    decode_tags = tf.squeeze(decode_tags, axis=[2])
    decode_tags = tf.concat([initial_state, decode_tags], axis=1)
    decode_tags = tf.reverse_sequence(decode_tags, sequence_length, seq_axis=1)

    best_score = tf.reduce_max(last_score, axis=1)
    return decode_tags, best_score

  return _multi_seq_fn()


def crf_constrained_decode(
    potentials,
    tag_bitmap,
    transition_params,
    sequence_length,
):
  """Decode the highest scoring sequence of tags under constraints.

  This is a function for tensor.

  Args:
    potentials: A [batch_size, max_seq_len, num_tags] tensor of unary
      potentials.
    tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
      representing all active tags at each index for which to calculate the
      unnormalized score.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
    sequence_length: A [batch_size] vector of true sequence lengths.

  Returns:
    decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                Contains the highest scoring tag indices.
    best_score: A [batch_size] vector, containing the score of `decode_tags`.
  """

  filtered_potentials = crf_filtered_inputs(potentials, tag_bitmap)
  return crf_decode(filtered_potentials, transition_params, sequence_length)
