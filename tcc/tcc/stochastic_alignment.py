# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Stochastic alignment between sampled cycles in the sequences in a batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.tcc.losses import classification_loss
from tcc.tcc.losses import regression_loss


def _align_single_cycle(cycle, embs, cycle_length, num_steps,
                        similarity_type, temperature):
  """Takes a single cycle and returns logits (simialrity scores) and labels."""
  # Choose random frame.
  n_idx = tf.random_uniform((), minval=0, maxval=num_steps, dtype=tf.int32)
  # Create labels
  onehot_labels = tf.one_hot(n_idx, num_steps)

  # Choose query feats for first frame.
  query_feats = embs[cycle[0], n_idx:n_idx+1]

  num_channels = tf.shape(query_feats)[-1]
  for c in range(1, cycle_length+1):
    candidate_feats = embs[cycle[c]]

    if similarity_type == 'l2':
      # Find L2 distance.
      mean_squared_distance = tf.reduce_sum(
          tf.squared_difference(tf.tile(query_feats, [num_steps, 1]),
                                candidate_feats), axis=1)
      # Convert L2 distance to similarity.
      similarity = -mean_squared_distance

    elif similarity_type == 'cosine':
      # Dot product of embeddings.
      similarity = tf.squeeze(tf.matmul(candidate_feats, query_feats,
                                        transpose_b=True))
    else:
      raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with
    # optimization.
    similarity /= tf.cast(num_channels, tf.float32)
    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature

    beta = tf.nn.softmax(similarity)
    beta = tf.expand_dims(beta, axis=1)
    beta = tf.tile(beta, [1, num_channels])

    # Find weighted nearest neighbour.
    query_feats = tf.reduce_sum(beta * candidate_feats,
                                axis=0, keepdims=True)

  return similarity, onehot_labels


def _align(cycles, embs, num_steps, num_cycles, cycle_length,
           similarity_type, temperature):
  """Align by finding cycles in embs."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = _align_single_cycle(cycles[i],
                                         embs,
                                         cycle_length,
                                         num_steps,
                                         similarity_type,
                                         temperature)
    logits_list.append(logits)
    labels_list.append(labels)

  logits = tf.stack(logits_list)
  labels = tf.stack(labels_list)

  return logits, labels


def gen_cycles(num_cycles, batch_size, cycle_length=2):
  """Generates cycles for alignment.

  Generates a batch of indices to cycle over. For example setting num_cycles=2,
  batch_size=5, cycle_length=3 might return something like this:
  cycles = [[0, 3, 4, 0], [1, 2, 0, 3]]. This means we have 2 cycles for which
  the loss will be calculated. The first cycle starts at sequence 0 of the
  batch, then we find a matching step in sequence 3 of that batch, then we
  find matching step in sequence 4 and finally come back to sequence 0,
  completing a cycle.

  Args:
    num_cycles: Integer, Number of cycles that will be matched in one pass.
    batch_size: Integer, Number of sequences in one batch.
    cycle_length: Integer, Length of the cycles. If we are matching between
      2 sequences (cycle_length=2), we get cycles that look like [0,1,0].
      This means that we go from sequence 0 to sequence 1 then back to sequence
      0. A cycle length of 3 might look like [0, 1, 2, 0].

  Returns:
    cycles: Tensor, Batch indices denoting cycles that will be used for
      calculating the alignment loss.
  """
  sorted_idxes = tf.tile(tf.expand_dims(tf.range(batch_size), 0),
                         [num_cycles, 1])
  sorted_idxes = tf.reshape(sorted_idxes, [batch_size, num_cycles])
  cycles = tf.reshape(tf.random.shuffle(sorted_idxes),
                      [num_cycles, batch_size])
  cycles = cycles[:, :cycle_length]
  # Append the first index at the end to create cycle.
  cycles = tf.concat([cycles, cycles[:, 0:1]], axis=1)
  return cycles


def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      seq_lens,
                                      num_steps,
                                      batch_size,
                                      loss_type,
                                      similarity_type,
                                      num_cycles,
                                      cycle_length,
                                      temperature,
                                      label_smoothing,
                                      variance_lambda,
                                      huber_delta,
                                      normalize_indices):
  """Compute cycle-consistency loss by stochastically sampling cycles.

  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size of
      the embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional information to the alignment loss.
    num_steps: Integer/Tensor, Number of timesteps in the embeddings.
    batch_size: Integer/Tensor, Batch size.
    loss_type: String, This specifies the kind of loss function to use.
      Currently supported loss functions: 'classification', 'regression_mse',
      'regression_mse_var', 'regression_huber'.
    similarity_type: String, Currently supported similarity metrics: 'l2',
      'cosine'.
    num_cycles: Integer, number of cycles to match while aligning
      stochastically.  Only used in the stochastic version.
    cycle_length: Integer, Lengths of the cycle to use for matching. Only used
      in the stochastic version. By default, this is set to 2.
    temperature: Float, temperature scaling used to scale the similarity
      distributions calculated using the softmax function.
    label_smoothing: Float, Label smoothing argument used in
      tf.keras.losses.categorical_crossentropy function and described in this
      paper https://arxiv.org/pdf/1701.06548.pdf.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities doesn't arise as sequence
      indices can be large numbers.

  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
      cycle-consistency loss.
  """
  # Generate cycles.
  cycles = gen_cycles(num_cycles, batch_size, cycle_length)

  logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length,
                          similarity_type, temperature)

  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:
    steps = tf.gather(steps, cycles[:, 0])
    seq_lens = tf.gather(seq_lens, cycles[:, 0])
    loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta)
  else:
    raise ValueError('Unidentified loss type %s. Currently supported loss '
                     'types are: regression_mse, regression_huber, '
                     'classification .'
                     % loss_type)
  return loss
