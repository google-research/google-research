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

"""Deterministic alignment between all pairs of sequences in a batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.tcc.losses import classification_loss
from tcc.tcc.losses import regression_loss


def pairwise_l2_distance(embs1, embs2):
  """Computes pairwise distances between all rows of embs1 and embs2."""
  norm1 = tf.reduce_sum(tf.square(embs1), 1)
  norm1 = tf.reshape(norm1, [-1, 1])
  norm2 = tf.reduce_sum(tf.square(embs2), 1)
  norm2 = tf.reshape(norm2, [1, -1])

  # Max to ensure matmul doesn't produce anything negative due to floating
  # point approximations.
  dist = tf.maximum(
      norm1 + norm2 - 2.0 * tf.matmul(embs1, embs2, False, True), 0.0)

  return dist


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
  """Returns similarity between each all rows of embs1 and all rows of embs2.

  The similarity is scaled by the number of channels/embedding size and
  temperature.

  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.

  Returns:
    similarity: Tensor, [M, N] tensor denoting similarity between embs1 and
      embs2.
  """
  channels = tf.cast(tf.shape(embs1)[1], tf.float32)
  # Go for embs1 to embs2.
  if similarity_type == 'cosine':
    similarity = tf.matmul(embs1, embs2, transpose_b=True)
  elif similarity_type == 'l2':
    similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
  else:
    raise ValueError('similarity_type can either be l2 or cosine.')

  # Scale the distance  by number of channels. This normalization helps with
  # optimization.
  similarity /= channels
  # Scale the distance by a temperature that helps with how soft/hard the
  # alignment should be.
  similarity /= temperature

  return similarity


def align_pair_of_sequences(embs1,
                            embs2,
                            similarity_type,
                            temperature):
  """Align a given pair embedding sequences.

  Args:
    embs1: Tensor, Embeddings of the shape [M, D] where M is the number of
      embeddings and D is the embedding size.
    embs2: Tensor, Embeddings of the shape [N, D] where N is the number of
      embeddings and D is the embedding size.
    similarity_type: String, Either one of 'l2' or 'cosine'.
    temperature: Float, Temperature used in scaling logits before softmax.
  Returns:
     logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
  """
  max_num_steps = tf.shape(embs1)[0]

  # Find distances between embs1 and embs2.
  sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
  # Softmax the distance.
  softmaxed_sim_12 = tf.nn.softmax(sim_12, axis=1)

  # Calculate soft-nearest neighbors.
  nn_embs = tf.matmul(softmaxed_sim_12, embs2)

  # Find distances between nn_embs and embs1.
  sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)

  logits = sim_21
  labels = tf.one_hot(tf.range(max_num_steps), max_num_steps)

  return logits, labels


def compute_deterministic_alignment_loss(embs,
                                         steps,
                                         seq_lens,
                                         num_steps,
                                         batch_size,
                                         loss_type,
                                         similarity_type,
                                         temperature,
                                         label_smoothing,
                                         variance_lambda,
                                         huber_delta,
                                         normalize_indices):
  """Compute cycle-consistency loss for all steps in each sequence.

  This aligns each pair of videos in the batch except with itself.
  When aligning it also matters which video is the starting video. So for N
  videos in the batch, we have N * (N-1) alignments happening.
  For example, a batch of size 3 has 6 pairs of sequence alignments.


  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size
      of the embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was
    done. This can provide additional information to the alignment loss.
    num_steps: Integer/Tensor, Number of timesteps in the embeddings.
    batch_size: Integer, Size of the batch.
    loss_type: String, This specifies the kind of loss function to use.
      Currently supported loss functions: 'classification', 'regression_mse',
      'regression_mse_var', 'regression_huber'.
    similarity_type: String, Currently supported similarity metrics: 'l2' ,
      'cosine' .
    temperature: Float, temperature scaling used to scale the similarity
      distributions calculated using the softmax function.
    label_smoothing: Float, Label smoothing argument used in
      tf.keras.losses.categorical_crossentropy function and described in this
      paper https://arxiv.org/pdf/1701.06548.pdf.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low
      results in high variance of the similarities (more uniform/random
      matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
    normalize_indices: Boolean, If True, normalizes indices by sequence
      lengths. Useful for ensuring numerical instabilities doesn't arise as
      sequence indices can be large numbers.
  Returns:
    loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
        cycle-consistency loss.
  """
  labels_list = []
  logits_list = []
  steps_list = []
  seq_lens_list = []

  for i in range(batch_size):
    for j in range(batch_size):
      # We do not align the sequence with itself.
      if i != j:
        logits, labels = align_pair_of_sequences(embs[i],
                                                 embs[j],
                                                 similarity_type,
                                                 temperature)
        logits_list.append(logits)
        labels_list.append(labels)
        steps_list.append(tf.tile(steps[i:i+1], [num_steps, 1]))
        seq_lens_list.append(tf.tile(seq_lens[i:i+1], [num_steps]))

  logits = tf.concat(logits_list, axis=0)
  labels = tf.concat(labels_list, axis=0)
  steps = tf.concat(steps_list, axis=0)
  seq_lens = tf.concat(seq_lens_list, axis=0)

  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:

    loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta)
  else:
    raise ValueError('Unidentified loss_type %s. Currently supported loss '
                     'types are: regression_mse, regression_huber, '
                     'classification.' % loss_type)

  return loss
