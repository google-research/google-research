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

"""Variants of the cycle-consistency loss described in TCC paper.

The Temporal Cycle-Consistency (TCC) Learning paper
(https://arxiv.org/pdf/1904.07846.pdf) describes a loss that enables learning
of self-supervised representations from sequences of embeddings that are good
at temporally fine-grained tasks like phase classification, video alignment etc.

These losses impose cycle-consistency constraints between sequences of
embeddings. Another interpretation of the cycle-consistency constraints is
that of mutual nearest-nieghbors. This means if state A in sequence 1 is the
nearest neighbor of state B in sequence 2 then it must also follow that B is the
nearest neighbor of A. We found that imposing this constraint on a dataset of
related sequences (like videos of people pitching a baseball) allows us to learn
generally useful visual representations.

This code allows the user to apply the loss while giving them the freedom to
choose the right encoder for their dataset/task. One advice for choosing an
encoder is to ensure that the encoder does not solve the mutual neighbor finding
task in a trivial fashion. For example, if one uses an LSTM or Transformer with
positional encodings, the matching between sequences may be done trivially by
counting the frame index with the encoder rather than learning good features.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tcc.tcc.deterministic_alignment import compute_deterministic_alignment_loss
from tcc.tcc.stochastic_alignment import compute_stochastic_alignment_loss


def compute_alignment_loss(embs,
                           batch_size,
                           steps=None,
                           seq_lens=None,
                           stochastic_matching=False,
                           normalize_embeddings=False,
                           loss_type='classification',
                           similarity_type='l2',
                           num_cycles=20,
                           cycle_length=2,
                           temperature=0.1,
                           label_smoothing=0.1,
                           variance_lambda=0.001,
                           huber_delta=0.1,
                           normalize_indices=True):
  """Computes alignment loss between sequences of embeddings.

  This function is a wrapper around different variants of the alignment loss
  described deterministic_alignment.py and stochastic_alignment.py files. The
  structure of the library is as follows:
  i) loss_fns.py - Defines the different loss functions.
  ii) deterministic_alignment.py - Performs the alignment between sequences by
  deterministically sampling all steps of the sequences.
  iii) stochastic_alignment.py - Performs the alignment between sequences by
  stochasticallty sub-sampling a fixed number of steps from the sequences.

  There are four major hparams that need to be tuned while applying the loss:
  i) Should the loss be applied with L2 normalization on the embeddings or
  without it?
  ii) Should we perform stochastic alignment of sequences? This means should we
  use all the steps of the embedding or only choose a random subset for
  alignment?
  iii) Should we apply cycle-consistency constraints using a classification loss
  or a regression loss? (Section 3 in paper)
  iv) Should the similarity metric be based on an L2 distance or cosine
  similarity?

  Other hparams that can be used to control how hard/soft we want the alignment
  between different sequences to be:
  i) temperature (all losses)
  ii) label_smoothing (classification)
  iii) variance_lambda (regression_mse_var)
  iv) huber_delta (regression_huber)
  Each of these params are used in their respective loss types (in brackets) and
  allow the application of the cycle-consistency constraints in a controllable
  manner but they do so in very different ways. Please refer to paper for more
  details.

  The default hparams work well for frame embeddings of videos of humans
  performing actions. Other datasets might need different values of hparams.


  Args:
    embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
      batch size, T is the number of timesteps in the sequence, D is the size of
      the embeddings.
    batch_size: Integer, Size of the batch.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
      If this is set to None, then we assume that the sampling was done in a
      uniform way and use tf.range(num_steps) as the steps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional information to the alignment loss. This is
      different from num_steps which is just the number of steps that have been
      sampled from the entire sequence.
    stochastic_matching: Boolean, Should the used for matching be sampled
      stochastically or deterministically? Deterministic is better for TPU.
      Stochastic is better for adding more randomness to the training process
      and handling long sequences.
    normalize_embeddings: Boolean, Should the embeddings be normalized or not?
      Default is to use raw embeddings. Be careful if you are normalizing the
      embeddings before calling this function.
    loss_type: String, This specifies the kind of loss function to use.
      Currently supported loss functions: classification, regression_mse,
      regression_mse_var, regression_huber.
    similarity_type: String, Currently supported similarity metrics: l2, cosine.
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

  ##############################################################################
  # Checking inputs and setting defaults.
  ##############################################################################

  # Get the number of timestemps in the sequence embeddings.
  num_steps = tf.shape(embs)[1]

  # If steps has not been provided assume sampling has been done uniformly.
  if steps is None:
    steps = tf.tile(tf.expand_dims(tf.range(num_steps), axis=0),
                    [batch_size, 1])

  # If seq_lens has not been provided assume is equal to the size of the
  # time axis in the emebeddings.
  if seq_lens is None:
    seq_lens = tf.tile(tf.expand_dims(num_steps, 0), [batch_size])

  if not tf.executing_eagerly():
    # Check if batch size embs is consistent with provided batch size.
    with tf.control_dependencies([tf.assert_equal(batch_size,
                                                  tf.shape(embs)[0])]):
      embs = tf.identity(embs)
    # Check if number of timesteps in embs is consistent with provided steps.
    with tf.control_dependencies([tf.assert_equal(num_steps,
                                                  tf.shape(steps)[1]),
                                  tf.assert_equal(batch_size,
                                                  tf.shape(steps)[0])]):
      steps = tf.identity(steps)
  else:
    tf.assert_equal(batch_size, tf.shape(steps)[0])
    tf.assert_equal(num_steps, tf.shape(steps)[1])
    tf.assert_equal(batch_size, tf.shape(embs)[0])

  ##############################################################################
  # Perform alignment and return loss.
  ##############################################################################

  if normalize_embeddings:
    embs = tf.nn.l2_normalize(embs, axis=-1)

  if stochastic_matching:
    loss = compute_stochastic_alignment_loss(
        embs=embs,
        steps=steps,
        seq_lens=seq_lens,
        num_steps=num_steps,
        batch_size=batch_size,
        loss_type=loss_type,
        similarity_type=similarity_type,
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        temperature=temperature,
        label_smoothing=label_smoothing,
        variance_lambda=variance_lambda,
        huber_delta=huber_delta,
        normalize_indices=normalize_indices)
  else:
    loss = compute_deterministic_alignment_loss(
        embs=embs,
        steps=steps,
        seq_lens=seq_lens,
        num_steps=num_steps,
        batch_size=batch_size,
        loss_type=loss_type,
        similarity_type=similarity_type,
        temperature=temperature,
        label_smoothing=label_smoothing,
        variance_lambda=variance_lambda,
        huber_delta=huber_delta,
        normalize_indices=normalize_indices)

  return loss
