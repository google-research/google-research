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

"""Tests for asr_divergence."""

from absl.testing import parameterized
import asr_divergence
from lingvo import compat as tf
import numpy as np


class AsrDivergenceTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Note that in practice, we compute unnormalized entropies and divergences,
    # since it is never the case that a discriminative ASR model is a delta
    # function. However, for testing purposes, we hardcode some values so that
    # the distributions are normalized.
    hardcode_ctc_p = -0.3769124084010098
    hardcode_ctc_q = -0.4204111095862999
    hardcode_rnnt_p = -1.0686916835303504
    hardcode_rnnt_q = -2.15345748597497

    # Set up example (x, y).
    self.input_seq_len = [3]
    self.output_seq_len = [2]
    self.output_labels = np.array([[1, 2]])

    # CTC distribution P(Y=y|X=x).
    self.ctc_logits_p = np.array([[
        [hardcode_ctc_p, -0.2, -0.3],
        [-0.4, -0.5, -0.6],
        [-0.7, -0.8, -0.9],
    ]])
    self.ctc_paths_p = np.array([
        np.sum([hardcode_ctc_p, -0.5, -0.9]),  # (b, 1, 2)
        np.sum([-0.2, -0.4, -0.9]),  # (1, b, 2)
        np.sum([-0.2, -0.5, -0.9]),  # (1, 1, 2)
        np.sum([-0.2, -0.6, -0.7]),  # (1, 2, b)
        np.sum([-0.2, -0.6, -0.9]),  # (1, 2, 2)
    ])

    # CTC distribution Q(Y=y|X=x).
    self.ctc_logits_q = np.array([[
        [hardcode_ctc_q, -0.51, -0.52],
        [-0.53, -0.54, -0.55],
        [-0.56, -0.57, -0.58],
    ]])
    self.ctc_paths_q = np.array([
        np.sum([hardcode_ctc_q, -0.54, -0.58]),  # (b, 1, 2)
        np.sum([-0.51, -0.53, -0.58]),  # (1, b, 2)
        np.sum([-0.51, -0.54, -0.58]),  # (1, 1, 2)
        np.sum([-0.51, -0.55, -0.56]),  # (1, 2, b)
        np.sum([-0.51, -0.55, -0.58]),  # (1, 2, 2)
    ])

    # RNN-T distribution P(Y=y|X=x).
    self.rnnt_s1_logits_p = np.array([[
        [-0.1, -0.2, -0.3],
        [hardcode_rnnt_p, -0.5, -0.6],
        [0.0, 0.0, -0.13],
    ]])
    self.rnnt_s2_logits_p = np.array([[
        [-0.7, -0.8, 0.0],
        [-0.9, -0.10, 0.0],
        [-0.11, -0.12, 0.0],
    ]])
    self.rnnt_paths_p = np.array([
        np.sum([-0.1, hardcode_rnnt_p, -0.11, -0.12,
                -0.13]),  # (S1, S1, S2, S2, S1)
        np.sum([-0.1, -0.9, -0.5, -0.12, -0.13]),  # (S1, S2, S1, S2, S1)
        np.sum([-0.1, -0.9, -0.10, -0.6, -0.13]),  # (S1, S2, S2, S1, S1)
        np.sum([-0.7, -0.2, -0.5, -0.12, -0.13]),  # (S2, S1, S1, S2, S1)
        np.sum([-0.7, -0.2, -0.10, -0.6, -0.13]),  # (S2, S1, S2, S1, S1)
        np.sum([-0.7, -0.8, -0.3, -0.6, -0.13]),  # (S2, S2, S1, S1, S1)
    ])

    # RNN-T distribution Q(Y=y|X=x).
    self.rnnt_s1_logits_q = np.array([[
        [-0.14, -0.15, -0.16],
        [hardcode_rnnt_q, -0.18, -0.19],
        [0.0, 0.0, -0.46],
    ]])
    self.rnnt_s2_logits_q = np.array([[
        [-0.40, -0.41, 0.0],
        [-0.42, -0.43, 0.0],
        [-0.44, -0.45, 0.0],
    ]])
    self.rnnt_paths_q = np.array([
        np.sum([-0.14, hardcode_rnnt_q, -0.44, -0.45,
                -0.46]),  # (S1, S1, S2, S2, S1)
        np.sum([-0.14, -0.42, -0.18, -0.45, -0.46]),  # (S1, S2, S1, S2, S1)
        np.sum([-0.14, -0.42, -0.43, -0.19, -0.46]),  # (S1, S2, S2, S1, S1)
        np.sum([-0.40, -0.15, -0.18, -0.45, -0.46]),  # (S2, S1, S1, S2, S1)
        np.sum([-0.40, -0.15, -0.43, -0.19, -0.46]),  # (S2, S1, S2, S1, S1)
        np.sum([-0.40, -0.41, -0.16, -0.19, -0.46]),  # (S2, S2, S1, S1, S1)
    ])

    # Arguments for CTC:CTC and RNN-T:RNN-T divergences.
    self.ctc_ctc_args = ((self.ctc_logits_p,
                          self.ctc_logits_q), self.output_labels,
                         self.input_seq_len, self.output_seq_len)
    self.rnnt_rnnt_args = ((self.rnnt_s1_logits_p, self.rnnt_s1_logits_q),
                           (self.rnnt_s2_logits_p, self.rnnt_s2_logits_q),
                           self.input_seq_len, self.output_seq_len)

  def DivergenceFormula(self,
                        divergence,
                        paths_p,
                        paths_q=None,
                        renyi_alpha=None):
    """Helper function to compute divergences manually given the paths."""
    exp_paths_p = tf.math.exp(paths_p)
    if paths_q is not None:
      exp_paths_q = tf.math.exp(paths_q)

    if divergence == 'entropy':
      return -tf.reduce_sum(exp_paths_p * paths_p, keepdims=True)
    elif divergence == 'log-entropy':
      return tf.math.log(self.DivergenceFormula('entropy', paths_p))
    elif divergence == 'reverse-kl':
      return tf.reduce_sum(exp_paths_q * (paths_q - paths_p), keepdims=True)
    elif divergence == 'log-reverse-kl':
      return tf.math.log(self.DivergenceFormula('reverse-kl', paths_p, paths_q))

  def testLogEntropyCTC(self):
    # Log Entropy for P.
    by_hand_p = self.DivergenceFormula('log-entropy', self.ctc_paths_p)
    nll_p, log_entropy_p = asr_divergence.log_entropy_ctc(
        self.ctc_logits_p, self.output_labels, self.input_seq_len,
        self.output_seq_len)
    self.assertAllClose(nll_p, [0.0])
    self.assertAllClose(by_hand_p, log_entropy_p, atol=1e-37)

    # Log Entropy for Q.
    by_hand_q = self.DivergenceFormula('log-entropy', self.ctc_paths_q)
    nll_q, log_entropy_q = asr_divergence.log_entropy_ctc(
        self.ctc_logits_q, self.output_labels, self.input_seq_len,
        self.output_seq_len)
    self.assertAllClose(nll_q, [0.0])
    self.assertAllClose(by_hand_q, log_entropy_q, atol=1e-37)

  def testLogEntropyRNNT(self):
    # Log Entropy for P.
    by_hand_p = self.DivergenceFormula('log-entropy', self.rnnt_paths_p)
    nll_p, log_entropy_p = asr_divergence.log_entropy_rnnt(
        self.rnnt_s1_logits_p, self.rnnt_s2_logits_p, self.input_seq_len,
        self.output_seq_len)
    self.assertAllClose(nll_p, [0.0])
    self.assertAllClose(by_hand_p, log_entropy_p, atol=1e-37)

    # Log Entropy for Q.
    by_hand_q = self.DivergenceFormula('log-entropy', self.rnnt_paths_q)
    nll_q, log_entropy_q = asr_divergence.log_entropy_rnnt(
        self.rnnt_s1_logits_q, self.rnnt_s2_logits_q, self.input_seq_len,
        self.output_seq_len)
    self.assertAllClose(nll_q, [0.0])
    self.assertAllClose(by_hand_q, log_entropy_q, atol=1e-37)

  @parameterized.parameters([
      ('log-reverse-kl', asr_divergence.log_reverse_kl_ctc_ctc,
       asr_divergence.log_reverse_kl_rnnt_rnnt),
  ])
  def testDivergence(self, div, ctc, rnnt):
    # Divergence between two CTC models.
    by_hand = self.DivergenceFormula(div, self.ctc_paths_p, self.ctc_paths_q)
    nll, divergence = ctc(*self.ctc_ctc_args)
    self.assertAllClose(nll, [0.0])
    self.assertAllClose(by_hand, divergence, atol=1e-37)

    # Divergence between two RNN-T models.
    by_hand = self.DivergenceFormula(div, self.rnnt_paths_p, self.rnnt_paths_q)
    nll, divergence = rnnt(*self.rnnt_rnnt_args)
    self.assertAllClose(nll, [0.0])
    self.assertAllClose(by_hand, divergence, atol=1e-37)


if __name__ == '__main__':
  tf.test.main()
