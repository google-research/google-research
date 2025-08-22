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

"""Tests for asr_loss."""

from absl.testing import parameterized
import asr_loss
from lingvo import compat as tf
import numpy as np
import semiring
import utils


class UtilsTest(tf.test.TestCase):

  def testInterleaveWithBlank(self):
    """Enumerate mock inputs by hand and compare."""
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    blank_1 = tf.constant([[0.0], [0.0]])
    blank_2 = tf.constant([[0.5, 0.5, 0.5]])

    output_1 = asr_loss.interleave_with_blank(x, blank_1, axis=1)
    output_2 = asr_loss.interleave_with_blank(x, blank_2, axis=0)

    expected_output_1 = tf.constant([[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                                     [0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0]])
    expected_output_2 = tf.constant([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0],
                                     [0.5, 0.5, 0.5], [4.0, 5.0, 6.0],
                                     [0.5, 0.5, 0.5]])

    self.assertAllClose(output_1, expected_output_1)
    self.assertAllClose(output_2, expected_output_2)


class ASRLossTest(tf.test.TestCase):

  def testCTCByHand(self):
    """Enumerate a very simple lattice by hand and compare."""
    input_logits = np.array([[
        [-1.0, -2.0, -3.0],
        [-4.0, -5.0, -6.0],
        [-7.0, -8.0, -9.0],
        [-10.0, -11.0, -12.0],
    ]])
    output_labels = np.array([[1, 2, 2]])

    loss = asr_loss.ctc(
        input_logits=input_logits,
        output_labels=output_labels,
        input_seq_len=[4],
        output_seq_len=[3],
    )

    by_hand = -tf.reduce_logsumexp(
        [
            np.sum([-2.0, -6.0, -7.0, -12.0]),  # (1, 2, b, 2)
        ],
        keepdims=True)

    self.assertAllClose(loss, by_hand)

    # Check that invalid losses are zero-ed out.
    loss = asr_loss.ctc(
        input_logits=input_logits,
        output_labels=output_labels,
        input_seq_len=[3],
        output_seq_len=[3],
    )

    by_hand = np.array([0.0])

    self.assertAllClose(loss, by_hand)

    # Check that the unused logits are masked out.
    loss = asr_loss.ctc(
        input_logits=input_logits,
        output_labels=output_labels,
        input_seq_len=[3],
        output_seq_len=[2],
    )

    by_hand = -tf.reduce_logsumexp(
        [
            np.sum([-1.0, -5.0, -9.0]),  # (b, 1, 2)
            np.sum([-2.0, -4.0, -9.0]),  # (1, b, 2)
            np.sum([-2.0, -5.0, -9.0]),  # (1, 1, 2)
            np.sum([-2.0, -6.0, -7.0]),  # (1, 2, b)
            np.sum([-2.0, -6.0, -9.0]),  # (1, 2, 2)
        ],
        keepdims=True)

    self.assertAllClose(loss, by_hand)

  def testRNNTByHand(self):
    """Enumerate a very simple lattice by hand and compare."""

    s1_logits = np.array([[
        [-1.0, -2.0, -3.0],
        [-4.0, -5.0, -6.0],
        [0.0, 0.0, -13.0],
    ]])
    s2_logits = np.array([[
        [-7.0, -8.0, 0.0],
        [-9.0, -10.0, 0.0],
        [-11.0, -12.0, 0.0],
    ]])

    loss = asr_loss.rnnt(
        s1_logits=s1_logits,
        s2_logits=s2_logits,
        s1_seq_len=[3],
        s2_seq_len=[2],
    )

    by_hand = -tf.reduce_logsumexp(
        [
            np.sum([-1.0, -4.0, -11.0, -12.0, -13.0]),  # (S1, S1, S2, S2, S1)
            np.sum([-1.0, -9.0, -5.0, -12.0, -13.0]),  # (S1, S2, S1, S2, S1)
            np.sum([-1.0, -9.0, -10.0, -6.0, -13.0]),  # (S1, S2, S2, S1, S1)
            np.sum([-7.0, -2.0, -5.0, -12.0, -13.0]),  # (S2, S1, S1, S2, S1)
            np.sum([-7.0, -2.0, -10.0, -6.0, -13.0]),  # (S2, S1, S2, S1, S1)
            np.sum([-7.0, -8.0, -3.0, -6.0, -13.0]),  # (S2, S2, S1, S1, S1)
        ],
        keepdims=True)

    self.assertAllClose(loss, by_hand)

    # Check that invalid losses are zero-ed out.
    loss_1 = asr_loss.rnnt(
        s1_logits=s1_logits,
        s2_logits=s2_logits,
        s1_seq_len=[0],
        s2_seq_len=[2],
    )
    loss_2 = asr_loss.rnnt(
        s1_logits=s1_logits,
        s2_logits=s2_logits,
        s1_seq_len=[3],
        s2_seq_len=[0],
    )
    zeros = np.array([0.0])

    self.assertAllClose(loss_1, zeros)
    self.assertAllClose(loss_2, zeros)

    # Check that the unused logits are masked out.
    s1_logits = np.array([[
        [-1.0, -2.0, -3.0],
        [-4.0, -5.0, -6.0],
        [1.0, 1.0, -13.0],
    ]])
    s2_logits = np.array([[
        [-7.0, -8.0, 1.0],
        [-9.0, -10.0, 1.0],
        [-11.0, -12.0, 1.0],
    ]])

    loss = asr_loss.rnnt(
        s1_logits=s1_logits,
        s2_logits=s2_logits,
        s1_seq_len=[3],
        s2_seq_len=[2])

    self.assertAllClose(loss, by_hand)


class SemiringLossTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Set up CTC inputs.
    self.ctc_logits_p = np.array([[
        [-1.0, -2.0, -3.0],
        [-4.0, -5.0, -6.0],
        [-7.0, -8.0, -9.0],
        [-10.0, -11.0, -12.0],
    ]])
    self.ctc_logits_q = np.array([[
        [-13.0, -14.0, -15.0],
        [-16.0, -17.0, -18.0],
        [-19.0, -20.0, -21.0],
        [-22.0, -23.0, -24.0],
    ]])
    self.ctc_short_paths_p = np.array([
        np.sum([-2.0, -6.0, -7.0, -12.0])  # (1, 2, b, 2)
    ])
    self.ctc_short_paths_q = np.array([
        np.sum([-14.0, -18.0, -19.0, -24.0]),  # (1, 2, b, 2)
    ])
    self.ctc_long_paths_p = np.array([
        np.sum([-1.0, -5.0, -9.0]),  # (b, 1, 2)
        np.sum([-2.0, -4.0, -9.0]),  # (1, b, 2)
        np.sum([-2.0, -5.0, -9.0]),  # (1, 1, 2)
        np.sum([-2.0, -6.0, -7.0]),  # (1, 2, b)
        np.sum([-2.0, -6.0, -9.0]),  # (1, 2, 2)
    ])
    self.ctc_long_paths_q = np.array([
        np.sum([-13.0, -17.0, -21.0]),  # (b, 1, 2)
        np.sum([-14.0, -16.0, -21.0]),  # (1, b, 2)
        np.sum([-14.0, -17.0, -21.0]),  # (1, 1, 2)
        np.sum([-14.0, -18.0, -19.0]),  # (1, 2, b)
        np.sum([-14.0, -18.0, -21.0]),  # (1, 2, 2)
    ])
    self.output_labels = np.array([[1, 2, 2]])
    self.input_seq_len = [4]
    self.output_seq_len = [3]
    self.invalid_input_seq_len = [3]
    self.invalid_output_seq_len = [3]
    self.unused_input_seq_len = [3]
    self.unused_output_seq_len = [2]

    # Set up RNN-T inputs.
    self.rnnt_s1_logits_p = np.array([[
        [-1.0, -2.0, -3.0],
        [-4.0, -5.0, -6.0],
        [0.0, 0.0, -13.0],
    ]])
    self.rnnt_s2_logits_p = np.array([[
        [-7.0, -8.0, 0.0],
        [-9.0, -10.0, 0.0],
        [-11.0, -12.0, 0.0],
    ]])
    self.rnnt_s1_logits_q = np.array([[
        [-14.0, -15.0, -16.0],
        [-17.0, -18.0, -19.0],
        [0.0, 0.0, -26.0],
    ]])
    self.rnnt_s2_logits_q = np.array([[
        [-20.0, -21.0, 0.0],
        [-22.0, -23.0, 0.0],
        [-24.0, -25.0, 0.0],
    ]])
    self.rnnt_paths_p = np.array([
        np.sum([-1.0, -4.0, -11.0, -12.0, -13.0]),  # (S1, S1, S2, S2, S1)
        np.sum([-1.0, -9.0, -5.0, -12.0, -13.0]),  # (S1, S2, S1, S2, S1)
        np.sum([-1.0, -9.0, -10.0, -6.0, -13.0]),  # (S1, S2, S2, S1, S1)
        np.sum([-7.0, -2.0, -5.0, -12.0, -13.0]),  # (S2, S1, S1, S2, S1)
        np.sum([-7.0, -2.0, -10.0, -6.0, -13.0]),  # (S2, S1, S2, S1, S1)
        np.sum([-7.0, -8.0, -3.0, -6.0, -13.0]),  # (S2, S2, S1, S1, S1)
    ])
    self.rnnt_paths_q = np.array([
        np.sum([-14.0, -17.0, -24.0, -25.0, -26.0]),  # (S1, S1, S2, S2, S1)
        np.sum([-14.0, -22.0, -18.0, -25.0, -26.0]),  # (S1, S2, S1, S2, S1)
        np.sum([-14.0, -22.0, -23.0, -19.0, -26.0]),  # (S1, S2, S2, S1, S1)
        np.sum([-20.0, -15.0, -18.0, -25.0, -26.0]),  # (S2, S1, S1, S2, S1)
        np.sum([-20.0, -15.0, -23.0, -19.0, -26.0]),  # (S2, S1, S2, S1, S1)
        np.sum([-20.0, -21.0, -16.0, -19.0, -26.0]),  # (S2, S2, S1, S1, S1)
    ])
    self.s1_seq_len = [3]
    self.s2_seq_len = [2]

  def ComputeLossByHand(self, sr_name, paths_p, paths_q):
    """Helper function to compute loss manually given the paths."""
    logp = tf.reduce_logsumexp(paths_p, keepdims=True)
    logq = tf.reduce_logsumexp(paths_q, keepdims=True)
    logminusplogq = tf.reduce_logsumexp(
        utils.logminus(paths_p, paths_q), keepdims=True)
    logminusqlogq = tf.reduce_logsumexp(
        utils.logminus(paths_q, paths_q), keepdims=True)
    logminusqlogp = tf.reduce_logsumexp(
        utils.logminus(paths_q, paths_p), keepdims=True)
    if sr_name == 'logentropy':
      return (logp, logminusplogq)
    elif sr_name == 'logreversekl':
      return (logp, logq, logminusqlogq, logminusqlogp)

  @parameterized.parameters([
      ('logentropy', semiring.LogEntropySemiring()),
      ('logreversekl', semiring.LogReverseKLSemiring()),
  ])
  def testCTCSemiring(self, sr_name, sr):
    loss = asr_loss.ctc_semiring(
        sr=sr,
        sr_inputs=(self.ctc_logits_p, self.ctc_logits_q),
        output_labels=self.output_labels,
        input_seq_len=self.input_seq_len,
        output_seq_len=self.output_seq_len)
    by_hand = self.ComputeLossByHand(sr_name, self.ctc_short_paths_p,
                                     self.ctc_short_paths_q)
    self.assertAllClose(loss, by_hand, atol=1e-37)

    # Check that invalid losses are zero-ed out.
    loss = asr_loss.ctc_semiring(
        sr=sr,
        sr_inputs=(self.ctc_logits_p, self.ctc_logits_q),
        output_labels=self.output_labels,
        input_seq_len=self.invalid_input_seq_len,
        output_seq_len=self.invalid_output_seq_len)
    for l in loss:
      self.assertAllClose(l, np.array([0.0]), atol=1e-37)

    # Check that the unused logits are masked out.
    loss = asr_loss.ctc_semiring(
        sr=sr,
        sr_inputs=(self.ctc_logits_p, self.ctc_logits_q),
        output_labels=self.output_labels,
        input_seq_len=self.unused_input_seq_len,
        output_seq_len=self.unused_output_seq_len)
    by_hand = self.ComputeLossByHand(sr_name, self.ctc_long_paths_p,
                                     self.ctc_long_paths_q)
    self.assertAllClose(loss, by_hand, atol=1e-37)

  @parameterized.parameters([
      ('logentropy', semiring.LogEntropySemiring()),
      ('logreversekl', semiring.LogReverseKLSemiring()),
  ])
  def testRNNTSemiring(self, sr_name, sr):
    loss = asr_loss.rnnt_semiring(
        sr=sr,
        s1_inputs=(self.rnnt_s1_logits_p, self.rnnt_s1_logits_q),
        s2_inputs=(self.rnnt_s2_logits_p, self.rnnt_s2_logits_q),
        s1_seq_len=self.s1_seq_len,
        s2_seq_len=self.s2_seq_len)
    by_hand = self.ComputeLossByHand(sr_name, self.rnnt_paths_p,
                                     self.rnnt_paths_q)
    self.assertAllClose(loss, by_hand, atol=1e-37)

    # Check that invalid losses are zero-ed out.
    loss_1 = asr_loss.rnnt_semiring(
        sr=sr,
        s1_inputs=(self.rnnt_s1_logits_p, self.rnnt_s1_logits_q),
        s2_inputs=(self.rnnt_s2_logits_p, self.rnnt_s2_logits_q),
        s1_seq_len=[0],
        s2_seq_len=self.s2_seq_len)
    loss_2 = asr_loss.rnnt_semiring(
        sr=sr,
        s1_inputs=(self.rnnt_s1_logits_p, self.rnnt_s1_logits_q),
        s2_inputs=(self.rnnt_s2_logits_p, self.rnnt_s2_logits_q),
        s1_seq_len=self.s1_seq_len,
        s2_seq_len=[0])
    zeros = tf.zeros_like(by_hand)

    self.assertAllClose(loss_1, zeros)
    self.assertAllClose(loss_2, zeros)

    # Check that the unused logits are masked out.
    rnnt_s1_logits_p = np.where(self.rnnt_s1_logits_p == 0.0, 1.23,
                                self.rnnt_s1_logits_p)
    rnnt_s1_logits_q = np.where(self.rnnt_s1_logits_q == 0.0, 1.23,
                                self.rnnt_s1_logits_q)
    rnnt_s2_logits_p = np.where(self.rnnt_s2_logits_p == 0.0, 1.23,
                                self.rnnt_s2_logits_p)
    rnnt_s2_logits_q = np.where(self.rnnt_s2_logits_q == 0.0, 1.23,
                                self.rnnt_s2_logits_q)

    loss = asr_loss.rnnt_semiring(
        sr=sr,
        s1_inputs=(rnnt_s1_logits_p, rnnt_s1_logits_q),
        s2_inputs=(rnnt_s2_logits_p, rnnt_s2_logits_q),
        s1_seq_len=self.s1_seq_len,
        s2_seq_len=self.s2_seq_len)
    by_hand = self.ComputeLossByHand(sr_name, self.rnnt_paths_p,
                                     self.rnnt_paths_q)
    self.assertAllClose(loss, by_hand, atol=1e-37)


if __name__ == '__main__':
  tf.test.main()
