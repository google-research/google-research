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

"""Compute Divergences between Speech Recognition Models.

For now, the only supported use case is when both models are the same, i.e. both
CTC or both RNN-T. We always calculate the NLL (negative log likelihood) and
return it as the first argument.
"""
import asr_loss
from lingvo import compat as tf
import semiring
import utils

SeqLen = utils.SeqLen
_NLLAndDivergence = tuple[tf.Tensor, tf.Tensor]


def log_entropy_ctc(input_logits, output_labels,
                    input_seq_len,
                    output_seq_len):
  """Computes log entropy [log(-sum{plogp})] of a CTC model."""
  logp, logminusplogp = asr_loss.ctc_semiring(
      sr=semiring.LogEntropySemiring(),
      sr_inputs=(input_logits, input_logits),
      output_labels=output_labels,
      input_seq_len=input_seq_len,
      output_seq_len=output_seq_len)
  return -logp, logminusplogp


def log_entropy_rnnt(s1_logits, s2_logits,
                     s1_seq_len,
                     s2_seq_len):
  """Computes log entropy [log(-sum{plogp})] of an RNN-T model."""
  logp, logminusplogp = asr_loss.rnnt_semiring(
      sr=semiring.LogEntropySemiring(),
      s1_inputs=(s1_logits, s1_logits),
      s2_inputs=(s2_logits, s2_logits),
      s1_seq_len=s1_seq_len,
      s2_seq_len=s2_seq_len)
  return -logp, logminusplogp


def log_reverse_kl_ctc_ctc(input_logits_pair,
                           output_labels, input_seq_len,
                           output_seq_len):
  """Computes log reverse KL divergence [log(sum{qlog(q/p)})] between two CTC models."""
  logp, _, logminusqlogq, logminusqlogp = asr_loss.ctc_semiring(
      sr=semiring.LogReverseKLSemiring(),
      sr_inputs=input_logits_pair,
      output_labels=output_labels,
      input_seq_len=input_seq_len,
      output_seq_len=output_seq_len)

  # Calculates Log(qlogq - qlogp).
  divergence = utils.weightedlogsumexp_list([logminusqlogq, logminusqlogp],
                                            [-1.0, 1.0])
  return -logp, divergence


def log_reverse_kl_rnnt_rnnt(s1_logits_pair,
                             s2_logits_pair,
                             s1_seq_len,
                             s2_seq_len):
  """Computes log reverse KL divergence [log(sum{qlog(q/p)})] between two RNN-T models."""
  logp, _, logminusqlogq, logminusqlogp = asr_loss.rnnt_semiring(
      sr=semiring.LogReverseKLSemiring(),
      s1_inputs=s1_logits_pair,
      s2_inputs=s2_logits_pair,
      s1_seq_len=s1_seq_len,
      s2_seq_len=s2_seq_len)

  # Calculates Log(qlogq - qlogp).
  divergence = utils.weightedlogsumexp_list([logminusqlogq, logminusqlogp],
                                            [-1.0, 1.0])
  return -logp, divergence
