# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Tests for fine tuning models."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
from protein_lm import domains
from protein_lm import fine_tune
from protein_lm import models

tf.enable_eager_execution()


def _get_lm(model_cls, domain, use_dropout=True):
  cfg = dict(batch_size=1, num_layers=2, num_heads=2, emb_dim=32,
             mlp_dim=32, qkv_dim=32)
  if not use_dropout:
    cfg.update(dict(dropout_rate=0., attention_dropout_rate=0.))
  return model_cls(domain, **cfg)


def _test_domain():
  vocab = domains.Vocabulary(
      tokens=['a', 'b', 'c'],
      include_bos=True,
      include_mask=True,
      include_pad=True)
  return domains.FixedLengthDiscreteDomain(vocab=vocab, length=3)


def _compute_logprob(inputs, model, weights):
  model.set_weights(weights)
  return models.compute_logprob(
      inputs, model, mask_token=None)


class FineTuneTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters((models.FlaxLM,), (models.FlaxBERT,))
  def test_fine_tuning_increases_likelihood(self, model_cls):
    domain = _test_domain()
    seqs = domain.sample_uniformly(6, seed=0)
    lm = _get_lm(model_cls, domain=domain)
    init_weights = lm.get_weights()
    init_logprob = _compute_logprob(seqs, lm, init_weights).mean()
    fine_tune_weights = fine_tune.fine_tune(
        lm, init_weights, seqs, num_epochs=1, batch_size=2, learning_rate=0.001)
    final_logprob = _compute_logprob(seqs, lm, fine_tune_weights).mean()
    self.assertGreater(final_logprob, init_logprob)

  @parameterized.parameters((models.FlaxLM,), (models.FlaxBERT,))
  def test_fine_tuning_zero_learning_rate(self, model_cls):
    domain = _test_domain()
    seqs = domain.sample_uniformly(6, seed=0)
    lm = _get_lm(model_cls, domain=domain)
    init_weights = lm.get_weights()
    init_logprob = _compute_logprob(seqs, lm, init_weights).mean()
    fine_tune_weights = fine_tune.fine_tune(
        lm, init_weights, seqs, num_epochs=1, batch_size=2, learning_rate=0.)
    final_logprob = _compute_logprob(seqs, lm, fine_tune_weights).mean()
    self.assertAllClose(final_logprob, init_logprob)

  @parameterized.parameters((models.FlaxLM,), (models.FlaxBERT,))
  def test_fine_tuning_with_example_weights(self, model_cls):
    domain = _test_domain()
    seqs = domain.sample_uniformly(2, seed=0)
    lm = _get_lm(model_cls, domain=domain)
    init_weights = lm.get_weights()
    init_logprobs = _compute_logprob(seqs, lm, init_weights)

    # Select the sequence that has higher initial likelihood and push
    # down its likelihood and pull up the likelihood of the other sequence.
    if init_logprobs[0] > init_logprobs[1]:
      seqs = seqs[::-1]

    example_weights = np.array([-1., 1.])
    fine_tune_weights = fine_tune.fine_tune(
        lm,
        init_weights,
        seqs,
        num_epochs=1,
        batch_size=2,
        example_weights=example_weights,
        learning_rate=0.001)
    final_logprobs = _compute_logprob(seqs, lm, fine_tune_weights)
    # Check that the ranking of their likelihoods is reversed.
    self.assertGreater(final_logprobs[1], final_logprobs[0])

  @parameterized.parameters((models.FlaxLM,), (models.FlaxBERT,))
  def test_deterministic(self, model_cls):
    domain = _test_domain()
    seqs = domain.sample_uniformly(2, seed=0)
    # Note that we are checking that it is deterministic when dropout is used.
    # All of the random number generation should be deterministic.
    lm = _get_lm(model_cls, domain=domain, use_dropout=True)
    init_weights = lm.get_weights()

    def run():
      fine_tune_weights = fine_tune.fine_tune(
          lm,
          init_weights,
          seqs,
          num_epochs=1,
          batch_size=2,
          shuffle=False,
          learning_rate=0.001)
      return _compute_logprob(seqs, lm, fine_tune_weights).mean()

    final_logprob1 = run()
    final_logprob2 = run()
    self.assertAllClose(final_logprob1, final_logprob2)

  # No BERT because each example in batch is masked differently.
  @parameterized.parameters(
      (models.FlaxLM,),)
  def test_duplicated_inputs(self, model_cls):
    domain = _test_domain()
    seq = domain.sample_uniformly(1, seed=0)
    lm = _get_lm(model_cls, domain=domain, use_dropout=False)
    init_weights = lm.get_weights()

    def run(n_copies):
      fine_tune_weights = fine_tune.fine_tune(
          lm,
          init_weights,
          np.tile(seq, [n_copies, 1]),
          num_epochs=1,
          batch_size=n_copies,
          shuffle=False,
          learning_rate=0.001)
      return _compute_logprob(seq, lm, fine_tune_weights).mean()

    final_logprob_1_copy = run(1)
    final_logprob_3_copies = run(3)
    self.assertAllClose(final_logprob_1_copy, final_logprob_3_copies)

  # No BERT because each example in batch is masked differently.
  @parameterized.parameters(
      (models.FlaxLM,),)
  def test_fine_tuning_with_zero_example_weight(self, model_cls):
    domain = _test_domain()
    lm = _get_lm(model_cls, domain=domain, use_dropout=False)
    init_weights = lm.get_weights()

    # Use 2 sequences of equal length, such that their contribution to
    # the loss (which is averages over non-pad tokens) is equal).
    seqs = domain.sample_uniformly(2, seed=0)

    # Fine tune with a weight of zero on the second example.
    fine_tune_weights_from_zero_weight = fine_tune.fine_tune(
        lm,
        init_weights,
        seqs,
        num_epochs=1,
        batch_size=2,
        example_weights=np.array([1., 0.]),
        shuffle=False,
        learning_rate=0.001)
    logprobs_from_zero_weight = _compute_logprob(
        seqs, lm, fine_tune_weights_from_zero_weight)

    # Fine tune using only the first example. Use an example weight of 2.0
    # to compensate for halving the number of tokens that the per-token
    # loss is computed over vs. above.
    fine_tune_weights_from_one_example = fine_tune.fine_tune(
        lm,
        init_weights,
        seqs[:1],
        num_epochs=1,
        batch_size=1,
        example_weights=np.array([2.]),
        shuffle=False,
        learning_rate=0.001)
    logprobs_from_one_example = _compute_logprob(
        seqs, lm, fine_tune_weights_from_one_example)

    self.assertAllClose(logprobs_from_zero_weight, logprobs_from_one_example)


if __name__ == '__main__':
  tf.test.main()
