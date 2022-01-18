# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""Tests for evaluation."""

import functools
import numpy as np
import tensorflow.compat.v1 as tf
from protein_lm import domains
from protein_lm import evaluation
from protein_lm import models

tf.enable_eager_execution()

lm_cls = functools.partial(
    models.FlaxLM,
    num_layers=1,
    num_heads=1,
    emb_dim=64,
    mlp_dim=64,
    qkv_dim=64)


class EvaluationTest(tf.test.TestCase):

  def test_empirical_baseline_construction(self):
    """Tests that EmpiricalBaseline construction is correct."""
    domain = domains.FixedLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=range(3), include_bos=True),
        length=2)
    train_data = np.array([[0, 1], [1, 0]])
    train_ds = tf.data.Dataset.from_tensor_slices((train_data,))
    eb = evaluation.EmpiricalBaseline(domain, train_ds, alpha=0)
    self.assertAllEqual(eb._empirical_dist, [0.5, 0.5, 0])

  def test_empirical_baseline_evaluation(self):
    """Tests that EmpiricalBaseline evaluation is correct."""
    domain = domains.FixedLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=range(2), include_bos=True),
        length=1)
    train_data = np.array([[0, 1], [1, 0]])
    train_ds = tf.data.Dataset.from_tensor_slices((train_data,))
    eval_data = np.array([[0, 1], [1, 0]])
    eval_ds = tf.data.Dataset.from_tensor_slices((eval_data,))
    eb = evaluation.EmpiricalBaseline(domain, train_ds)
    metrics = evaluation.evaluate(eb, eval_ds)
    self.assertAllEqual(np.asarray(metrics['accuracy']), 0.5)
    self.assertAllClose(np.asarray(metrics['perplexity']), 2)
    self.assertAllClose(np.asarray(metrics['loss']), 0.69, atol=0.1)

  def test_flaxlm_evaluation(self):
    """Tests that FlaxLM evaluation runs."""
    domain = domains.FixedLengthDiscreteDomain(
        vocab=domains.Vocabulary(tokens=range(2), include_bos=True),
        length=1)
    eval_data = np.array([[0, 1], [1, 0]])
    eval_ds = tf.data.Dataset.from_tensor_slices((eval_data,))
    lm = lm_cls(domain=domain)
    evaluation.evaluate(lm, eval_ds)


if __name__ == '__main__':
  tf.test.main()
