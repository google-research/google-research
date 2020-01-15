# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for Transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.data_generators import problem_hparams

import tensorflow.compat.v1 as tf

from state_of_sparsity.sparse_transformer.models import sparse_transformer

BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10


def get_model(
    hparams=None,
    mode=tf.estimator.ModeKeys.TRAIN,
    has_input=True,
    model_cls=sparse_transformer.SparseTransformer):
  if hparams is None:
    hparams = sparse_transformer.sparse_transformer_tiny()
  hparams.hidden_size = 8
  hparams.filter_size = 32
  hparams.num_heads = 1
  hparams.layer_prepostprocess_dropout = 0.0

  p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE,
                                                   VOCAB_SIZE,
                                                   hparams)
  if not has_input:
    del p_hparams.modality["inputs"]
  hparams.problem_hparams = p_hparams

  inputs = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
  targets = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  return model_cls(hparams, mode, p_hparams), features


class SparseTransformerTest(tf.test.TestCase):

  def testTransformerBase(self):
    model, features = get_model(sparse_transformer.sparse_transformer_tiny())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerVariationalDropout(self):
    model, features = get_model(
        sparse_transformer.sparse_transformer_tiny_variational_dropout())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerMagnitudePruning(self):
    model, features = get_model(
        sparse_transformer.sparse_transformer_tiny_magnitude_pruning())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerRandomPruning(self):
    model, features = get_model(
        sparse_transformer.sparse_transformer_tiny_random_pruning())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerL0Regularization(self):
    model, features = get_model(
        sparse_transformer.sparse_transformer_tiny_l0_regularization())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerSplitHeadMagnitudePruning(self):
    model, features = get_model(
        sparse_transformer.sparse_transformer_tiny_shmp())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

if __name__ == "__main__":
  tf.test.main()
