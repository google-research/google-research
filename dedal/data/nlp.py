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

"""Implements data transforms for pretraining of (protein) LMs."""

from typing import Tuple

import gin
import tensorflow as tf
from tensorflow import math
import tensorflow_probability as tfp

from dedal import vocabulary
from dedal.data import transforms


@gin.configurable
class DynamicLanguageModelMasker(transforms.Transform):
  """Dynamically masks input to create a BERT-like Cloze task."""

  def __init__(self,
               masked_lm_prob = 0.15,
               mask_token_prob = 0.80,
               resample_token_prob = 0.50,
               deterministic_n_lm_tokens = True,
               **kwargs):
    super().__init__(**kwargs)
    self._masked_lm_prob = masked_lm_prob
    self._mask_token_prob = mask_token_prob
    self._resample_token_prob = resample_token_prob
    self._deterministic_n_lm_tokens = deterministic_n_lm_tokens

    # tf.where-based "branching" for BERT's Cloze task
    self._branch1_sampler = (
        tfp.distributions.Uniform() if self._deterministic_n_lm_tokens
        else tfp.distributions.Bernoulli(
            probs=self._masked_lm_prob, dtype=tf.bool)
        )
    self._branch2_sampler = tfp.distributions.Bernoulli(
        probs=self._mask_token_prob, dtype=tf.bool)
    self._branch3_sampler = tfp.distributions.Bernoulli(
        probs=self._resample_token_prob, dtype=tf.bool)

    # Resample (integer-valued) tokens uniformly at random, ignoring any special
    # tokens in the vocabulary
    self._resample_sampler = vocabulary.Sampler(self._vocab)

  def _get_lm_mask(
      self, inputs_mask, branch1):
    """Dynamically selects tokens to be part of the Cloze task."""
    inputs_shape = tf.shape(inputs_mask)
    batched = len(inputs_shape) == 2
    batch_size = inputs_shape[0] if batched else None
    padded_len = inputs_shape[1] if batched else inputs_shape[0]
    # Select a proportion self._masked_lm_prob of the tokens in each sequence
    # uniformly at random...
    if self._deterministic_n_lm_tokens:
      # tf.function is finnicky with types...
      inputs_mask = tf.cast(inputs_mask, tf.float32)
      masked_lm_prob = tf.cast(self._masked_lm_prob, tf.float32)
      padded_len = tf.cast(padded_len, tf.float32)

      n_normal_tokens_per_seq = tf.reduce_sum(inputs_mask, -1)
      max_lm_tokens_per_seq = tf.cast(masked_lm_prob * padded_len, tf.int32)
      lm_tokens_per_seq = tf.maximum(
          1, tf.cast(masked_lm_prob * n_normal_tokens_per_seq, tf.int32))

      indices = lm_tokens_per_seq - 1
      if batched:
        indices = tf.stack((tf.range(batch_size), indices), axis=-1)
      branch1 *= inputs_mask
      values, _ = math.top_k(branch1, k=max_lm_tokens_per_seq)
      cutoffs = (tf.gather(values, indices) if not batched
                 else tf.gather_nd(values, indices)[:, None])

      return branch1 >= cutoffs
    # ... otherwise, select them via i.i.d. coin tosses with success probability
    # equal to self._masked_lm_prob
    return tf.logical_and(inputs_mask, branch1)

  def call(self, inputs):
    """Dynamically masks a batch of tokens for BERT-like pretraining.

    Args:
      inputs: A tf.Tensor<int32>[batch, len] of sequences.

    Returns:
      A 3-tuple of tf.Tensor<int32>[batch, len] representing the masked inputs,
      the target output (equal to the unmasked input) and the mask.
    """
    inputs_shape = tf.shape(inputs)
    # inputs_mask[i][j] = True iff token of sequence i at position j is a normal
    # token, i.e. neither padding nor <CLS> / <EOS>
    inputs_mask = self._vocab.padding_mask(inputs)
    inputs_mask = tf.math.logical_and(
        inputs_mask, self._vocab.special_token_mask(inputs))

    # Sample Bernoulli-distributed tensors that will decide the "branching" on a
    # token by token basis...
    branch1 = self._branch1_sampler.sample(inputs_shape)
    branch2 = self._branch2_sampler.sample(inputs_shape)
    branch3 = self._branch3_sampler.sample(inputs_shape)
    # ...as well as the tokens resampled uniformly at random
    resampled_inputs = self._resample_sampler.sample(inputs_shape)

    # Mask indicating which tokens form the Cloze task in this call
    lm_mask = self._get_lm_mask(inputs_mask, branch1)

    # Build the final input
    mask_token_cond = tf.logical_and(lm_mask, branch2)
    resample_token_cond = tf.logical_and(
        tf.logical_and(lm_mask, tf.logical_not(branch2)), branch3)

    masked_inputs = tf.where(mask_token_cond,
                             tf.cast(self._vocab.mask_code, inputs.dtype),
                             inputs)
    masked_inputs = tf.where(resample_token_cond,
                             tf.cast(resampled_inputs, masked_inputs.dtype),
                             masked_inputs)

    return masked_inputs, inputs, tf.cast(lm_mask, tf.float32)


@gin.configurable
class DynamicLanguageModelShifter(transforms.Transform):
  """Dynamically shifts input to create an autoregressive LM task.

  Attributes:
    batched: Whether the transformation will be applied to batches (2D tensors)
      or individual examples (1D tensors). Always True for this class.
  """

  def __init__(self, eos_as_token = False, **kwargs):
    super().__init__(**kwargs)
    self._eos_as_token = eos_as_token

  def call(self, inputs):
    """Dynamically masks a batch of tokens for BERT-like pretraining.

    Args:
      inputs: A tf.Tensor<int32>[batch, len] of sequences.

    Returns:
      A 3-tuple of tf.Tensor<int32>[batch, len] representing the shifted inputs,
      the target output (equal to to the unshited input), and the mask.
    """
    result = inputs
    result = tf.pad(result[:, :-1], [[0, 0], [1, 0]],
                    constant_values=self._vocab.get('CLS'))
    weights = self._vocab.padding_mask(inputs)
    if not self._eos_as_token:
      weights = tf.logical_and(weights, self._vocab.special_token_mask(inputs))

    return result, inputs, weights
