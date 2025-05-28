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

"""Top-k, topp and temperature sampling methods."""

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

NEG_INF = -1.0e7  # Masking purpose


def sampling(logits, rng, topk=0, topp=0, temperature=1.0):
  """top-k and nucleus sampling method.

  The nucleus sampling method is proposed in the paper `The Curious Case of
  Neural Text Degeneration (https://arxiv.org/pdf/1904.09751.pdf)`

  Args:
    logits: array: [batch_size, ..., vocab_size] logits at the current position.
    rng: JAX random state.
    topk: int: only the top-k logits will be considered to sample next token.
      If topk is zero, sample from full logits.
    topp: float: the smallest number of logits whose cumulative sum of probs
                 adds up to topp. The next token will be sampled from these
                 tokens. If zero, sample from full logits.
    temperature: float: sampling temperature factor. As it approaches zero this
                        becomes equivalent to greedy sampling. As it approaches
                        inf, this becomes uniform sampling.

  Returns:
    Array of sampled sequences: [batch_size, ...]
  """
  if topp < 0 or topk < 0:
    raise ValueError("`topp` and `topk` must be non-negative")
  if topp and topk:
    raise ValueError("At most one of `topp` or `topk` can be non-zero.")
  if topp > 0:
    logits = top_p_logits(logits, topp)
  if topk > 0:
    return top_k_sampling(logits, rng, topk, temperature)
  else:
    return jax.random.categorical(rng, logits / temperature)


def top_k_sampling(logits, rng, topk=0, temperature=1.0):
  """Only top k probable tokens should be considered for sampling."""
  topk_logits, topk_idxs = lax.top_k(logits, topk)
  topk_token = jnp.expand_dims(
      random.categorical(rng, topk_logits/temperature).astype(jnp.int32),
      axis=-1)
  # Returns the original indices corresponding to the sampled top-k tokens.
  # [batch_size, ...]
  sampled_tokens = jnp.squeeze(
      jnp.take_along_axis(topk_idxs, topk_token, axis=-1),
      axis=-1).astype(jnp.int32)
  return sampled_tokens


def top_p_logits(logits, topp=0):
  """Finds the top logits with cumulative probability >= top and mask others."""
  logits_sorted = jnp.sort(logits, axis=-1)[Ellipsis, ::-1]  # sort descending
  sorted_cum_probs = jnp.cumsum(
      jax.nn.softmax(logits_sorted, axis=-1), axis=-1)  # get cumsum probs
  cutoff_index = jnp.sum(
      sorted_cum_probs < topp, axis=-1, keepdims=True)  # find cutoff index
  cutoff_logit = jnp.take_along_axis(logits_sorted, cutoff_index, axis=-1)
  logits = jnp.where(logits < cutoff_logit,
                     jnp.full_like(logits, NEG_INF), logits)
  return logits
