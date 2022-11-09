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

"""Support for incremental processing on Transformers.

We provide two functions, `prefill` and `generate`, which operate on the
`Chunk` and `ChunkResult` types from chunk.py.
* The function `prefill`, sometimes also called 'encode' in Transformer
  codebases, runs a single forwards pass over a batch of input sequences,
  returning scores and KV caches for those tokens.
* The function `generate`, sometimes also called 'decode' in Transformer
  codebases, generates new text autoregressively, in a sequential loop that
  generates one token at a time per sequence.

Example use cases follow. Each example builds upon the previous.

Example 1: scoring some text
============================

We create a `Chunk` of input text and then run `prefill` on it.

```
jitted_model = JittedModel(...)

# Create a batch=1 input chunk of text.
few_shot_examples = Chunk.tokenize(
    vocab, ["Cows have 4 legs. Fish have 0 legs."], is_first_chunk=True)
few_shot_examples_result = jitted_model.prefill(weights, [], few_shot_examples)
print(few_shot_examples_result.per_token_scores)
```

Example 2: generating text using the prompt
===========================================

We use the `few_shot_examples_result` from the previous example as attention
context (the KV cache) from which we generate new text.

```
# Controls random sampling
my_sampling = Sampling(temperature=0.7)
# 4 random seeds, so that we generate 4 different outputs.
sample_ids = jnp.arange(4, jnp.int32)
generated_text, generated_text_result = jitted_model.generate(
    weights, my_sampling, [few_shot_examples_result], sample_ids)
# Print all 4 samples
for s in generated_text.detokenize(vocab):
  print(s)
```

Example 3: Multiple prompts sharing a common prefix
===================================================

In a few-shot-prompted scenario, we typically have a common prefix (the few-shot
prompts), shared over a batch of tasks, and for each task we generate multiple
samples. By splitting each of these steps into its own `prefill` or `generate`
call, we can do this in a way that maximally exploits the sharing.

In Example 1 we already called `prefill` on single shared sequence which has the
few-shot examples. Next we call `prefill` on the batch of tasks, using the
few-shot examples as attention context. It is permissible to have more tasks
than few-shot examples, as we demonstrate here:

```
# Prefill a batch=3 set of tasks.
tasks = Chunk.tokenize(vocab, ["Humans have", "Potatos have", "Dinosaurs have"])
tasks_result = jitted_model.prefill(weights, [few_shot_examples_result], tasks)
# Generate 2 samples for each task. This sums to 6 samples in total.
sample_ids = jnp.arange(6, jnp.int32)
task_samples, task_samples_results = jitted_model.generate(
    weights, my_sampling, [few_shot_examples_result, tasks_result], sample_ids)
```

Example 4: appending even more text, and then generating some more
==================================================================

If we were in a chatbot scenario, at this point we might append some more
user-provided text to the context, and then generate yet another response. This
consists of another call to `prefill` followed by another call to `generate`.
As this example shows, they can be arbitrarily combined any fashion.

```
# Add the user response, using `prefill`.
user_response = Chunk.tokenize(vocab, ["How many legs does a chicken have?"])
user_response_result = jitted_model.prefill(
    weights, [few_shot_examples_result, generated_text_result], user_response)
# Generate another AI response, using `generate`.
ai_response_text, ai_response_result = jitted_model.generate(
    weights, my_sampling,
    [few_shot_examples_result, generated_text_result, user_response_result],
    sample_ids
)
# Print all 4 samples
for s in generated_text.detokenize(vocab):
  print(s)
```

TODO(reinerp): Example 4 uses an ever-increasing list of ChunkResults as
context arguments. In a more realistic chatbot scenario we would concatenate all
the ChunkResults into a single longer ChunkResult, subject to batch size
restrictions.
"""

from functools import partial  # pylint: disable=g-importing-member
from typing import Any, List, Optional, Sequence, Tuple

import jax
from jax import lax
from jax.experimental import pjit
from jax.experimental.maps import Mesh
import jax.numpy as jnp
import jax.scipy
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import ChunkResult
from scaling_transformer_inference_efficiency.chunk import InferFn
from scaling_transformer_inference_efficiency.sampling import Sampling

Weights = Any
P = pjit.PartitionSpec


def _bos_logits(vocab_size):
  """Logits that put assign probability 1.0 to on _BOS_ID."""
  logits = jnp.full((vocab_size,), -1e10)
  return logits.at[0].set(0.0)


class JittedModel:
  """A model with JIT-compiled prefill and generate functions."""

  def __init__(self, hparams, eos_id,
               infer_fn, weights_physical_axes):
    self._hparams = hparams
    self._eos_id = eos_id
    self._infer = infer_fn
    self._weights_axes = weights_physical_axes
    self._mesh = partitioning.make_mesh()
    # _prefill_p: maps num_prefixes -> jitted _prefill_impl function
    self._prefill_p = {}
    # _score_p: maps num_prefixes -> jitted _generate_impl function
    self._generate_p = {}

  @property
  def mesh(self):
    """Gets the mesh used for this model."""
    return self._mesh

  def prefix_axes(
      self, prefix):
    num_prefixes = len(prefix)
    kv_cache_axes = jax.tree_map(partitioning.logical_to_physical,
                                 attention.KVCache.logical_axes())
    return [kv_cache_axes] * num_prefixes

  @partial(jax.jit, static_argnums=(0,))
  def _prefill_impl(
      self,
      weights,
      prefix,
      chunk,
      prev_logits,
  ):
    """Analyzes a chunk of input text with the neural net. Jitted function."""
    return self._infer(weights, prefix,
                       chunk).to_chunk_result(prev_logits, chunk)

  def prefill(self, weights, prefix,
              chunk):
    """Non-generative inference for a batch.

    Args:
      weights: Model weights.
      prefix: Already-processed tokens in the prefix, if any.
      chunk: The tokens to prefill.

    Returns:
      Scores for the batch.
    """
    with self._mesh:
      if prefix:
        prev_logits = prefix[-1].next_token_logits
      else:
        prev_logits = None

      result_axes = jax.tree_map(partitioning.logical_to_physical,
                                 ChunkResult.logical_axes())
      prefix = [p.kv_cache for p in prefix]
      return pjit.pjit(
          self._prefill_impl,
          in_axis_resources=(self._weights_axes, self.prefix_axes(prefix), None,
                             pjit.FROM_GDA),
          out_axis_resources=result_axes)(weights, prefix, chunk, prev_logits)

  @partial(jax.jit, static_argnums=(0, 1))
  def _generate_impl(self, steps, weights, sampling,
                     prefix,
                     prev_chunk_next_token_logits,
                     sample_ids):
    """Generates a chunk of text, given some prefixes. Jitted function."""
    batch, = sample_ids.shape

    # Seeding of the RNG itself is deterministic. To generate different samples,
    # users can provide sample_number_offset.
    sample_rngs = jax.vmap(
        jax.random.fold_in, in_axes=(None, 0))(jax.random.PRNGKey(0),
                                               sample_ids)
    token_indexes_start = attention.prefix_lengths(prefix)
    token_indexes_start = attention.flat_broadcast(token_indexes_start, batch)

    # Generation loop.

    def loop_body(token_i, state):
      # We have two different chunks at play in this loop:
      # 1. `chunk`/`chunk_result`, of shape (batch, steps). This is the
      #    mutable state that we fill one token at a time, starting at step=0
      #    and going up to step=steps-1.
      # 2. `token_chunk`/`token_chunk_result`/`token_full_chunk_result`, of
      #    shape (batch, 1). These are the input/output of a single forwards
      #    pass through the model, which processes just one chunk per batch
      #    element.
      #
      # We write token_* into chunk/chunk_result at a new position each loop
      # iteration.
      chunk, chunk_result = state
      step_rngs = jax.vmap(jax.random.fold_in)(sample_rngs,
                                               token_indexes_start + token_i)
      next_token = sampling.sample(chunk_result.next_token_logits, step_rngs)
      # ^ next_token: [batch]
      token_chunk = Chunk(
          tokens=next_token[:, np.newaxis],
          lengths=jnp.full((batch,), 1, jnp.int32))
      # ^ token_chunk: Chunk[batch, 1]

      token_full_chunk_result = self._infer(weights,
                                            prefix + [chunk_result.kv_cache],
                                            token_chunk)
      chunk = chunk.update(token_i, token_chunk)
      chunk_result = chunk_result.update(token_i, token_chunk,
                                         token_full_chunk_result)
      return chunk, chunk_result

    last_logits = prev_chunk_next_token_logits
    if last_logits is None:
      last_logits = _bos_logits(self._hparams.vocab)[np.newaxis, :]
    last_logits = attention.flat_broadcast(last_logits, batch)

    chunk = Chunk.zeros(batch, steps)
    chunk_result = ChunkResult.zeros(self._hparams, batch, steps)
    chunk_result = chunk_result.replace(next_token_logits=last_logits)

    chunk, chunk_result = lax.fori_loop(0, steps, loop_body,
                                        (chunk, chunk_result))

    # The length of the chunk is the index of the first EOS ID. We don't
    # calculate this during the generation loop, so instead we calculate it now.
    is_eos = chunk.tokens == self._eos_id
    token_iota = lax.broadcasted_iota(jnp.int32, (batch, steps), 1)
    chunk = chunk.replace(
        lengths=jnp.min(jnp.where(is_eos, token_iota, steps), axis=1))

    return chunk, chunk_result

  def generate(self, steps, weights, sampling,
               prefix,
               sample_ids):
    """Generative inference for a batch.

    Note about random number seeding:
    We provide strong guarantees about random number seeding, to make it
    possible for callers to get deterministic results that are independent of
    batch packing and independent of splitting into `Chunk`s for incremental
    processing. Specifically, we guarantee that random numbers are constructed
    by:

    ```
    def rng_for_token(sample_id: int, token_index: int) -> jax.random.KeyArray:
      rng = jax.random.PRNGKey(0)
      rng = jax.random.fold_in(rng, sample_id)
      rng = jax.random.fold_in(rng, token_index)
      return rng
    ```

    Here, `sample_id` is taken from the `sample_ids` array provided by the
    caller, and `token_index` is the number of non-padding tokens in this sample
    prior to the token currently being generated. This scheme that any text
    generated with the same `sample_id`, from the same prefix, using the same
    sampling hyperparameters, will make the same random decisions and will
    therefore be deterministic, independent of batch packing or chunk splitting.

    Args:
      steps: Number of steps to generate.
      weights: Model weights.
      sampling: Controls how we sample from the model.
      prefix: Already-processed tokens in the prefix, if any.
      sample_ids: Per-sample random seeds to use for sampling. By convention,
        you should generally number these sequentially starting from 0.
        int32[num_samples]

    Returns:
      The generated text, together with its processed results.
    """
    with self._mesh:
      if prefix:
        prev_chunk_next_token_logits = prefix[-1].next_token_logits
      else:
        prev_chunk_next_token_logits = None

      prefix = [p.kv_cache for p in prefix]
      result_axes = jax.tree_map(partitioning.logical_to_physical,
                                 ChunkResult.logical_axes())
      return pjit.pjit(
          self._generate_impl,
          static_argnums=0,
          in_axis_resources=(self._weights_axes, None, self.prefix_axes(prefix),
                             pjit.FROM_GDA, None),
          out_axis_resources=(None, result_axes))(steps, weights, sampling,
                                                  prefix,
                                                  prev_chunk_next_token_logits,
                                                  sample_ids)
