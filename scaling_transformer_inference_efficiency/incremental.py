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
few_shot_examples_result = jitted_model.prefill(params, [], few_shot_examples)
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
    params, my_sampling, [few_shot_examples_result], sample_ids)
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
tasks_result = jitted_model.prefill(params, [few_shot_examples_result], tasks)
# Generate 2 samples for each task. This sums to 6 samples in total.
sample_ids = jnp.arange(6, jnp.int32)
task_samples, task_samples_results = jitted_model.generate(
    params, my_sampling, [few_shot_examples_result, tasks_result], sample_ids)
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
    params, [few_shot_examples_result, generated_text_result], user_response)
# Generate another AI response, using `generate`.
ai_response_text, ai_response_result = jitted_model.generate(
    params, my_sampling,
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
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import jax
from jax import lax
from jax.experimental import pjit
from jax.experimental.maps import Mesh
from jax.experimental.maps import xmap
import jax.numpy as jnp
import jax.scipy
import numpy as np

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import global_to_per_device
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import ChunkResult
from scaling_transformer_inference_efficiency.chunk import FullChunkResult
from scaling_transformer_inference_efficiency.chunk import InferFn
from scaling_transformer_inference_efficiency.global_to_per_device import shard_map
from scaling_transformer_inference_efficiency.sampling import Sampling

Weights = weights.Weights
P = pjit.PartitionSpec


def _bos_logits(vocab_size):
  """Logits that put assign probability 1.0 to on _BOS_ID."""
  logits = jnp.full((vocab_size,), -1e10)
  return logits.at[0].set(0.0)


class JittedModel:
  """A model with JIT-compiled prefill and generate functions."""

  def __init__(self, hparams, eos_id,
               infer_fn, weights_logical_axes,
               rules):
    self._hparams = hparams
    self._eos_id = eos_id
    self._infer = infer_fn
    self._mesh = partitioning.make_mesh()
    self.rules = partitioning.PartitioningRules(rules)
    with self.rules:
      self._weights_axes = jax.tree_map(partitioning.logical_to_physical,
                                        weights_logical_axes)
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
      params,
      prefix,
      chunk,
      prev_logits,
  ):
    """Analyzes a chunk of input text with the neural net. Jitted function."""
    return self._infer(params, prefix,
                       chunk).to_chunk_result(prev_logits, chunk)

  def prefill(self, params, prefix,
              chunk):
    """Non-generative inference for a batch.

    Args:
      params: Model weights.
      prefix: Already-processed tokens in the prefix, if any.
      chunk: The tokens to prefill.

    Returns:
      Scores for the batch.
    """
    with self._mesh, self.rules:
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
          out_axis_resources=result_axes)(params, prefix, chunk, prev_logits)

  @partial(jax.jit, static_argnums=(0, 1))
  def _generate_impl(self, steps, params, sampling,
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

      token_full_chunk_result = self._infer(params,
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

  def generate(self, steps, params, sampling,
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
      params: Model weights.
      sampling: Controls how we sample from the model.
      prefix: Already-processed tokens in the prefix, if any.
      sample_ids: Per-sample random seeds to use for sampling. By convention,
        you should generally number these sequentially starting from 0.
        int32[num_samples]

    Returns:
      The generated text, together with its processed results.
    """
    with self._mesh, self.rules:
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
                             pjit.FROM_GDA,
                             partitioning.logical_to_physical(
                                 P('logit_batch'))),
          out_axis_resources=(None, result_axes))(steps, params, sampling,
                                                  prefix,
                                                  prev_chunk_next_token_logits,
                                                  sample_ids)


########################### Xmap version ######################################


class XmapModel:
  """A model with xmapped JIT-compiled prefill and generate functions."""

  def __init__(self, hparams, eos_id,
               infer_fn, weights_logical_axes,
               rules):
    self._hparams = hparams
    self._eos_id = eos_id
    self._infer = infer_fn
    self._mesh = partitioning.make_mesh()
    self.rules = partitioning.PartitioningRules(rules)
    with self.rules:
      self._weights_axes = jax.tree_map(partitioning.logical_to_physical,
                                        weights_logical_axes)
    # _prefill_p: maps num_prefixes -> jitted _prefill_impl function
    self._prefill_p = {}
    # _score_p: maps num_prefixes -> jitted _generate_impl function
    self._generate_p = {}

  @property
  def mesh(self):
    """Gets the mesh used for this model."""
    return self._mesh

  def rotate_weights(self, params, latency = True):
    """Rotate the weights for the collectives.

    Assumed to occur in a per device form. Assumes 2D partitioning.
    q_wi: [layers, heads.YZ, dmodel.X, q_wi_per_head]
    o_wo: [layers, heads.YZ, owo_per_head, dmodel.X]

    Args:
      params: unmodified paramaters
      latency: Whether to do latency collectives

    Returns:
      params: new parameters, rotated for a given collective
    """

    def rotate(params):
      new_layer = params.layer
      if latency:
        new_layer = new_layer.replace(
            q_wi=collectives.preshuffle_for_reducescatter_bidirectional_latency(
                new_layer.q_wi, sharded_dim='x', scatter_dim=1))
        new_layer = new_layer.replace(
            o_wo=collectives.preshuffle_for_async_matmul_allgather_latency(
                new_layer.o_wo, shuffle_axis=1, shard_axis='x'))
      else:
        new_layer = new_layer.replace(
            q_wi=collectives
            .preshuffle_for_reducescatter_bidirectional_throughput(
                new_layer.q_wi, 'x', scatter_dim=1, subsplit_dim=3))
        new_layer = new_layer.replace(
            o_wo=collectives.preshuffle_for_async_matmul_allgather_throughput(
                new_layer.o_wo, shuffle_axis=1, shard_axis='x'))

      return params.replace(layer=new_layer)

    with self._mesh, self.rules:
      params = jax.jit(
          shard_map(
              rotate,
              self._mesh,
              in_pspecs=(self._weights_axes,),
              out_pspecs=self._weights_axes))(
                  params)

    return params

  @partial(jax.jit, static_argnums=(0,))
  def prepare_inputs(
      self,
      params,
      prefix,
      sample_ids = None
  ):
    """Prepares inputs for hard xmap pass by pulling out axes to be mapped."""

    params_xmap, _ = global_to_per_device.fold_out_tree(self._mesh, params,
                                                        Weights.logical_axes())

    cache = [p.kv_cache for p in prefix]
    if prefix:
      cache_logical = [attention.KVCache.logical_axes() for _ in cache]
      cache_xmap, _ = global_to_per_device.fold_out_tree(
          self._mesh, cache, cache_logical)
    else:
      cache_xmap = []

    if prefix:
      # prev_logits = prefix[-1].next_token_logits
      prev_logits, _ = global_to_per_device.fold_out_tree(
          self._mesh, prefix[-1].next_token_logits,
          ChunkResult.logical_axes().next_token_logits)
    else:
      prev_logits = None

    if sample_ids is not None:
      sample_ids, _ = global_to_per_device.fold_out(self._mesh, sample_ids,
                                                    P('logit_batch'))
    return params_xmap, cache_xmap, prev_logits, sample_ids

  @partial(jax.jit, static_argnums=(0,))
  def prepare_chunk(self, chunk):
    chunk_xmap, _ = global_to_per_device.fold_out_tree(self._mesh, chunk,
                                                       Chunk.logical_axes())
    return chunk_xmap

  @partial(jax.jit, static_argnums=(0,))
  def to_chunk_result(self, full_chunk_result,
                      prev_logits,
                      chunk):
    """Maps to only the info we need for future processing."""

    full_chunk_result = jax.tree_util.tree_map(global_to_per_device.fold_in,
                                               full_chunk_result,
                                               FullChunkResult.logical_axes())
    prev_logits = jax.tree_util.tree_map(
        global_to_per_device.fold_in, prev_logits,
        ChunkResult.logical_axes().next_token_logits)

    res = full_chunk_result.to_chunk_result(prev_logits, chunk)

    result_axes = jax.tree_map(partitioning.logical_to_physical,
                               ChunkResult.logical_axes())
    result_named_sharding = jax.tree_map(
        partial(jax.sharding.NamedSharding, self._mesh), result_axes)
    return jax.tree_util.tree_map(jax.device_put, res, result_named_sharding)

  def prefill(self, params, prefix,
              chunk):
    """Non-generative inference for a batch.

    Args:
      params: Model weights.
      prefix: Already-processed tokens in the prefix, if any.
      chunk: The tokens to prefill.

    Returns:
      Scores for the batch.
    """
    with self.rules:
      chunk_layout = jax.tree_map(global_to_per_device.logical_to_layout,
                                  Chunk.logical_axes())
      params_layouts = jax.tree_map(global_to_per_device.logical_to_layout,
                                    Weights.logical_axes())
      result_layout = jax.tree_map(global_to_per_device.logical_to_layout,
                                   FullChunkResult.logical_axes())
      cache_layout = jax.tree_map(
          global_to_per_device.logical_to_layout,
          [attention.KVCache.logical_axes() for _ in prefix])

      with self._mesh:
        # TODO(sholto): We may need to pull this out with real size weights
        # Or just get shardmap / xmap in pjit actually working.
        params_xmap, cache_xmap, prev_logits, _ = self.prepare_inputs(
            params, prefix)

        chunk_xmap = self.prepare_chunk(chunk)
        # 2D: logits: [batch.x, time, vocab.YZ]
        #     kv_cache: [.., ..., prefixbatch, ...]
        full_chunk_result = xmap(
            self._infer,
            in_axes=(params_layouts, cache_layout, chunk_layout),
            out_axes=result_layout,
            axis_resources={
                'x': 'x',
                'y': 'y',
                'z': 'z'
            })(params_xmap, cache_xmap, chunk_xmap)

        return self.to_chunk_result(full_chunk_result, prev_logits, chunk)

  # pylint: disable = protected-access
  @staticmethod
  def _generate_impl(model, steps, sampling, params,
                     prefix,
                     prev_chunk_next_token_logits,
                     sample_ids):
    """Generates a chunk of text. Xmapped version."""
    logit_sharded_batch = sample_ids.shape[0]
    logit_sharding_axis_size = partitioning.get_sharding_divisor(
        P('logit_batch'))
    global_batch = logit_sharded_batch * logit_sharding_axis_size
    # We are very deliberate about savings comms.
    # By default, chunk info is small, so we don't shard by batch except
    # the kv cache. Otherwise we would be doing frequent all-gathers on
    # tiny chunks before each inference step.
    # Logits are huge! So we shard them by X in batch and YZ in vocab.
    # this means that we track two different sized batch dims.
    # Seeding of the RNG itself is deterministic. To generate different samples,
    # users can provide sample_number_offset.
    sample_rngs = jax.vmap(
        jax.random.fold_in, in_axes=(None, 0))(jax.random.PRNGKey(0),
                                               sample_ids)
    # jax.debug.breakpoint()
    token_indexes_start = attention.prefix_lengths(prefix)
    token_indexes_start = attention.flat_broadcast(token_indexes_start,
                                                   logit_sharded_batch)

    last_logits = prev_chunk_next_token_logits
    if last_logits is None:
      last_logits = _bos_logits(model._hparams.vocab)[np.newaxis, :]

    last_logits = attention.flat_broadcast(last_logits, logit_sharded_batch)

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
      # logits: float32[batch.X, vocab.YZ], step_rngs: [batch] (sliced later)
      # ->  sample: int32[batch]
      next_token = sampling.sample_manual(chunk_result.next_token_logits,
                                          step_rngs)
      # ^ next_token: [batch] - Note, as yet unsharded
      token_chunk = Chunk(
          tokens=next_token[:, np.newaxis],
          lengths=jnp.full((global_batch,), 1, jnp.int32))
      # ^ token_chunk: Chunk[batch, 1]
      token_full_chunk_result = model._infer(params,
                                             prefix + [chunk_result.kv_cache],
                                             token_chunk)
      chunk = chunk.update(token_i, token_chunk)
      chunk_result = chunk_result.update(
          token_i, token_chunk, token_full_chunk_result, per_device=True)
      return chunk, chunk_result

    chunk = Chunk.zeros(global_batch, steps)
    kv_batch = global_batch // partitioning.get_sharding_divisor(
        P('attn_batch'))
    chunk_result = ChunkResult.zeros(model._hparams, global_batch, steps,
                                     kv_batch)
    chunk_result = chunk_result.replace(next_token_logits=last_logits)

    chunk, chunk_result = lax.fori_loop(0, steps, loop_body,
                                        (chunk, chunk_result))

    # The length of the chunk is the index of the first EOS ID. We don't
    # calculate this during the generation loop, so instead we calculate it now.
    is_eos = chunk.tokens == model._eos_id
    token_iota = lax.broadcasted_iota(jnp.int32, (global_batch, steps), 1)
    chunk = chunk.replace(
        lengths=jnp.min(jnp.where(is_eos, token_iota, steps), axis=1))

    return chunk, chunk_result

  def instantiate_generating_fn(
      self, steps,
      sampling):  # pylint: disable = g-bare-generic
    """Create partial fn, because xmap cannot mark args as static."""
    return partial(self._generate_impl, self, steps, sampling)

  def generate(
      self,
      params,
      generate_fn,  # pylint: disable = g-bare-generic
      prefix,
      sample_ids):
    """Generative inference for a batch. See pjit version for details."""

    with self.rules:
      params_layouts = jax.tree_map(global_to_per_device.logical_to_layout,
                                    Weights.logical_axes())
      result_layout = jax.tree_map(global_to_per_device.logical_to_layout,
                                   ChunkResult.logical_axes())
      cache_layout = jax.tree_map(
          global_to_per_device.logical_to_layout,
          [attention.KVCache.logical_axes() for _ in prefix])

      prev_chunk_next_token_layout = jax.tree_map(
          global_to_per_device.logical_to_layout,
          ChunkResult.logical_axes().next_token_logits)

      # 2D: [batch.X,]
      sample_ids_layout = global_to_per_device.logical_to_layout(
          P('logit_batch'))

      with self._mesh:
        # TODO(sholto): We may need to pull this out with real size weights
        # Or just get shardmap / xmap in pjit actually working.
        (params_xmap, cache_xmap, prev_chunk_next_token_logits,
         sample_ids_xmap) = self.prepare_inputs(params, prefix, sample_ids)

        print('kv_cacheextern', jax.tree_map(jnp.shape, cache_xmap))
        return xmap(
            generate_fn,
            in_axes=(params_layouts, cache_layout, prev_chunk_next_token_layout,
                     sample_ids_layout),
            out_axes=({}, result_layout),  # fully replicate sample indices
            axis_resources={
                'x': 'x',
                'y': 'y',
                'z': 'z'
            })(params_xmap, cache_xmap, prev_chunk_next_token_logits,
               sample_ids_xmap)
