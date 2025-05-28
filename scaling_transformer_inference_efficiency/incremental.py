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

from dataclasses import dataclass  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import jax
from jax import lax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import jax.scipy
from jax.sharding import Mesh
import numpy as np
from seqio.vocabularies import Vocabulary

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.chunk import Chunk
from scaling_transformer_inference_efficiency.chunk import ChunkResult
from scaling_transformer_inference_efficiency.chunk import FullChunkResult
from scaling_transformer_inference_efficiency.chunk import InferFn
from scaling_transformer_inference_efficiency.sampling import SampleFn
from scaling_transformer_inference_efficiency.sampling import SamplingHyperParams

Weights = weights.Weights
P = jax.sharding.PartitionSpec


# pylint: disable = g-bare-generic
@dataclass
class StreamClient:
  """Used to handle streaming results."""

  prev_token_decoded: Optional[jnp.ndarray] = None
  prev_token: Optional[jnp.ndarray] = None
  stream_callback: Callable = lambda x: print(x, end='')
  stream_done_callback: Callable = lambda: None

  def find_new_chars(self, vocab: Vocabulary, next_token: np.ndarray):
    """We decode pairs because the tokenizer strips whitespace."""
    prefix = self.prev_token_decoded
    whole = (
        vocab.decode_tf(np.concatenate([self.prev_token, next_token], -1))
        .numpy()
        .decode('utf-8')
    )
    new_text = whole[len(prefix) :]
    return new_text

  def stream_result(
      self, logits: jax.Array, vocab: Vocabulary, x: int, y: int, z: int
  ):
    """Steam result back to std. For the moment only stream first element."""

    if x == 0 and y == 0 and z == 0:
      logits = np.array(logits)
      current_token = np.array(logits[0:1])
      if self.prev_token is None:
        new_chars = vocab.decode_tf(current_token).numpy().decode('utf-8')
      else:
        new_chars = self.find_new_chars(vocab, current_token)

      self.stream_callback(new_chars)
      self.prev_token = current_token  # pytype: disable=annotation-type-mismatch  # jax-ndarray
      self.prev_token_decoded = new_chars.lstrip(' ').rstrip(' ')

  def clear_prev_token(self):
    self.prev_token = None
    self.stream_done_callback()


def _bos_logits(vocab_size: int, bos_id: int = 0) -> jnp.ndarray:
  """Logits that put assign probability 1.0 to on _BOS_ID."""
  logits = jnp.full((vocab_size,), -1e10)
  return logits.at[bos_id].set(0.0)


class InferenceModel:
  """A model with xmapped JIT-compiled prefill and generate functions."""

  def __init__(
      self,
      hparams: checkpoint.HParams,
      eos_id: int,
      infer_fn: InferFn,
      sample_fn: SampleFn,
      mesh: Mesh,
      rules: Sequence[Tuple[str, Any]],
      vocab: Optional[Vocabulary] = None,
      bos_id: Optional[int] = None,  # Allow to overwrite the default value.
  ):
    self._hparams = hparams
    self._eos_id = eos_id
    self._infer = infer_fn
    self._sample = sample_fn
    self.mesh = mesh
    self.rules = partitioning.PartitioningRules(rules)
    with self.rules:
      self.sample_ids_sharding = partitioning.logical_to_physical(
          P('logit_batch')
      )
      self.embeddings_logical = P(
          'residual_batch', 'residual_time', 'residual_embed'
      )
      self.embeddings_sharding = jax.tree.map(
          partitioning.logical_to_physical, self.embeddings_logical
      )
    self.vocab = vocab
    if bos_id is None:
      if vocab is not None:
        bos_id = vocab.bos_id
      else:
        bos_id = 0
    self.bos_id = bos_id
    # _prefill_p: maps num_prefixes -> jitted _prefill_impl function
    self._prefill_p = {}
    # _score_p: maps num_prefixes -> jitted _generate_impl function
    self._generate_p = {}

  def rotate_weights(self, params: Weights, latency: bool = True) -> Weights:
    """Rotate the weights for the collectives.

    Assumed to occur in a per device form. Assumes 2D partitioning.
    q_wi: [layers, heads.YZ, dmodel.X, q_wi_per_head]
    o_wo: [layers, heads.YZ, owo_per_head, dmodel.X]

    Args:
      params: unmodified
      latency: Whether to do latency collectives

    Returns:
      params: new parameters, rotated for a given collective
    """

    def rotate(params):
      new_layer = params.layer
      if latency:
        new_layer = new_layer.replace(
            q_wi=collectives.preshuffle_for_reducescatter_latency(
                new_layer.q_wi, scatter_axis=1, axis_name='x'
            )
        )
        new_layer = new_layer.replace(
            o_wo=collectives.preshuffle_for_allgather_matmul_latency(
                new_layer.o_wo, shuffle_axis=1, axis_name='x'
            )
        )
      else:
        new_layer = new_layer.replace(
            q_wi=collectives.preshuffle_for_reducescatter_throughput(
                new_layer.q_wi, scatter_axis=1, subsplit_axis=3, axis_name='x'
            )
        )
        new_layer = new_layer.replace(
            o_wo=collectives.preshuffle_for_allgather_matmul_throughput(
                new_layer.o_wo, shuffle_axis=1, axis_name='x'
            )
        )

      return params.replace(layer=new_layer)

    with self.mesh, self.rules:
      params = jax.jit(
          shard_map(
              rotate,
              self.mesh,
              in_specs=(params.physical_axes(),),
              out_specs=params.physical_axes(),
              check_rep=False,
          ),
          donate_argnums=(0,),
      )(params)

      return params

  # pylint: disable = g-bare-generic
  # pylint: disable = protected-access
  @staticmethod
  def _prefill_impl(
      model,
      params: Weights,
      cache: Sequence[ChunkResult],
      chunk: Chunk,
      prev_logits: Optional[jnp.ndarray],
      pre_embedded_inputs: Optional[jax.Array] = None,
      return_full_chunk: bool = False,
  ) -> Union[ChunkResult, FullChunkResult]:
    """Wrap both prefill and results formatting in a single xmap call."""
    if pre_embedded_inputs is not None:
      full_chunk_result = model._infer(
          params, cache, chunk, pre_embedded_inputs=pre_embedded_inputs
      )
    else:
      full_chunk_result = model._infer(params, cache, chunk)
    if return_full_chunk:
      return full_chunk_result
    else:
      return full_chunk_result.to_chunk_result(
          prev_logits, chunk, bos_id=model.bos_id
      )

  def instantiate_prefill_fn(self, return_full_chunk: bool = False):
    return partial(
        self._prefill_impl,
        self,
        return_full_chunk=return_full_chunk,
    )

  def prefill(
      self,
      params: Weights,
      prefill_impl: Callable,
      prefix: Sequence[ChunkResult],
      chunk: Chunk,
      pre_embedded_inputs: Optional[jax.Array] = None,
  ) -> Union[ChunkResult, FullChunkResult]:
    """Non-generative inference for a batch.

    Args:
      params: Model weights.
      prefill_impl: Partialed prefillimpl
      prefix: Already-processed tokens in the prefix, if any.
      chunk: The tokens to prefill.
      pre_embedded_inputs: If we want to do the embeddings outside (e.g. for
        prompt tuning)

    Returns:
      Scores for the batch.
    """
    with self.mesh, self.rules:
      if prefix:
        prev_logits = prefix[-1].next_token_logits
      else:
        prev_logits = None

      prefix = [p.kv_cache for p in prefix]
      return jax.jit(prefill_impl)(
          params, prefix, chunk, prev_logits, pre_embedded_inputs
      )

  @staticmethod
  def create_output_buffer(
      hparams: checkpoint.HParams,
      sample_ids: jnp.ndarray,
      prefix: List[attention.KVCache],
      length: int,
      prev_chunk_next_token_logits: Optional[jnp.ndarray] = None,
      circular: bool = False,
      bos_id: int = 0,
  ):
    """Create everything we need to deterministically write output samples."""
    # Seeding of the RNG itself is deterministic.
    # To generate different samples, users can provide sample_number_offset.
    (batch,) = sample_ids.shape
    sample_rngs = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(  # pytype: disable=wrong-arg-types  # jax-ndarray
        jax.random.PRNGKey(0), sample_ids
    )
    token_indexes_start = attention.prefix_lengths(prefix)
    token_indexes_start = attention.flat_broadcast(token_indexes_start, batch)

    # Generation loop.
    last_logits = prev_chunk_next_token_logits
    if last_logits is None:
      last_logits = _bos_logits(hparams.vocab, bos_id)[np.newaxis, :]
    last_logits = attention.flat_broadcast(last_logits, batch)
    chunk = Chunk.zeros(batch, length)
    chunk_result = ChunkResult.zeros(hparams, batch, length, circular=circular)
    chunk_result = chunk_result.replace(next_token_logits=last_logits)
    return sample_rngs, token_indexes_start, chunk, chunk_result

  @staticmethod
  def sample_infer_write(
      model,
      params: Weights,
      prefix: List[attention.KVCache],
      sample_params: SamplingHyperParams,
      token_indexes_start: jnp.ndarray,
      sample_rngs: jnp.ndarray,
      write_index: int,
      state: Tuple[Chunk, ChunkResult],
  ):
    """Samples prev inference, infers, writes to cache.

    We sample first then do the next inference step because we already have
    prexisting logits from prefill, so it saves us a step. Additionally, it
    lowers better.
    We have two different chunks at play in this loop:
    1. `chunk`/`chunk_result`, of shape (batch, steps). This is the
       mutable state that we fill one token at a time, starting at step=0
       and going up to step=steps-1.
    2. `token_chunk`/`token_chunk_result`/`token_full_chunk_result`, of
       shape (batch, 1). These are the input/output of a single forwards
       pass through the model, which processes just one chunk per batch
       element.

    We write token_* into chunk/chunk_result at a new position each loop
    iteration.

    Args:
      model: InferenceModel
      params: Model weights
      prefix: pre_existing kv_cache
      sample_params: Temperature etc
      token_indexes_start: Per element token index in the full sequence, for
        fully deterministic sampling.
      sample_rngs: Rng per element
      write_index: current index in the state. Always the same for all elements
        so that it is a slice, not a scatter.
      state: chunk / chunk_result pair described above. May be a circular buffer
    Returns:
      chunk: Written to (for tokens)
      chunk_result: Written to (for KV_cache)
    """
    batch, _ = sample_rngs.shape
    chunk, chunk_result = state
    step_rngs = jax.vmap(jax.random.fold_in)(  # pytype: disable=wrong-arg-types  # jax-ndarray
        sample_rngs, token_indexes_start + chunk.lengths
    )
    next_token = model._sample(
        chunk_result.next_token_logits, step_rngs, sample_params, model.mesh,
    )
    # ^ next_token: [batch]
    token_chunk = Chunk(
        tokens=next_token[:, np.newaxis],
        lengths=jnp.full((batch,), 1, jnp.int32),
    )
    # ^ token_chunk: Chunk[batch, 1]

    token_full_chunk_result = model._infer(
        params, prefix + [chunk_result.kv_cache], token_chunk
    )
    chunk = chunk.update(write_index, token_chunk)
    chunk_result = chunk_result.update(
        write_index,
        token_chunk,
        token_full_chunk_result,
        bos_id=model.bos_id,
    )
    return chunk, chunk_result

  # pylint: disable = protected-access
  # pylint: disable = g-long-lambda
  # pylint: disable = unnecessary-lambda
  # pytype: disable=attribute-error
  # pytype: disable=bad-unpacking
  @partial(jax.jit, static_argnums=(0, 6, 7, 8))
  def _generate_impl(
      self,
      params: Weights,
      prefix: List[attention.KVCache],
      prev_chunk_next_token_logits: jnp.ndarray,
      sample_ids: jnp.ndarray,
      sample_params: SamplingHyperParams,
      steps: int,
      stream: Optional[StreamClient] = None,
      return_all_logits=False,
  ) -> Union[Tuple[Chunk, ChunkResult], Tuple[Chunk, ChunkResult, jax.Array]]:
    """Generates a chunk of text, given some prefixes. Jitted function."""
    del return_all_logits  # TODO(sholto): For all logit scoring
    (batch,) = sample_ids.shape
    del stream  # TODO(sholto): reimplement once shardmap callback is done

    sample_rngs, token_indexes_start, chunk, chunk_result = (
        self.create_output_buffer(
            self._hparams,
            sample_ids,
            prefix,
            steps,
            prev_chunk_next_token_logits,
            circular=False,
            bos_id=self.bos_id,
        )
    )

    loop_body = partial(
        self.sample_infer_write,
        self,
        params,
        prefix,
        sample_params,
        token_indexes_start,
        sample_rngs,
    )

    chunk, chunk_result = lax.fori_loop(
        0, steps, loop_body, (chunk, chunk_result)
    )

    # The length of the chunk is the index of the first EOS ID. We don't
    # calculate this during the generation loop, so instead we calculate it now.
    is_eos = chunk.tokens == self._eos_id
    token_iota = lax.broadcasted_iota(jnp.int32, (batch, steps), 1)
    chunk = chunk.replace(
        lengths=jnp.min(jnp.where(is_eos, token_iota, steps), axis=1)
    )

    return chunk, chunk_result

  def instantiate_generating_fn(
      self,
      steps: int,
      stream: Optional[StreamClient] = None,
      return_all_logits=False,
  ) -> Callable:  # pylint: disable = g-bare-generic
    """Create partial fn to ensure caching."""

    return partial(
        self._generate_impl,
        steps=steps,
        stream=stream,
        return_all_logits=return_all_logits,
    )

  def generate(
      self,
      params: Weights,
      generate_fn: Callable,  # pylint: disable = g-bare-generic
      prefix: Sequence[ChunkResult],
      sample_ids: jnp.ndarray,
      sample_params: SamplingHyperParams,
  ) -> Tuple[Chunk, ChunkResult]:
    """Generative inference for a batch.

    Note about random number seeding:
    We provide strong guarantees about random number seeding, to make it
    possible for callers to get deterministic results that are independent of
    batch packing and independent of splitting into `Chunk`s for incremental
    processing. Specifically, we guarantee that random numbers are constructed
    by:

    ```
    def rng_for_token(sample_id: int, token_index: int) -> jax.Array:
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
      params: Model weights.
      generate_fn: Cached generation fn
      prefix: Already-processed tokens in the prefix, if any.
      sample_ids: Per-sample random seeds to use for sampling. By convention,
        you should generally number these sequentially starting from 0.
        int32[num_samples]
      sample_params: sampling parameters

    Returns:
      The generated text, together with its processed results.
    """
    with self.mesh, self.rules:
      if prefix:
        prev_chunk_next_token_logits = prefix[-1].next_token_logits
      else:
        prev_chunk_next_token_logits = None

      cache = [p.kv_cache for p in prefix]

      return generate_fn(
          params=params,
          prefix=cache,
          prev_chunk_next_token_logits=prev_chunk_next_token_logits,
          sample_ids=sample_ids,
          sample_params=sample_params,
      )
