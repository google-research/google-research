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

"""Xmap compatability till grad(shard_map) is fixed."""

from functools import partial  # pylint: disable=g-importing-member
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from flax import struct
import jax
from jax import lax
from jax.experimental.maps import xmap
import jax.numpy as jnp
import jax.scipy
from jax.sharding import Mesh
import numpy as np
import seqio
import typing_extensions

from scaling_transformer_inference_efficiency import attention
from scaling_transformer_inference_efficiency import checkpoint
from scaling_transformer_inference_efficiency import collectives
from scaling_transformer_inference_efficiency import partitioning
from scaling_transformer_inference_efficiency import special2
from scaling_transformer_inference_efficiency import weights
from scaling_transformer_inference_efficiency.layers import two_d_parallel_xmap
from scaling_transformer_inference_efficiency.maps import shard_map

Weights = weights.Weights
P = jax.sharding.PartitionSpec

HParams = checkpoint.HParams
Weights = weights.Weights
QuantizedWeights = weights.QuantizedWeights
Layer = weights.Layer
QuantizedLayer = weights.QuantizedLayer

_BOS_ID = 0


@struct.dataclass
class Chunk:
  """A chunk of token IDs. These are typically used as the input to a model."""

  tokens: Union[np.ndarray, jnp.ndarray]  # int32[batch, max_len]
  lengths: Union[np.ndarray, jnp.ndarray]  # int32[batch]

  @classmethod
  def logical_axes(cls):
    return Chunk(  # pytype: disable=wrong-arg-types  # jax-ndarray
        tokens=P('batch', 'time'),
        lengths=P('batch'),
    )

  @classmethod
  def zeros(cls, batch, seqlen):
    """Returns an all-zeros Chunk of the specified shape.

    The returned Chunk doesn't have a useful meaning as text. This function is
    primarily intended to be used as initial loop state, which will subsequently
    be overwritten with meaningful token IDs.

    Args:
      batch: batch size.
      seqlen: number of tokens in the chunk.

    Returns:
      A Chunk with zeros in all locations.
    """
    return Chunk(
        tokens=jnp.zeros((batch, seqlen), jnp.int32),
        lengths=jnp.zeros((batch,), jnp.int32),
    )

  @classmethod
  def tokenize(
      cls, vocab, texts, is_first_chunk
  ):
    """Parses the text into token IDs and creates a Chunk from them.

    For example:

    ```
    # A chunk with batch size 2.
    chunk = Chunk.tokenize(
        vocab,
        ["Humans have 2 legs.",
        "Humans have 3 legs.",
        ],
        is_first_chunk=True)
    ```

    Alternatively, the same text split over multiple chunks:

    ```
    # Prefix has batch size 1.
    prefix_chunk = Chunk.tokenize(vocab, ["Humans have"], is_first_chunk=True)
    # Suffix has batch size 2.
    suffix_chunk = Chunk.tokenize(vocab, ["2 legs.", "3 legs."],
                                  is_first_chunk=False)
    chunks = [prefix_chunk, suffix_chunk]
    ```

    Args:
      vocab: The vocabulary with which to parse the text into tokens.
      texts: The batch of sequences to parse.
      is_first_chunk: Whether this is the first chunk in a logical sequence. If
        so, as part of tokenization we will prepend the special
        "beginning-of-sequence" token (token ID = 0), which informs the model
        that this is indeed the beginning of the sequence.

    Returns:
      The batch of sequences, parsed into a Chunk. The result's sequence length
      equals the longest sequence length of any input string.
    """
    # t5x/models.py;l=643
    # t5x/google/prediction_service/handler.py;l=514
    # t5x/google/prediction_service/handler.py;l=425
    # seqio/dataset_providers.py;l=1106
    #   - task evaluation
    # seqio/dataset_providers.py;l=943
    #   - preprocessor evaluation
    # prediction_service.gin;l=41
    #   - Gin config
    # seqio.DecoderFeatureConverter

    # First we:
    # * parse the strings into token IDs
    # * pad all the sequences so their lengths equal the longest sequences's
    #   length, and form a batch
    lengths = []  # List[int]
    batch_tokens = []  # List[int32[seqlen]]. Each seqlen can be different.
    max_length = 0
    for text in texts:
      # Parsing:
      tokens = vocab.encode_tf(text)
      (length,) = tokens.shape
      if length > max_length:
        max_length = length
      lengths.append(length)
      batch_tokens.append(np.array(tokens))

    # Padding to max length, and then concatenating into a batch
    # pylint: disable = g-complex-comprehension
    batch_tokens = np.array(
        [
            np.pad(
                tokens,
                (0, max_length - tokens.shape[0]),
                constant_values=vocab.pad_id,
            )
            for tokens in batch_tokens
        ]
    )
    # ^ batch_tokens: int32[batch, seqlen]
    lengths = np.array(lengths)
    # ^ lengths: int32[batch]

    # The model expects a beginning-of-sequence token (id equal to _BOS_ID) at
    # the beginning of the logical string of text. If this is the first chunk in
    # a list, we should add it. Otherwise, if it's a later chunk in the list,
    # then the beginning-of-sequence token has already been added to the first
    # one, so we don't need it again.
    if is_first_chunk:
      batch_tokens = jnp.concatenate(
          [
              jnp.full((batch_tokens.shape[0], 1), _BOS_ID, jnp.int32),
              batch_tokens,
          ],
          axis=1,
      )
      lengths = lengths + 1

    # After padding and beginning-of-sequence insertion, an example output would
    # be:
    #
    # batch_tokens:
    # [[0, 123, 456, 789, 0,     0],
    #  [0, 234, 567, 890, 123, 456],
    # ]
    # lengths:
    # [4, 6]
    #
    # Alternatively, for the same text but without beginning-of-sequence
    # insertion:
    #
    # batch_tokens:
    # [[123, 456, 789, 0,     0],
    #  [234, 567, 890, 123, 456],
    # ]
    # lengths:
    # [3, 5]
    return Chunk(
        tokens=batch_tokens,
        lengths=lengths,
    )

  def detokenize(self, vocab):
    """Turns a chunk back into text.

    ```
    orig_texts = ["Humans have 2 legs.",
        "Humans have 3 legs.",
        ]
    chunk = Chunk.tokenize(vocab, orig_texts, is_first_chunk=True)
    texts = chunk.detokenize(vocab)
    assert(texts == orig_texts)
    ```

    Args:
      vocab: Vocabulary for detokenization.

    Returns:
      Text form of the chunk.
    """
    me = self.copy_to_host()
    # Mask out everything above 'lengths', by replacing it with the
    # end-of-sequence token ID. Then vocab.decode_tf won't decode past the
    # end-of-sequence token.
    masked_tokens = np.where(
        np.array(me.token_mask), np.array(me.tokens), vocab.eos_id
    )
    return list(vocab.decode_tf(masked_tokens).numpy())

  @property
  def token_mask(self):
    """Gets a mask which is true for in-bounds tokens. bool[batch, seqlen]."""
    token_index = jax.lax.broadcasted_iota(jnp.int32, self.tokens.shape, 1)
    return token_index < self.lengths[:, np.newaxis]

  def copy_to_host(self):
    """Copies the data from the device to the host."""
    return Chunk(np.array(self.tokens), np.array(self.lengths))

  def split_at(self, n):
    """Splits a chunk into two chunks, where the first has length `n`."""
    assert n < self.tokens.shape[1]
    me = self.copy_to_host()
    first = Chunk(me.tokens[:, :n], np.minimum(me.lengths, n))
    second = Chunk(me.tokens[:, n:], np.maximum(me.lengths, n) - n)
    return first, second

  def pad_to_length(self, n):
    """Pads the chunk to the target length."""
    seqlen = self.tokens.shape[1]
    assert n >= seqlen
    tokens = jnp.pad(self.tokens, ((0, 0), (0, n - seqlen)))
    return Chunk(tokens, self.lengths)

  def update(self, token_i, token):
    """Writes the batch of tokens to the specified token index."""
    assert token.tokens.shape[1] == 1, 'token must have seqlen=1'
    return Chunk(
        tokens=lax.dynamic_update_index_in_dim(
            self.tokens, token.tokens[:, 0], token_i, 1
        ),
        lengths=jnp.maximum(self.lengths, token_i + 1),
    )


@struct.dataclass
class ChunkResult:
  """Result of analyzing a `Chunk` by the neural net.

  This is returned at JIT boundaries.
  """

  # Scores and other candidates for the _current_ token (not the next one).
  per_token_scores: jnp.ndarray  # float32[batch, seqlen]
  top_token_ids: jnp.ndarray  # int32[batch, seqlen, top_k]
  top_token_probs: jnp.ndarray  # float32[batch, seqlen, top_k]

  # Logits for the _next_ token
  next_token_logits: jnp.ndarray  # float32[batch, vocab_size]

  # KV cache.
  kv_cache: attention.KVCache

  @classmethod
  def logical_axes(cls):
    return ChunkResult(  # pytype: disable=wrong-arg-types  # jax-ndarray
        per_token_scores=P('batch', 'time'),
        top_token_ids=P('batch', 'time', 'top_k'),
        top_token_probs=P('batch', 'time', 'top_k'),
        next_token_logits=P('logit_batch', 'vocab'),
        kv_cache=attention.KVCache.logical_axes(),
    )

  def copy_to_host(self):
    return jax.tree_map(jax.device_get, self)

  @classmethod
  def zeros(
      cls,
      hparams,
      batch,
      seqlen,
      kv_batch = None,
  ):
    """Creates an all-zeros ChunkResult of the specified shape."""
    cache_batch = kv_batch if kv_batch is not None else batch
    return ChunkResult(
        kv_cache=attention.KVCache.zeros(hparams, cache_batch, seqlen),
        per_token_scores=jnp.zeros((batch, seqlen), jnp.float32),
        top_token_ids=jnp.zeros((batch, seqlen, _TOP_K), jnp.int32),
        top_token_probs=jnp.zeros((batch, seqlen, _TOP_K), jnp.float32),
        next_token_logits=jnp.zeros((batch, hparams.vocab), jnp.float32),
    )

  def update(
      self,
      token_i,
      token_chunk,
      token_full_result,
      per_device = False,
  ):
    """Writes a single-token FullChunkResult to the specified index of this.

    The index token_i is assumed to be the last token written to this
    ChunkResult so far.

    Args:
      token_i: The seqlen index to write to.
      token_chunk: The input tokens with which to write. Shape Chunk[batch, 1].
      token_full_result: The results to write. Shape FullChunkResult[batch, 1].
      per_device: Whether this is used in a per device or global context.

    Returns:
      This, but with token written at index token_i.
    """
    token_batch, token_seqlen, token_vocab = token_full_result.logits.shape
    batch, vocab = self.next_token_logits.shape
    assert batch == token_batch
    assert token_seqlen == 1
    assert token_vocab == vocab

    token_small = token_full_result.to_chunk_result(
        self.next_token_logits, token_chunk, per_device
    )

    return ChunkResult(
        kv_cache=self.kv_cache.write_token(token_i, token_full_result.kv_cache),
        per_token_scores=lax.dynamic_update_index_in_dim(
            self.per_token_scores,
            token_small.per_token_scores[:, 0],
            token_i,
            1,
        ),
        top_token_ids=lax.dynamic_update_index_in_dim(
            self.top_token_ids, token_small.top_token_ids[:, 0, :], token_i, 1
        ),
        top_token_probs=lax.dynamic_update_index_in_dim(
            self.top_token_probs,
            token_small.top_token_probs[:, 0, :],
            token_i,
            1,
        ),
        next_token_logits=token_full_result.logits[:, 0, :],
    )


_TOP_K = 4
_BOS_ID = 0


def _bos_logits(vocab_size):
  """Logits that put assign probability 1.0 to on _BOS_ID."""
  logits = jnp.full((vocab_size,), -1e10)
  return logits.at[_BOS_ID].set(0.0)


@struct.dataclass
class FullChunkResult:
  """Result produced by an 'infer' call."""

  logits: jnp.ndarray  # float32[batch, seqlen, vocab_size]
  kv_cache: attention.KVCache

  @classmethod
  def logical_axes(cls):
    return FullChunkResult(  # pytype: disable=wrong-arg-types  # jax-ndarray
        logits=P('logit_batch', 'time', 'vocab'),
        kv_cache=attention.KVCache.logical_axes(),
    )

  def to_chunk_result(
      self,
      prev_logits,
      chunk,
      per_device = False,
  ):
    """Converts this to its more minimal form, ChunkResult.

    Args:
      prev_logits: The `next_token_logits` of the previous chunk in the
        sequence, or None if this is the first chunk in the sequence.
        float32[batch, vocab_size]. In 2D [batch.x, time, vocab.yz]
      chunk: Input token IDs for this chunk.
      per_device: Whether this is used in a per device or global context.

    Returns:
      This, but in its minimized form.
    """
    # Example 1 (first chunk in a sequence):
    #
    #   prev_logits = None
    #   tokens = [0, 123, 456, 789]
    #   self.logits = [logits_123, logits_456, logits_789, logits_next]
    #
    # Here `logits_123` is a set of logits that assigns a reasonably high
    # probability to the token ID 123. Note that `self.logits`` is shifted left
    # by 1 from `tokens`, because `self.logits` is predicting the next token.
    #
    # We compute scores for the 4 tokens we've seen, by shifting `self.logits`
    # right by 1. We need a probability distribution for the first token. Since
    # the first token in the sequence must always be the beginning-of-sequence
    # token (ID=0), we use the special `_bos_logits` for the first token, which
    # assigns probability 1.0 to ID=0.
    #
    # So we compute per-token scores as:
    #
    #  shifted_logits = [_bos_logits, logits_123, logits_456, logits_789]
    #  per_token_scores = shifted_logits[self.logits]
    #    = [_bos_logits[0], logits_123[123], logits_456[456], logits_789[789]]
    #
    # The values `logits_next` have "fallen off the end" of the chunk. They are
    # not useful in computing scores for this chunk, but we remember them
    # because we'll use them to compute scores for the beginning of the next
    # chunk. We store this in `ChunkResult.next_token_logits`.
    #
    # Example 2 (second chunk in a sequence):
    #
    #   prev_logits = <some jnp.ndarray>
    #   tokens = [987, 654, 321]
    #   self.logits = [logits_654, logits_321, logits_next]
    #
    # This time when computing `shifted_logits`, we don't use `_bos_logits`.
    # Instead we use `prev_chunk.next_token_logits`. That yields:
    #
    #   shifted_logits = [prev_logits, logits_654, logits_321]
    #   per_token_scores = shifted_logits[self.logits]
    #    = [prev_logits[987], logits_654[654], logits_321[321]]
    #
    # Example 3 (second chunk in a sequence is empty):
    #
    #   prev_chunk = <some ChunkResult>
    #   tokens = []
    #   self.logits = []
    #
    # This is mostly degenerate but there's an important special case we need
    # to handle: `ChunkResult.next_token_logits` doesn't come from
    # `self.logits[-1]` like usual (because that would be empty); instead it
    # comes from `prev_chunk.next_token_logits`.

    batch, seqlen, vocab_size = self.logits.shape

    # we need to slice into lengths because logits is sharded
    if per_device:
      logit_sharding_axis = partitioning.get_sharding_divisor(P('logit_batch'))
      # TODO(sholto): This assumes logit batch is only ever sharded along x
      #               or is unsharded
      physical_logit_batch = partitioning.logical_to_physical(P('logit_batch'))
      if 'y' in physical_logit_batch or 'z' in physical_logit_batch:
        raise NotImplementedError('Only one sampling partitioning implemented.')
      x_index = lax.axis_index('x') * logit_sharding_axis
      lengths = lax.dynamic_slice_in_dim(
          chunk.lengths, x_index * batch, batch  # batch is per device
      )
    else:
      lengths = chunk.lengths

    # First figure out what logits to use for the first token.
    if prev_logits is None:
      # Use beginning-of-sequence marker as the logits.
      prev_logits = jnp.broadcast_to(
          _bos_logits(vocab_size), (batch, vocab_size)
      )
      # ^ prev_logits: f32[batch, vocab]
    else:
      prev_logits = attention.flat_broadcast(prev_logits, batch)
      # ^ prev_logits: f32[batch, vocab]

    # Now shift in the prev_logits and shift out the last token's logits.
    shifted_logits = jnp.concatenate(
        [prev_logits[:, np.newaxis, :], self.logits[:, :-1, :]], axis=1
    )
    batch_iota = lax.broadcasted_iota(jnp.int32, (batch,), 0)
    next_token_logits = self.logits[batch_iota, lengths - 1, :]
    # ^ next_token_logits: f32[batch, vocab]
    length_is_zero = lengths == 0
    # ^ length_is_zero: f32[batch]
    length_is_zero = length_is_zero[:, np.newaxis]
    # length_is_zero: bool[batch, 1]
    # Special handling for the case where the sequence length is zero, see
    # Example 3 above.
    next_token_logits = jnp.where(
        length_is_zero, prev_logits, next_token_logits
    )

    if per_device:
      # To do this we'd need to do some collective ops,
      # these are unnecessary auxiliary outputs for the moment, so
      # just zero them out.
      # TODO(sholto): Ideally put these somewhere in sampling,
      #       where we have the fully materialised vocab dim
      global_batch = batch * logit_sharding_axis
      per_token_scores = jnp.zeros((global_batch, seqlen), jnp.float32)
      top_ids = jnp.zeros((global_batch, seqlen, _TOP_K), jnp.int32)
      top_probs = jnp.zeros((global_batch, seqlen, _TOP_K), jnp.float32)

    else:
      # Now compute the compressed representation of shifted_logits, extracting
      # per-token scores, and per-token top token IDs.
      batch_iota = lax.broadcasted_iota(jnp.int32, (batch, seqlen), 0)
      token_iota = lax.broadcasted_iota(jnp.int32, (batch, seqlen), 1)
      logits_max = jnp.max(shifted_logits, axis=-1)
      logits_sumexp = jnp.sum(
          special2.exp2(shifted_logits - logits_max[:, :, np.newaxis]), axis=-1
      )
      logits_sum = jnp.log2(logits_sumexp) + logits_max
      per_token_scores = (
          shifted_logits[batch_iota, token_iota, chunk.tokens] - logits_sum
      ) * special2.LN_2
      top_logits, top_ids = lax.top_k(shifted_logits, k=_TOP_K)

      top_probs = special2.exp2(top_logits - logits_max[:, :, np.newaxis]) * (
          1.0 / logits_sumexp[:, :, np.newaxis]
      )

    return ChunkResult(
        per_token_scores=per_token_scores,
        top_token_ids=top_ids,
        top_token_probs=top_probs,
        next_token_logits=next_token_logits,
        kv_cache=self.kv_cache,
    )


def xmap_embed(
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    token_chunk,
    shard_seqlen_vs_batch = False,
    batch_unsharded = False,
):
  """Embeds a chunk of logits.

  Args:
    params: Weights object
    kv_caches: List of chunks preprocessed earlier
    token_chunk: An unsharded token chunk. Assume .tokens is int32[batch,
      maxlen]
    shard_seqlen_vs_batch: Whether to shard seqlen or batch by z.
    batch_unsharded:  global_batch is less than z so we cannot shard along

  Returns:
    embeddings: bfloat16[[batch.Z, time, embed.XY] || [batch, time, embed.XYZ]
    sin: RoPE embeddings starting at the appropriate index determined by
         pre-existing kv_cache for each index in the batch.
    cos: ""
  """

  z_axis = lax.psum(1, 'z')
  # Start indices are the sums of the lengths of the KV caches.
  start_indices = attention.prefix_lengths(kv_caches)
  (prefix_batch,) = start_indices.shape
  batch, max_length = token_chunk.tokens.shape
  assert batch % prefix_batch == 0, 'Incompatible batch sizes'

  # Do RoPE lookups in the sin/cos tables. Only needed once per prefix_batch.
  def slice_at(index, table):
    # table: [precomputed_length, qkv // 2]
    return lax.dynamic_slice_in_dim(table, index, max_length)

  def slices_at(indices, table):
    return jax.vmap(slice_at, in_axes=(0, None))(indices, table)

  sin = slices_at(start_indices, params.sin)
  cos = slices_at(start_indices, params.cos)
  # sin, cos: bf16[prefix_batch, max_length, qkv // 2]

  # x: int32[batch, maxlen]
  # embed: bfloat16[vocab.YZ, embed.X]
  x = token_chunk.tokens
  vocab_yz, _ = params.embedding.shape

  yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
  vocab_start = yz_index * vocab_yz

  # Initial embedding lookup:
  with jax.named_scope('embed'):
    one_x = x - vocab_start
    embeds = params.embedding[one_x, :]
    one_x = one_x[:, :, jnp.newaxis]
    embeds = jnp.where((one_x >= 0) & (one_x < vocab_yz), embeds, 0)
    # [batch, time, embed.XY]
    embeds = lax.psum_scatter(embeds, 'y', scatter_dimension=2, tiled=True)

    if shard_seqlen_vs_batch:
      # [batch, time.Z, embed.XY]
      embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=1, tiled=True)
    else:
      if batch_unsharded:
        # [batch, time, embed.XYZ]
        embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=2, tiled=True)
      else:
        # [batch.Z, time, embed.XY]
        embeds = lax.psum_scatter(embeds, 'z', scatter_dimension=0, tiled=True)

  return embeds, sin, cos


@struct.dataclass
class Sampling:
  """Hyperparameters controlling sampling from a model."""

  temperature: float

  # TODO(reinerp): topk/topp support.

  def sample(self, logits, step_rngs):
    """Samples from the output logits of a model.

    Args:
      logits: The output logits to sample from. float32[batch, vocab_size].
      step_rngs: For each batch element, the RNG state for sampling.
        jax.random.PRNGKey[batch]

    Returns:
      The selected samples, as token IDs. int32[batch].
    """

    def sample_nonzero():
      # jax.random.categorical expects just one rng. We use vmap to extend it to
      # support a batch of rngs.
      return jnp.int32(
          jax.vmap(jax.random.categorical)(step_rngs, logits / self.temperature)
      )

    def sample_zero():
      return jnp.int32(jnp.argmax(logits, -1))

    # To avoid numerical instability when dividing by very small temperatures,
    # we sample deterministically (greedily) when the temperature is
    # sufficiently close to zero.
    return lax.cond(self.temperature > 1e-4, sample_nonzero, sample_zero)

  def sample_manual(
      self, logits, step_rngs
  ):
    """Samples from the output logits when within xmap."""

    with jax.named_scope('sample'):
      # logits:
      # float32[batch.X, vocab.YZ]
      #   -> float32[batch.XYZ, vocab]
      y_axis = lax.psum(1, 'y')
      z_axis = lax.psum(1, 'z')
      yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
      batch_x, _ = logits.shape
      padded_batch_x = max(batch_x, y_axis * z_axis)
      if padded_batch_x > batch_x:
        logits = jnp.pad(
            logits,
            pad_width=((0, padded_batch_x - batch_x), (0, 0), (0, 0)),
            mode='constant',
        )
      # We all to all so that we get the full logit on each, but shard batch
      # as much as possible
      logits = lax.all_to_all(
          logits, ('y', 'z'), split_axis=0, concat_axis=1, tiled=True
      )
      # need to only take the relevant part of this
      split_size = batch_x // y_axis // z_axis
      step_rngs = lax.dynamic_slice_in_dim(
          step_rngs,
          yz_index * split_size,
          (batch_x // y_axis // z_axis),
          axis=0,
      )
      # TODO(sholto): Confirm this is the best way of doing it
      # logits = binary_search.topp_mask(logits, 0.9, -1e10)
      # TODO(sholto): maybe put t5x binary search back in
      sample = jnp.int32(
          jax.vmap(jax.random.categorical)(step_rngs, logits / self.temperature)
      )
      # sample: int32[batch]
      sample = lax.all_gather(sample, ('x', 'y', 'z'), axis=0, tiled=True)
    return sample

  def sample_manual_batch_unsharded(
      self, logits, step_rngs
  ):
    """Samples from output logits within xmap, with batch unshardedable.

    Args:
      logits: [batch, vocab.YZX]
      step_rngs: [batch]

    Returns:
      sample" int32[batch]
    """

    with jax.named_scope('sample'):
      # multi-part all gather not implemented for xmap in jit see lax.parallel
      logits = lax.all_gather(logits, 'x', axis=1, tiled=True)
      logits = lax.all_gather(logits, 'z', axis=1, tiled=True)
      logits = lax.all_gather(logits, 'y', axis=1, tiled=True)
      assert logits.shape[0] == step_rngs.shape[0]
      sample = jnp.int32(
          jax.vmap(jax.random.categorical)(step_rngs, logits / self.temperature)
      )
    return sample


def div_up(x, y):
  return (x + y - 1) // y


# pylint: disable = redefined-outer-name
class InferFn(typing_extensions.Protocol):
  """A function providing a forwards pass through a model."""

  def __call__(
      self,
      weights,
      kv_caches,
      chunk,
  ):
    Ellipsis


# pylint: disable = g-bare-generic
# pylint: disable = invalid-name
def infer_xmap(
    h,
    _transformer_layer_fn,
    params,  # pylint: disable=g-bare-generic, invalid-name
    kv_caches,
    chunk,
    attn_all_to_all,
    latency_collectives,
    shard_seqlen_vs_batch,
    batch_unsharded = False,
    intermediate_dtype = jnp.bfloat16,
    pre_embedded_inputs = None,
):
  """Forward pass through xmap path, returning per-token logits."""

  # flaxformer/architectures/t5/t5_architecture.py;l=1516;
  x_axis = lax.psum(1, 'x')
  y_axis = lax.psum(1, 'y')
  z_axis = lax.psum(1, 'z')

  if attn_all_to_all == partitioning.AttnAllToAll.NONE:
    attn_batch_sharding = 1
  elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
    attn_batch_sharding = z_axis
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
    attn_batch_sharding = y_axis * z_axis
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
    attn_batch_sharding = y_axis * z_axis * x_axis
  else:
    raise NotImplementedError('Ensure you pass in a matching object')

  batch, max_length = chunk.tokens.shape

  # Start indices are the sums of the lengths of the KV caches.
  x, sin, cos = xmap_embed(
      params, kv_caches, chunk, shard_seqlen_vs_batch, batch_unsharded
  )
  # Used for prompt tuning (where we want to take gradients w.r.t the inputs)
  if pre_embedded_inputs is not None:
    x = pre_embedded_inputs

  def loop_body(layer, carry):
    x, k, v = carry
    x, layer_k, layer_v = _transformer_layer_fn(
        h,
        layer,
        params.layer,
        sin,
        cos,
        kv_caches,
        x,
        x_axis,
        y_axis,
        z_axis,
        attn_all_to_all,
        latency_collectives,
        shard_seqlen_vs_batch,
        batch_unsharded,
        intermediate_dtype,
    )
    k = lax.dynamic_update_index_in_dim(
        k, jnp.swapaxes(layer_k, 0, 1), layer, 0
    )
    v = lax.dynamic_update_index_in_dim(
        v, jnp.swapaxes(layer_v, 0, 1), layer, 0
    )
    return x, k, v

  # Initialize output KV cache.
  k = jnp.zeros(
      (h.layers, max_length, div_up(batch, attn_batch_sharding), h.qkv),
      intermediate_dtype,
  )
  v = jnp.zeros(
      (h.layers, max_length, div_up(batch, attn_batch_sharding), h.qkv),
      intermediate_dtype,
  )
  x, k, v = jax.lax.fori_loop(0, h.layers, loop_body, (x, k, v))

  k = jnp.swapaxes(k, 0, 1)
  v = jnp.swapaxes(v, 0, 1)

  # [batch, maxlen, embed.X]
  xnorm, _ = two_d_parallel_xmap.allgather_layernorm(
      x, shard_seqlen_vs_batch, batch_unsharded
  )

  # x: bfloat16[batch, maxlen, dmodel.X] # [vocab.YZ, embed.X]
  with jax.named_scope('unembed'):
    logits_unreduced = jnp.einsum(
        'bte,ve->btv', jnp.float32(xnorm), jnp.float32(params.embedding)
    )
    # x: [batch, maxlen, vocab.YZ] {X unreduced}
    if batch_unsharded:
      # logits: float32[batch.X, maxlen, vocab.YZ]
      logits = lax.psum_scatter(
          logits_unreduced, 'x', scatter_dimension=2, tiled=True
      )
    else:
      # logits: float32[batch, maxlen, vocab.YZX]
      logits = lax.psum_scatter(
          logits_unreduced, 'x', scatter_dimension=0, tiled=True
      )

  k, v = jnp.bfloat16(k), jnp.bfloat16(v)

  # We need to get only the part of lengths which corresponds to that
  # shard of the kv cache, which can be sharded across batch
  # NOTE: This will not currently support MHA being sharded over heads
  #  -only multiquery attention but neither will any of the code above
  # where k and v are sliced into.
  # A MHA kv cache would require a heads dimension!
  # That being said, we don't have any parallel-layers MHA models.

  # chunk.lengths: [batch] -> [batch.attn_batch_sharding]
  # TODO(sholto): Make this simpler
  if attn_all_to_all == partitioning.AttnAllToAll.NONE:
    cache_lengths = chunk.lengths
  elif attn_all_to_all == partitioning.AttnAllToAll.AXIS_Z:
    slice_size = batch // attn_batch_sharding
    z_index = lax.axis_index('z') * slice_size
    cache_lengths = lax.dynamic_slice_in_dim(chunk.lengths, z_index, slice_size)
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZ:
    slice_size = batch // attn_batch_sharding
    yz_index = (lax.axis_index('y') * z_axis + lax.axis_index('z')) * slice_size
    cache_lengths = lax.dynamic_slice_in_dim(
        chunk.lengths, yz_index, slice_size
    )
  elif attn_all_to_all == partitioning.AttnAllToAll.AXES_YZX:
    slice_size = batch // attn_batch_sharding
    yzx_index = (
        lax.axis_index('y') * (z_axis * x_axis)
        + lax.axis_index('z') * x_axis
        + lax.axis_index('x')
    ) * slice_size
    cache_lengths = lax.dynamic_slice_in_dim(
        chunk.lengths, yzx_index, slice_size
    )
  # should equal batch dim as sharded for kv cache
  assert cache_lengths.shape[0] == batch // attn_batch_sharding
  assert cache_lengths.shape[0] == k.shape[2]

  return FullChunkResult(
      logits=logits,
      kv_cache=attention.KVCache(
          cache_lengths, k, v, jnp.zeros([0], jnp.int32)
      ),
  )


class XmapModel:
  """A model with xmapped JIT-compiled prefill and generate functions.

  We assume that all inputs have been prepared appropriately for xmap mode,
  this includes both params and the cache. Routines are provided to support
  this. As a result, the outputs of xmap layers stay 'folded out'. In
  future, when shard_map is complete, this will not be necessary and everything
  will be jit(shard_map using the above class.
  """

  def __init__(
      self,
      hparams,
      eos_id,
      infer_fn,
      weights_logical_axes,
      rules,
      vocab = None,
      devices = None,
  ):  # used for sampling
    self._hparams = hparams
    self._eos_id = eos_id
    self._infer = infer_fn
    self._mesh = partitioning.make_mesh(devices)
    self.rules = partitioning.PartitioningRules(rules)
    with self.rules:
      self.chunk_layout = jax.tree_map(
          shard_map.logical_to_layout, Chunk.logical_axes()
      )
      self._weights_logical_axes = weights_logical_axes
      self._weights_axes = jax.tree_map(
          partitioning.logical_to_physical, weights_logical_axes
      )
      self.params_layouts = jax.tree_map(
          shard_map.logical_to_layout, weights_logical_axes
      )
      self.result_layout = jax.tree_map(
          shard_map.logical_to_layout, ChunkResult.logical_axes()
      )
      self.full_chunk_layout = jax.tree_map(
          shard_map.logical_to_layout, FullChunkResult.logical_axes()
      )
      self.cache_layout = jax.tree_map(
          shard_map.logical_to_layout, attention.KVCache.logical_axes()
      )
      self.sample_ids_layout = shard_map.logical_to_layout(P('logit_batch'))
      self.prev_chunk_next_token_layout = jax.tree_map(
          shard_map.logical_to_layout,
          ChunkResult.logical_axes().next_token_logits,
      )
      self.prev_logits_layout = self.result_layout.next_token_logits
      self.all_logits_logical = P('time', 'logit_batch', 'vocab')
      self.all_logits_layout = jax.tree_map(
          shard_map.logical_to_layout, self.all_logits_logical
      )
      self.embeddings_logical = P(
          'residual_batch', 'residual_time', 'residual_embed'
      )
      self.embeddings_layout = jax.tree_map(
          shard_map.logical_to_layout, self.embeddings_logical
      )
    self.vocab = vocab
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

    with self._mesh, self.rules:
      params = shard_map.shard_map(
          rotate,
          self._mesh,
          in_specs=(self._weights_axes,),
          out_specs=self._weights_axes,
          donate_argnums=(0,),
      )(params)

      return params

  @partial(jax.jit, static_argnums=(0,))
  def prepare_params(
      self,
      params,
  ):
    """Prepares inputs for hard xmap pass by pulling out axes to be mapped."""

    with self.rules, self._mesh:
      params_xmap, _ = shard_map.fold_out_tree(
          self._mesh, params, params.logical_axes()
      )

      return params_xmap

  @partial(jax.jit, static_argnums=(0,))
  def prepare_sample_ids(
      self, sample_ids
  ):
    """Prepares inputs for hard xmap pass by pulling out axes to be mapped."""

    with self.rules, self._mesh:
      sample_ids, _ = shard_map.fold_out(
          self._mesh, sample_ids, P('logit_batch')
      )
      return sample_ids  # pytype: disable=bad-return-type  # jax-ndarray

  # pylint: disable = g-bare-generic
  # pylint: disable = protected-access
  @staticmethod
  def _prefill_impl(
      model,
      params_xmap,
      cache_xmap,
      chunk_xmap,
      prev_logits,
      pre_embedded_inputs = None,
      return_full_chunk = False,
  ):
    """Wrap both prefill and results formatting in a single xmap call."""
    full_chunk_result = model._infer(
        params_xmap,
        cache_xmap,
        chunk_xmap,
        pre_embedded_inputs=pre_embedded_inputs,
    )
    if return_full_chunk:
      return full_chunk_result
    else:
      with jax.named_scope('to_chunk_result'):
        chunk_result = full_chunk_result.to_chunk_result(
            prev_logits, chunk_xmap, per_device=True
        )
      return chunk_result

  def instantiate_prefill_fn(self, return_full_chunk = False):
    return partial(
        self._prefill_impl, self, return_full_chunk=return_full_chunk
    )

  def prefill(
      self,
      params_xmap,
      prefill_impl,
      prefix_xmap,
      chunk_xmap,
      pre_embedded_inputs = None,
      return_full_chunk = False,
  ):
    """Non-generative inference for a batch.

    Args:
      params_xmap: Model weights.
      prefill_impl: Partialed prefillimpl
      prefix_xmap: Already-processed tokens in the prefix, if any.
      chunk_xmap: The tokens to prefill.
      pre_embedded_inputs: If we want to do the embeddings outside (e.g. for
        prompt tuning)
      return_full_chunk: We may want all logits

    Returns:
      Scores for the batch.
    """
    with self.rules, self._mesh:
      if prefix_xmap:
        prev_logits = prefix_xmap[-1].next_token_logits
      else:
        prev_logits = None
      # 2D: logits: [batch.x, time, vocab.YZ]
      #     kv_cache: [.., ..., prefixbatch, ...]
      cache = [p.kv_cache for p in prefix_xmap]

      if return_full_chunk:
        out_axes = self.full_chunk_layout
      else:
        out_axes = self.result_layout

      embeddings_layout = (
          None if pre_embedded_inputs is None else self.embeddings_layout
      )

      chunk_result = xmap(
          prefill_impl,
          in_axes=(
              self.params_layouts,
              [self.cache_layout for _ in prefix_xmap],
              self.chunk_layout,
              self.prev_logits_layout,
              embeddings_layout,
          ),
          out_axes=out_axes,
          axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
      )(params_xmap, cache, chunk_xmap, prev_logits, pre_embedded_inputs)

      return chunk_result

  # pylint: disable = protected-access
  # pylint: disable = g-long-lambda
  # pylint: disable = unnecessary-lambda
  # pytype: disable=attribute-error
  # pytype: disable=bad-unpacking
  @staticmethod
  def _generate_impl(
      model,
      steps,
      sampling,
      params,
      prefix,
      prev_chunk_next_token_logits,
      sample_ids,
      batch_unsharded = False,
      return_all_logits=False,
  ):
    """Generates a chunk of text. Xmapped version."""
    logit_sharded_batch = sample_ids.shape[0]
    logit_sharding_axis_size = partitioning.get_sharding_divisor(
        P('logit_batch')
    )
    global_batch = logit_sharded_batch * logit_sharding_axis_size
    # We are very deliberate about savings comms.
    # By default, chunk info is small, so we don't shard by batch except
    # the kv cache. Otherwise we would be doing frequent all-gathers on
    # tiny chunks before each inference step.
    # Logits are huge! So we shard them by X in batch and YZ in vocab.
    # this means that we track two different sized batch dims.
    # Seeding of the RNG itself is deterministic. To generate different samples,
    # users can provide sample_number_offset.
    sample_rngs = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(  # pytype: disable=wrong-arg-types  # jax-ndarray
        jax.random.PRNGKey(0), sample_ids
    )

    token_indexes_start = attention.prefix_lengths(prefix)
    token_indexes_start = attention.flat_broadcast(
        token_indexes_start, logit_sharded_batch
    )

    last_logits = prev_chunk_next_token_logits
    if last_logits is None:
      last_logits = _bos_logits(model._hparams.vocab)[np.newaxis, :]

    last_logits = attention.flat_broadcast(last_logits, logit_sharded_batch)

    if return_all_logits:
      batch, vocab = last_logits.shape
      # [batch.x, vocab.YZ] || [batch, vocab.yzx]
      all_logits = jnp.zeros((steps, batch, vocab))
    # Generation loop.

    def loop_body(
        token_i,
        state,
    ):
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
      if return_all_logits:
        chunk, chunk_result, all_logits = state
      else:
        chunk, chunk_result = state
      step_rngs = jax.vmap(jax.random.fold_in)(  # pytype: disable=wrong-arg-types  # jax-ndarray
          sample_rngs, token_indexes_start + token_i
      )
      # logits: float32[batch.X, vocab.YZ], step_rngs: [batch] (sliced later)
      # ->  sample: int32[batch]
      # TODO(sholto): In the event we are returning all logits
      # No need to sample, so remove
      if batch_unsharded:
        next_token = sampling.sample_manual_batch_unsharded(
            chunk_result.next_token_logits, step_rngs
        )
      else:
        next_token = sampling.sample_manual(
            chunk_result.next_token_logits, step_rngs
        )
      # ^ next_token: [batch] - Note, as yet unsharded
      token_chunk = Chunk(
          tokens=next_token[:, np.newaxis],
          lengths=jnp.full((global_batch,), 1, jnp.int32),
      )
      # ^ token_chunk: Chunk[batch, 1]
      token_full_chunk_result = model._infer(
          params, prefix + [chunk_result.kv_cache], token_chunk
      )
      chunk = chunk.update(token_i, token_chunk)
      chunk_result = chunk_result.update(
          token_i, token_chunk, token_full_chunk_result, per_device=True
      )

      if return_all_logits:
        # slice in for that step, swap time and batch
        all_logits = jax.lax.dynamic_update_slice(
            all_logits,
            jnp.swapaxes(token_full_chunk_result.logits, 0, 1),
            (token_i, 0, 0),
        )

      if return_all_logits:
        return chunk, chunk_result, all_logits
      else:
        return chunk, chunk_result

    chunk = Chunk.zeros(global_batch, steps)
    kv_batch = global_batch // partitioning.get_sharding_divisor(
        P('attn_batch')
    )
    chunk_result = ChunkResult.zeros(
        model._hparams, global_batch, steps, kv_batch
    )
    chunk_result = chunk_result.replace(next_token_logits=last_logits)

    if return_all_logits:
      chunk, chunk_result, all_logits = lax.fori_loop(
          0, steps, loop_body, (chunk, chunk_result, all_logits)
      )
    else:
      chunk, chunk_result = lax.fori_loop(
          0, steps, loop_body, (chunk, chunk_result)
      )

    # The length of the chunk is the index of the first EOS ID. We don't
    # calculate this during the generation loop, so instead we calculate it now.
    is_eos = chunk.tokens == model._eos_id
    token_iota = lax.broadcasted_iota(jnp.int32, (global_batch, steps), 1)
    chunk = chunk.replace(
        lengths=jnp.min(jnp.where(is_eos, token_iota, steps), axis=1)
    )

    if return_all_logits:
      return chunk, chunk_result, all_logits
    else:
      return chunk, chunk_result

  # pytype: enable=attribute-error
  # pytype: enable=bad-unpacking

  def instantiate_generating_fn(
      self,
      steps,
      sampling,
      batch_unsharded = False,
      return_all_logits=False,
  ):  # pylint: disable = g-bare-generic
    """Create partial fn, because xmap cannot mark args as static."""
    return partial(
        self._generate_impl,
        self,
        steps,
        sampling,
        batch_unsharded=batch_unsharded,
        return_all_logits=return_all_logits,
    )

  def generate(
      self,
      params_xmap,
      generate_fn,  # pylint: disable = g-bare-generic
      prefix_xmap,
      sample_ids_xmap,
      return_all_logits = False,
  ):
    """Generative inference for a batch. See pjit version for details."""

    with self.rules, self._mesh:
      if prefix_xmap:
        prev_chunk_next_token_logits = prefix_xmap[-1].next_token_logits
      else:
        prev_chunk_next_token_logits = None

      cache = [p.kv_cache for p in prefix_xmap]

      if return_all_logits:
        out_axes = out_axes = ({}, self.result_layout, self.all_logits_layout)
      else:
        out_axes = ({}, self.result_layout)

      return xmap(
          generate_fn,
          in_axes=(
              self.params_layouts,
              [self.cache_layout for _ in prefix_xmap],
              self.prev_chunk_next_token_layout,
              self.sample_ids_layout,
          ),
          out_axes=out_axes,  # fully replicate sample indices
          axis_resources={'x': 'x', 'y': 'y', 'z': 'z'},
      )(params_xmap, cache, prev_chunk_next_token_logits, sample_ids_xmap)
