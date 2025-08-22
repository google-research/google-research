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

"""Fast decoding routines for layout generation."""

from typing import Any

from . import sampling
import flax
import jax
from jax import lax
import jax.numpy as jnp

# Constants
# We assume the default End-of-Sentence token id is 2.
EOS_ID = 2


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jax.Array  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  cur_seqs: jax.Array  # int32: [batch_size, max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: jax.Array  # bool: [batch_size]
  # The current state of the autoregressive decoding caches.
  cache: Any  # Any pytree of arrays, e.g. flax attention Cache object.
  rng: jax.Array  # Sampling random state.


def state_init(batch_size, max_decode_len, cache, rng):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(0)
  cur_seqs0 = jnp.ones((batch_size, max_decode_len), jnp.int32)
  finished_flags0 = jnp.zeros((batch_size,), jnp.bool_)
  return State(
      cur_index=cur_index0,
      cur_seqs=cur_seqs0,
      finished_flags=finished_flags0,
      cache=cache,
      rng=rng)


def decode(inputs,
           cache,
           tokens_to_logits,
           eos_id=EOS_ID,
           max_decode_len=None,
           sampling_method='topp',
           rng=None,
           logit_masks=None,
           conditional='none'):
  """Fast decoding for autoregressive layout generation.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    cache: flax attention cache.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    eos_id: int: id of end-of-sentence token for target vocabulary.
    max_decode_len: int: maximum length of decoded translations.
    sampling_method: str: sampling method.
    rng: jnp.DeviceArray: sampling random state.
    logit_masks: array: [1, vocab_size], step-specific logit mask.
    conditional: str: conditional type.

  Returns:
    [batch_size, max_decode_len] layout sequences
  """
  inputs = inputs.astype('int32')
  batch_size = inputs.shape[0]
  assert max_decode_len is not None
  end_marker = jnp.array(eos_id)

  # initialize state
  init_state = state_init(batch_size, max_decode_len, cache, rng)
  # Positions in one asset less than or equal to the position info which use
  # the grouth truth tokens.
  # Conditions on asset.
  if conditional == 'a':
    conditional_info = jnp.array(0)
  # Conditions on asset and size.
  elif conditional == 'a+s':
    conditional_info = jnp.array(2)
  # Unconditional generation.
  elif conditional == 'none':
    conditional_info = jnp.array(-1)
  else:
    raise ValueError(f"Unknown conditional type '{conditional}'")

  def loop_cond_fn(state):
    """decoding loop termination condition."""
    # Have we reached max decoding length?
    not_at_end = state.cur_index < max_decode_len
    # Are all sequences finished?
    all_finished = jnp.all(state.finished_flags)
    return not_at_end & (~all_finished)

  def loop_body_fn(state):
    """decoding loop state update function."""
    # Current input ids --> [batch, 1].
    cur_ids = lax.dynamic_slice(
        state.cur_seqs,
        (0, state.cur_index),
        (batch_size, 1))

    # Calls model on current tokens to get next-position logits and cache.
    # --> [batch, 1]
    logits, new_cache = tokens_to_logits(cur_ids, state.cache, state.cur_index)
    # Masks logits at the given step.
    logits_mask = lax.dynamic_slice(
        logit_masks,
        (0, state.cur_index%5, 0),
        (1, 1, logits.shape[-1]))
    logits = jnp.where(logits_mask > 0, -1e7, logits)
    rng = state.rng
    if sampling_method == 'greedy':
      sampled_ids = jnp.argmax(logits, axis=-1)
    else:
      # Sampling next token.
      rng, sample_rng = jax.random.split(rng, 2)
      sampled_ids = sampling.sampling(logits, sample_rng, topp=0.9)

    real_ids = lax.dynamic_slice(inputs, (0, state.cur_index), (batch_size, 1))
    sampled_ids = jnp.where(
        jnp.greater_equal(conditional_info, state.cur_index % 5), real_ids,
        sampled_ids)
    topk_seq = lax.dynamic_update_slice(state.cur_seqs, sampled_ids,
                                        (0, state.cur_index + 1))
    # Updates finished state.
    newly_finished = (topk_seq[:, state.cur_index + 1] == end_marker)
    all_finished = newly_finished | state.finished_flags

    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=topk_seq,
        finished_flags=all_finished,
        cache=new_cache,
        rng=rng)

  # Run while loop and get final state.
  final_state = lax.while_loop(loop_cond_fn,
                               loop_body_fn,
                               init_state)
  return final_state.cur_seqs
