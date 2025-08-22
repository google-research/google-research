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

import functools

from .  import sampling
import flax
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jax.Array  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  cur_seqs: jax.Array  # int32: [batch_size, beam_size, max_decode_len]
  rng: jax.Array  # Sampling random state.
  final_seqs: jax.Array


def state_init(masked_batch, rng, total_iteration_num):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(0)
  cur_seqs0 = masked_batch
  final_seqs0 = jnp.expand_dims(masked_batch, 1)
  final_seqs0 = jnp.tile(final_seqs0, (1, total_iteration_num, 1))
  return State(
      cur_index=cur_index0,
      cur_seqs=cur_seqs0,
      rng=rng,
      final_seqs=final_seqs0)


def decode(inputs,
           tokens_to_logits,
           sampling_method='topp',
           rng=None,
           logit_masks=None,
           iterative_nums=None,
           layout_dim=2):
  """Fast decoding for layout generation.

  In the decoding alogrithm, we also first generate asset classes, then based on
  generated asset classes, asset sizes is generated and finally conditioning on
  generated asset classes and sizes, we generate asset positions. During
  generating each attribute, there are some iterations to refine them.

  Args:
    inputs: array: [batch_size, length] int32 sequence of tokens.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    sampling_method: str: sampling method.
    rng: jnp.DeviceArray: sampling random state.
    logit_masks: array: [1, seq_len, vocab_size], step-specific logit mask.
    iterative_nums: array: iterative numbers for class, size and position.
    layout_dim: int: the dimension of layout.

  Returns:
     Tuple of:
       [batch_size, max_decode_len] layout sequences
  """
  inputs = inputs.astype('int32')
  total_dim = layout_dim * 2 + 1

  cum_iterative_num = np.cumsum(iterative_nums)
  total_iterations = np.sum(iterative_nums)
  # initialize state
  init_state = state_init(inputs, rng, total_iterations)
  position_ids = jnp.arange(inputs.shape[-1])[None, :]
  is_asset = position_ids % total_dim == 0
  # is_size = (position_ids % 5 == 1) | (position_ids % 5 == 2)
  is_size = functools.reduce(
      lambda x, y: x | y,
      [position_ids % total_dim == i for i in range(1, layout_dim + 1)])
  # is_position = (position_ids % 5 == 3) | (position_ids % 5 == 4)
  is_position = functools.reduce(
      lambda x, y: x | y,
      [position_ids % total_dim == i for i in range(layout_dim + 1, total_dim)])
  special_symbol_mask = jnp.ones((1, logit_masks.shape[1], 4))
  logit_masks = jax.lax.dynamic_update_slice(logit_masks, special_symbol_mask,
                                             (0, 0, 0))

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    # Have we reached the max iteration numbers?
    not_at_end = (state.cur_index < total_iterations)
    return not_at_end

  def loop_body_fn(state):
    """Beam search loop state update function."""
    # Current input ids --> [batch, 1].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    logits = tokens_to_logits(cur_ids)
    logits = jax.nn.log_softmax(logits, axis=-1)
    logits = jnp.where(logit_masks > 0, -1e7, logits)
    rng = state.rng
    step = state.cur_index
    if sampling_method == 'greedy':
      sampled_ids = jnp.argmax(logits, axis=-1)
    else:
      # Sampling next token.
      rng, sample_rng = jax.random.split(rng, 2)
      # sampled_ids = sampling.sampling(logits, sample_rng, topp=0.5)
      sampled_ids = jnp.argmax(logits, axis=-1)
      sampled_ids_2nd = sampling.sampling(
          logits, sample_rng, topk=5, temperature=1.5)
      # Chooses which attribute will be refine given the current step.
      cur_attribute = jnp.array(
          step >= cum_iterative_num[1], dtype='int32') + jnp.array(
              step >= cum_iterative_num[0], dtype='int32')
      sampled_ids = jnp.where(cur_attribute > 1, sampled_ids, sampled_ids_2nd)

    def position_iteration_mask(sampled_ids):
      """Iterations to refine positions of assets.

      In these iterations, we will only mask positions of asses based on their
      confidence scores and regenerated these masked tokens.

      Args:
        sampled_ids: layouts sampled from current logits.
      Returns:
        sampled_ids: generated layouts.
        masked_ratio: masked ratio at the current iteration.
        target_mask: masking only consider position tokens.
      """
      # We just update the tokens which are masks in the model input.
      sampled_ids = jnp.where(cur_ids == 3, sampled_ids, cur_ids)
      masked_ratio = (cum_iterative_num[2] - step - 1.) / iterative_nums[2]
      # Possible masking candiates can only be position tokens and masked tokens
      # (the index of which is 3) in the model input.
      target_mask = (cur_ids == 3) & is_position
      return sampled_ids, masked_ratio, target_mask

    def size_iteration_mask(sampled_ids):
      """Iterations to refine sizes of assets."""
      sampled_ids = jnp.where((cur_ids == 3) & (~is_position), sampled_ids,  # pylint: disable=invalid-unary-operand-type]
                              cur_ids)
      masked_ratio = (cum_iterative_num[1] - step - 1.) / iterative_nums[1]
      target_mask = (cur_ids == 3) & is_size
      return sampled_ids, masked_ratio, target_mask

    def asset_iteration_mask(sampled_ids):
      """Iterations to refine asset class of assets."""
      sampled_ids = jnp.where((cur_ids == 3) & (is_asset), sampled_ids, cur_ids)
      masked_ratio = (cum_iterative_num[0] - step - 1.) / iterative_nums[0]
      target_mask = (cur_ids == 3) & is_asset
      return sampled_ids, masked_ratio, target_mask

    # Chooses which attribute will be refine given the current step.
    cur_attribute = jnp.array(
        step >= cum_iterative_num[1], dtype='int32') + jnp.array(
            step >= cum_iterative_num[0], dtype='int32')
    cur_mask_info = jax.lax.switch(
        cur_attribute,
        [asset_iteration_mask, size_iteration_mask, position_iteration_mask],
        sampled_ids)

    sampled_ids, masked_ratio, target_mask = cur_mask_info
    # Updates final seqs with the current layouts.
    final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))

    probs = jax.nn.softmax(logits, axis=-1)
    selected_probs = jnp.squeeze(
        jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids, -1), -1), -1)
    # Only tokens in target mask could be selected.
    selected_probs = jnp.where(target_mask,
                               selected_probs, 2.)
    sorted_selected_probs = jnp.sort(selected_probs, axis=-1)
    # Gets mask lens for each sample in the batch according to the mask ratio.
    masked_len = jnp.expand_dims(
        jnp.ceil(
            jnp.sum(target_mask, axis=1) * masked_ratio),
        1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = jnp.take_along_axis(sorted_selected_probs, masked_len, axis=-1)
    # Masks tokens with lower confidence.
    sampled_ids = jnp.where(selected_probs < cut_off, 3, sampled_ids)

    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        rng=rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(loop_cond_fn,
                               loop_body_fn,
                               init_state)
  return final_state.final_seqs
