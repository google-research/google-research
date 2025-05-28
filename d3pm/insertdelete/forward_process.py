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

# pyformat: disable
"""Sentinel-based insert/delete forward process.

Computing forward process probabilities for insert-delete diffusion can be made
much easier by enforcing that there's always a single unique alignment between
any adjacent pair of sequences.

To do this, we can introduce "sentinels": markers that indicate that an insert
or delete is about to happen. For instance, we might observe sequences

  t=1    A  B  C          D  E  F
  t=2    A  B DEL INS INS D DEL F     INS
  t=3    A DEL     G   H  D     F INS  J
  t=4    A         G   H  D     F  K   J

where an INS token identifies positions that are about to have an insert, and
a DEL token identifies positions that are being deleted.

Note that, if we strip out the delete tokens in the first sequence and the
insert tokens in the second sequence, the remainder is two sequences of matched
tokens that transitioned independently. Thus, we do not have to marginalize
over possible alignments; we already know exaclty what alignment generated the
step.

Another useful property is that, conditioned on a given time t, the only thing
we need to predict about DEL tokens at time t-1 is how many there were! We
do not have to predict values for them, because we know that the only token
value preceeding a deletion is the DEL token.

As such, the task of our model is simply to predict

  - whether each token at time t was an INS token at time t-1
  - how many DEL tokens occurred in the space before this token and the previous
  - for any token that wasn't an INS token, what value it had at time t-1

which is unambiguous and straighforward to parameterize.

Note: When counting DEL tokens, we choose to count the number of DEL tokens
(at time t-1) between the current token and the previous token that isn't an
INS token (at time t). This is because the alignment of DEL and INS tokens
isn't well defined.

This module includes functions for:

- computing parameters of q(x_{t+1} | x_0) from q(x_y | x_0), q(x_{t+1} | x_t)
- sampling from q(x_{t+1} | x_t) == q(x_{t+1}, alignment(t, t+1) | x_t)
  since the alignment is deterministic
- recomputing alignment(t, t+1) from x_{t+1} and x_t
- sampling from q(x_t, alignment(0, t) | x_0)
- computing log q(x_t, alignment(0, t) | x_0)
- computing log q(x_t | x_0) by marginalizing over alignments
- sampling an alignment from q(alignment(0, t) | x_0, x_t)
"""
# pyformat: enable

import dataclasses
import functools
import operator
import types
from typing import Any, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np

from d3pm.insertdelete import distributions
from d3pm.insertdelete import dynamic_programs
from d3pm.insertdelete import math_util
from d3pm.insertdelete import transition_operator
from d3pm.insertdelete import util

PyTree = Any
NDArray = Any
PRNGKey = NDArray

# pylint: disable=invalid-name,possibly-unused-variable

# BAD_VALUE = np.nan  # To propagate errors better when testing.
BAD_VALUE = -911911.  # To make gradients safe


@flax.struct.dataclass
class OneStepDistn:
  """Parameters of a one-step sentinel insert/delete distribution.

  Attributes:
    lp_delete: Log probability of replacing a token with a DEL sentinel token.
    lp_insert: Geometric log-prob of inserting new INS tokens between any two
      pairs of other tokens (after removing DEL tokens).
    A: float[vocab_size, vocab_size] transition matrix of probs
    D_insert_logits: float[vocab_size] Distribution of token logits that INS
      tokens transform into.
  """
  lp_delete: NDArray
  lp_insert: NDArray
  A: transition_operator.TransitionOperator
  D_insert_logits: NDArray

  @property
  def lp_no_delete(self):  # pylint: disable=g-missing-from-attributes
    return math_util.log_not(self.lp_delete)

  @property
  def lp_no_insert(self):  # pylint: disable=g-missing-from-attributes
    return math_util.log_not(self.lp_insert)


@flax.struct.dataclass
class ManyStepDistn:
  """Parameters of a many-step cumulative sentinel insert/delete distribution.

  Represents the composition of multiple OneStepDistns.

  lp_insert, lp_silent_insert, lp_silent_insert_to_delete are all mutually
  geometric: they must sum to less than one, and at each time we decide whether
  or not to insert, we choose between
    (lp_insert, lp_silent_insert, lp_silent_insert_to_delete, 1 - total)
  and repeat unless we chose the `1 - total` "stop inserting" option.

  Attributes:
    A: float[vocab_size, vocab_size] transition matrix of probs
    D_silent_insert_logits: float[vocab_size] Distribution of tokens that we
      insert silently with probability lp_silent_insert.
    D_reroll_logits: float[vocab_size] Distribution of tokens that we change to
      with our reroll probability.
    D_sentinel_insert_logits: float[vocab_size] Distribution of tokens that INS
      tokens transform into. Only used if the initial sequence contains INS
      tokens, which is usually not the case.
    lp_sentinel_delete: Probability of replacing a token with a DEL sentinel
      token.
    lp_silent_delete: Probability of removing a token without inserting a
      sentinel first (for instance, if the sentinel token appeared at time 1 and
      we are currently simulating step 2.)
    lp_sentinel_insert: Probability of inserting new INS tokens between any two
      pairs of other tokens (after removing DEL tokens).
    lp_silent_insert: Probability of silently inserting a new token without an
      INS sentinel.
    lp_silent_insert_to_delete: Probability of silently inserting a DEL sentinel
      token.
    lp_reroll: Probability of rerolling a kept token.
    precomputed_delete_count_marginals: Optional precomputed marginal
      distribution, as computed by `delete_count_marginals`.
  """
  A: transition_operator.TransitionOperator
  D_silent_insert_logits: NDArray
  D_reroll_logits: NDArray
  D_sentinel_insert_logits: NDArray
  lp_sentinel_delete: NDArray
  lp_silent_delete: NDArray
  lp_sentinel_insert: NDArray
  lp_silent_insert: NDArray
  lp_silent_insert_to_delete: NDArray
  lp_reroll: NDArray
  precomputed_delete_count_marginals: Optional[Tuple[
      distributions.RandomVariablePDF, distributions.RandomVariablePDF]] = None

  @property
  def lp_no_delete(self):  # pylint: disable=g-missing-from-attributes
    return math_util.log_not_any(self.lp_sentinel_delete, self.lp_silent_delete)

  @property
  def lp_keep(self):  # pylint: disable=g-missing-from-attributes
    return math_util.log_not_any(self.lp_sentinel_delete, self.lp_silent_delete,
                                 self.lp_reroll)

  @property
  def lp_no_insert(self):  # pylint: disable=g-missing-from-attributes
    return math_util.log_not_any(self.lp_sentinel_insert, self.lp_silent_insert,
                                 self.lp_silent_insert_to_delete)

  def probs(self, as_dict=False):
    fn = dict if as_dict else types.SimpleNamespace
    return fn(
        A=self.A.prob_matrix(),
        D_silent_insert=jnp.exp(self.D_silent_insert_logits),
        D_reroll=jnp.exp(self.D_reroll_logits),
        D_sentinel_insert=jnp.exp(self.D_sentinel_insert_logits),
        sentinel_delete=jnp.exp(self.lp_sentinel_delete),
        silent_delete=jnp.exp(self.lp_silent_delete),
        sentinel_insert=jnp.exp(self.lp_sentinel_insert),
        silent_insert=jnp.exp(self.lp_silent_insert),
        silent_insert_to_delete=jnp.exp(self.lp_silent_insert_to_delete),
        reroll=jnp.exp(self.lp_reroll))

  def then(self, after, check=False):
    """Compose a ManyStepDistribution and a OneStepDistribution."""
    self_lp_insert_kept_token = math_util.safe_logaddexp(
        self.lp_sentinel_insert, self.lp_silent_insert)

    # Sentinel delete: Keep, then sentinel delete at step 2; or, sentinel
    # delete, then insert at step 1, then sentinel delete at step 2.
    lp_sentinel_delete = math_util.safe_logaddexp(
        self.lp_no_delete + after.lp_delete,
        (self.lp_sentinel_delete +
         arbitrarily_many_times(self.lp_silent_insert_to_delete) +
         self_lp_insert_kept_token + after.lp_delete))

    # Silent delete: Delete silently in first step, or delete with sentinel and
    # choose not to insert anything after the sentinel.
    lp_silent_delete = math_util.safe_logaddexp(
        self.lp_silent_delete,
        (self.lp_sentinel_delete +
         arbitrarily_many_times(self.lp_silent_insert_to_delete) +
         self.lp_no_insert))

    # Sentinel insert: Sentinel inserts from step 2 take priority, and no other
    # inserts introduce sentinels.
    lp_sentinel_insert = after.lp_insert
    # Distribution of sentinel inserts: Optional, compute the transformation
    # of a sentinel insert.
    D_sentinel_insert_logits = after.A.apply(
        before=self.D_sentinel_insert_logits, is_distn=True, log=True)

    # Silent insert: Don't insert in second step, then an arbitrary number of
    # ignored insert-to-deletes in first step, then insert either silently
    # or with sentinel, then keep it at second step.
    lp_silent_insert = (
        after.lp_no_insert +
        arbitrarily_many_times(self.lp_silent_insert_to_delete) +
        self_lp_insert_kept_token + after.lp_no_delete)
    # Distribution of silent inserts: We either silently inserted it earlier
    # then perturbed it at step 2, or inserted it with a sentinel then replaced
    # it at step 2.
    D_silent_insert_logits = math_util.safe_logaddexp(
        math_util.safe_sub_or_ninf(self.lp_sentinel_insert,
                                   self_lp_insert_kept_token) +
        after.D_insert_logits,
        math_util.safe_sub_or_ninf(
            self.lp_silent_insert, self_lp_insert_kept_token) + after.A.apply(
                before=self.D_silent_insert_logits, is_distn=True, log=True))

    # Insert-to-delete: Don't insert in second step, then an arbitrary number of
    # ignored insert-to-deletes in first step, then insert either silently
    # or with sentinel, then delete it at second step.
    lp_silent_insert_to_delete = (
        after.lp_no_insert +
        arbitrarily_many_times(self.lp_silent_insert_to_delete) +
        self_lp_insert_kept_token + after.lp_delete)

    # Reroll: Either reroll immediately and keep, or delete with sentinel, then
    # insert and keep.
    lp_reroll_immediate = self.lp_reroll + after.lp_no_delete
    lp_reroll_from_delete = (
        self.lp_sentinel_delete +
        arbitrarily_many_times(self.lp_silent_insert_to_delete) +
        self_lp_insert_kept_token + after.lp_no_delete)
    lp_reroll = math_util.safe_logaddexp(lp_reroll_immediate,
                                         lp_reroll_from_delete)
    # Either a perturbed previous reroll, or the same as our silent insert
    # distribution.
    D_reroll_logits = math_util.safe_logaddexp(
        math_util.safe_sub_or_ninf(lp_reroll_immediate, lp_reroll) +
        after.A.apply(before=self.D_reroll_logits, is_distn=True, log=True),
        math_util.safe_sub_or_ninf(lp_reroll_from_delete, lp_reroll) +
        D_silent_insert_logits)

    if check:
      # Sanity check by computing the implied probabilities also.
      # Probability of keeping and not rerolling:
      lp_no_reroll_no_delete = (
          math_util.log_not_any(self.lp_sentinel_delete, self.lp_silent_delete,
                                self.lp_reroll) + after.lp_no_delete)
      # Probability of deciding to stop inserting:
      lp_no_insert = (
          after.lp_no_insert +
          arbitrarily_many_times(self.lp_silent_insert_to_delete) +
          self.lp_no_insert)

      total_from_take = math_util.safe_logsumexp(
          jnp.stack([
              lp_sentinel_delete,
              lp_silent_delete,
              lp_reroll,
              lp_no_reroll_no_delete,
          ]))
      total_from_gen = math_util.safe_logsumexp(
          jnp.stack([
              lp_sentinel_insert,
              lp_silent_insert,
              lp_silent_insert_to_delete,
              lp_no_insert,
          ]))
      if (np.isnan(total_from_take) or np.isnan(total_from_gen) or
          np.maximum(np.abs(total_from_take), np.abs(total_from_gen)) > 1e-4):
        raise ValueError("Failure to sum to 1")

    return ManyStepDistn(
        A=self.A.then(after.A),
        D_silent_insert_logits=D_silent_insert_logits,
        D_sentinel_insert_logits=D_sentinel_insert_logits,
        D_reroll_logits=D_reroll_logits,
        lp_sentinel_delete=lp_sentinel_delete,
        lp_silent_delete=lp_silent_delete,
        lp_sentinel_insert=lp_sentinel_insert,
        lp_silent_insert=lp_silent_insert,
        lp_silent_insert_to_delete=lp_silent_insert_to_delete,
        lp_reroll=lp_reroll)

  @staticmethod
  def identity_for(one_step):
    return ManyStepDistn(
        A=one_step.A.left_fold_identity(),
        D_reroll_logits=jnp.full_like(one_step.D_insert_logits, -np.inf),
        D_silent_insert_logits=jnp.full_like(one_step.D_insert_logits, -np.inf),
        D_sentinel_insert_logits=jnp.full_like(one_step.D_insert_logits,
                                               -np.inf),
        lp_sentinel_delete=-np.inf,
        lp_silent_delete=-np.inf,
        lp_sentinel_insert=-np.inf,
        lp_silent_insert=-np.inf,
        lp_silent_insert_to_delete=-np.inf,
        lp_reroll=-np.inf)

  @staticmethod
  def from_single_step(one_step):
    return ManyStepDistn(
        A=one_step.A,
        D_reroll_logits=jnp.full_like(one_step.D_insert_logits, -np.inf),
        D_silent_insert_logits=jnp.full_like(one_step.D_insert_logits, -np.inf),
        D_sentinel_insert_logits=one_step.D_insert_logits,
        lp_sentinel_delete=one_step.lp_delete,
        lp_sentinel_insert=one_step.lp_insert,
        lp_silent_delete=-np.inf,
        lp_silent_insert=-np.inf,
        lp_silent_insert_to_delete=-np.inf,
        lp_reroll=-np.inf)

  def delete_count_marginals(self, max_len):
    """Compute distributions of how many deletions we see at time t.

    Suppose we have k tokens at x0, and we delete all of them somewhere between
    time 0 and time t. How many delete sentinels would we see at time t?

    Args:
      max_len: Maximum length of a sequence; also upper bound for the number of
        insertions we can imagine inserting and then deleting.

    Returns:
      lp_between_two_undeleted: float32[max_len + 1, max_len + 1] such that
        lp_between_two_undeleted[i, j] is the log probability of observing
        `j` delete sentinels at time `t`, given that there were `i` tokens in
        x0 that were deleted (without inserting) between two tokens that were
        not delete sentinels (e.g. two kept tokens).
      lp_before_reroll: float32[max_len + 1, max_len + 1] such that
        lp_before_reroll[i, j] is the log probability of observing
        `j` delete sentinels at time `t`, given that there were `i` tokens in
        x0 that were deleted (without inserting), plus one more that was deleted
        and produced a sentinel at time t, then was followed by an insert, which
        would then be combined into a reroll.
    """
    if self.precomputed_delete_count_marginals:
      (lp_between_two_undeleted,
       lp_before_reroll) = self.precomputed_delete_count_marginals
      if max_len + 1 != lp_between_two_undeleted.log_probs.shape[0]:
        raise ValueError("Precomputed matrix was the wrong size!")
      return lp_between_two_undeleted, lp_before_reroll
    # How likely are we to delete silently? (Then, always, not insert.)
    # (Case 1 in sample_intermediate:handle_delete.)
    lp_delete_no_t_sentinel = self.lp_silent_delete

    # How likely are we to leave a delete sentinel, then not insert a token?
    # (But we do allow inserting a silent delete sentinel.)
    # (Case 2 in sample_intermediate:handle_delete)
    lp_delete_with_t_sentinel = (
        self.lp_sentinel_delete +
        arbitrarily_many_times(self.lp_silent_insert_to_delete) +
        self.lp_no_insert)

    # Every time we delete a token but leave a sentinel, we have some chance of
    # inserting and then immediately deleting a few tokens. How many actual
    # delete sentinels do we see, including the `k` we deleted from x0 along
    # with the `k+1` geometric random variables we might have added?
    def _compute_kplus1_geometrics_plus_k(k):
      return distributions.negative_binomial_log_pdf(
          self.lp_silent_insert_to_delete, k + 1, max_len + 1
      ).shift(k)

    distn_kplus1_geometrics_plus_k = jax.vmap(
        _compute_kplus1_geometrics_plus_k
    )(jnp.arange(max_len + 1))

    # Now, if we deleted `n` tokens from x0, it's not guaranteed that we deleted
    # those at time `t` instead of before.
    def get_mixtures(num_prev_x0_deletions):
      # First, how many produced delete sentinels at time t?
      number_of_deletions_at_t = distributions.binomial_log_pdf(
          num_prev_x0_deletions, lp_delete_with_t_sentinel,
          lp_delete_no_t_sentinel, max_len + 1)
      # Now, given that, how many extra delete sentinels would we see for each
      # of the "between_two_undeleted" and "before_reroll" cases?
      # When we reroll, that's like adding one more deleted token that we know
      # showed up at time t.
      return (
          # between two undeleted
          number_of_deletions_at_t.mixture_of(distn_kplus1_geometrics_plus_k),
          # before reroll
          number_of_deletions_at_t.shift(
              by=1).mixture_of(distn_kplus1_geometrics_plus_k),
      )

    mixtures = jax.vmap(get_mixtures)(jnp.arange(max_len + 1))
    return mixtures

  def with_precomputed_delete_count_marginals(self, max_len):
    (lp_lp_between_two_undeleted,
     lp_before_reroll) = self.delete_count_marginals(max_len)
    return dataclasses.replace(
        self,
        precomputed_delete_count_marginals=(lp_lp_between_two_undeleted,
                                            lp_before_reroll))


def arbitrarily_many_times(log_prob):
  """Returns log(1 + exp(prob) + exp(prob)^2 + exp(prob)^3 + ...)."""
  # log (1/(1 - exp(prob))) = -log(1 - exp(prob)) = -log1p(-exp(log_prob))
  return -jnp.log1p(-jnp.exp(log_prob))


def shift_to_front(sequence,
                   keep_mask,
                   fill_with = None):
  """Shift all values where keep_mask is True to the front."""
  # This can be implemented in XLA as a sort on booleans.
  if fill_with is not None:
    sequence = jnp.where(keep_mask, sequence, fill_with)
  permutation = jnp.argsort(jnp.logical_not(keep_mask))
  return sequence[permutation]
  # (_, result) = jax.lax.sort((jnp.logical_not(keep_mask), sequence))
  # return result


@flax.struct.dataclass
class DynamicLengthSentinelSequence:
  """A sequence of statically-unknown length with sentinels.

  Attributes:
    tokens: int32[max_len] array, which either contains the sentinel indices or
      a vocab token index.
    length: int32, givint the length of the sequence including sentinels.
  """
  INSERT_SENTINEL = -1
  DELETE_SENTINEL = -2
  ERROR = -3
  tokens: NDArray
  length: NDArray

  def is_valid(self):
    errors = self.token_mask() & (
        self.tokens == DynamicLengthSentinelSequence.ERROR)
    return ((self.length <= self.tokens.shape[0]) &
            (jnp.count_nonzero(errors) == 0))

  def token_mask(self):
    return jnp.arange(self.tokens.shape[0]) < self.length

  def fill_padding(self, fill_value=ERROR):
    return DynamicLengthSentinelSequence(
        tokens=jnp.where(self.token_mask(), self.tokens, fill_value),
        length=self.length)

  def insert_sentinel_mask(self):
    return self.tokens == DynamicLengthSentinelSequence.INSERT_SENTINEL

  def delete_sentinel_mask(self):
    return self.tokens == DynamicLengthSentinelSequence.DELETE_SENTINEL

  def strip_sentinels(self, which):
    if which == "insert":
      mask = ~self.insert_sentinel_mask()
    elif which == "delete":
      mask = ~self.delete_sentinel_mask()
    else:
      raise NotImplementedError(which)
    new_length = jnp.count_nonzero(mask)
    return DynamicLengthSentinelSequence(
        shift_to_front(self.tokens, mask), new_length)


@flax.struct.dataclass
class SequenceAlignment:
  """An alignment of a sequence with a previous sequence.

  Two tokens are aligned if either:
  - The first token was perturbed with the transition matrix to become the
    second token.
  - The first token was deleted (at an unobserved step, so it no longer shows
    a DEL sentinel), and the second token was inserted in the gap after the
    first token. (This is a "reroll".)

  Attributes:
    backpointers: int32[max_len] array, which is either an index into a previous
      sequence (if this token came from a previous position), or -1 to indicate
      that the token was inserted randomly.
    delete_sentinels: bool[max_len] array, tracking which in the second output
      are delete sentinels.
    before_length: Length of the original sequence.
    after_length: Length of the new sequence.
    is_single_step: Whether this represents a single step. If so, this alignment
      is a deterministic function of the two sequences (every insert is tagged
      with INS, and every delete is tagged with DEL).
    reroll_mask: Optional bool[max_len] array indicating if a match is due to a
      reroll.
  """
  NO_BACKPOINTER = -1
  ERROR = -2
  backpointers: NDArray
  delete_sentinels: NDArray
  before_length: NDArray
  after_length: NDArray
  is_single_step: bool = flax.struct.field(pytree_node=False)
  reroll_mask: Optional[NDArray] = None

  def fill_padding(self, fill_with=ERROR):
    """Add error markers based on length."""
    return dataclasses.replace(
        self,
        backpointers=jnp.where(
            jnp.arange(self.backpointers.shape[0]) < self.after_length,
            self.backpointers, fill_with))

  def without_reroll_mask(self):
    """Remove reroll information."""
    return dataclasses.replace(self, reroll_mask=None)

  def insert_mask(self):
    """Returns a mask of which tokens afterward were inserts."""
    return ((self.backpointers == SequenceAlignment.NO_BACKPOINTER) &
            (jnp.arange(self.backpointers.shape[0]) < self.after_length))

  def number_of_deletions(self):
    """Returns a number of deletions preceding each position or EOS."""
    padded_insert_mask = jnp.pad(self.insert_mask(), [(0, 1)], "constant")
    padded_delete_mask = jnp.pad(self.delete_mask(), [(0, 1)], "constant")
    padded_backptr = jnp.pad(self.backpointers, [(0, 1)], "constant")
    padded_backptr = padded_backptr.at[self.after_length].set(
        self.before_length)

    deletion_counts = jnp.cumsum(padded_delete_mask)
    deletions_at_matched_target = deletion_counts[padded_backptr]
    deletions_at_prev_matched_target = util.previous_active_value(
        deletions_at_matched_target, ~padded_insert_mask)

    each_deletions = (
        deletions_at_matched_target - deletions_at_prev_matched_target)
    return each_deletions

  def delete_mask(self):
    """Returns a mask of which tokens in the initial sequence were deleted."""
    was_copied = jnp.full([self.backpointers.shape[0]], False)
    safe_backpointers = (
        self.fill_padding(self.backpointers.shape[0] + 1).backpointers)
    was_copied = was_copied.at[safe_backpointers].set(True)
    return (~was_copied &
            (jnp.arange(was_copied.shape[0]) < self.before_length))

  def then(
      self,
      second,
  ):
    """Composes two alignments."""
    # Subtle point: Re-align inserts for rerolls.
    # Original:
    #    A B       C
    #    | |       |
    #    A x * + + C
    #    |     | | |
    #    A     D E x
    # Re-aligned:
    #    A B       C
    #    |  \      |
    #    |   \     |
    #    |    \    |
    #    A x * + + C
    #    |     | | |
    #    A     D E x
    # Composed:
    #    A B   C
    #    | |   |
    #    | |   |
    #    | |   |
    #    A D E x
    # We may or may not also want to track rerolls.
    assert (self.reroll_mask is None) == (second.reroll_mask is None)

    mid_insert_mask = self.insert_mask()
    mid_delete_mask = second.delete_mask()

    if self.reroll_mask is None:
      self_reroll_mask = jnp.zeros(self.backpointers.shape, bool)
      second_reroll_mask = jnp.zeros(second.backpointers.shape, bool)
    else:
      self_reroll_mask = self.reroll_mask
      second_reroll_mask = second.reroll_mask

    def step(realign_pointer, mid_idx):
      cases = []
      # If this was copied and then deleted, update our pointer to track a
      # possible reroll.
      cases.append({
          "if": ~mid_insert_mask[mid_idx] & mid_delete_mask[mid_idx],
          "state": self.backpointers[mid_idx],
          "output": (SequenceAlignment.ERROR, False),
      })
      # If this was copied twice, clear our reroll pointer.
      cases.append({
          "if": ~mid_insert_mask[mid_idx] & ~mid_delete_mask[mid_idx],
          "state": SequenceAlignment.NO_BACKPOINTER,
          "output": (self.backpointers[mid_idx], self_reroll_mask[mid_idx]),
      })
      # If this was inserted and then copied, use and clear our reroll pointer.
      cases.append({
          "if":
              mid_insert_mask[mid_idx] & ~mid_delete_mask[mid_idx],
          "state":
              SequenceAlignment.NO_BACKPOINTER,
          "output": (realign_pointer,
                     realign_pointer != SequenceAlignment.NO_BACKPOINTER),
      })
      # If this was inserted and then deleted, do nothing.
      cases.append({
          "if": mid_insert_mask[mid_idx] & mid_delete_mask[mid_idx],
          "state": realign_pointer,
          "output": (SequenceAlignment.ERROR, False),
      })
      # Choose a case.
      return util.tree_select(
          [case["if"] for case in cases],
          [(case["state"], case["output"]) for case in cases])

    _, (realigned_first_step_pointers, realigned_self_rerolls) = jax.lax.scan(
        step,
        init=SequenceAlignment.NO_BACKPOINTER,
        xs=jnp.arange(mid_insert_mask.shape[0]))

    backpointers = jnp.where(
        second.backpointers == SequenceAlignment.NO_BACKPOINTER,
        SequenceAlignment.NO_BACKPOINTER,
        realigned_first_step_pointers[second.backpointers])
    if self.reroll_mask is None:
      reroll_mask = None
    else:
      # Reroll if:
      # - second sequence did not have a delete sentinel
      # - second token was aligned with something
      # - the thing it was aligned with, after reroll correction, was a reroll
      #   at step 1.
      reroll_mask = jnp.where(
          ((second.backpointers == SequenceAlignment.NO_BACKPOINTER)
           | second.delete_sentinels), False,
          second_reroll_mask | realigned_self_rerolls[second.backpointers])

    return SequenceAlignment(
        backpointers,
        second.delete_sentinels,
        self.before_length,
        second.after_length,
        is_single_step=False,
        reroll_mask=reroll_mask).fill_padding()


def sample_noise_step(
    sequence,
    distn,
    rng,
    precomputed_xA = None,
):
  """Run a forward step of an insert-delete diffusion model.

  Args:
    sequence: Input sequence.
    distn: Distribution to noise by. Either one step or many.
    rng: PRNG key.
    precomputed_xA: Optional precomputed distn.A.apply(sequence.tokens)

  Returns:
    New sequence and it's alignment with the initial sequence.
  """
  if isinstance(distn, OneStepDistn):
    is_single_step = True
    distn = ManyStepDistn.from_single_step(distn)
  else:
    assert isinstance(distn, ManyStepDistn)
    is_single_step = False

  (sequence, distn) = jax.tree.map(jnp.array, (sequence, distn))

  max_len, = sequence.tokens.shape
  k1, k2, k3, k4 = jax.random.split(rng, 4)

  # Choose which tokens to perturb, reroll, silently delete, and delete with
  # sentinel.
  take_logits = jnp.stack([
      distn.lp_keep,
      distn.lp_reroll,
      distn.lp_silent_delete,
      distn.lp_sentinel_delete,
  ])
  take_choices = jax.random.categorical(k1, logits=take_logits, shape=[max_len])
  take_choices = jnp.where(jnp.any(jnp.isnan(take_logits)), 4, take_choices)
  del k1

  if precomputed_xA is None:
    precomputed_xA = jax.vmap(
        functools.partial(distn.A.apply, is_distn=False, log=True))(
            sequence.tokens)

  # Apply the changes.
  def process_one_token(take_choice, token, key, precomputed_xA_logits):
    from_insert_sentinel = jax.random.categorical(
        key, distn.D_sentinel_insert_logits)
    from_insert_sentinel = jnp.where(
        distn.D_sentinel_insert_logits[from_insert_sentinel] == -np.inf,
        DynamicLengthSentinelSequence.ERROR, from_insert_sentinel)
    new_value = jnp.choose(
        take_choice, [
            jnp.where(
                token == DynamicLengthSentinelSequence.INSERT_SENTINEL,
                from_insert_sentinel,
                jax.random.categorical(key, precomputed_xA_logits),
            ),
            jax.random.categorical(key, distn.D_reroll_logits),
            DynamicLengthSentinelSequence.ERROR,
            DynamicLengthSentinelSequence.DELETE_SENTINEL,
            DynamicLengthSentinelSequence.ERROR,
        ],
        mode="clip")
    new_value = jnp.where(token == DynamicLengthSentinelSequence.ERROR,
                          DynamicLengthSentinelSequence.ERROR, new_value)
    mask = ((token != DynamicLengthSentinelSequence.DELETE_SENTINEL) &
            (take_choice != 2))
    return new_value, mask

  kept_tokens_unshifted, kept_mask = (
      jax.vmap(process_one_token)(take_choices, sequence.tokens,
                                  jax.random.split(k2, max_len),
                                  precomputed_xA))
  del k2

  kept_mask = kept_mask & sequence.token_mask()
  kept_tokens = shift_to_front(kept_tokens_unshifted, kept_mask)
  kept_length = jnp.count_nonzero(kept_mask)

  # kept_backpointers_unshifted = jnp.where(take_choices == 1, -1,
  #                                         jnp.arange(max_len))
  kept_backpointers_unshifted = jnp.arange(max_len)
  kept_backpointers = shift_to_front(kept_backpointers_unshifted, kept_mask)

  was_reroll_unshifted = (take_choices == 1)
  was_reroll = shift_to_front(was_reroll_unshifted, kept_mask)

  # Choose how many tokens to insert. At each position in the new sequence,
  # we choose whether it is a sentinel insert, a silent insert, an
  # insert-to-delete, or a position where we stopped inserting and took from
  # our kept sequence (which produces a slot we will copy into).
  insert_logits = jnp.stack([
      distn.lp_sentinel_insert,
      distn.lp_silent_insert,
      distn.lp_silent_insert_to_delete,
      distn.lp_no_insert,
  ])
  insert_choices = jax.random.categorical(
      k3, logits=insert_logits, shape=[max_len])
  insert_choices = jnp.where(
      jnp.any(jnp.isnan(insert_logits)), 4, insert_choices)
  del k3

  # Pick which token to insert in each place.
  insert_tokens = jnp.choose(
      insert_choices, [
          DynamicLengthSentinelSequence.INSERT_SENTINEL,
          jax.random.categorical(
              k4, distn.D_silent_insert_logits, shape=[max_len]),
          DynamicLengthSentinelSequence.DELETE_SENTINEL,
          DynamicLengthSentinelSequence.ERROR,
          DynamicLengthSentinelSequence.ERROR,
      ],
      mode="clip")
  del k4

  # Identify the number of slots and their indices.
  mask_slot = (insert_choices == 3)
  num_slots = jnp.count_nonzero(mask_slot)
  # Get the indices of all false values (followed by indices of true ones)
  slot_indices = jnp.argsort(~mask_slot)
  # Mask out the indices that weren't slots
  slot_indices = jnp.where(
      jnp.arange(max_len) < num_slots, slot_indices, max_len + 1)

  # Scatter our new tokens into the slots.
  new_sequence = insert_tokens.at[slot_indices].set(kept_tokens)
  new_backpointers = (
      jnp.full([max_len], -1).at[slot_indices].set(kept_backpointers))
  new_reroll = (jnp.full([max_len], False).at[slot_indices].set(was_reroll))

  # New location of end-of-sequence was the position of the first slot we didn't
  # fill. Note that there's a chance we filled all the slots; in that case we
  # return a length of max_len + 1 to report failure of representing the result.
  new_seq_len = slot_indices[kept_length]
  new_sequence_result = DynamicLengthSentinelSequence(
      new_sequence, new_seq_len).fill_padding()

  return (
      new_sequence_result,
      SequenceAlignment(
          new_backpointers,
          new_sequence_result.delete_sentinel_mask(),
          sequence.length,
          new_seq_len,
          is_single_step=is_single_step,
          reroll_mask=new_reroll).fill_padding(),
  )


def align_one_step(xt,
                   xtplus1):
  """Infer the alignment between two adjacent sequences."""
  xt_indices = jnp.arange(xt.tokens.shape[0])
  kept_indices = shift_to_front(xt_indices, ~xt.delete_sentinel_mask())
  xtplus1_indices = jnp.arange(xtplus1.tokens.shape[0])
  copied_indices = shift_to_front(
      xtplus1_indices,
      ~xtplus1.insert_sentinel_mask(),
      fill_with=xtplus1.tokens.shape[0] + 1)
  backpointers = (
      jnp.full([xtplus1.tokens.shape[0]],
               -1).at[copied_indices].set(kept_indices))
  return SequenceAlignment(
      backpointers,
      xtplus1.delete_sentinel_mask(),
      xt.length,
      xtplus1.length,
      is_single_step=True,
      reroll_mask=jnp.zeros(xtplus1.tokens.shape, bool))


def sentinel_one_step_log_prob(
    xt,
    xtplus1,
    d_t_to_tplus1,
    precomputed_xt_Attplus1 = None,
):
  """Log probs of a single insert-delete step.

  Args:
    xt: Sequence before the step.
    xtplus1: Sequence after the step.
    d_t_to_tplus1: Distribution of the step.
    precomputed_xt_Attplus1: Optional precomputed
      d_t_to_tplus1.A.apply(xt.tokens)

  Returns:
    log q(x_{t+1} | x_t)
  """
  xt, xtplus1, d_t_to_tplus1 = jax.tree.map(jnp.array,
                                            (xt, xtplus1, d_t_to_tplus1))
  # Log probs of new insert sentinels.
  insertion_mask = xtplus1.insert_sentinel_mask()
  insert_count = jnp.count_nonzero(insertion_mask)
  copy_count = xtplus1.length - insert_count
  insert_sentinel_log_probs = (
      jnp.where(insert_count == 0, 0., insert_count * d_t_to_tplus1.lp_insert) +
      (copy_count + 1) * d_t_to_tplus1.lp_no_insert)

  if precomputed_xt_Attplus1 is None:
    precomputed_xt_Attplus1 = jax.vmap(
        functools.partial(d_t_to_tplus1.A.apply, is_distn=False, log=True))(
            xt.tokens)

  # Log probs of each of the aligned pairs of tokens (ignoring DEL in xt and
  # INS in xtplus1)
  aligned_xt = shift_to_front(xt.tokens,
                              xt.token_mask() & ~xt.delete_sentinel_mask())
  aligned_xt_Attplus1 = shift_to_front(
      precomputed_xt_Attplus1,
      xt.token_mask() & ~xt.delete_sentinel_mask())

  aligned_xtplus1 = shift_to_front(
      xtplus1.tokens,
      xtplus1.token_mask() & ~xtplus1.insert_sentinel_mask())

  def compute_token_logprob(out_of_bounds, xt_token, xtplus1_token,
                            aligned_xt_Attplus1_logits):
    cases = []
    # Out of bounds: ignore.
    cases.append({"if": out_of_bounds, "then": 0.0})
    # Impossible: should have been stripped out.
    cases.append({
        "if": (xt_token == DynamicLengthSentinelSequence.ERROR) |
              (xt_token == DynamicLengthSentinelSequence.DELETE_SENTINEL) |
              (xtplus1_token == DynamicLengthSentinelSequence.ERROR) |
              (xtplus1_token == DynamicLengthSentinelSequence.INSERT_SENTINEL),
        "then":
            BAD_VALUE
    })
    # If this was deleted, add deletion logprob.
    cases.append({
        "if": xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
        "then": d_t_to_tplus1.lp_delete
    })
    # Otherwise, it wasn't deleted. If it used to be an insert sentinel, we
    # drew this token from our insert distn.
    cases.append({
        "if":
            xt_token == DynamicLengthSentinelSequence.INSERT_SENTINEL,
        "then":
            d_t_to_tplus1.lp_no_delete +
            d_t_to_tplus1.D_insert_logits[xtplus1_token]
    })
    # Finally, if it was an ordinary token, we must have perturbed.
    default = (
        d_t_to_tplus1.lp_no_delete + aligned_xt_Attplus1_logits[xtplus1_token])
    return jnp.select(
        condlist=[case["if"] for case in cases],
        choicelist=[case["then"] for case in cases],
        default=default)

  values = jax.vmap(compute_token_logprob)(
      jnp.arange(xt.tokens.shape[0]) >= copy_count, aligned_xt, aligned_xtplus1,
      aligned_xt_Attplus1)
  aligned_logprobs = jnp.sum(values)

  return insert_sentinel_log_probs + aligned_logprobs


def _get_copy_chance(s_token,
                     t_token,
                     known_reroll,
                     distn,
                     precomputed_xA_logits=None):
  """Computes the log probability of modifying `s_token` into `d_token`."""
  cases = []
  if known_reroll is None:
    lp_reroll = distn.lp_reroll
    lp_keep = distn.lp_keep
  else:
    lp_reroll = jnp.where(known_reroll, distn.lp_reroll, -jnp.inf)
    lp_keep = jnp.where(known_reroll, -jnp.inf, distn.lp_keep)
  # If either was an error, propagate.
  cases.append({
      "if": (s_token == DynamicLengthSentinelSequence.ERROR) |
            (t_token == DynamicLengthSentinelSequence.ERROR),
      "then":
          BAD_VALUE
  })
  # Impossible to copy a delete sentinel or produce an insert sentinel from a
  # copy. Assign zero probability.
  cases.append({
      "if": (s_token == DynamicLengthSentinelSequence.DELETE_SENTINEL) |
            (t_token == DynamicLengthSentinelSequence.INSERT_SENTINEL),
      "then":
          -jnp.inf
  })
  # Otherwise, if we produced a delete sentinel, use delete sentinel log prob.
  cases.append({
      "if": t_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      "then": distn.lp_sentinel_delete
  })
  # Otherwise, if we produced a token from an insert sentinel, look up the
  # right logit. Note that we have to watch out for rerolls.
  cases.append({
      "if":
          s_token == DynamicLengthSentinelSequence.INSERT_SENTINEL,
      "then":
          math_util.safe_logaddexp(
              lp_reroll + distn.D_reroll_logits[t_token],
              lp_keep + distn.D_sentinel_insert_logits[t_token])
  })
  # Otherwise, if we saw two ordinary tokens, look up the right logit.
  if precomputed_xA_logits is None:
    precomputed_xA_logits = distn.A.apply(s_token, is_distn=False, log=True)
  default = math_util.safe_logaddexp(
      (lp_reroll + distn.D_reroll_logits[t_token]),
      (lp_keep + precomputed_xA_logits[t_token]))

  return jnp.select(
      condlist=[case["if"] for case in cases],
      choicelist=[case["then"] for case in cases],
      default=default)


def _get_insert_chance(t_token, distn):
  """Computes the log probability of inserting `t_token` over multiple steps."""
  cases = []
  # Propagate error.
  cases.append({
      "if": t_token == DynamicLengthSentinelSequence.ERROR,
      "then": BAD_VALUE
  })
  # Silently inserted a delete sentinel.
  cases.append({
      "if": t_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      "then": distn.lp_silent_insert_to_delete
  })
  # Inserted an insert sentinel.
  cases.append({
      "if": t_token == DynamicLengthSentinelSequence.INSERT_SENTINEL,
      "then": distn.lp_sentinel_insert
  })
  # Silent insert of a token.
  default = distn.lp_silent_insert + distn.D_silent_insert_logits[t_token]

  return jnp.select(
      condlist=[case["if"] for case in cases],
      choicelist=[case["then"] for case in cases],
      default=default)


def _get_delete_chance(s_token, distn):
  """Computes the log probability of deleting `s_token` over multiple steps."""
  cases = []
  # Propagate error.
  cases.append({
      "if": s_token == DynamicLengthSentinelSequence.ERROR,
      "then": BAD_VALUE
  })
  # Delete sentinels always get deleted.
  cases.append({
      "if": s_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      "then": 0.0
  })
  # Otherwise, this was a silent deletion.
  default = distn.lp_silent_delete

  return jnp.select(
      condlist=[case["if"] for case in cases],
      choicelist=[case["then"] for case in cases],
      default=default)


def multi_step_aligned_log_prob(
    source,
    target,
    alignment,
    distn,
    precomputed_xA = None,
):
  """Log probs over many insert-delete steps.

  Args:
    source: Sequence before the step.
    target: Sequence after the step.
    alignment: Alignment of source and target.
    distn: Distribution of the step.
    precomputed_xA: Optional precomputed distn.A.apply(source.tokens)

  Returns:
    log q(target, alignment | source)
  """
  (source, target, alignment,
   distn) = jax.tree.map(jnp.array, (source, target, alignment, distn))

  # Log probs of inserts.
  insertion_mask = alignment.insert_mask()
  insert_count = jnp.count_nonzero(insertion_mask)
  copy_count = target.length - insert_count

  insert_log_probs_per_token = jnp.where(
      insertion_mask,
      jax.vmap(functools.partial(_get_insert_chance,
                                 distn=distn))(target.tokens), 0.0)
  insert_log_probs = jnp.sum(insert_log_probs_per_token)
  stop_insert_log_probs = (copy_count + 1) * distn.lp_no_insert

  # Log probs of deletes.
  deletion_mask = alignment.delete_mask()
  delete_log_probs_per_token = jnp.where(
      deletion_mask,
      jax.vmap(functools.partial(_get_delete_chance,
                                 distn=distn))(source.tokens), 0.0)
  delete_log_probs = jnp.sum(delete_log_probs_per_token)

  # Log probs of copies.
  aligned_source = shift_to_front(source.tokens,
                                  source.token_mask() & ~deletion_mask)
  aligned_target = shift_to_front(target.tokens,
                                  target.token_mask() & ~insertion_mask)
  if alignment.reroll_mask is None:
    aligned_known_reroll = None
  else:
    aligned_known_reroll = shift_to_front(alignment.reroll_mask,
                                          target.token_mask() & ~insertion_mask)

  if precomputed_xA is None:
    aligned_precomputed_xA = jax.vmap(
        functools.partial(distn.A.apply, is_distn=False, log=True))(
            aligned_source)
  else:
    aligned_precomputed_xA = shift_to_front(
        precomputed_xA,
        source.token_mask() & ~deletion_mask)

  aligned_log_probs_per_token = jnp.where(
      jnp.arange(target.tokens.shape[0]) < copy_count,
      jax.vmap(functools.partial(_get_copy_chance, distn=distn))(
          aligned_source,
          aligned_target,
          aligned_known_reroll,
          precomputed_xA_logits=aligned_precomputed_xA), 0.0)
  aligned_log_probs = jnp.sum(aligned_log_probs_per_token)

  return (insert_log_probs + stop_insert_log_probs + delete_log_probs +
          aligned_log_probs)


@dataclasses.dataclass
class DynamicBandSchedule(dynamic_programs.DynamicProgrammingSchedule):
  """Schedule that only writes to indices along a dynamic band.

  Writes along diagonals of length `band_width` that linearly interpolate to
  the dynamic bounds. For instance, for band_width 4, dynamic bounds (6, 8),
  we write to each index in the table at these times:

     0,  1,  2,  3,  4,  5,  _,  _
     1,  2,  3,  4,  5,  6,  7,  8
     2,  3,  4,  5,  6,  7,  8,  9
     3,  4,  5,  6,  7,  8,  9, 10
     _,  _,  6,  7,  8,  9, 10, 11
     _,  _,  _,  _,  9, 10, 11, 12

  Attributes:
    max_shape: Static size of the maximum table.
    band_width: Number of elements in each diagonal to compute.
    dynamic_bounds: Coordinates of the last endpoint to fill.
  """
  max_shape: Tuple[int, int]
  band_width: int
  dynamic_bounds: Tuple[NDArray, NDArray]

  @property
  def num_steps(self):
    return np.sum(self.max_shape) - 1

  def indices(self, t):
    shift_min_t = self.band_width - 1
    shift_max_t = np.sum(self.dynamic_bounds) - self.band_width + 1

    shift_min_ix_0 = self.band_width - 1
    shift_max_ix_0 = self.dynamic_bounds[0]

    # shift_min_ix_1 = 0
    # shift_max_ix_1 = self.dynamic_bounds[1] - self.band_width

    shift_0 = (
        shift_min_ix_0 + (t - shift_min_t) *
        (shift_max_ix_0 - shift_min_ix_0) // (shift_max_t - shift_min_t))
    shift_1 = t - shift_0

    indices_0 = shift_0 - jnp.arange(self.band_width)
    indices_1 = shift_1 + jnp.arange(self.band_width)

    valid = ((indices_0 >= 0) & (indices_0 < self.dynamic_bounds[0]) &
             (indices_1 >= 0) & (indices_1 < self.dynamic_bounds[1]))

    return util.tree_where(valid, (indices_0, indices_1), self.max_shape)


def compute_multi_step_alignment_table(
    source,
    target,
    distn,
    precomputed_xA = None,
    pack_block_size=None,
    use_dynamic_band=False,
):
  """Compute forward alignment table.

  Should only be used for metrics and evaluation purposes, not intended for
  training.

  Args:
    source: int[max_len] sequence of token indices.
    target: int[max_len] sequence of token indices.
    distn: Distribution to noise by.
    precomputed_xA: Optional precomputed distn.A.apply(source.tokens)
    pack_block_size: Optional packing size, used to compute a blocked dynamic
      program schedule.
    use_dynamic_band: Whether to use a dynamic band, which only fills a subset
      of entries in the DP table. This can be used to marginalize over likely
      alignments only.

  Returns:
    (log_gen, log_take), each of shape int[max_len+1, max_len+1], where:
      log_gen[i, j] is the log-prob of consuming source[:i], producing
      target[:j],
        and arriving at state GEN (deciding whether to generate more tokens)
      log_take[i, j] is the log-prob of consuming source[:i], producing
      target[:j],
        and arriving at state TAKE (deciding whether to take tokens from the
        source; also the terminal state)
      Note that the table may contain garbage after the end of the sequences,
      so the result should be indexed into appropriately.
  """
  source, target, distn = jax.tree.map(jnp.array, (source, target, distn))
  source_max_len, = source.tokens.shape
  target_max_len, = target.tokens.shape

  # copy_chances = jax.vmap(
  #     jax.vmap(
  #         functools.partial(_get_copy_chance, known_reroll=None, distn=distn),
  #         in_axes=(None, 0)),
  #     in_axes=(0, None))(source.tokens, target.tokens)
  insert_chances = jax.vmap(functools.partial(_get_insert_chance, distn=distn))(
      target.tokens)
  delete_chances = jax.vmap(functools.partial(_get_delete_chance, distn=distn))(
      source.tokens)

  if precomputed_xA is None:
    precomputed_xA = jax.vmap(
        functools.partial(distn.A.apply, is_distn=False, log=True))(
            source.tokens)

  def lookback_fn(indices):
    i, j = indices
    log_gen_lookback = {"i, j - 1": (i, j - 1)}
    log_take_lookback = {"i - 1, j - 1": (i - 1, j - 1), "i - 1, j": (i - 1, j)}
    return log_gen_lookback, log_take_lookback

  def kernel_fn(reads, indices):
    i, j = indices

    log_gen, log_take = reads
    copy_chance = _get_copy_chance(
        source.tokens[i - 1],
        target.tokens[j - 1],
        known_reroll=None,
        precomputed_xA_logits=precomputed_xA[i - 1],
        distn=distn)

    log_gen_ij = math_util.safe_logaddexp(
        jnp.where(j == 0, -jnp.inf,
                  log_gen["i, j - 1"] + insert_chances[j - 1]),
        jnp.where((i == 0) | (j == 0), -jnp.inf,
                  log_take["i - 1, j - 1"] + copy_chance))
    # base case
    log_gen_ij = jnp.where((i == 0) & (j == 0), 0., log_gen_ij)

    log_take_ij = math_util.safe_logaddexp(
        jnp.where(i == 0, -jnp.inf,
                  log_take["i - 1, j"] + delete_chances[i - 1]),
        log_gen_ij + distn.lp_no_insert)
    return log_gen_ij, log_take_ij

  if use_dynamic_band:
    assert pack_block_size

    schedule = DynamicBandSchedule(
        max_shape=(source_max_len + 1, target_max_len + 1),
        band_width=pack_block_size,
        dynamic_bounds=(source.length + 1, target.length + 1))

  elif pack_block_size:

    def dep_fn(indices):
      i, j = indices
      deps = []
      if i > 0:
        deps.append((i - 1, j))
      if j > 0:
        deps.append((i, j - 1))
      if i > 0 and j > 0:
        deps.append((i - 1, j - 1))
      return deps

    schedule = dynamic_programs.packed_block_schedule(
        shape=(source_max_len + 1, target_max_len + 1),
        block_size=pack_block_size,
        dependency_fn=dep_fn)

  else:
    schedule = dynamic_programs.SerialSchedule(
        shape=(source_max_len + 1, target_max_len + 1))

  if use_dynamic_band:
    # Treat off-band entries as impossible.
    default_value = -jnp.inf
  else:
    # Fill with sentinel value so that we notice if we ever forget to fill.
    default_value = BAD_VALUE
  empty_table = jnp.full((source_max_len + 1, target_max_len + 1),
                         default_value)

  # @experimental_cpu_call.cpu_call
  def execute_dynamic_program():
    return dynamic_programs.dynamic_program((empty_table, empty_table),
                                            lookback_fn=lookback_fn,
                                            kernel_fn=kernel_fn,
                                            schedule=schedule)

  log_gen, log_take = execute_dynamic_program()

  return log_gen, log_take


def multi_step_log_prob(source,
                        target,
                        distn,
                        pack_block_size=None):
  _, log_take = compute_multi_step_alignment_table(
      source=source,
      target=target,
      distn=distn,
      pack_block_size=pack_block_size)
  return log_take[source.length, target.length]


@flax.struct.dataclass
class AlignmentSampleState:
  source_index: NDArray
  target_index: NDArray
  is_in_gen: NDArray
  backpointers: NDArray
  reroll_mask: NDArray
  rng: Optional[NDArray] = None


def sample_alignment_from_table(
    source,
    target,
    distn,
    alignment_table,
    rng,
    precomputed_xA = None):
  """Sample an alignment from a table.

  Args:
    source: Source sequence.
    target: Target sequence.
    distn: Distribution to noise by.
    alignment_table: Alignment table.
    rng: PRNGKey to use.
    precomputed_xA: Optional precomputed distn.A.apply(source.tokens)

  Returns:
    Random alignment conditioned on source and target.
  """
  source, target, distn = jax.tree.map(jnp.array, (source, target, distn))
  log_gen, log_take = alignment_table

  # Start with the accepting state of the FSM:
  initial_state = AlignmentSampleState(
      source_index=source.length,
      target_index=target.length,
      is_in_gen=False,  # we stop in state take after consuming the entire input
      backpointers=jnp.full([target.tokens.shape[0]], -1),
      reroll_mask=jnp.full([target.tokens.shape[0]], False),
      rng=rng)

  if precomputed_xA is None:
    precomputed_xA = jax.vmap(
        functools.partial(distn.A.apply, is_distn=False, log=True))(
            source.tokens)

  def cond(state):
    # Continue executing until we reach the initial state.
    return jnp.logical_not((state.source_index == 0) & (state.target_index == 0)
                           & state.is_in_gen)

  def body(state):
    # Run a single two-level transition in reverse.
    i = state.source_index
    j = state.target_index
    which_rng, next_rng = jax.random.split(state.rng)

    log_gen_ij = log_gen[i, j]
    log_gen_i_jm1 = jnp.where(j == 0, -jnp.inf, log_gen[i, j - 1])
    log_take_im1_jm1 = (
        jnp.where((i == 0) | (j == 0), -jnp.inf, log_take[i - 1, j - 1]))
    log_take_im1_j = jnp.where(i == 0, -jnp.inf, log_take[i - 1, j])

    source_token = source.tokens[i - 1]
    target_token = target.tokens[j - 1]
    precomputed_xA_logits = precomputed_xA[i - 1]
    copy_keep_chance = _get_copy_chance(source_token, target_token, False,
                                        distn, precomputed_xA_logits)
    copy_reroll_chance = _get_copy_chance(source_token, target_token, True,
                                          distn, precomputed_xA_logits)
    insert_chance = _get_insert_chance(target_token, distn)
    delete_chance = _get_delete_chance(source_token, distn)

    # Leading to gen_ij
    case_1_logit = log_gen_i_jm1 + insert_chance
    case_1_state = AlignmentSampleState(
        source_index=i,
        target_index=j - 1,
        is_in_gen=True,  # started in gen
        backpointers=state.backpointers,
        reroll_mask=state.reroll_mask)

    case_2a_logit = log_take_im1_jm1 + copy_keep_chance
    case_2a_state = AlignmentSampleState(
        source_index=i - 1,
        target_index=j - 1,
        is_in_gen=False,  # started in take
        backpointers=state.backpointers.at[j - 1].set(i - 1),
        reroll_mask=state.reroll_mask)
    case_2b_logit = log_take_im1_jm1 + copy_reroll_chance
    case_2b_state = AlignmentSampleState(
        source_index=i - 1,
        target_index=j - 1,
        is_in_gen=False,  # started in take
        backpointers=state.backpointers.at[j - 1].set(i - 1),
        reroll_mask=state.reroll_mask.at[j - 1].set(True))

    # Leading to take_ij
    case_3_logit = log_take_im1_j + delete_chance
    case_3_state = AlignmentSampleState(
        source_index=i - 1,
        target_index=j,
        is_in_gen=False,  # started in take
        backpointers=state.backpointers,
        reroll_mask=state.reroll_mask)

    case_4_logit = log_gen_ij + distn.lp_no_insert
    case_4_state = AlignmentSampleState(
        source_index=i,
        target_index=j,
        is_in_gen=True,  # started in gen
        backpointers=state.backpointers,
        reroll_mask=state.reroll_mask)

    chosen_case = jnp.where(
        state.is_in_gen,
        jax.random.categorical(
            which_rng, jnp.stack([case_1_logit, case_2a_logit, case_2b_logit])),
        3 + jax.random.categorical(which_rng,
                                   jnp.stack([case_3_logit, case_4_logit])))

    return dataclasses.replace(
        util.tree_choose(chosen_case, [
            case_1_state, case_2a_state, case_2b_state, case_3_state,
            case_4_state
        ]),
        rng=next_rng)

  # Run this in a loop until we stop.
  finished_state = jax.lax.while_loop(cond, body, initial_state)
  return SequenceAlignment(
      finished_state.backpointers,
      target.delete_sentinel_mask(),
      source.length,
      target.length,
      is_single_step=isinstance(distn, OneStepDistn),
      reroll_mask=finished_state.reroll_mask)


def rao_blackwellize_dynamic_band_alignment(
    source,
    target,
    distn,
    alignment,
    alignment_table,
    rng,
):
  """If the alignment falls in the dynamic band, resample it.

  Suppose we want to compute an unbiased estimate of some function of an
  alignment. Notice that the following algorithm produces the right
  distribution:
  - Sample an alignment
  - If the alignment satisfies some property P, resample an alignment
    conditioned on sampling property P.
  - If the alignment does not satisfy property P, return the original alignment.

  This is essentially Rao-Blackwellizing over the quantity

    "yes" if property P else ("no", alignment)

  Here we let property P be "this alignment falls within the dynamic band we
  computed in the alignment table".

  Args:
    source: Source sequence.
    target: Target sequence.
    distn: Distribution to noise by.
    alignment: Sampled alignment.
    alignment_table: Table of alignments we can efficiently re-sample from.
    rng: RNG key.

  Returns:
    Another sample of an alignment, if efficiently re-sampleable, otherwise a
    copy of the original alignment.
  """
  log_gen, log_take = alignment_table

  source_delete_mask = alignment.delete_mask()
  source_valid = (
      jnp.arange(source_delete_mask.shape[0]) < alignment.before_length)
  source_copy_mask = (source_valid & ~source_delete_mask)

  target_insert_mask = alignment.insert_mask()
  target_valid = (
      jnp.arange(target_insert_mask.shape[0]) < alignment.after_length)
  target_copy_mask = (target_valid & ~target_insert_mask)

  # How many target tokens have we consumed after processing each source token,
  # and vice versa?
  safe_backptr = alignment.backpointers
  # safe_backptr = safe_backptr.at[target.length].set(source.length)
  safe_backptr = jnp.where(
      target_valid & (safe_backptr != SequenceAlignment.NO_BACKPOINTER),
      safe_backptr, source_delete_mask.shape[0])

  sources_before_each_source = jnp.arange(source_delete_mask.shape[0])
  sources_before_each_target = util.previous_active_value(
      sources_before_each_source[safe_backptr],
      target_copy_mask,
      inclusive=True)

  targets_before_each_target = jnp.arange(target_insert_mask.shape[0])
  targets_before_each_source = util.previous_active_value(
      jnp.full_like(sources_before_each_source,
                    -1).at[safe_backptr].set(targets_before_each_target),
      source_copy_mask,
      inclusive=True)

  gen_indices = (sources_before_each_target + 1, targets_before_each_target + 1)
  take_indices = (sources_before_each_source + 1,
                  targets_before_each_source + 1)

  log_gen, log_take = alignment_table
  used_gen = log_gen[gen_indices]
  used_take = log_take[take_indices]
  used_gen_ok = jnp.where(target_valid, used_gen > -np.inf, True)
  used_take_ok = jnp.where(source_valid, used_take > -np.inf, True)

  all_valid = jnp.all(used_gen_ok) & jnp.all(used_take_ok)

  return util.tree_where(
      all_valid,
      sample_alignment_from_table(
          source=source,
          target=target,
          distn=distn,
          alignment_table=alignment_table,
          rng=rng), alignment)


@flax.struct.dataclass
class IntermediateSampleState:
  source_index: NDArray
  target_index: NDArray
  intermediate_index: NDArray
  intermediate_buffer: NDArray
  alignment_0_t_backpointers: NDArray
  alignment_0_t_reroll: NDArray
  alignment_t_tplus1_backpointers: NDArray
  debug_total_logprob: NDArray
  max_steps: NDArray
  rng: Optional[NDArray] = None


def _get_intermediate_insert_logits(xtplus1_token, d_0_to_t, d_t_to_tplus1,
                                    precomputed_Attplus1_xtplus1):
  """Computes log probabilities for inserts over multiple steps."""
  if precomputed_Attplus1_xtplus1 is None:
    precomputed_Attplus1_xtplus1 = d_t_to_tplus1.A.observe(
        xtplus1_token, is_distn=False, log=True)

  # Case 1: We inserted an INS sentinel at time 2. It did not appear at time
  # 1 and had no other effects.
  is_case_1 = xtplus1_token == DynamicLengthSentinelSequence.INSERT_SENTINEL

  # Case 2: We didn't insert at time 2. At time 1, we inserted an unknown
  # geometric number of delete sentinels, and then inserted some other token.
  # We decided to transform it into what we observed (which is not an INS
  # sentinel).
  # At time 1, we see the delete sentinels followed by our inserted token.
  observe_at_tplus1_logits = jnp.where(
      xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      d_t_to_tplus1.lp_delete,
      (
          d_t_to_tplus1.lp_no_delete
          + d_t_to_tplus1.A.observe(xtplus1_token, is_distn=False, log=True)
      ),
  )
  observe_at_tplus1_insert_sentinel_logit = jnp.where(
      xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      d_t_to_tplus1.lp_delete,
      (
          d_t_to_tplus1.lp_no_delete
          + d_t_to_tplus1.D_insert_logits[xtplus1_token]
      ),
  )

  case_2_shared = d_t_to_tplus1.lp_no_insert + arbitrarily_many_times(
      d_0_to_t.lp_silent_insert_to_delete
  )
  lp_was_ins_sentinel_at_step_1 = (
      case_2_shared
      + d_0_to_t.lp_sentinel_insert
      + observe_at_tplus1_insert_sentinel_logit
  )
  lp_was_token_at_step_1 = (
      case_2_shared
      + d_0_to_t.lp_silent_insert
      + d_0_to_t.D_silent_insert_logits
      + observe_at_tplus1_logits
  )
  lp_case_2 = math_util.safe_logaddexp(
      lp_was_ins_sentinel_at_step_1,
      math_util.safe_logsumexp(lp_was_token_at_step_1),
  )

  return types.SimpleNamespace(**locals())


def _get_intermediate_copy_logits(x0_token, xtplus1_token, known_reroll,
                                  d_0_to_t, d_t_to_tplus1, precomputed_x0_A0t,
                                  precomputed_Attplus1_xtplus1):
  """Computes log probabilities for copies/modifications over multiple steps."""
  if known_reroll is None:
    observe_reroll = 0.0
    observe_no_reroll = 0.0
  else:
    # If we have a reroll mask, zero out probabilities for any path that
    # causes us to reroll / not reroll inconsistently. Note that if we chose
    # to delete this token, the reroll mask will be False regardless of
    # whether or not we decided to reroll at step 1.
    (observe_reroll, observe_no_reroll) = util.tree_select(
        condlist=[
            xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
            known_reroll,
        ],
        choicelist=[(0.0, 0.0), (0.0, -jnp.inf)],
        default=(-jnp.inf, 0.0))

  if precomputed_x0_A0t is None:
    precomputed_x0_A0t = d_0_to_t.A.apply(x0_token, is_distn=False, log=True)

  if precomputed_Attplus1_xtplus1 is None:
    precomputed_Attplus1_xtplus1 = d_t_to_tplus1.A.observe(
        xtplus1_token, is_distn=False, log=True)

  # Case 1: We perturbed/rerolled at time 1, then perturbed/marked it for
  # deletion to transform it into what we observed.
  # At time 1, we see the half-perturbed version of this token.
  observe_at_tplus1_logits = jnp.where(
      xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      d_t_to_tplus1.lp_delete,
      (d_t_to_tplus1.lp_no_delete + precomputed_Attplus1_xtplus1))

  case_1_no_reroll_token_logits = (
      d_0_to_t.lp_keep + precomputed_x0_A0t + observe_no_reroll +
      observe_at_tplus1_logits)
  case_1_reroll_token_logits = (
      d_0_to_t.lp_reroll + d_0_to_t.D_reroll_logits + observe_reroll +
      observe_at_tplus1_logits)
  case_1_token_logits = math_util.safe_logaddexp(case_1_no_reroll_token_logits,
                                                 case_1_reroll_token_logits)
  lp_case_1 = math_util.safe_logsumexp(case_1_token_logits)

  # Case 2: We transformed it to a sentinel delete token at time 1, followed
  # by inserting some unknown geometric number of delete sentinels, and then
  # chose to insert some other token at time 1. Finally we transformed this
  # into what we observed at time 2.
  # At time 1, we see the delete sentinels followed by our newly inserted
  # token.
  case_2_shared = (
      d_0_to_t.lp_sentinel_delete +
      arbitrarily_many_times(d_0_to_t.lp_silent_insert_to_delete) +
      observe_reroll)
  observe_at_tplus1_insert_sentinel_logit = jnp.where(
      xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
      d_t_to_tplus1.lp_delete, (d_t_to_tplus1.lp_no_delete +
                                d_t_to_tplus1.D_insert_logits[xtplus1_token]))
  lp_was_ins_sentinel_at_step_1 = (
      case_2_shared + d_0_to_t.lp_sentinel_insert +
      observe_at_tplus1_insert_sentinel_logit)
  case_2_lp_was_token_at_step_1 = (
      case_2_shared + d_0_to_t.lp_silent_insert +
      d_0_to_t.D_silent_insert_logits + observe_at_tplus1_logits)
  lp_case_2 = math_util.safe_logaddexp(
      lp_was_ins_sentinel_at_step_1,
      math_util.safe_logsumexp(case_2_lp_was_token_at_step_1))

  return types.SimpleNamespace(**locals())


def sample_intermediate(
    x0,
    xtplus1,
    alignment,
    d_0_to_t,
    d_t_to_tplus1,
    rng,
    precomputed_x0_A0t = None,
    precomputed_Attplus1_xtplus1 = None,
    debug = False
):
  """Sample an intermediate value conditioned on an aligned outer value.

  Samples from
  q(x_t, alignment(0, t), alignment(t, t+1) | x_0, x_{t+1}, alignment(0, t+1)).

  Args:
    x0: Initial sequence.
    xtplus1: Final sequence.
    alignment: Alignment between sequences.
    d_0_to_t: Distribution of the first steps.
    d_t_to_tplus1: Distribution of the last step.
    rng: Random generator.
    precomputed_x0_A0t: Optional precomputed d_0_to_t.A.apply(x0.tokens)
    precomputed_Attplus1_xtplus1: Optional precomputed
      d_t_to_tplus1.A.observe(xtplus1.tokens)
    debug: Whether to compute total log prob of all alignments while sampling
      one.

  Returns:
    xt, alignment from 0 to t, and alignment from t to t+1.
  """
  (x0, xtplus1, alignment, d_0_to_t, d_t_to_tplus1) = jax.tree.map(
      jnp.array, (x0, xtplus1, alignment, d_0_to_t, d_t_to_tplus1))

  # Algorithm sketch:
  # Consider each event in the alignment, which is a transition in our aggregate
  # FSM. At each point, decide what we did, and insert the corresponding
  # entries into our middle states.

  insert_mask = alignment.insert_mask()
  delete_mask = alignment.delete_mask()

  if precomputed_x0_A0t is None:
    precomputed_x0_A0t = jax.vmap(
        functools.partial(d_0_to_t.A.apply, is_distn=False, log=True))(
            x0.tokens)

  if precomputed_Attplus1_xtplus1 is None:
    precomputed_Attplus1_xtplus1 = jax.vmap(
        functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True))(
            xtplus1.tokens)

  def handle_insert(state, rng):
    xtplus1_token = xtplus1.tokens[state.target_index]
    insert_info = _get_intermediate_insert_logits(
        xtplus1_token, d_0_to_t, d_t_to_tplus1,
        precomputed_Attplus1_xtplus1[state.target_index])
    # Case 1: We inserted an INS sentinel at time 2. It did not appear at time
    # 1 and had no other effects.
    case_1_state = dataclasses.replace(
        state,
        target_index=state.target_index + 1,
        debug_total_logprob=state.debug_total_logprob + d_t_to_tplus1.lp_insert)

    # Case 2: We didn't insert at time 2. At time 1, we inserted an unknown
    # geometric number of delete sentinels, and then inserted some other token.
    # We decided to transform it into what we observed (which is not an INS
    # sentinel).
    # At time 1, we see the delete sentinels followed by our inserted token.
    k1, k2 = jax.random.split(rng, 2)
    num_ephemeral_sentinels = distributions.random_geometric(
        k1, d_0_to_t.lp_silent_insert_to_delete)
    del k1
    case_2_token_logits_stacked = jnp.concatenate([
        insert_info.lp_was_ins_sentinel_at_step_1[None],
        insert_info.lp_was_token_at_step_1
    ])
    case_2_token_stacked = jax.random.categorical(k2,
                                                  case_2_token_logits_stacked)
    case_2_token = jnp.where(case_2_token_stacked == 0,
                             DynamicLengthSentinelSequence.INSERT_SENTINEL,
                             case_2_token_stacked - 1)
    case_2_token_position = state.intermediate_index + num_ephemeral_sentinels

    intermediate_buffer = (
        state.intermediate_buffer.at[case_2_token_position].set(case_2_token))
    alignment_t_tplus1_backpointers = (
        state.alignment_t_tplus1_backpointers.at[state.target_index].set(
            case_2_token_position))
    case_2_state = dataclasses.replace(
        state,
        target_index=state.target_index + 1,
        intermediate_index=case_2_token_position + 1,
        intermediate_buffer=intermediate_buffer,
        alignment_t_tplus1_backpointers=alignment_t_tplus1_backpointers,
        debug_total_logprob=state.debug_total_logprob + insert_info.lp_case_2)

    return util.tree_where(insert_info.is_case_1, case_1_state, case_2_state)

  def handle_stop_inserting(state, rng):
    # We must have decided not to insert in step 2, inserted an unknown
    # geometric number of delete sentinels, and finally chose to stop
    # inserting.
    lp_stop_inserting = (
        d_t_to_tplus1.lp_no_insert +
        arbitrarily_many_times(d_0_to_t.lp_silent_insert_to_delete) +
        d_0_to_t.lp_no_insert)
    num_ephemeral_sentinels = distributions.random_geometric(
        rng, d_0_to_t.lp_silent_insert_to_delete)
    next_state = dataclasses.replace(
        state,
        intermediate_index=state.intermediate_index + num_ephemeral_sentinels,
        debug_total_logprob=state.debug_total_logprob + lp_stop_inserting)
    return next_state

  def handle_delete(state, rng):
    x0_token = x0.tokens[state.source_index]
    case_key, k1 = jax.random.split(rng)
    # Case 1: Token was a delete sentinel at time 0. It was dropped silently
    # with no other effects.
    # At time 1, we see nothing.
    is_case_1 = (x0_token == DynamicLengthSentinelSequence.DELETE_SENTINEL)
    case_1_state = dataclasses.replace(
        state, source_index=state.source_index + 1)

    # Case 2: Token was not a delete sentinel.
    # We deleted it silently at step 1 with no other effects.
    # At time 1, we see nothing.
    lp_case_2 = d_0_to_t.lp_silent_delete
    case_2_state = case_1_state

    # Case 3: Token was not a delete sentinel.
    # We replaced it with a delete sentinel at step 1. We then inserted an
    # unknown geometric number of delete sentinels, and finally chose to stop
    # inserting. All of them were removed at step 2.
    # At time 1, we see these delete sentinels.
    # Note that the matched token comes before the extra tokens in the
    # alignment!
    lp_case_3 = (
        d_0_to_t.lp_sentinel_delete +
        arbitrarily_many_times(d_0_to_t.lp_silent_insert_to_delete) +
        d_0_to_t.lp_no_insert)
    num_ephemeral_sentinels = distributions.random_geometric(
        k1, d_0_to_t.lp_silent_insert_to_delete)
    del k1

    alignment_0_t_backpointers = (
        state.alignment_0_t_backpointers.at[state.intermediate_index].set(
            state.source_index))
    case_3_state = dataclasses.replace(
        state,
        source_index=state.source_index + 1,
        intermediate_index=(state.intermediate_index + 1 +
                            num_ephemeral_sentinels),
        alignment_0_t_backpointers=alignment_0_t_backpointers)

    # Combine cases
    case_2_or_3_state = dataclasses.replace(
        util.tree_choose(
            jax.random.categorical(case_key, jnp.stack([lp_case_2, lp_case_3])),
            [case_2_state, case_3_state]),
        debug_total_logprob=(state.debug_total_logprob +
                             math_util.safe_logaddexp(lp_case_2, lp_case_3)))

    return util.tree_where(is_case_1, case_1_state, case_2_or_3_state)

  def handle_copy(state, rng):
    case_key, k1, k2, k3, k4 = jax.random.split(rng, 5)
    x0_token = x0.tokens[state.source_index]
    xtplus1_token = xtplus1.tokens[state.target_index]

    if alignment.reroll_mask is None:
      known_reroll = None
    else:
      known_reroll = alignment.reroll_mask[state.target_index]
    copy_info = _get_intermediate_copy_logits(
        x0_token, xtplus1_token, known_reroll, d_0_to_t, d_t_to_tplus1,
        precomputed_x0_A0t[state.source_index],
        precomputed_Attplus1_xtplus1[state.target_index])

    # Case 1: We perturbed/rerolled at time 1, then perturbed/marked it for
    # deletion to transform it into what we observed.
    # At time 1, we see the half-perturbed version of this token.
    case_1_token = jax.random.categorical(k1, copy_info.case_1_token_logits)
    del k1
    case_1_was_reroll = jax.random.categorical(
        k4,
        jnp.stack([
            copy_info.case_1_no_reroll_token_logits[case_1_token],
            copy_info.case_1_reroll_token_logits[case_1_token],
        ])).astype(bool)
    del k4

    intermediate_buffer = (
        state.intermediate_buffer.at[state.intermediate_index].set(case_1_token)
    )
    alignment_0_t_backpointers = (
        state.alignment_0_t_backpointers.at[state.intermediate_index].set(
            state.source_index))
    alignment_0_t_reroll = (
        state.alignment_0_t_reroll.at[state.intermediate_index].set(
            case_1_was_reroll))
    alignment_t_tplus1_backpointers = (
        state.alignment_t_tplus1_backpointers.at[state.target_index].set(
            state.intermediate_index))

    case_1_state = dataclasses.replace(
        state,
        source_index=state.source_index + 1,
        target_index=state.target_index + 1,
        intermediate_index=state.intermediate_index + 1,
        intermediate_buffer=intermediate_buffer,
        alignment_0_t_backpointers=alignment_0_t_backpointers,
        alignment_0_t_reroll=alignment_0_t_reroll,
        alignment_t_tplus1_backpointers=alignment_t_tplus1_backpointers)

    # Case 2: We transformed it to a sentinel delete token at time 1, followed
    # by inserting some unknown geometric number of delete sentinels, and then
    # chose to insert some other token at time 1. Finally we transformed this
    # into what we observed at time 2.
    # At time 1, we see the delete sentinels followed by our newly inserted
    # token.
    case_2_token_logits_stacked = jnp.concatenate([
        copy_info.lp_was_ins_sentinel_at_step_1[None],
        copy_info.case_2_lp_was_token_at_step_1
    ])
    case_2_token_stacked = jax.random.categorical(k2,
                                                  case_2_token_logits_stacked)
    del k2
    case_2_token = jnp.where(case_2_token_stacked == 0,
                             DynamicLengthSentinelSequence.INSERT_SENTINEL,
                             case_2_token_stacked - 1)

    num_ephemeral_sentinels = distributions.random_geometric(
        k3, d_0_to_t.lp_silent_insert_to_delete)
    del k3

    case_2_token_position = (
        state.intermediate_index + 1 + num_ephemeral_sentinels)

    intermediate_buffer = (
        state.intermediate_buffer.at[case_2_token_position].set(case_2_token))
    alignment_0_t_backpointers = (
        state.alignment_0_t_backpointers.at[state.intermediate_index].set(
            state.source_index))
    alignment_t_tplus1_backpointers = (
        state.alignment_t_tplus1_backpointers.at[state.target_index].set(
            case_2_token_position))

    case_2_state = dataclasses.replace(
        state,
        source_index=state.source_index + 1,
        target_index=state.target_index + 1,
        intermediate_index=case_2_token_position + 1,
        intermediate_buffer=intermediate_buffer,
        alignment_0_t_backpointers=alignment_0_t_backpointers,
        alignment_t_tplus1_backpointers=alignment_t_tplus1_backpointers)

    return dataclasses.replace(
        util.tree_choose(
            jax.random.categorical(
                case_key, jnp.stack([copy_info.lp_case_1,
                                     copy_info.lp_case_2])),
            [case_1_state, case_2_state]),
        debug_total_logprob=(
            state.debug_total_logprob +
            math_util.safe_logaddexp(copy_info.lp_case_1, copy_info.lp_case_2)))

  def cond(state):
    return (state.max_steps >= 0) & ((state.source_index < x0.length) |
                                     (state.target_index < xtplus1.length))

  def body(state):
    # Following the order that the FSM processes events:
    # - Process any insertions, in order.
    # - Process any (non-rerolling) deletions, in order.
    # - If we reached EOS, stop.
    # - Process copies or rerolls (deletions that are followed by insertions).
    step1_rng, step2_rng, next_rng = jax.random.split(state.rng, 3)
    state = dataclasses.replace(state, rng=None)
    was_deletion = ((state.source_index < x0.length)
                    & delete_mask[state.source_index])
    was_insertion = ((state.target_index < xtplus1.length)
                     & insert_mask[state.target_index])
    keep_inserting_after = ((state.target_index + 1 < xtplus1.length)
                            & insert_mask[state.target_index + 1])

    # If needed, process an insertion. Maybe also process stop-inserting.
    state_after_insertion = handle_insert(state, step1_rng)
    state_after_insertion_maybe_stop_insert = util.tree_where(
        keep_inserting_after, state_after_insertion,
        handle_stop_inserting(state_after_insertion, step2_rng))

    # If needed, process a deletion.
    state_after_deletion = handle_delete(state, step1_rng)

    # If needed, process a copy/reroll. Maybe also process stop-inserting.
    state_after_copy = handle_copy(state, step1_rng)
    state_after_copy_maybe_stop_insert = util.tree_where(
        keep_inserting_after, state_after_copy,
        handle_stop_inserting(state_after_copy, step2_rng))

    # Select the right action.
    return dataclasses.replace(
        util.tree_where(
            was_insertion, state_after_insertion_maybe_stop_insert,
            util.tree_where(was_deletion, state_after_deletion,
                            state_after_copy_maybe_stop_insert)),
        rng=next_rng,
        max_steps=state.max_steps - 1)

  k1, k2 = jax.random.split(rng)
  del rng
  initial_state = IntermediateSampleState(
      source_index=0,
      target_index=0,
      intermediate_index=0,
      intermediate_buffer=jnp.full_like(
          xtplus1.tokens, DynamicLengthSentinelSequence.DELETE_SENTINEL),
      alignment_0_t_backpointers=jnp.full_like(
          xtplus1.tokens, SequenceAlignment.NO_BACKPOINTER),
      alignment_0_t_reroll=jnp.zeros(xtplus1.tokens.shape, bool),
      alignment_t_tplus1_backpointers=jnp.full_like(
          xtplus1.tokens, SequenceAlignment.NO_BACKPOINTER),
      debug_total_logprob=0.0,
      max_steps=x0.length + xtplus1.length + 1,
      rng=k1)
  del k1
  # Before entering the loop, handle initial stop-inserting transition.
  initial_state = util.tree_where(insert_mask[0], initial_state,
                                  handle_stop_inserting(initial_state, k2))
  del k2

  if debug:

    def scan_step(state, _):
      out = util.tree_where(cond(state), body(state), state)
      return out, out

    final_state, intermediates = jax.lax.scan(
        scan_step,
        init=initial_state,
        xs=None,
        length=(x0.length + xtplus1.length + 1))
    intermediates = jax.tree.map(
        lambda a, b: jnp.concatenate([a[None], b]), initial_state,
        intermediates)
  else:
    final_state = jax.lax.while_loop(cond, body, initial_state)

  xt = DynamicLengthSentinelSequence(final_state.intermediate_buffer,
                                     final_state.intermediate_index)
  alignment_0_to_t = SequenceAlignment(
      final_state.alignment_0_t_backpointers,
      xt.delete_sentinel_mask(),
      x0.length,
      xt.length,
      False,
      reroll_mask=final_state.alignment_0_t_reroll)
  alignment_t_to_tplus1 = SequenceAlignment(
      final_state.alignment_t_tplus1_backpointers,
      xtplus1.delete_sentinel_mask(),
      xt.length,
      xtplus1.length,
      True,
      reroll_mask=jnp.zeros(final_state.alignment_t_tplus1_backpointers.shape,
                            bool))

  if debug:
    return (xt, alignment_0_to_t, alignment_t_to_tplus1, final_state,
            intermediates)  # pytype: disable=bad-return-type
  else:
    return xt, alignment_0_to_t, alignment_t_to_tplus1


@flax.struct.dataclass
class ReverseProcessMarginalDistribution:
  """Outputs or targets for a model.

  Attributes:
    was_missing: bool[max_len] indicating whether each token in the current
      sequence was inserted right now, and didn't exist in the previous step.
      All other values are arbitrary where this is true.
    was_insert_log_prob: float32[max_len] Log prob that each token was an INS
      token in the last step.
    previous_token_log_probs: float32[max_len, vocab_size] Log prob that each
      token was each of the vocab tokens in the last step. Should sum to log(1 -
      exp(was_insert_log_prob)), so that it represents a log probability
      including insert as an option.
    log_prob_number_of_preceding_deletes: float32[max_len + 1, max_len] log prob
      of having observed each number of deletes before each token. Contains a
      value at length+1, for the number of deletes before the EOS token (i.e.
      before ending the sequence)
    length: Length of the sequence.
  """
  was_missing: NDArray
  was_insert_log_prob: NDArray
  previous_token_log_probs: NDArray
  log_prob_number_of_preceding_deletes: NDArray
  length: NDArray

  @staticmethod
  def point_mass(xt,
                 xtplus1, vocab_size):
    """Point mass at a single aligned previous sequence."""
    alignment = align_one_step(xt, xtplus1)
    was_missing = xtplus1.insert_sentinel_mask()
    was_insert = xt.insert_sentinel_mask()[alignment.backpointers]
    previous_token_probs = (
        jax.vmap(lambda t: jax.nn.one_hot(t, vocab_size))(
            xt.tokens[alignment.backpointers]))
    max_len = xtplus1.tokens.shape[0]
    number_of_preceding_deletes = (
        jax.vmap(lambda t: jax.nn.one_hot(t, max_len + 1))(
            alignment.number_of_deletions()))
    return ReverseProcessMarginalDistribution(
        was_missing, jnp.log(was_insert), jnp.log(previous_token_probs),
        jnp.log(number_of_preceding_deletes), xtplus1.length)

  @staticmethod
  def cross_entropy(
      samples_from,
      model_output,
      return_extra=False,
  ):
    """Computes cross entropy between model outputs and true sample distn."""
    (max_len,) = samples_from.was_insert_log_prob.shape
    valid_token_mask = ~samples_from.was_missing & (
        jnp.arange(max_len) < samples_from.length
    )
    valid_token_or_eos_mask = (
        jnp.pad(valid_token_mask, [(0, 1)],
                "constant").at[samples_from.length].set(True))

    delete_term = jnp.sum(
        jnp.where(
            valid_token_or_eos_mask[:, None],
            math_util.safe_exp_weighted(
                samples_from.log_prob_number_of_preceding_deletes,
                model_output.log_prob_number_of_preceding_deletes), 0.0))

    insert_term = jnp.sum(
        jnp.where(
            valid_token_mask,
            math_util.safe_exp_weighted(samples_from.was_insert_log_prob,
                                        model_output.was_insert_log_prob), 0.0))

    # import pdb
    # pdb.set_trace()
    token_term = jnp.sum(
        jnp.where(
            valid_token_mask[:, None],
            math_util.safe_exp_weighted(samples_from.previous_token_log_probs,
                                        model_output.previous_token_log_probs),
            0.0))
    total = (delete_term + insert_term + token_term)
    if return_extra:
      return total, {
          "delete_term": delete_term,
          "insert_term": insert_term,
          "token_term": token_term
      }
    return total

  def mean_across_batch(self):
    """Average across a batch dimension."""
    return ReverseProcessMarginalDistribution(
        was_missing=self.was_missing[0],
        was_insert_log_prob=math_util.logmeanexp(
            self.was_insert_log_prob, axis=0),
        previous_token_log_probs=math_util.logmeanexp(
            self.previous_token_log_probs, axis=0),
        log_prob_number_of_preceding_deletes=math_util.logmeanexp(
            self.log_prob_number_of_preceding_deletes, axis=0),
        length=self.length[0])

  def sample(self, rng):
    """Sample a sequence from the conditionally-indep. marginal distribution."""
    max_len, = self.was_insert_log_prob.shape
    was_missing_padded = jnp.pad(self.was_missing, [(0, 1)], "constant")
    # Sample each quantity.
    k1, k2, k3 = jax.random.split(rng, 3)
    del rng
    was_insert = jax.random.bernoulli(k1, jnp.exp(self.was_insert_log_prob))
    del k1
    prev_token_if_not_insert = jax.random.categorical(
        k2, self.previous_token_log_probs)
    del k2
    prev_token = jnp.where(was_insert,
                           DynamicLengthSentinelSequence.INSERT_SENTINEL,
                           prev_token_if_not_insert)
    deletes_before = jax.random.categorical(
        k3, self.log_prob_number_of_preceding_deletes)
    deletes_before = jnp.where(was_missing_padded, 0, deletes_before)
    del k3
    # Assemble.
    cumulative_deletes_before = jnp.cumsum(deletes_before)
    cumulative_copies_before = jnp.cumsum(~was_missing_padded) - 1
    cumulative_outputs_before = (
        cumulative_deletes_before + cumulative_copies_before)
    write_pointers = jnp.where(self.was_missing, max_len,
                               cumulative_outputs_before[:-1])
    dest_tokens = jnp.full([max_len],
                           DynamicLengthSentinelSequence.DELETE_SENTINEL)
    dest_tokens = dest_tokens.at[write_pointers].set(prev_token)
    new_length = cumulative_outputs_before[self.length]
    return DynamicLengthSentinelSequence(dest_tokens, new_length)


def intermediate_marginals(
    x0,
    xtplus1,
    alignment,
    d_0_to_t,
    d_t_to_tplus1,
    precomputed_x0_A0t = None,
    precomputed_Attplus1_xtplus1 = None,
):
  """Obtain marginals for each decision.

  Args:
    x0: Initial sequence.
    xtplus1: Final sequence.
    alignment: Alignment of x0 and xtplus1.
    d_0_to_t: Distribution of the first steps. (Note: t > 0 since we assume
      insert sentinels are handled.)
    d_t_to_tplus1: Distribution of the last step.
    precomputed_x0_A0t: Optional precomputed d_0_to_t.A.apply(x0.tokens)
    precomputed_Attplus1_xtplus1: Optional precomputed
      d_t_to_tplus1.A.observe(xtplus1.tokens)

  Returns:
    - Posterior marginals conditioned on x0 and xtplus1, for use as a target.
    - The quantity E_{q(x_t | x0, xtplus1, alignment)}[ log q(xtplus1 | xt) ]
  """
  (x0, xtplus1, alignment, d_0_to_t, d_t_to_tplus1) = jax.tree.map(
      jnp.array, (x0, xtplus1, alignment, d_0_to_t, d_t_to_tplus1))
  # d_0_to_tplus1 = d_0_to_t.then(d_t_to_tplus1)
  max_len, = xtplus1.tokens.shape

  if precomputed_x0_A0t is None:
    precomputed_x0_A0t = jax.vmap(
        functools.partial(d_0_to_t.A.apply, is_distn=False, log=True))(
            x0.tokens)

  if precomputed_Attplus1_xtplus1 is None:
    precomputed_Attplus1_xtplus1 = jax.vmap(
        functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True))(
            xtplus1.tokens)

  # Algorithm sketch:
  # Precompute:
  # - the marginal distribution of how many delete sentinels you see at
  #   time 1, conditioned on having k deleted things before time 2.
  # - the marginal distribution of how many delete sentinels you see in a single
  #   emphemeral gap.
  # Then, walk through xtplus1, and compute marginals over each insert and
  # copy decision, allocating delete sentinels as needed.

  # Deletions: Sentinel deletions add nothing, so we can strip them out first.
  # Silent deletions may or may not add insert sentinels depending on whether
  # they were silent before step 2 or whether they added things. However, since
  # these are all independent, we can simply precompute the marginal
  # distribution over how many DEL sentinels we see given the number of deleted
  # things in x0.
  (unshifted_binomial_mixture,
   shifted_binomial_mixture) = d_0_to_t.delete_count_marginals(max_len)

  # Insertions: Sentinel insertions can be safely stripped out first. Insertions
  # at earlier steps have some chance of turning into an INS sentinel and some
  # chance of turning into something else. In either case, there may be some
  # additional number of DEL sentinels. Note, however, that deletions never
  # occur before insertions, according to the FSM (since deletion followed by
  # insertion is represented as a rerolling copy.)

  # Copies and rerolls: The most interesting case. In some situations, we
  # simply take the sentinels from preceeding deletions. In others, we insert
  # additional ephemeral sentinels because of a reroll. The number of deletions
  # we observe in the previous step is actually correlated with the token
  # value, but here we just are concerned with the marginal distribution.

  # First mark aligned pairs in x0 and xt:
  # A "nontrivial" deletion is one that occured AFTER the first step.
  # A "nontrivial" insertion is one that` occured BEFORE time t+1.
  x0_matched = x0.token_mask() & ~alignment.delete_mask()
  x0_sentinel_deletion = x0.delete_sentinel_mask()
  x0_nontrivial_deletion = alignment.delete_mask() & ~x0.delete_sentinel_mask()
  xtplus1_matched = xtplus1.token_mask() & ~alignment.insert_mask()
  xtplus1_sentinel_insertion = xtplus1.insert_sentinel_mask()
  xtplus1_nontrivial_insertion = (
      alignment.insert_mask() & ~xtplus1.insert_sentinel_mask())

  # Pad by 1 and imagine we've aligned an EOS token. This makes some of our
  # later manipulation easier.
  padzero_end = lambda x: jnp.pad(x, [(0, 1)], "constant")
  x0_matched = padzero_end(x0_matched).at[x0.length].set(True)
  x0_sentinel_deletion = padzero_end(x0_sentinel_deletion)
  x0_nontrivial_deletion = padzero_end(x0_nontrivial_deletion)
  xtplus1_matched = padzero_end(xtplus1_matched).at[xtplus1.length].set(True)
  xtplus1_sentinel_insertion = padzero_end(xtplus1_sentinel_insertion)
  xtplus1_nontrivial_insertion = padzero_end(xtplus1_nontrivial_insertion)
  backptrs = padzero_end(alignment.backpointers).at[xtplus1.length].set(
      x0.length)

  # Count how many insertions/deletions have occured at each point.
  x0_deletion_cumulative_count = jnp.cumsum(x0_nontrivial_deletion)
  # xtplus1_insertion_cumulative_count = jnp.cumsum(
  #     xtplus1_nontrivial_insertion)
  xtplus1_backptr_deletion_count = (x0_deletion_cumulative_count[backptrs])

  # At each matched position, compute the number of inserts and deletes
  # since the last matched position.
  xtplus1_prev_cumulative_deletes_at_match = util.previous_active_value(
      xtplus1_backptr_deletion_count, xtplus1_matched, default=0)
  xtplus1_deletes_immediately_before = (
      xtplus1_backptr_deletion_count - xtplus1_prev_cumulative_deletes_at_match)

  def handle_silent_insert(xtplus1_index):
    xtplus1_token = xtplus1.tokens[xtplus1_index]
    # Figure out what this token was at time t-1.
    insert_info = _get_intermediate_insert_logits(
        xtplus1_token, d_0_to_t, d_t_to_tplus1,
        precomputed_Attplus1_xtplus1[xtplus1_index])

    # Ignore case 1; this is a non-silent insert and will be handled elsewhere.
    # Case 2: We didn't insert at time 2. At time 1, we inserted an unknown
    # geometric number of delete sentinels, and then inserted some other token.
    # We decided to transform it into what we observed (which is not an INS
    # sentinel).
    was_insert = math_util.safe_sub_or_ninf(
        insert_info.lp_was_ins_sentinel_at_step_1, insert_info.lp_case_2)
    was_token = math_util.safe_sub_or_ninf(
        insert_info.lp_was_token_at_step_1, insert_info.lp_case_2
    )
    expected_forward_lp_of_token = math_util.safe_exp_weighted(
        was_insert, d_t_to_tplus1.D_insert_logits[xtplus1_token]
    ) + jnp.sum(
        math_util.safe_exp_weighted(
            was_token,
            d_t_to_tplus1.A.observe(xtplus1_token, is_distn=False, log=True),
        )
    )
    expected_forward_lp = d_t_to_tplus1.lp_no_insert + jnp.where(
        xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
        d_t_to_tplus1.lp_delete,
        d_t_to_tplus1.lp_no_delete + expected_forward_lp_of_token,
    )
    return {
        "was_insert": was_insert,
        "was_token": was_token,
        "delete_sentinels": distributions.geometric_log_pdf(
            d_0_to_t.lp_silent_insert_to_delete, max_len + 1
        ).log_probs,
        "expected_forward_logprob": expected_forward_lp,
        "debug_expected_1": jnp.where(
            xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
            1.0,
            (
                math_util.safe_exp_weighted(was_insert, 1.0)
                + jnp.sum(math_util.safe_exp_weighted(was_token, 1.0))
            ),
        ),
    }

  def handle_copy(xtplus1_index):
    x0_index = alignment.backpointers[xtplus1_index]
    x0_token = x0.tokens[x0_index]
    is_eos = (xtplus1_index == xtplus1.length)
    xtplus1_token = xtplus1.tokens[xtplus1_index]
    num_prev_x0_deletions = xtplus1_deletes_immediately_before[xtplus1_index]
    if alignment.reroll_mask is None:
      known_reroll = None
    else:
      known_reroll = alignment.reroll_mask[xtplus1_index]

    copy_info = _get_intermediate_copy_logits(
        x0_token, xtplus1_token, known_reroll, d_0_to_t, d_t_to_tplus1,
        precomputed_x0_A0t[x0_index],
        precomputed_Attplus1_xtplus1[xtplus1_index])

    # Combine our cases.
    was_insert = copy_info.lp_was_ins_sentinel_at_step_1
    was_token = math_util.safe_logaddexp(
        copy_info.case_1_token_logits, copy_info.case_2_lp_was_token_at_step_1)
    lp_total = math_util.safe_logaddexp(copy_info.lp_case_1,
                                        copy_info.lp_case_2)
    (was_insert, was_token, lp_case_1_normalized,
     lp_case_2_normalized) = jax.tree.map(
         lambda v: math_util.safe_sub_or_ninf(v, lp_total),
         (was_insert, was_token, copy_info.lp_case_1, copy_info.lp_case_2))

    # If this is the post-processing EOS "token", we need to adjust this, since
    # the EOS token is deterministically "copied".
    (was_insert, was_token, lp_case_1_normalized,
     lp_case_2_normalized) = util.tree_where(
         is_eos,
         (BAD_VALUE, BAD_VALUE, 0.0, -np.inf),
         (was_insert, was_token, lp_case_1_normalized, lp_case_2_normalized),
     )

    # Tricky bit: how many deletions did we see before this?
    # If we hit case 1, every preceding deletion had some chance of adding one
    # plus a geometric number of deletion sentinels. In particular, we take a
    # binomial mixture of negative binomials. If no deletion appeared at time t,
    # we also have a single geometric due to the decision to stop inserting.
    # If we hit case 2, there was also an additional one plus geometric number
    # added by the reroll; we can obtain this by shifting by one before applying
    # our mixture.
    delete_sentinels = math_util.safe_logaddexp(
        (
            lp_case_1_normalized
            + unshifted_binomial_mixture.log_probs[num_prev_x0_deletions]
        ),
        (
            lp_case_2_normalized
            + shifted_binomial_mixture.log_probs[num_prev_x0_deletions]
        ),
    )
    expected_forward_lp_of_token = math_util.safe_exp_weighted(
        was_insert, d_t_to_tplus1.D_insert_logits[xtplus1_token]
    ) + jnp.sum(
        math_util.safe_exp_weighted(
            was_token,
            d_t_to_tplus1.A.observe(xtplus1_token, is_distn=False, log=True),
        )
    )
    expected_forward_lp = d_t_to_tplus1.lp_no_insert + jnp.where(
        xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
        d_t_to_tplus1.lp_delete,
        d_t_to_tplus1.lp_no_delete + expected_forward_lp_of_token,
    )
    expected_forward_lp = jnp.where(
        is_eos, d_t_to_tplus1.lp_no_insert, expected_forward_lp
    )

    return {
        "was_insert": was_insert,
        "was_token": was_token,
        "delete_sentinels": delete_sentinels,
        "expected_forward_logprob": expected_forward_lp,
        "debug_expected_1": jnp.where(
            is_eos
            | (xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL),
            1.0,
            (
                math_util.safe_exp_weighted(was_insert, 1.0)
                + jnp.sum(math_util.safe_exp_weighted(was_token, 1.0))
            ),
        ),
    }

  # Compute what we need for each token in xtplus1.
  stuff_for_copy = jax.vmap(handle_copy)(
      jnp.arange(xtplus1.tokens.shape[0] + 1))
  stuff_for_trivial_insert = {
      "was_insert": BAD_VALUE,
      "was_token": BAD_VALUE,
      "delete_sentinels": BAD_VALUE,
      "expected_forward_logprob": d_t_to_tplus1.lp_insert,
      "debug_expected_1": 1.0,
  }
  stuff_for_insert = jax.vmap(handle_silent_insert)(
      jnp.arange(xtplus1.tokens.shape[0] + 1))
  stuff_for_out_of_bounds = {
      "was_insert": BAD_VALUE,
      "was_token": BAD_VALUE,
      "delete_sentinels": BAD_VALUE,
      "expected_forward_logprob": BAD_VALUE,
      "debug_expected_1": BAD_VALUE,
  }
  combiner_fn = util.vmap_with_kwargs(
      util.tree_select, condlist_axes=[0, 0, 0], choicelist_axes=[0, None, 0])
  stuff_combined = combiner_fn(
      condlist=[
          xtplus1_matched, xtplus1_sentinel_insertion,
          xtplus1_nontrivial_insertion
      ],
      choicelist=[stuff_for_copy, stuff_for_trivial_insert, stuff_for_insert],
      default=stuff_for_out_of_bounds)

  expected_forward_logprob_total = jnp.sum(
      jnp.where(
          jnp.arange(max_len + 1) <= xtplus1.length,
          stuff_combined["expected_forward_logprob"], 0.0))

  return (ReverseProcessMarginalDistribution(
      was_missing=xtplus1.insert_sentinel_mask(),
      was_insert_log_prob=stuff_combined["was_insert"][:-1],
      previous_token_log_probs=stuff_combined["was_token"][:-1],
      log_prob_number_of_preceding_deletes=stuff_combined["delete_sentinels"],
      length=xtplus1.length), expected_forward_logprob_total)


def expected_forward_logprob(
    xt_marginals,
    xtplus1,
    d_t_to_tplus1,
    precomputed_Attplus1_xtplus1 = None,
):
  """Compute expected logprob over marginals."""
  (xt_marginals, xtplus1,
   d_t_to_tplus1) = jax.tree.map(jnp.array,
                                 (xt_marginals, xtplus1, d_t_to_tplus1))
  # Log probs of new insert sentinels.
  insertion_mask = xtplus1.insert_sentinel_mask()
  insert_count = jnp.count_nonzero(insertion_mask)
  copy_count = xtplus1.length - insert_count
  insert_sentinel_log_probs = (
      insert_count * d_t_to_tplus1.lp_insert +
      (copy_count + 1) * d_t_to_tplus1.lp_no_insert)

  if precomputed_Attplus1_xtplus1 is None:
    precomputed_Attplus1_xtplus1 = jax.vmap(
        functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True))(
            xtplus1.tokens)

  # Log probs of each of the aligned pairs of tokens (ignoring DEL in xt and
  # INS in xtplus1)
  def compute_token_logprob(i):
    xtplus1_token = xtplus1.tokens[i]
    cases = []
    # Out of bounds or INS: ignore.
    cases.append({"if": insertion_mask[i] | i >= xtplus1.length, "then": 0.0})
    # If this was deleted, add deletion logprob.
    cases.append({
        "if": xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
        "then": d_t_to_tplus1.lp_delete
    })
    # Otherwise, it wasn't deleted. If it used to be an insert sentinel, we
    # drew this token from our insert distn.
    # Finally, if it was an ordinary token, we must have perturbed.
    prob_was_ins = jnp.exp(xt_marginals.was_insert_log_prob[i])
    val_for_insert = (
        d_t_to_tplus1.lp_no_delete +
        d_t_to_tplus1.D_insert_logits[xtplus1_token])
    prob_was_token_if_not_ins = jax.nn.softmax(
        xt_marginals.previous_token_log_probs[i])
    val_for_token = (
        d_t_to_tplus1.lp_no_delete + precomputed_Attplus1_xtplus1[i])
    default = (
        jnp.where(prob_was_ins == 0, 0, prob_was_ins * val_for_insert) +
        (1 - prob_was_ins) * jnp.sum(
            jnp.where(prob_was_token_if_not_ins == 0, 0,
                      prob_was_token_if_not_ins * val_for_token)))
    return jnp.select(
        condlist=[case["if"] for case in cases],
        choicelist=[case["then"] for case in cases],
        default=default)

  values = jax.vmap(compute_token_logprob)(jnp.arange(xtplus1.tokens.shape[0]))
  aligned_logprobs = jnp.sum(values)
  return insert_sentinel_log_probs + aligned_logprobs


def apply_x0_parameterization(
    approximate_x0_guess,
    xtplus1,
    d_0_to_t,
    d_t_to_tplus1,
    precomputed_Attplus1_xtplus1 = None
):
  r"""Computes x0 parameterization for insert/delete.

  We can choose to parameterize a diffusion model as

    p_theta(x_t | x_{t+1}) =
      \sum_{x_0} f(x_0 | x_{t+1}) q(x_{t+1} | x_0, x_{t+1}) / Z

  which applies weights based on the forward process. This is a bit awkward
  for an insert/delete model, but we can still do it, by assuming the model
  is predicting both the tokens of x_0 and a reroll-compatible alignment of
  it with x_{t+1}. The weird thing is that the model will likely learn to
  align tokens in x_{t+1} with tokens in x_0 that are totally unrelated,
  simply because in our forward process marginalization, we re-align inserts
  and deletes to make inference tractable.

  Despite this being maybe a bad idea, it's probably still worth trying for
  consistency and thoroughness.

  Args:
    approximate_x0_guess: A guess at marginals for x0 and their alignment with
      x_{t+1}, taking the form of f(x_0 | x_{t+1}). Note that `f` does NOT
      represent the model's "real" posterior  p_theta(x_0 | x_{t+1}), and it
      will likely be significantly different from this.
    xtplus1: Value to condition on.
    d_0_to_t: Distribution from 0 to t.
    d_t_to_tplus1: Distribution from t to t+1.
    precomputed_Attplus1_xtplus1: Optional precomputed
      d_t_to_tplus1.A.observe(xtplus1.tokens)

  Returns:
    Adjusted marginals, so that they now represent p_theta(x_t | x_{t+1})
  """
  (approximate_x0_guess, xtplus1, d_0_to_t, d_t_to_tplus1) = jax.tree.map(
      jnp.array, (approximate_x0_guess, xtplus1, d_0_to_t, d_t_to_tplus1))
  max_len, = xtplus1.tokens.shape

  if precomputed_Attplus1_xtplus1 is None:
    precomputed_Attplus1_xtplus1 = jax.vmap(
        functools.partial(d_t_to_tplus1.A.observe, is_distn=False, log=True))(
            xtplus1.tokens)

  # Sketch:
  # If model predicts something is part of x0:
  # - Token comes from the standard A1.apply + A2.observe
  # - Number of inserts is based on taking its number of inserts prediction
  #   and doing a non-rerolling binomial
  # If model predicts something is NOT part of x0:
  # - Token comes from the silent insert distribution
  # - Number of inserts is based on taking its number of inserts prediction
  #   and doing a non-rerolling binomial, PLUS shifting it by one geometric
  #   to account for the things inserted before this token.
  # Why does this make sense? Well, if model says not part of x0, then it's
  # either a reroll or a non-reroll insert. Non-reroll inserts are just
  # inserts that don't have a deletion before them, and the difference in the
  # number of DEL sentinels that get added in the two cases is just how many
  # things in x0 were deleted before we got there.

  distn_deletes_at_t_from_deletes_at_x0, _ = d_0_to_t.delete_count_marginals(
      max_len)

  def process_token(insert_logprob, x0_token_distn, xtplus1_token,
                    precomputed_Attplus1_xtplus1_logits):
    # not_insert_logprob = math_util.safe_logsumexp(x0_token_distn, axis=-1)
    # How likely to see xtplus1_token from each other token?
    observe_from_token_logits = jnp.where(
        xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
        d_t_to_tplus1.lp_delete, precomputed_Attplus1_xtplus1_logits)
    # How likely to see xtplus1_token from an insert at time t?
    observe_from_insert_logit = jnp.where(
        xtplus1_token == DynamicLengthSentinelSequence.DELETE_SENTINEL,
        d_t_to_tplus1.lp_delete, d_t_to_tplus1.D_insert_logits[xtplus1_token])
    # Given that there was an insert, when did it happen?
    total_insert_lp = math_util.safe_logaddexp(d_0_to_t.lp_sentinel_insert,
                                               d_0_to_t.lp_silent_insert)
    lp_insert_is_sentinel = math_util.safe_sub_or_ninf(
        d_0_to_t.lp_sentinel_insert, total_insert_lp)
    lp_insert_is_silent = math_util.safe_sub_or_ninf(d_0_to_t.lp_silent_insert,
                                                     total_insert_lp)

    # Suppose this was a token from x0, according to our guess.
    xt_token_logits_from_keep = (
        d_0_to_t.A.apply(x0_token_distn, is_distn=True, log=True) +
        observe_from_token_logits)

    # Suppose this was an insert, and suppose it happened at time t.
    xt_insert_logit_from_insert = (
        insert_logprob + lp_insert_is_sentinel + observe_from_insert_logit)

    # Suppose this was an insert, but it happened before time t.
    xt_token_logits_from_insert = (
        insert_logprob + lp_insert_is_silent + d_0_to_t.D_silent_insert_logits +
        observe_from_token_logits)

    # Combine our cases.
    xt_token_logits = math_util.safe_logaddexp(xt_token_logits_from_keep,
                                               xt_token_logits_from_insert)
    xt_insert_logit = xt_insert_logit_from_insert
    denominator = math_util.safe_logaddexp(
        xt_insert_logit, math_util.safe_logsumexp(xt_token_logits))
    xt_token_logits_normalized = xt_token_logits - denominator
    xt_insert_logit_normalized = xt_insert_logit - denominator

    return (xt_token_logits_normalized, xt_insert_logit_normalized)

  token_logits, insert_logit = jax.vmap(process_token)(
      approximate_x0_guess.was_insert_log_prob,
      approximate_x0_guess.previous_token_log_probs, xtplus1.tokens,
      precomputed_Attplus1_xtplus1)

  # In any of these situations, there's always at least one chance to add
  # geometric-r.v.s of ephemeral insert-to-delete sentinels. If we think there
  # were more tokens between this and the previous in x0, we may have more.
  # We let the model guess how many were in x0, and compute how many we'd
  # see here as a function of that. It's likely correlated with whether or
  # not this was an insert, and where it was, but due to conditional
  # independence assumptions we ignore those interactions.
  #
  # Note: We're folding rerolling into a special case of "having deletions
  # before an insert".
  def adjust_deletes(num_deletes_before):
    # import pdb
    # pdb.set_trace()
    return distributions.RandomVariablePDF(num_deletes_before).mixture_of(
        distn_deletes_at_t_from_deletes_at_x0).log_probs

  deletes_adjusted = jax.vmap(adjust_deletes)(
      approximate_x0_guess.log_prob_number_of_preceding_deletes)

  return ReverseProcessMarginalDistribution(
      was_missing=approximate_x0_guess.was_missing,
      was_insert_log_prob=insert_logit,
      previous_token_log_probs=token_logits,
      log_prob_number_of_preceding_deletes=deletes_adjusted,
      length=approximate_x0_guess.length)


def terminal_sentinel_distribution(x0,
                                   d_0_to_T,
                                   max_len = None):
  """Compute log q(x_T | x_0) assuming the distribution only deletes."""
  if max_len is None:
    max_len = x0.tokens.shape[0]

  num_actual_tokens = jnp.count_nonzero(x0.token_mask()
                                        & ~x0.delete_sentinel_mask())

  is_valid_terminal_distribution = functools.reduce(operator.or_, [
      d_0_to_T.lp_sentinel_insert == -np.inf,
      d_0_to_T.lp_silent_insert == -np.inf,
      d_0_to_T.lp_silent_insert_to_delete == -np.inf,
      d_0_to_T.lp_reroll == -np.inf,
      math_util.safe_logaddexp(d_0_to_T.lp_sentinel_delete,
                               d_0_to_T.lp_silent_delete) > jnp.log(.999),
  ])

  # Number of deletions we would see due to some number of tokens in x0 that
  # were deleted at time T (including new deletions of inserted tokens between
  # 0 and T).
  between_two_undeleted, _ = d_0_to_T.delete_count_marginals(max_len)
  total_deletes = between_two_undeleted.log_probs[num_actual_tokens]

  return jnp.where(is_valid_terminal_distribution, total_deletes, BAD_VALUE)


def compute_in_bounds_log_probs(xt,
                                d_t_to_tplus1,
                                max_len = None):
  """Compute log probability of sampling something at most `max_len` long.

  Directly computes the quantity by considering it as the CDF of a binomial
  mixture of negative binomial distributions, and using the CDF of a negative
  binomial distribution.

  Args:
    xt: Sequence to start with,
    d_t_to_tplus1: Single-step distribution we are sampling from.
    max_len: Maximum length we don't want to exceed.

  Returns:
    Log probability of generating something shorter than or equal to max_len.
  """
  if max_len is None:
    max_len = xt.tokens.shape[0]

  num_kept_tokens = jnp.count_nonzero(xt.token_mask()
                                      & ~xt.delete_sentinel_mask())

  # We generate a negative binomial number of INS tokens in addition to the
  # tokens we keep.
  # In this case, it's the CDF of the quantity `k + NB(k + 1, p_ins)`
  # evaluated at max_len, which is the CDF of `NB(k + 1, p_ins)` evaluated
  # at `max_len - k`
  result = jnp.log1p(-jax.scipy.special.betainc(
      max_len - num_kept_tokens + 1,
      num_kept_tokens + 1,
      jnp.exp(d_t_to_tplus1.lp_insert),
  ))
  return result
