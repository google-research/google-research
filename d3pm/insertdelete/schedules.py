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

"""Insert-delete sentinel diffusion schedule."""

import dataclasses
from typing import Any, Callable, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from d3pm.insertdelete import forward_process
from d3pm.insertdelete import transition_operator

NDArray = Any


@dataclasses.dataclass
class SentinelInsertDeleteSchedule:
  """Schedule of stacked insert-delete distributions.

  Attributes:
    steps: OneStepDistn with an extra batch dimension, such that steps[t] gives
      the operator that will apply to x_t to produce x_{t+1}
    cumulative: ManyStepDistn with an extra batch dimension, such that
      cumulative[t] gives the operator that will apply to x_0 to produce x_t.
    weights: Weights of each step for sampling, to allow choosing some steps
      more often than others. weights[t] gives the probability of computing
      scores based on the posterior p(x_t | x_{t+1}).
    num_steps: (Inferred) Number of steps in the schedule.
  """
  steps: forward_process.OneStepDistn
  cumulative: forward_process.ManyStepDistn
  weights: NDArray

  def sample_step_number(self, rng):
    """Sample a timestep proportional to the weights."""
    return jax.random.categorical(rng, jnp.log(self.weights))

  def distns_at_step(self, step_number):
    """Extract distributions for p(x_{step_number}, x_{step_number+1} | x_0).

    Args:
      step_number: Step number to use.

    Returns:
      Tuple of distributions (d_start_to_here, d_here_to_next).
    """
    d_start_to_here = jax.tree.map(lambda x: x[step_number], self.cumulative)
    d_here_to_next = jax.tree.map(lambda x: x[step_number], self.steps)
    return d_start_to_here, d_here_to_next

  def sample_step_distns(self, rng):
    """Sample a step number and the relevant joint distributions."""
    step_number = self.sample_step_number(rng)
    d_start_to_here, d_here_to_next = self.distns_at_step(step_number)
    t_weight = self.weights[step_number]
    return step_number, t_weight, d_start_to_here, d_here_to_next

  @property
  def num_steps(self):
    return self.weights.shape[0]

  def terminal_distn(self):
    return jax.tree.map(lambda x: x[-1], self.cumulative)


def build_schedule(
    steps,
    weights = None,
    add_deletion_step = True):
  """Build a schedule given a set of per-step distributions.

  Args:
    steps: List of step distributions of the same type. Transition matrices must
      have a left fold identity defined, and must be mutually composable.
    weights: Weights to use when sampling a step.
    add_deletion_step: Whether to add an extra step at the end that deletes the
      entire sequence.

  Returns:
    Stacked schedule suitable for training.
  """
  if add_deletion_step:
    deletion_step = dataclasses.replace(
        steps[-1], lp_delete=0.0, lp_insert=-np.inf)
    steps = steps + [deletion_step]

  accumulator = forward_process.ManyStepDistn.identity_for(steps[0])

  cumulative = [accumulator]
  for step in steps:
    accumulator = accumulator.then(step)
    cumulative.append(accumulator)

  stacker = lambda *args: jnp.stack(list(args))
  stacked_steps = jax.tree.map(stacker, *steps)
  stacked_cumulative = jax.tree.map(stacker, *cumulative)

  if weights is None:
    stacked_weights = jnp.full((len(steps),), 1. / len(steps))
  else:
    stacked_weights = jnp.array(list(weights))

  return SentinelInsertDeleteSchedule(stacked_steps, stacked_cumulative,
                                      stacked_weights)


def build_uniform_insert_delete_schedule(
    num_steps,
    vocab_size,
    insert_prob,
    token_denylist = ()
):
  """Build a uniform schedule over a vocab.

  The returned schedule is as uniform as possible in a few different ways:
  - Inserts are chosen uniformly from all tokens.
  - Tokens are equally likely to be inserted at any step, except for the last
    step, which doesn't insert anything to ensure we map to the empty sequence.
  - For each token, it is guaranteed to be deleted before the end of the
    sequence, but the time at which it is deleted is sampled uniformly over all
    timesteps.

  Args:
    num_steps: Number of noising steps.
    vocab_size: Size of the token vocabulary.
    insert_prob: Insertion probability for the first step.
    token_denylist: Tokens that should never be inserted.

  Returns:
    A uniform schedule.
  """
  d_insert = np.ones((vocab_size,))
  d_insert[list(token_denylist)] = 0.0
  d_insert = jax.nn.log_softmax(d_insert)
  steps = []
  for t in range(num_steps - 1):
    steps.append(
        forward_process.OneStepDistn(
            lp_insert=jnp.log(insert_prob),
            lp_delete=1.0 / (num_steps - t),
            A=transition_operator.IdentityOperator(vocab_size),
            D_insert_logits=d_insert,
        )
    )
  return build_schedule(steps, add_deletion_step=True)


def schedule_from_interpolators(
    num_steps,
    transition_distn_fn,
    insertion_distn_fn,
    relative_size_fn,
    refresh_prob_fn,
    weight_fn = None,
    interpolator = None,
):
  """Build a schedule that interpolates between key points.

  Args:
    num_steps: How many steps of diffusion to run. Shouldn't affect the overall
      shape of the diffusion, so if you double it, it's like inserting one new
      step in between each existing pair.
    transition_distn_fn: Given a time interval [t1, t2], return an operator that
      takes values from t1 to t2. Should obey the property that
      transition_distn_fn(a, b).then(transition_distn_fn(b, c)) ==
      transition_distn_fn(a, c)
    insertion_distn_fn: Given a time point, return the logits for inserts that
      happen at that time.
    relative_size_fn: Given a time point, return the approximate desired ratio
      of the current length to the original length. Variance will depend on
      refresh_prob_fn. Must be nonnegative.
    refresh_prob_fn: Given a time point, return a probability of deleting a
      token that didn't otherwise need to be deleted in order to maintain the
      desired relative size. Must be increasing and be bounded between 0 and 1.
    weight_fn: Optional weight fn, giving a weight for samples at each given
      time.
    interpolator: Optional function that maps times between [0, 1] to times that
      the other functions are expecting.

  Returns:
    A schedule that jointly interpolates according to the provided functions,
    then deletes everything when it reaches t=1.
  """
  endpoints = np.linspace(0., 1., num_steps, endpoint=True)
  if interpolator:
    endpoints = jax.vmap(interpolator)(endpoints)
  steps = []
  weights = []
  for prev, current in zip(endpoints[:-1], endpoints[1:]):
    center = (prev + current) / 2
    # log(length(t)) ~= log(length(t-1)) + log(new tokens per old token)
    # log(new tokens per old token) ~= log(length(t)) - log(length(t-1))
    #                 ~= log(length(t)/length(0)) - log(length(t-1)/length(0))
    # new tokens per old token ~=
    #   (1 - p_delete) new tokens per old token due to deletions
    #   + p_insert / (1 - p_insert) new tokens per old token due to insertions
    #
    # First, find the smallest p_delete that allows us to maintain the right
    # insertion rate:
    #  1 - p_delete_min = new tokens per old token
    #  p_delete_min = 1 - new tokens per old token
    log_current_relative_size = jnp.log(relative_size_fn(current))
    log_prev_relative_size = jnp.log(relative_size_fn(prev))
    log_expected_new_tokens = log_current_relative_size - log_prev_relative_size
    min_p_delete = jnp.maximum(0, -jnp.expm1(log_expected_new_tokens))
    # Now, figure out how many extra deletes we want.
    # del_before(t+1) = del_before(t) + (1 - del_before(t)) del_at(t+1)
    # => del_at(t+1) = (del_before(t+1) - del_before(t))/(1 - del_before(t))
    lp_extra_delete = (
        jnp.log(refresh_prob_fn(current) - refresh_prob_fn(prev)) -
        jnp.log1p(-refresh_prob_fn(prev)))
    # Total deletion probability:
    lp_delete = jnp.logaddexp(
        jnp.log(min_p_delete),
        jnp.log1p(-min_p_delete) + lp_extra_delete)
    # Now figure out our insertion probability to bring us back to our target.
    # p_insert / (1 - p_insert) = (new tokens per old token) - (1 - p_delete)
    insert_odds = jnp.expm1(log_expected_new_tokens) + jnp.exp(lp_delete)
    lp_insert = jnp.where(insert_odds <= 0, -jnp.inf,
                          jax.nn.log_sigmoid(jnp.log(insert_odds)))

    steps.append(
        forward_process.OneStepDistn(
            lp_insert=lp_insert,
            lp_delete=lp_delete,
            A=transition_distn_fn(prev, current),
            D_insert_logits=insertion_distn_fn(center)))
    if weight_fn:
      weights.append(weight_fn(center))

  return build_schedule(
      steps, weights if weight_fn else None, add_deletion_step=True)
