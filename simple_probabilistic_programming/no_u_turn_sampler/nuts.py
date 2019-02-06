# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""No U-Turn Sampler via an Eager-only single-chain implementation.

The implementation uses minimal abstractions and data structures: it applies
Python callables, lists, and Tensors. It closely follows [1; Algorithm 3] in
that there exists a "build tree" function that recursively builds the No-U-Turn
Sampler trajectory. The path length is set adaptively; the step size is fixed.

Future work may abstract this code as part of a Markov chain Monte Carlo
library.

#### References

[1]: Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively
     Setting Path Lengths in Hamiltonian Monte Carlo.
     In _Journal of Machine Learning Research_, 15(1):1593-1623, 2014.
     http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfe = tf.contrib.eager

__all__ = [
    "kernel",
]


def kernel(target_log_prob_fn,
           current_state,
           step_size,
           seed=None,
           current_target_log_prob=None,
           current_grads_target_log_prob=None,
           name=None):
  """Simulates a No-U-Turn Sampler (NUTS) trajectory.

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `*current_state` and returns its (possibly unnormalized) log-density under
      the target distribution.
    current_state: List of `Tensor`s representing the states to simulate from.
    step_size: List of `Tensor`s representing the step sizes for the leapfrog
      integrator. Must have same shape as `current_state`.
    seed: Integer to seed the random number generator.
    current_target_log_prob: Scalar `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`.
    current_grads_target_log_prob: List of `Tensor`s representing gradient of
      `current_target_log_prob` with respect to `current_state`. Must have same
      shape as `current_state`.
    name: A name for the operation.

  Returns:
    next_state: List of `Tensor`s representing the next states of the NUTS
      trajectory. Has same shape as `current_state`.
    next_target_log_prob: Scalar `Tensor` representing the value of
      `target_log_prob_fn` at `next_state`.
    next_grads_target_log_prob: List of `Tensor`s representing the gradient of
      `next_target_log_prob` with respect to `next_state`.

  Raises:
    NotImplementedError: If the execution mode is not eager.
  """
  if not tf.executing_eagerly():
    raise NotImplementedError("`kernel` is only available in Eager mode.")

  with tf.name_scope(name,
                     default_name="nuts_kernel",
                     values=[current_state, step_size, seed,
                             current_target_log_prob,
                             current_grads_target_log_prob]):
    with tf.name_scope("initialize"):
      current_state = [tf.convert_to_tensor(s) for s in current_state]
      step_size = [tf.convert_to_tensor(s) for s in step_size]
      value_and_gradients_fn = tfe.value_and_gradients_function(
          target_log_prob_fn)
      value_and_gradients_fn = _embed_no_none_gradient_check(
          value_and_gradients_fn)
      if (current_target_log_prob is None or
          current_grads_target_log_prob is None):
        (current_target_log_prob,
         current_grads_target_log_prob) = value_and_gradients_fn(*current_state)

      seed_stream = tfd.SeedStream(seed, "nuts_kernel")
      current_momentum = []
      for state_tensor in current_state:
        momentum_tensor = tf.random_normal(shape=tf.shape(state_tensor),
                                           dtype=state_tensor.dtype,
                                           seed=seed_stream())
        current_momentum.append(momentum_tensor)

      # Draw a slice variable u ~ Uniform(0, p(initial state, initial
      # momentum)) and compute log u. For numerical stability, we perform this
      # in log space where log u = log (u' * p(...)) = log u' + log
      # p(...) and u' ~ Uniform(0, 1).
      log_slice_sample = tf.log(tf.random_uniform([], seed=seed_stream()))
      log_slice_sample += _log_joint(current_target_log_prob,
                                     current_momentum)

      # Initialize loop variables. It comprises a collection of information
      # about a "reverse" state, a collection of information about a "forward"
      # state, a collection of information about the next state,
      # the trajectory's tree depth, the number of candidate states, and
      # whether to continue the trajectory.
      reverse_state = current_state
      reverse_target_log_prob = current_target_log_prob
      reverse_grads_target_log_prob = current_grads_target_log_prob
      reverse_momentum = current_momentum
      forward_state = current_state
      forward_target_log_prob = current_target_log_prob
      forward_grads_target_log_prob = current_grads_target_log_prob
      forward_momentum = current_momentum
      next_state = current_state
      next_target_log_prob = current_target_log_prob
      next_grads_target_log_prob = current_grads_target_log_prob
      depth = 0
      num_states = 1
      continue_trajectory = True

    while continue_trajectory:
      # Grow the No-U-Turn Sampler trajectory by choosing a random direction and
      # simulating Hamiltonian dynamics in that direction. This extends either
      # the forward or reverse state.
      direction = tfp.math.random_rademacher([], seed=seed_stream())
      if direction < 0:
        [
            reverse_state,
            reverse_target_log_prob,
            reverse_grads_target_log_prob,
            reverse_momentum,
            _,
            _,
            _,
            _,
            next_state_in_subtree,
            next_target_log_prob_in_subtree,
            next_grads_target_log_prob_in_subtree,
            num_states_in_subtree,
            continue_trajectory,
        ] = _build_tree(
            value_and_gradients_fn=value_and_gradients_fn,
            current_state=reverse_state,
            current_target_log_prob=reverse_target_log_prob,
            current_grads_target_log_prob=reverse_grads_target_log_prob,
            current_momentum=reverse_momentum,
            direction=direction,
            depth=depth,
            step_size=step_size,
            log_slice_sample=log_slice_sample,
            seed=seed_stream())
      else:
        [
            _,
            _,
            _,
            _,
            forward_state,
            forward_target_log_prob,
            forward_grads_target_log_prob,
            forward_momentum,
            next_state_in_subtree,
            next_target_log_prob_in_subtree,
            next_grads_target_log_prob_in_subtree,
            num_states_in_subtree,
            continue_trajectory,
        ] = _build_tree(
            value_and_gradients_fn=value_and_gradients_fn,
            current_state=forward_state,
            current_target_log_prob=forward_target_log_prob,
            current_grads_target_log_prob=forward_grads_target_log_prob,
            current_momentum=forward_momentum,
            direction=direction,
            depth=depth,
            step_size=step_size,
            log_slice_sample=log_slice_sample,
            seed=seed_stream())

      if continue_trajectory:
        # If the built tree did not terminate, accept the tree's next state
        # with a certain probability.
        accept_state_in_subtree = _random_bernoulli(
            [],
            probs=tf.minimum(1., num_states_in_subtree / num_states),
            dtype=tf.bool,
            seed=seed_stream())
        if accept_state_in_subtree:
          next_state = next_state_in_subtree
          next_target_log_prob = next_target_log_prob_in_subtree
          next_grads_target_log_prob = next_grads_target_log_prob_in_subtree

      # Continue the NUTS trajectory if the tree-building did not terminate, and
      # if the reverse-most and forward-most states do not exhibit a U-turn.
      has_no_u_turn = tf.logical_and(
          _has_no_u_turn(forward_state, reverse_state, forward_momentum),
          _has_no_u_turn(forward_state, reverse_state, reverse_momentum))
      continue_trajectory = continue_trajectory and has_no_u_turn
      num_states += num_states_in_subtree
      depth += 1

    return next_state, next_target_log_prob, next_grads_target_log_prob


def _build_tree(value_and_gradients_fn,
                current_state,
                current_target_log_prob,
                current_grads_target_log_prob,
                current_momentum,
                direction,
                depth,
                step_size,
                log_slice_sample,
                max_simulation_error=1000.,
                seed=None):
  """Builds a tree at a given tree depth and at a given state.

  The `current` state is immediately adjacent to, but outside of,
  the subtrajectory spanned by the returned `forward` and `reverse` states.

  Args:
    value_and_gradients_fn: Python callable which takes an argument like
      `*current_state` and returns a tuple of its (possibly unnormalized)
      log-density under the target distribution and its gradient with respect to
      each state.
    current_state: List of `Tensor`s representing the current states of the
      NUTS trajectory.
    current_target_log_prob: Scalar `Tensor` representing the value of
      `target_log_prob_fn` at the `current_state`.
    current_grads_target_log_prob: List of `Tensor`s representing gradient of
      `current_target_log_prob` with respect to `current_state`. Must have same
      shape as `current_state`.
    current_momentum: List of `Tensor`s representing the momentums of
      `current_state`. Must have same shape as `current_state`.
    direction: int that is either -1 or 1. It determines whether to perform
      leapfrog integration backwards (reverse) or forward in time respectively.
    depth: non-negative int that indicates how deep of a tree to build.
      Each call to `_build_tree` takes `2**depth` leapfrog steps.
    step_size: List of `Tensor`s representing the step sizes for the leapfrog
      integrator. Must have same shape as `current_state`.
    log_slice_sample: The log of an auxiliary slice variable. It is used
      together with `max_simulation_error` to avoid simulating trajectories with
      too much numerical error.
    max_simulation_error: Maximum simulation error to tolerate before
      terminating the trajectory. Simulation error is the
      `log_slice_sample` minus the log-joint probability at the simulated state.
    seed: Integer to seed the random number generator.

  Returns:
    reverse_state: List of `Tensor`s representing the "reverse" states of the
      NUTS trajectory. Has same shape as `current_state`.
    reverse_target_log_prob: Scalar `Tensor` representing the value of
      `target_log_prob_fn` at the `reverse_state`.
    reverse_grads_target_log_prob: List of `Tensor`s representing gradient of
      `reverse_target_log_prob` with respect to `reverse_state`. Has same shape
      as `reverse_state`.
    reverse_momentum: List of `Tensor`s representing the momentums of
      `reverse_state`. Has same shape as `reverse_state`.
    forward_state: List of `Tensor`s representing the "forward" states of the
      NUTS trajectory. Has same shape as `current_state`.
    forward_target_log_prob: Scalar `Tensor` representing the value of
      `target_log_prob_fn` at the `forward_state`.
    forward_grads_target_log_prob: List of `Tensor`s representing gradient of
      `forward_target_log_prob` with respect to `forward_state`. Has same shape
      as `forward_state`.
    forward_momentum: List of `Tensor`s representing the momentums of
      `forward_state`. Has same shape as `forward_state`.
    next_state: List of `Tensor`s representing the next states of the NUTS
      trajectory. Has same shape as `current_state`.
    next_target_log_prob: Scalar `Tensor` representing the value of
      `target_log_prob_fn` at `next_state`.
    next_grads_target_log_prob: List of `Tensor`s representing the gradient of
      `next_target_log_prob` with respect to `next_state`.
    num_states: Number of acceptable candidate states in the subtree. A state is
      acceptable if it is "in the slice", that is, if its log-joint probability
      with its momentum is greater than `log_slice_sample`.
    continue_trajectory: bool determining whether to continue the simulation
      trajectory. The trajectory is continued if no U-turns are encountered
      within the built subtree, and if the log-probability accumulation due to
      integration error does not exceed `max_simulation_error`.
  """
  if depth == 0:  # base case
    # Take a leapfrog step. Terminate the tree-building if the simulation
    # error from the leapfrog integrator is too large. States discovered by
    # continuing the simulation are likely to have very low probability.
    [
        next_state,
        next_target_log_prob,
        next_grads_target_log_prob,
        next_momentum,
    ] = _leapfrog(
        value_and_gradients_fn=value_and_gradients_fn,
        current_state=current_state,
        current_grads_target_log_prob=current_grads_target_log_prob,
        current_momentum=current_momentum,
        step_size=direction * step_size)
    next_log_joint = _log_joint(next_target_log_prob, next_momentum)
    num_states = tf.cast(next_log_joint > log_slice_sample, dtype=tf.int32)
    continue_trajectory = (next_log_joint >
                           log_slice_sample - max_simulation_error)
    return [
        next_state,
        next_target_log_prob,
        next_grads_target_log_prob,
        next_momentum,
        next_state,
        next_target_log_prob,
        next_grads_target_log_prob,
        next_momentum,
        next_state,
        next_target_log_prob,
        next_grads_target_log_prob,
        num_states,
        continue_trajectory,
    ]

  # Build a tree at the current state.
  seed_stream = tfd.SeedStream(seed, "build_tree")
  [
      reverse_state,
      reverse_target_log_prob,
      reverse_grads_target_log_prob,
      reverse_momentum,
      forward_state,
      forward_target_log_prob,
      forward_grads_target_log_prob,
      forward_momentum,
      next_state,
      next_target_log_prob,
      next_grads_target_log_prob,
      num_states,
      continue_trajectory,
  ] = _build_tree(value_and_gradients_fn=value_and_gradients_fn,
                  current_state=current_state,
                  current_target_log_prob=current_target_log_prob,
                  current_grads_target_log_prob=current_grads_target_log_prob,
                  current_momentum=current_momentum,
                  direction=direction,
                  depth=depth - 1,
                  step_size=step_size,
                  log_slice_sample=log_slice_sample,
                  seed=seed_stream())
  if continue_trajectory:
    # If the just-built subtree did not terminate, build a second subtree at
    # the forward or reverse state, as appropriate.
    if direction < 0:
      [
          reverse_state,
          reverse_target_log_prob,
          reverse_grads_target_log_prob,
          reverse_momentum,
          _,
          _,
          _,
          _,
          far_state,
          far_target_log_prob,
          far_grads_target_log_prob,
          far_num_states,
          far_continue_trajectory,
      ] = _build_tree(
          value_and_gradients_fn=value_and_gradients_fn,
          current_state=reverse_state,
          current_target_log_prob=reverse_target_log_prob,
          current_grads_target_log_prob=reverse_grads_target_log_prob,
          current_momentum=reverse_momentum,
          direction=direction,
          depth=depth - 1,
          step_size=step_size,
          log_slice_sample=log_slice_sample,
          seed=seed_stream())
    else:
      [
          _,
          _,
          _,
          _,
          forward_state,
          forward_target_log_prob,
          forward_grads_target_log_prob,
          forward_momentum,
          far_state,
          far_target_log_prob,
          far_grads_target_log_prob,
          far_num_states,
          far_continue_trajectory,
      ] = _build_tree(
          value_and_gradients_fn=value_and_gradients_fn,
          current_state=forward_state,
          current_target_log_prob=forward_target_log_prob,
          current_grads_target_log_prob=forward_grads_target_log_prob,
          current_momentum=forward_momentum,
          direction=direction,
          depth=depth - 1,
          step_size=step_size,
          log_slice_sample=log_slice_sample,
          seed=seed_stream())

    # Propose either `next_state` (which came from the first subtree and so is
    # nearby) or the new forward/reverse state (which came from the second
    # subtree and so is far away).
    num_states += far_num_states
    accept_far_state = _random_bernoulli(
        [],
        probs=far_num_states / num_states,
        dtype=tf.bool,
        seed=seed_stream())
    if accept_far_state:
      next_state = far_state
      next_target_log_prob = far_target_log_prob
      next_grads_target_log_prob = far_grads_target_log_prob

    # Continue the NUTS trajectory if the far subtree did not terminate either,
    # and if the reverse-most and forward-most states do not exhibit a U-turn.
    has_no_u_turn = tf.logical_and(
        _has_no_u_turn(forward_state, reverse_state, forward_momentum),
        _has_no_u_turn(forward_state, reverse_state, reverse_momentum))
    continue_trajectory = far_continue_trajectory and has_no_u_turn

  return [
      reverse_state,
      reverse_target_log_prob,
      reverse_grads_target_log_prob,
      reverse_momentum,
      forward_state,
      forward_target_log_prob,
      forward_grads_target_log_prob,
      forward_momentum,
      next_state,
      next_target_log_prob,
      next_grads_target_log_prob,
      num_states,
      continue_trajectory,
  ]


def _embed_no_none_gradient_check(value_and_gradients_fn):
  """Wraps value and gradients function to assist with None gradients."""
  @functools.wraps(value_and_gradients_fn)
  def func_wrapped(*args, **kwargs):
    """Wrapped function which checks for None gradients."""
    value, grads = value_and_gradients_fn(*args, **kwargs)
    if any(grad is None for grad in grads):
      raise ValueError("Gradient is None for a state.")
    return value, grads
  return func_wrapped


def _has_no_u_turn(state_one, state_two, momentum):
  """If two given states and momentum do not exhibit a U-turn pattern."""
  dot_product = sum([tf.reduce_sum((s1 - s2) * m)
                     for s1, s2, m in zip(state_one, state_two, momentum)])
  return dot_product > 0


def _leapfrog(value_and_gradients_fn,
              current_state,
              current_grads_target_log_prob,
              current_momentum,
              step_size):
  """Runs one step of leapfrog integration."""
  mid_momentum = [
      m + 0.5 * step * g for m, step, g in
      zip(current_momentum, step_size, current_grads_target_log_prob)]
  next_state = [
      s + step * m for s, step, m in
      zip(current_state, step_size, mid_momentum)]
  next_target_log_prob, next_grads_target_log_prob = value_and_gradients_fn(
      *next_state)
  next_momentum = [
      m + 0.5 * step * g for m, step, g in
      zip(mid_momentum, step_size, next_grads_target_log_prob)]
  return [
      next_state,
      next_target_log_prob,
      next_grads_target_log_prob,
      next_momentum,
  ]


def _log_joint(current_target_log_prob, current_momentum):
  """Log-joint probability given a state's log-probability and momentum."""
  momentum_log_prob = -sum([tf.reduce_sum(0.5 * (m ** 2.))
                            for m in current_momentum])
  return current_target_log_prob + momentum_log_prob


def _random_bernoulli(shape, probs, dtype=tf.int32, seed=None, name=None):
  """Returns samples from a Bernoulli distribution."""
  with tf.name_scope(name, "random_bernoulli", [shape, probs]):
    probs = tf.convert_to_tensor(probs)
    random_uniform = tf.random_uniform(shape, dtype=probs.dtype, seed=seed)
    return tf.cast(tf.less(random_uniform, probs), dtype)
