# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Estimation of automaton behavior using RL-style trajectory rollouts."""

import functools
from typing import Any, Union

import dataclasses
import jax
import jax.numpy as jnp

from gfsa import automaton_builder
from gfsa import jax_util


@jax_util.register_dataclass_pytree
@dataclasses.dataclass
class RollOutState:
  """The state of a single agent rollout.

  Attributes:
    rng: PRNGKey determining how samples are drawn.
    at_initial: Whether agent is in the initial state and receives the initial
      state observation.
    succeeded: Whether the agent has taken the FINISH action.
    failed: Whether the agent has taken the FAIL action.
    special_from_initial: If the agent took a special action, whether it was at
      the initial state at the time.
    itn_state_index: If the agent is at an input-tagged-node instead of the
      initial state, the index of that state.
    final_node: The final location of the agent.
    log_prob: Log probability of the sequence of choices made so far; used to
      compute REINFORCE gradients.
  """
  rng: Any  # PRNGKey
  at_initial: Any = True  # <bool[]>
  succeeded: Any = False  # <bool[]>
  failed: Any = False  # <bool[]>
  special_from_initial: Any = False  # <bool[]>
  itn_state_index: Any = -1  # <int32[]>
  final_node: Any = None  # optional <int32[]>
  log_prob: Any = 0.0  # <float32[]>


def roll_out_transitions(
    builder,
    transition_matrix,
    variant_weights, start_machine_state,
    node_index, steps, rng,
    max_possible_transitions):
  """Roll out transitions for a transition matrix.

  Args:
    builder: The automaton builder associated with this Markov chain.
    transition_matrix: The per-variant transition matrix for this Markov chain.
    variant_weights: Weights to assign to each routing variant for each node, as
      a <float[num_nodes, num_variants]> array (which should sum to 1 across the
      last axis).
    start_machine_state: Initial machine state distribution for the starting
      nodes, as a <float[num_fsm_states]> array (which should sum to 1).
    node_index: Initial node index where the solve should start, as a int32
      scalar.
    steps: How many steps to unroll.
    rng: Random number generator used to sample.
    max_possible_transitions: Max transitions to truncate to.

  Returns:
    Final RollOutState.
  """
  (variants, nodes, fsm_states, in_tagged_nodes,
   _) = transition_matrix.initial_to_in_tagged.shape
  itn_states = in_tagged_nodes * fsm_states

  def add_special_dests(in_tagged_dests):
    return jnp.concatenate(
        [in_tagged_dests,
         jnp.arange(itn_states, itn_states + 3)])

  # Collapse our transition matrix into a more useful form:
  # - Combine nodes and FSM states
  # - Extract a subset of possible destinations
  # - Combine special and move actions into a single probability table
  k = min(max_possible_transitions, in_tagged_nodes) * fsm_states
  initial_to_in_tagged_combo = transition_matrix.initial_to_in_tagged.reshape(
      (variants, nodes, fsm_states, itn_states))
  _, initial_to_in_tagged_dests = jax.lax.top_k(
      jnp.sum(initial_to_in_tagged_combo, axis=(0, 2)), k=k)
  initial_to_in_tagged_probs = jax.vmap(
      lambda c, i: c[:, :, i], in_axes=(1, 0),
      out_axes=1)(initial_to_in_tagged_combo, initial_to_in_tagged_dests)
  initial_probs = jnp.concatenate(
      [initial_to_in_tagged_probs, transition_matrix.initial_to_special], -1)
  initial_dests = jax.vmap(add_special_dests)(initial_to_in_tagged_dests)

  in_tagged_to_in_tagged_combo = transition_matrix.in_tagged_to_in_tagged.reshape(
      (variants, itn_states, itn_states))
  _, in_tagged_to_in_tagged_dests = jax.lax.top_k(
      jnp.sum(in_tagged_to_in_tagged_combo, axis=0), k=k)
  in_tagged_to_in_tagged_probs = jax.vmap(
      lambda c, i: c[:, i], in_axes=(1, 0),
      out_axes=1)(in_tagged_to_in_tagged_combo, in_tagged_to_in_tagged_dests)
  in_tagged_probs = jnp.concatenate([
      in_tagged_to_in_tagged_probs,
      transition_matrix.in_tagged_to_special.reshape((variants, itn_states, 3))
  ], -1)
  in_tagged_dests = jax.vmap(add_special_dests)(in_tagged_to_in_tagged_dests)

  start_node_variant_weights = variant_weights[node_index]
  initial_probs_from_here = jnp.einsum("v,s,vsj->j", start_node_variant_weights,
                                       start_machine_state,
                                       initial_probs[:, node_index, :, :])
  initial_dests_from_here = initial_dests[node_index]

  per_itn_variants = variant_weights[transition_matrix.in_tagged_node_indices]

  # Set up the initial state
  initial_state = RollOutState(rng=rng)

  @jax.remat
  def scan_body(state, ignored_input):
    assert ignored_input is None
    rng, key = jax.random.split(state.rng)

    # If we are in the initial state, sample the initial transition.
    def at_initial_info():
      return initial_probs_from_here, initial_dests_from_here

    # If we are in a normal state, sample the next action.
    def at_normal_info():
      cur_variant_weights = per_itn_variants[state.itn_state_index //
                                             fsm_states]
      next_step_probs = jnp.einsum("v,vj->j", cur_variant_weights,
                                   in_tagged_probs[:, state.itn_state_index, :])

      return next_step_probs, in_tagged_dests[state.itn_state_index, :]

    # Figure out which to do, and sample from the appropriate probabilities
    step_probs, step_dests = jax.tree_multimap(
        functools.partial(jnp.where, state.at_initial), at_initial_info(),
        at_normal_info())

    next_idx = jax.random.categorical(key, jnp.log(step_probs))
    log_prob = jnp.log(step_probs[next_idx])
    dest = step_dests[next_idx]
    did_special = dest >= itn_states

    state_after_move = RollOutState(
        at_initial=False,
        succeeded=False,
        failed=False,
        special_from_initial=False,
        itn_state_index=dest,
        final_node=None,
        log_prob=state.log_prob + log_prob,
        rng=rng)

    special_idx = dest - itn_states
    state_after_special = RollOutState(
        at_initial=(special_idx == builder.special_actions.index(
            automaton_builder.SpecialActions.BACKTRACK)),
        succeeded=(special_idx == builder.special_actions.index(
            automaton_builder.SpecialActions.FINISH)),
        failed=(special_idx == builder.special_actions.index(
            automaton_builder.SpecialActions.FAIL)),
        special_from_initial=state.at_initial,
        itn_state_index=state.itn_state_index,
        final_node=None,
        log_prob=state.log_prob + log_prob,
        rng=rng)

    # Choose the right branch to take
    def choose(move, special, done):
      return jnp.where(state.succeeded | state.failed, done,
                       jnp.where(did_special, special, move))

    new_state = jax.tree_multimap(choose, state_after_move, state_after_special,
                                  state)

    return new_state, None

  final_state, _ = jax.lax.scan(scan_body, initial_state, None, length=steps)
  final_node = jnp.where(
      final_state.special_from_initial, node_index,
      transition_matrix.in_tagged_node_indices[final_state.itn_state_index //
                                               fsm_states])
  final_state = dataclasses.replace(final_state, final_node=final_node)
  return final_state


def one_node_particle_estimate(
    builder,
    transition_matrix,
    variant_weights,
    start_machine_state,
    node_index,
    num_valid_nodes,
    steps,
    num_rollouts,
    max_possible_transitions,
    rng,
):
  """Approximately solve for absorbing probabilities using multiple rollouts.

  Note that this is not differentiable! Used for tests.

  Args:
    builder: The automaton builder associated with this Markov chain.
    transition_matrix: The per-variant transition matrix for this Markov chain.
    variant_weights: Weights to assign to each routing variant for each node, as
      a <float[num_nodes, num_variants]> array (which should sum to 1 across the
      last axis).
    start_machine_state: Initial machine state distribution for the starting
      nodes, as a <float[num_fsm_states]> array (which should sum to 1).
    node_index: Initial node index where the solve should start, as a int32
      scalar.
    num_valid_nodes: How many valid nodes there are.
    steps: How many steps to unroll.
    num_rollouts: How many rollout particles to simulate.
    max_possible_transitions: Maximum number of possible transitions.
    rng: Random number generator used to sample.,

  Returns:
    Estimated probabilities of absorbing at each node, as a
    <float[num_nodes]> array.
  """
  max_num_nodes = variant_weights.shape[0]

  # Run each of the rollouts.
  all_states = jax_util.vmap_with_kwargs(
      roll_out_transitions, rng_axis=0)(
          builder=builder,
          transition_matrix=transition_matrix,
          variant_weights=variant_weights,
          start_machine_state=start_machine_state,
          node_index=node_index,
          steps=steps,
          rng=jax.random.split(rng, num_rollouts),
          max_possible_transitions=max_possible_transitions)

  # Add up successes.
  success_counts = (
      jnp.zeros([max_num_nodes], jnp.int32).at[all_states.final_node].add(
          all_states.succeeded.astype(jnp.int32)))

  # Use this to estimate probability
  approx_out_distn = success_counts / num_rollouts
  approx_out_distn = jnp.where(
      jnp.arange(max_num_nodes) < num_valid_nodes, approx_out_distn, 0)

  return approx_out_distn


def all_nodes_particle_estimate(
    builder,
    transition_matrix,
    variant_weights,
    start_machine_states,
    num_valid_nodes,
    steps,
    num_rollouts,
    max_possible_transitions,
    rng,
):
  """Particle-based estimate of absorbing probabilities.

  Not differentiable!

  Args:
    builder: The automaton builder associated with this Markov chain.
    transition_matrix: The per-variant transition matrix for this Markov chain.
    variant_weights: Weights to assign to each routing variant for each node, as
      a <float[num_nodes, num_nodes, num_variants]> array (which should sum to 1
      across the last axis).
    start_machine_states: Initial machine state distribution for the starting
      nodes, as a <float[num_nodes, num_fsm_states]> array (which should sum to
      1 across the last axis).
    num_valid_nodes: How many valid nodes there are.
    steps: How many steps of the automaton to simulate.
    num_rollouts: How many rollout particles to simulate.
    max_possible_transitions: Maximum number of possible transitions.
    rng: Random number generator for rollouts.

  Returns:
    Estimated probabilities of absorbing at each node, as a
    <float[num_nodes, num_nodes]> array.
  """
  num_nodes = start_machine_states.shape[0]

  # Map our rollout across nodes.
  all_nodes_roll_out_fn = jax_util.vmap_with_kwargs(
      one_node_particle_estimate,
      variant_weights_axis=0,
      start_machine_state_axis=0,
      node_index_axis=0,
      rng_axis=0)

  return all_nodes_roll_out_fn(
      builder=builder,
      transition_matrix=transition_matrix,
      variant_weights=variant_weights,
      start_machine_state=start_machine_states,
      node_index=jnp.arange(num_nodes),
      num_valid_nodes=num_valid_nodes,
      steps=steps,
      num_rollouts=num_rollouts,
      max_possible_transitions=max_possible_transitions,
      rng=jax.random.split(rng, num_nodes))
