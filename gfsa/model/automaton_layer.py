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

# Lint as: python3
"""Flax module for the GFSA layer."""

from typing import Any, Callable, Optional, TypeVar

import flax
import gin
import jax
import jax.numpy as jnp

from gfsa import automaton_builder
from gfsa import automaton_sampling
from gfsa import jax_util
from gfsa.model import model_util
from gfsa.model import side_outputs

T = TypeVar("T")


def _unshaped_initializer(fn,
                          **kwargs):
  """Build a flax-style initializer that ignores shapes.

  Args:
    fn: Callable that takes a random number generator and some keyword
      arguments, and returns an initial parameter value.
    **kwargs: Arguments to pass to fn.

  Returns:
    Callable that takes a PRNGKey and an (unused) shape, and returns the value
    returned by fn.
  """

  def _initialize(rng_key, unused_shape):
    return fn(rng_key, **kwargs)

  return _initialize


class FiniteStateGraphAutomaton(flax.nn.Module):
  """GFSA layer, with different edge types as states."""

  @gin.configurable("FiniteStateGraphAutomaton")
  def apply(
      self,
      encoded_graph,
      dynamic_metadata,
      static_metadata,
      builder,
      num_out_edges = 1,
      num_intermediate_states = 0,
      variant_weights = None,
      steps = None,
      share_states_across_edges = True,
      backtrack_fails_prob = 0.001,
      initialization_noise_factor = 0.01,
      legacy_initialize = False,
      initialize_smoothing = 0.001,
      use_gate_parameterization = False,
      gate_noise = 0.2,
      logit_scaling = "learned",
      dynamic_scaling_absolute_shift = 1.0,
      dynamic_scaling_relative_shift = 1.0,
      estimator_type = "solver",
      sampling_max_possible_transitions = None,
  ):
    """Apply the graph finite-state automaton layer.

    Args:
      encoded_graph: Graph structure to run the automaton on.
      dynamic_metadata: Metadata about the actual unpadded graph. Unused.
      static_metadata: Statically-known metadata about the graph size. If
        encoded_graph is padded, this should reflect the padded size, not the
        original size.
      builder: Builder object that encoded the graph. Determines the shape of
        the automaton parameters.
      num_out_edges: How many distinct "edge types" to produce. Each of these
        corresponds to a different automaton start state.
      num_intermediate_states: How many intermediate states to include. These
        states are only used inside the automaton, and don't get exposed as
        start states for an edge type. If share_states_across_edges is True, all
        of the edge types will share the same set of intermediate states. If
        share_states_across_edges is False, each edge type will have its OWN set
        of `num_intermediate_states` states.
      variant_weights: Optional <float32[num_nodes, num_nodes, num_variants]> or
        <float32[num_nodes, num_nodes, num_out_edges, num_variants]> array that
        has nonnegative elements that sum to 1 along the last axis. If provided,
        variant_weights[i, j, (e,) v] specifies how much policy variant v should
        be used (for edge type e) when starting from node i and arriving at
        intermediate node j. Variants correspond to the start-node-conditioned
        observations described in Appendix C.2.
      steps: How many steps to use when solving the automaton. If not provided,
        uses the number of nodes in the graph. Smaller numbers of steps may make
        it impossible to reach certain nodes in the graph; larger numbers may
        allow additional backtracking and state changes at the expense of
        additional compute time.
      share_states_across_edges: Whether the different edge types share the same
        set of states. If True, any state can transition to any other state
        (even to a different start state). If False, every edge type gets a
        separate set of `1 + num_intermediate_states` that are not shared; in
        other words, every output edge uses a distinct finite state machine.
      backtrack_fails_prob: Backtracking decay factor; determines how often the
        automaton halts when it tries to take the BACKTRACK action. If the
        automaton attempts to backtrack with close to 100% probability, this
        ensures numerical stability and counteracts noise.
      initialization_noise_factor: How much noise to use when initializing the
        automaton policy (see AutomatonBuilder.initialize_routing_params)
      legacy_initialize: Whether to use legacy initialization, which sets the
        log-space softmax weights as Dirichlet random samples (instead of
        setting the softmax output distribution as Dirichlet random samples).
      initialize_smoothing: Controls how much we smooth the initial parameters
        toward a uniform distribution. For small values, this is effectively a
        lower bound of the probability we take each action. If zero, we can
        sample arbitrarily small starting probabilities from the Dirichlet
        distribution. (Specifically, we adjust the sampled probabilities as
        `p_init(x) = (p_sampled(x) + c)/(1 + n*c)` where `n` is the number of
        possible actions and `c = initialize_smoothing`.
      use_gate_parameterization: Whether to use gate parameterization instead of
        default parameterization. If so, the other initialization args are
        ignored. Not used for experiments in the paper.
      gate_noise: Logistic noise for gate parameterization.
      logit_scaling: One of "none", "learned", "dynamic". The "dynamic" mode
        is experimental and was not used for experiments in the paper.
      dynamic_scaling_absolute_shift: For dynamic scaling, how much extra to
        shift logits, in an absolute sense, after shifting for mean magnitude.
      dynamic_scaling_relative_shift: For dynamic scaling, how much extra to
        shift logits, relative to standard deviation of magnitude, after
        shifting for mean magnitude
      estimator_type: One of "solver", "one_sample".
      sampling_max_possible_transitions: Number of possible outgoing transitions
        for any given start node. Used to accelerate the sampling process when
        estimator is "one_sample".

    Returns:
      <float32[num_out_edges, num_nodes, num_nodes]> weighted adjacency
      matrix for `num_out_edges` new edge types.
    """
    del dynamic_metadata
    num_nodes = static_metadata.num_nodes
    steps = steps if steps is not None else num_nodes
    assert logit_scaling in ("none", "learned", "dynamic")

    if variant_weights is None:
      num_variants = 1
      variant_out_edge_axis = None
      variant_weights = jnp.ones((num_nodes, num_nodes, num_variants))
    elif variant_weights.ndim == 3:
      num_variants = variant_weights.shape[-1]
      variant_out_edge_axis = None
      if variant_weights.shape != (num_nodes, num_nodes, num_variants):
        raise ValueError(
            f"variant_weights shape {variant_weights.shape} doesn't match "
            f"expected shape ({num_nodes}, {num_nodes}, <anything>)")
    elif variant_weights.ndim == 4:
      num_variants = variant_weights.shape[-1]
      variant_out_edge_axis = 2
      if variant_weights.shape != (num_nodes, num_nodes, num_out_edges,
                                   num_variants):
        raise ValueError(
            f"variant_weights shape {variant_weights.shape} doesn't match "
            "expected shape"
            f"({num_nodes}, {num_nodes}, {num_out_edges}, <anything>)")
    else:
      raise ValueError(f"Invalid variant_weights shape {variant_weights.shape};"
                       " expected 3 or 4 axes")

    if share_states_across_edges:
      num_fsm_states = num_out_edges + num_intermediate_states

      # Initialize or retrieve the automaton parameters; these parameters are
      # shared across all edge types.
      if use_gate_parameterization:

        def shared_routing_initializer(rng_key, unused_shape):
          return builder.initialize_routing_gates(
              key=rng_key,
              num_fsm_states=num_fsm_states,
              num_variants=num_variants,
              logistic_noise=gate_noise)

        routing_gates = self.param(
            "routing_gate_logits_shared",
            shape=None,
            initializer=shared_routing_initializer)
        routing_gates = side_outputs.encourage_discrete_logits(
            routing_gates,
            distribution_type="binary",
            name="routing_gate_logits_shared")
        routing_params = builder.routing_gates_to_probs(routing_gates)

      else:

        def shared_routing_initializer(rng_key, unused_shape):
          routing_probs = builder.initialize_routing_params(
              key=rng_key,
              num_fsm_states=num_fsm_states,
              num_variants=num_variants,
              noise_factor=initialization_noise_factor)
          if legacy_initialize:
            return routing_probs
          return jax.tree_map(lambda x: jnp.log(x + initialize_smoothing),
                              routing_probs)

        log_routing_params = self.param(
            "log_routing_params_shared",
            shape=None,
            initializer=shared_routing_initializer)
        routing_params = builder.routing_softmax(log_routing_params)

      # Don't precompute constants if we are tracing an XLA computation; wait
      # until we know a value for our parameters by adding a fake data
      # dependence.
      trigger = jax.tree_leaves(routing_params)[0]
      variant_weights = jax.lax.tie_in(trigger, variant_weights)

      # Build the automaton on the provided graph.
      transition_matrix = builder.build_transition_matrix(
          routing_params, encoded_graph, static_metadata)

      # Each edge type is a start state.
      if num_intermediate_states > 0:
        start_machine_states = jnp.concatenate([
            jax.lax.tie_in(trigger, jnp.eye(num_out_edges)),
            jax.lax.tie_in(trigger,
                           jnp.zeros((num_out_edges, num_intermediate_states)))
        ], 1)
      else:
        start_machine_states = jax.lax.tie_in(trigger, jnp.eye(num_out_edges))

      start_machine_states = jnp.broadcast_to(
          start_machine_states, (num_nodes, num_out_edges, num_fsm_states))

      # Solve for absorbing distribution for each of the starting states by
      # vmapping across the dimensions that depend on the start state.
      if estimator_type == "solver":
        absorbing_solution = jax_util.vmap_with_kwargs(
            automaton_builder.all_nodes_absorbing_solve,
            variant_weights_axis=variant_out_edge_axis,
            start_machine_states_axis=1)(
                builder=builder,
                transition_matrix=transition_matrix,
                variant_weights=variant_weights,
                start_machine_states=start_machine_states,
                steps=steps,
                backtrack_fails_prob=backtrack_fails_prob)
      elif estimator_type == "one_sample":
        assert sampling_max_possible_transitions is not None
        rollout_each_node_fn = jax_util.vmap_with_kwargs(
            automaton_sampling.roll_out_transitions,
            variant_weights_axis=0,
            start_machine_state_axis=0,
            node_index_axis=0,
            rng_axis=0)
        rollout_each_edgetype_fn = jax_util.vmap_with_kwargs(
            rollout_each_node_fn,
            variant_weights_axis=variant_out_edge_axis,
            start_machine_state_axis=1,
            rng_axis=0)
        all_states = rollout_each_edgetype_fn(
            builder=builder,
            transition_matrix=transition_matrix,
            variant_weights=variant_weights,
            start_machine_state=start_machine_states,
            node_index=jnp.arange(num_nodes),
            steps=steps,
            max_possible_transitions=sampling_max_possible_transitions,
            rng=jax.random.split(flax.nn.make_rng(),
                                 num_out_edges * num_nodes).reshape(
                                     [num_out_edges, num_nodes, -1]))

        def set_absorbing(final_node, succeeded):
          return jnp.zeros([num_nodes]).at[final_node].set(succeeded)

        # absorbing_solution is [num_out_edges, num_nodes, num_nodes]
        absorbing_solution = jax.vmap(jax.vmap(set_absorbing))(
            all_states.final_node, all_states.succeeded)

        side_outputs.SideOutput(
            all_states.log_prob, name="one_sample_log_prob_per_edge_per_node")

        # Somewhat of a hack: associate the log prob with its own learned
        # baseline as a side output, so it can be trained alongside the rest
        # of the model, but don't do anything with it until the loss function.
        one_sample_reward_baseline = self.param(
            "one_sample_reward_baseline",
            shape=(),
            initializer=jax.nn.initializers.zeros)
        side_outputs.SideOutput(
            one_sample_reward_baseline, name="one_sample_reward_baseline")
      else:
        raise ValueError(f"Invalid estimator {estimator_type}")

      # Rescale the logits.
      logits = model_util.safe_logit(absorbing_solution)
      if logit_scaling == "learned":
        # Learned scaling and shifting.
        logits = model_util.ScaleAndShift(logits)
      elif logit_scaling == "dynamic":
        # Dynamic scaling only implemented with gates.
        assert use_gate_parameterization
        # First, quantify how discrete the gates are. Conceptually, we want
        # to quantify how far away from zero the logits are, in a differentiable
        # way. To make it smooth, use logsumexp:
        relevant_gates = [routing_gates.move_gates, routing_gates.accept_gates]
        soft_abs_logits = [jnp.logaddexp(g, -g) for g in relevant_gates]
        # Take a mean and standard deviation over these logits to summarize.
        logit_mean = (
            sum(jnp.sum(x) for x in soft_abs_logits) /
            sum(x.size for x in soft_abs_logits))
        logit_var = (
            sum(jnp.sum(jnp.square(x - logit_mean)) for x in soft_abs_logits) /
            sum(x.size for x in soft_abs_logits))
        logit_std = jnp.sqrt(logit_var)
        side_outputs.SideOutput(logit_mean, name="gate_logit_abs_mean")
        side_outputs.SideOutput(logit_std, name="gate_logit_abs_std")
        # Now, use these to choose an adjustment factor. Intuitively, the "off"
        # gates should be centered around (-logit_mean). So anything that
        # gets sufficiently more mass than that should be "on". We consider
        # two notions of "sufficiently": either relative to the variance in
        # logits, or absolute.
        shift_threshold = (-logit_mean + dynamic_scaling_absolute_shift +
                           dynamic_scaling_relative_shift * logit_std)
        side_outputs.SideOutput(shift_threshold, name="shift_threshold")
        # Adjust so that values at the shift threshold are mapped to edges of
        # weight 0.5
        logits = logits - shift_threshold

    else:
      num_fsm_states = 1 + num_intermediate_states

      if estimator_type != "solver":
        raise NotImplementedError(
            "Sampling estimators not implemented for unshared states.")

      # Different automaton parameters for each start state
      if use_gate_parameterization:

        def unshared_routing_initializer(rng_key, unused_shape):
          key_per_edge = jax.random.split(rng_key, num_out_edges)
          return jax_util.vmap_with_kwargs(
              builder.initialize_routing_gates, key_axis=0)(
                  key=key_per_edge,
                  num_fsm_states=num_fsm_states,
                  num_variants=num_variants,
                  logistic_noise=gate_noise)

        routing_gates = self.param(
            "routing_gate_logits_unshared",
            shape=None,
            initializer=unshared_routing_initializer)
        routing_gates = side_outputs.encourage_discrete_logits(
            routing_gates,
            distribution_type="binary",
            name="routing_gate_logits_unshared")
        stacked_routing_params = jax.vmap(builder.routing_gates_to_probs)(
            routing_gates)

      else:

        def unshared_routing_initializer(rng_key, unused_shape):
          key_per_edge = jax.random.split(rng_key, num_out_edges)
          routing_probs = jax_util.vmap_with_kwargs(
              builder.initialize_routing_params, key_axis=0)(
                  key=key_per_edge,
                  num_fsm_states=num_fsm_states,
                  num_variants=num_variants,
                  noise_factor=initialization_noise_factor)
          if legacy_initialize:
            return routing_probs
          return jax.tree_map(lambda x: jnp.log(x + initialize_smoothing),
                              routing_probs)

        log_routing_params = self.param(
            "log_routing_params_unshared",
            shape=None,
            initializer=unshared_routing_initializer)
        stacked_routing_params = jax.vmap(builder.routing_softmax)(
            log_routing_params)

      def solve_one(one_edge_routing_params, one_edge_variant_weights):
        """Run one of the edge-specific automata."""
        # Build the automaton on the provided graph.
        transition_matrix = builder.build_transition_matrix(
            one_edge_routing_params, encoded_graph, static_metadata)

        # Start state is always state 0.
        start_machine_states = jnp.broadcast_to(
            (jnp.arange(num_fsm_states) == 0)[None, :],
            (num_nodes, num_fsm_states))

        return automaton_builder.all_nodes_absorbing_solve(
            builder=builder,
            transition_matrix=transition_matrix,
            variant_weights=one_edge_variant_weights,
            start_machine_states=start_machine_states,
            steps=steps,
            backtrack_fails_prob=backtrack_fails_prob)

      absorbing_solution = jax.vmap(
          solve_one,
          in_axes=(0, variant_out_edge_axis),
      )(stacked_routing_params, variant_weights)

      # Rescale the logits.
      logits = model_util.safe_logit(absorbing_solution)
      if logit_scaling == "learned":
        # Learned scaling and shifting.
        logits = model_util.ScaleAndShift(logits)
      elif logit_scaling == "dynamic":
        raise NotImplementedError(
            "Dynamic scaling not implemented for unshared")

    logits = side_outputs.encourage_discrete_logits(
        logits, distribution_type="binary", name="edge_logits")
    result = jax.nn.sigmoid(logits)
    return result
