# coding=utf-8
# Copyright 2021 The Google Research Authors.
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
"""Models and helpers for gridworld maze task."""

import contextlib
import time
import typing
from typing import Any, Callable, Dict, Tuple

from absl import logging
import dataclasses

import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np

from gfsa import automaton_builder
from gfsa import jax_util
from gfsa.datasets import data_loading
from gfsa.datasets import graph_bundle
from gfsa.datasets.mazes import maze_task
from gfsa.model import automaton_layer
from gfsa.training import train_util


def iterative_fixed_point(
    fun,
    x,
    guess,
    iterations,
):
  """Finds a fixed point of a given function by iterating it.

  In particular, returns a y such that we approximately have

    y = fun(x, y)

  This function makes no convergence guarantees; it is up to the caller to
  ensure that iterating fun(x, *) converges.

  Args:
    fun: Function to find the fixed point of.
    x: Inputs that determine the fixed point; will be differentiated through.
    guess: Initial guess about the solution; will NOT be differentiated through.
    iterations: Number of iterations to run the equation for.

  Returns:
    Approximate fixed point, using implicit differentiation to compute gradients
    if requested.
  """

  @jax.custom_jvp
  def solve_forward(x, guess):
    """Iterates the system for a fixed number of iterations."""
    return jax.lax.fori_loop(0, iterations, lambda _, y: fun(x, y), guess)

  def solve_jvp(primals, tangents):
    """Solves the system with implicit differentiation."""
    x, guess = primals
    dx, _ = tangents

    # Temporary workaround: cast `dx` to float because `jax.jvp` expects integer
    # primals to have float0 derivatives. We would prefer them to just be
    # closed over by `fun` or fed into `nondiff_argnums`. Unfortunately that is
    # currently broken because custom_jvp can't handle batch (vmap) tracers in
    # that case. Once https://github.com/google/jax/pull/4112 merges we should
    # be able to remove this.
    def int_tangents_to_float(v):
      if np.issubdtype(v.dtype, np.integer):
        return np.zeros(v.shape, jax.dtypes.float0)
      else:
        return v

    dx = jax.tree_map(int_tangents_to_float, dx)

    # Solve the system.
    y = solve_forward(x, guess)
    # y = f(x, y)
    # dy = f_x(x,y) dx + f_y(x,y) dy
    # (I - f_y(x,y)) dy = f_x(x,y) dx
    # dy = (I - f_y(x,y))^-1 f_x(x,y) dx
    _, f_jvp_x = jax.jvp(lambda x: fun(x, y), (x,), (dx,))
    eye_minus_fy = jax.jacobian(lambda y: y - fun(x, y))(y)
    dy = jnp.linalg.solve(eye_minus_fy, f_jvp_x)
    return y, dy

  solve_forward.defjvp(solve_jvp)

  return solve_forward(x, guess)


@gin.configurable
def soft_maze_values(
    actions,
    target_state_index,
    estimates=None,
    temperature=1.0,
):
  """Compute the value function for a maximum-entropy-optimal policy in a maze.

  Tis function assumes we have an un-discounted reward function that is -1
  everywhere except for a specific goal location, which has 0 reward and
  instantly terminates the episode. It then returns the value of being at each
  state for an optimal entropy-regularized policy that attempts to maximize

    E[ sum_t r_t + temperature * H(policy) ].

  It does this by iterating the soft Bellman equations

    V(s) = t * log(mean(exp(Q(s, a)/t)))  (arithmetic mean over actions)
    Q(s, a) = r_{s,a} + E[V(s')]          (expectation over env dynamics)

  See https://arxiv.org/abs/1702.08165 ("RL with Deep Energy-Based Policies")

  Args:
    actions: <float32[num_states, num_states, num_actions]> which determines a
      distribution over where you end up if you take each given action. The
      policy will choose a distribution over actions to take at each state by
      considering the expected value of taking each action, but does not get to
      decide where it ends up within the selected action's distribution.
    target_state_index: State that terminates the episode.
    estimates: Initial estimates of values.
    temperature: Temperature of the soft updates, which controls how much
      implicit entropy regularization there is. For temperatures near zero, the
      policy becomes a deterministic one and the values represent shortest
      paths. For larger temperatures, the policy will have more entropy and
      paths may be longer than necessary. Larger temperatures are likely to be
      easier to differentiate through. (note: currently the output is NOT
        differentiable w.r.t. the temerature hyperparam.)

  Returns:
    values: <float[num_states]> giving the expected future reward of being in
      each state under the max-ent-optimal policy.
    q_values: <float[num_states, num_actions]> giving q values of each action.
    policy: <float[num_states, num_actions]> giving the probability of taking
      each action at each state.
  """
  num_states, _, num_actions = actions.shape
  if estimates is None:
    # Worst case values, assuming a connected graph: suppose the graph was a
    # linear chain, and the policy walks completely randomly from the start to
    # the end. Then the expected number of steps that it would take would be
    # `num_states**2` (see
    # https://en.wikipedia.org/wiki/Random_walk#One-dimensional_random_walk,
    # since this is equivalent to starting at 0 and walking to either
    # -num_states or num_states)
    estimates = jnp.full([num_states], -float(num_states)**2)

  def soft_bellman_backup(stuff, current_values):
    # Need to pass target_state_index because custom gradient machinery doesn't
    # understand it when closing over batched values.
    actions, target_state_index = stuff
    # Compute Q values for taking each action at each state
    # (every non-goal state has immediate reward -1).
    q_values = -1 + jnp.einsum("sta,t->sa", actions, current_values)
    # Compute value of current states by max-ent Bellman.
    new_v_values = (jax.scipy.special.logsumexp(q_values / temperature, -1) -
                    jnp.log(num_actions)) * temperature
    # Goal state is fixed to value 0 (episode terminates before the agent takes
    # any action).
    new_v_values = jax.ops.index_update(new_v_values,
                                        jax.ops.index[target_state_index], 0.0)
    return new_v_values

  # Must iterate at least num_states to guarantee everything is reachable;
  # we iterate a bit longer to make sure things converge.
  soft_v_values = iterative_fixed_point(
      soft_bellman_backup, (actions, target_state_index),
      estimates,
      iterations=num_states * 2)

  # Also extract the final Q values and entropy-regularized optimal policy.
  soft_q_values = -1 + jnp.einsum("sta,t->sa", actions, soft_v_values)
  policy = jax.nn.softmax(soft_q_values / temperature, -1)

  return soft_v_values, soft_q_values, policy


@gin.configurable
def loss_fn(
    model,
    example,
    static_metadata = None,  # pylint:disable=bad-whitespace
    num_goals = 1):
  """Computes the loss function for an automaton for a random maze goal.

  Args:
    model: Module that computes an additional action matrix for the maze agent.
    example: Tuple of a maze and an example id, which is used to select a goal
      location.
    static_metadata: Unused static metadata.
    num_goals: How many random goals to choose.

  Returns:
    Negative of average value across all (valid) possible initial states, along
    with some metrics.
  """
  assert static_metadata is None
  graph, example_id = example
  num_valid_nodes = graph.graph_metadata.num_nodes

  # Pick random targets.
  # TODO(ddjohnson) Remove the cast once JAX fixes its types.
  targets = jax.random.randint(
      jax.random.PRNGKey(typing.cast(int, example_id)), (num_goals,), 0,
      num_valid_nodes)

  # Compute automaton outputs, ordered as [source, dest, action]
  option_actions = model(
      graph.automaton_graph, dynamic_metadata=graph.graph_metadata).transpose(
          (1, 2, 0))
  static_num_nodes, _, num_options = option_actions.shape

  # Rewrite failure as staying in place, since we want to make sure the agent
  # can't do "terminate episode here" as an action. (Note: we could disallow
  # the automaton to take a fail action, but that wouldn't prevent the automaton
  # from just never accepting and waiting until the solve times out.)
  success_prob = jnp.sum(option_actions, axis=1, keepdims=True)
  option_actions_rewrite = (
      option_actions +
      (1 - success_prob) * jnp.eye(static_num_nodes)[:, :, None])
  # Only do the rewrite if there's missing probability mass. If we have too much
  # mass (from numerical issues) then just re-normalize.
  option_actions = jnp.where(success_prob < 1, option_actions_rewrite,
                             option_actions / success_prob)

  # Construct the node-to-node action matrix.
  primitive_actions = graph.edges.apply_add(
      in_array=jnp.eye(4),
      out_array=jnp.zeros([static_num_nodes, static_num_nodes, 4]),
      in_dims=(0,),
      out_dims=(0, 1))

  all_actions = jnp.concatenate([primitive_actions, option_actions], axis=-1)

  # Get soft values for each target.
  soft_v_values, _, policy = jax.vmap(soft_maze_values, (None, 0))(all_actions,
                                                                   targets)

  # Average over possible starting positions.
  valid_mask = jnp.arange(static_num_nodes) < num_valid_nodes
  average_expected_reward = jnp.sum(
      jnp.where(valid_mask[None, :], soft_v_values,
                jnp.zeros_like(soft_v_values))) / (
                    num_goals * num_valid_nodes)

  loss = -average_expected_reward

  # Compute additional metrics:
  # - Entropy of the automaton outputs
  valid_pair_mask = (valid_mask[:, None, None] & valid_mask[None, :, None])
  ok_mask = ((option_actions > 0) & valid_pair_mask)
  p_log_p = jnp.where(ok_mask, option_actions * jnp.log(option_actions), 0)
  option_entropy = -jnp.sum(p_log_p) / (4 * num_valid_nodes)

  # - Probability of taking an automaton action
  prob_of_choosing_any_option = (
      jnp.sum(jnp.where(valid_mask[None, :, None], policy[:, :, 4:], 0)) /
      (num_goals * num_valid_nodes))

  # - How distinct the automaton actions are, measured as the mutual information
  #   between destination and automaton action, conditioned on the source, and
  #   assuming we choose automaton actions uniformly at random.
  joint_action_and_destination = option_actions / num_options
  marginal_destinations = jnp.sum(joint_action_and_destination, -1)
  mutual_info = joint_action_and_destination * (
      jnp.log(joint_action_and_destination) + jnp.log(num_options) -
      jnp.log(marginal_destinations[:, :, None]))
  mutual_info = jnp.where((joint_action_and_destination > 0) & valid_pair_mask,
                          mutual_info, 0)
  mutual_info = jnp.sum(mutual_info) / num_valid_nodes

  return loss, {
      "option_entropy_of_dest": option_entropy,
      "prob_of_choosing_any_option": prob_of_choosing_any_option,
      "mutual_info_option_and_dest": mutual_info,
  }


@gin.configurable
def train(
    runner,
    dataset_paths = gin.REQUIRED,
    prefetch = 4,
    batch_size_per_device = gin.REQUIRED,
    validation_example_count = gin.REQUIRED,
):
  """Train the maze automaton.

  Args:
    runner: Helper object that runs the experiment.
    dataset_paths: Dictionary of dataset paths, with keys:
      - "train_dataset": Path to training dataset files.
      - "eval_dataset": Path to validation dataset files.
    prefetch: Maximum number of examples to prefetch in a background thread.
    batch_size_per_device: Batch size for each device.
    validation_example_count: How many examples to use when computing validation
      metrics.

  Returns:
    Optimizer at the end of training (for interactive debugging).
  """
  num_devices = jax.local_device_count()
  logging.info("Found %d devices: %s", num_devices, jax.devices())

  with contextlib.ExitStack() as exit_stack:
    logging.info("Setting up datasets...")
    raw_train_iterator = runner.build_sampling_iterator(
        dataset_paths["train_dataset"], example_type=graph_bundle.GraphBundle)

    raw_valid_iterator_factory = runner.build_one_pass_iterator_factory(
        dataset_paths["eval_dataset"],
        example_type=graph_bundle.GraphBundle,
        truncate_at=validation_example_count)

    # Add the example id into the example itself, so that we can use it to
    # randomly choose a goal.
    def reify_id(it):
      for item in it:
        yield dataclasses.replace(item, example=(item.example, item.example_id))

    def reify_id_and_batch(it):
      return data_loading.batch(
          reify_id(it), (num_devices, batch_size_per_device),
          remainder_behavior=data_loading.BatchRemainderBehavior.PAD_ZERO)

    train_iterator = reify_id_and_batch(raw_train_iterator)
    valid_iterator_factory = (
        lambda: reify_id_and_batch(raw_valid_iterator_factory()))

    if prefetch:
      train_iterator = exit_stack.enter_context(
          data_loading.ThreadedPrefetcher(train_iterator, prefetch))

    logging.info("Setting up model...")
    padding_config = maze_task.PADDING_CONFIG
    model_def = automaton_layer.FiniteStateGraphAutomaton.partial(
        static_metadata=padding_config.static_max_metadata,
        builder=maze_task.BUILDER)

    # Initialize parameters randomly.
    _, initial_params = model_def.init(
        jax.random.PRNGKey(int(time.time() * 1000)),
        graph_bundle.zeros_like_padded_example(padding_config).automaton_graph,
        dynamic_metadata=padding_config.static_max_metadata)

    model = flax.nn.Model(model_def, initial_params)
    optimizer = flax.optim.Adam().create(model)

    extra_artifacts = {
        "builder.pickle": maze_task.BUILDER,
    }

    return runner.training_loop(
        optimizer=optimizer,
        train_iterator=train_iterator,
        loss_fn=loss_fn,
        validation_fn=train_util.build_averaging_validator(
            loss_fn, valid_iterator_factory),
        extra_artifacts=extra_artifacts)
