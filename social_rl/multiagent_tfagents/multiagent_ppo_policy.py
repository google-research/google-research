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

"""A multi-agent meta-controller policy that runs policies for agents within it.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ppo import ppo_utils
from tf_agents.distributions import reparameterized_sampling
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts_library

tfd = tfp.distributions


@gin.configurable
class MultiagentPPOPolicy(tf_policy.TFPolicy):
  """A Multiagent PPO policy that aggregates actions of all agents.

  This centralized policy is responsible for distributing observations to the
  appropriate agent and collecting their actions and states into an aggregated
  representation.

  When the networks have state (RNNs, LSTMs) you must be careful to pass the
  state for the actor network to `action()` and the state of the value network
  to `apply_value_network()`. Use `get_initial_value_state()` to access
  the state of the value network. The states of all agents are concatenated
  together and distributed by this policy.
  """

  # Building policy out of sub-policies, so pylint:disable=protected-access

  def __init__(self,
               agent_policies,
               time_step_spec=None,
               action_spec=None,
               clip=True,
               collect=True,
               inactive_agent_ids=None):
    """Builds a Multiagent policy out of several other agents' policies.

    Note this policy does not have an observation normalizer; it relies on the
    agents within it to normalize their own observations.

    Args:
      agent_policies: An array of already-built policies for each agent.
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      clip: Whether to clip actions to spec before returning them.  Default
        True. Most policy-based algorithms (PCL, PPO, REINFORCE) use unclipped
        continuous actions for training.
      collect: If True, creates ops for actions_log_prob, value_preds, and
        action_distribution_params. (default True)
      inactive_agent_ids: Integer IDs of agents who will not train or act in the
        environment, but will simply return a no-op action.

    Raises:
      ValueError: if actor_network or value_network is not of type
        tf_agents.networks.network.Network.
    """
    self._agent_policies = agent_policies
    self.n_agents = len(agent_policies)
    self.inactive_agent_ids = inactive_agent_ids
    if inactive_agent_ids:
      self.learning_agents = \
        [a for a in range(self.n_agents) if a not in inactive_agent_ids]
    else:
      self.learning_agents = range(self.n_agents)

    for agent_policy in agent_policies:
      if collect and isinstance(agent_policy, greedy_policy.GreedyPolicy):
        raise ValueError('Trying to create a collecting meta-agent policy '
                         'with greedy agents')
      elif not collect and not isinstance(
          agent_policy, greedy_policy.GreedyPolicy) and agent_policy._collect:
        raise ValueError('Trying to create an eval meta-agent policy with '
                         'collecting agents')

    self._collect = collect

    info_spec = self._make_info_spec(time_step_spec)

    # Make multi-agent policy_state spec
    # All policies must have the same state spec.
    n_agents = len(agent_policies)

    def make_multi_policy_state_spec(spec):
      new_shape = (n_agents,) + spec.shape
      return tensor_spec.TensorSpec(shape=new_shape, dtype=spec.dtype,
                                    name=spec.name)

    if collect:
      single_policy_state_spec = agent_policies[0]._actor_network.state_spec
    else:
      single_policy_state_spec = agent_policies[
          0]._wrapped_policy._actor_network.state_spec
    multi_policy_state_spec = tf.nest.map_structure(
        make_multi_policy_state_spec, single_policy_state_spec)
    multi_policy_state_spec = {'actor_network_state': multi_policy_state_spec}

    super(MultiagentPPOPolicy, self).__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=multi_policy_state_spec,
        info_spec=info_spec,
        clip=clip)

    self._action_fn = self._action  # Necessary to override _action in child

  def _make_info_spec(self, time_step_spec):
    # Make multi-agent info spec
    if self._collect:
      info_spec = [p.info_spec for p in self._agent_policies]
    else:
      info_spec = ()

    return info_spec

  def _apply_actor_network(self, time_step, policy_states, training=False):
    actions = [None] * self.n_agents
    new_states = {'actor_network_state': [None] * self.n_agents}
    for agent_id, policy in enumerate(self._agent_policies):
      # Fixed agents do not act. Used for debugging
      if self.inactive_agent_ids and agent_id in self.inactive_agent_ids:
        actions[agent_id] = tf.ones_like(time_step.discount, dtype=tf.int64) * 6
        new_states['actor_network_state'][agent_id] = \
            policy_states['actor_network_state'][agent_id]
        continue

      agent_time_step = self._get_obs_for_agent(time_step, agent_id)
      if isinstance(policy, greedy_policy.GreedyPolicy):
        policy = policy._wrapped_policy
      agent_policy_state = [
          state[:, agent_id] for state in policy_states['actor_network_state']
      ]
      actions[agent_id], new_states['actor_network_state'][agent_id] = \
          policy._apply_actor_network(
              agent_time_step,
              agent_policy_state, training)
    new_states = {
        'actor_network_state': [
            tf.stack(i, axis=1)
            for i in list(zip(*new_states['actor_network_state']))
        ]
    }
    return actions, new_states

  def _get_obs_for_agent(self, time_step, agent_id):
    """Pull out the observation of a particular agent."""

    def get_single_obs(observation):
      if len(observation.shape) == 2:
        # Need at least one additional dim besides batch
        return tf.expand_dims(observation[:, agent_id], -1)
      else:
        return observation[:, agent_id]

    single_obs = tf.nest.map_structure(get_single_obs, time_step.observation)

    return ts_library.TimeStep(
        time_step.step_type, time_step.reward, time_step.discount, single_obs)

  def _variables(self):
    var_list = []
    for policy in self._agent_policies:
      if isinstance(policy, greedy_policy.GreedyPolicy):
        policy = policy._wrapped_policy
      var_list += policy._variables()
    return var_list

  def _distribution(self, time_step, policy_state, training=False):
    # Actor network outputs a list of distributions or actions (one for each
    # agent), and a list of policy states for each agent
    actions_or_distributions, policy_state = self._apply_actor_network(
        time_step, policy_state, training=training)

    def _to_distribution(action_or_distribution):
      if isinstance(action_or_distribution, tf.Tensor):
        # This is an action tensor, so wrap it in a deterministic distribution.
        return tfp.distributions.Deterministic(loc=action_or_distribution)
      return action_or_distribution

    distributions = tf.nest.map_structure(_to_distribution,
                                          actions_or_distributions)

    # Prepare policy_info.
    if self._collect:
      policy_info = ppo_utils.get_distribution_params(distributions, False)

      # Wrap policy info to be comptabile with new spec
      for a in range(len(policy_info)):
        if not self.inactive_agent_ids or a not in self.inactive_agent_ids:
          policy_info[a] = {'dist_params': policy_info[a]}

      # Fake logits for fixed agents.
      if self.inactive_agent_ids and self.learning_agents:
        for a in self.inactive_agent_ids:
          policy_info[a] = {'dist_params': {'logits': tf.zeros_like(
              policy_info[self.learning_agents[0]]['dist_params']['logits'])}}

      # PolicyStep has actions, state, info
      step_result = policy_step.PolicyStep(distributions, policy_state,
                                           policy_info)
    else:
      # I was not able to use a GreedyPolicy wrapper and also override _action,
      # so I replicated the greedy functionality here.
      def dist_fn(dist):
        try:
          greedy_action = dist.mode()
        except NotImplementedError:
          raise ValueError("Your network's distribution does not implement "
                           "mode making it incompatible with a greedy policy.")

        return greedy_policy.DeterministicWithLogProb(loc=greedy_action)

      actions = tf.nest.map_structure(dist_fn, distributions)
      step_result = policy_step.PolicyStep(actions, policy_state, ())

    return step_result

  # Subclasses MAY optionally override _action.
  def _action(self, time_step, policy_state, seed):
    """Implementation of `action`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A Tensor, or a nested dict, list or tuple of Tensors
        representing the previous policy_state.
      seed: Seed to use if action performs sampling (optional).

    Returns:
      A `PolicyStep` named tuple containing:
        `action`: An action Tensor matching the `action_spec()`.
        `state`: A policy state tensor to be fed into the next call to action.
        `info`: Optional side information such as action log probabilities.
    """
    seed_stream = tfp.util.SeedStream(seed=seed, salt='ppo_policy')
    distribution_step = self._distribution(time_step, policy_state)
    actions = tf.nest.map_structure(
        lambda d: reparameterized_sampling.sample(d, seed=seed_stream()),
        distribution_step.action)
    info = distribution_step.info

    if self.emit_log_probability:
      try:
        log_probability = tf.nest.map_structure(lambda a, d: d.log_prob(a),
                                                actions,
                                                distribution_step.action)
        info = policy_step.set_log_probability(info, log_probability)
      except:
        raise TypeError('%s does not support emitting log-probabilities.' %
                        type(self).__name__)

    # Stack actions into multi-agent action
    # actions = tf.stack(actions, axis=1)

    return distribution_step._replace(action=actions, info=info)
