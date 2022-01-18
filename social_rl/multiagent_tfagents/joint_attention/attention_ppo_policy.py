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

"""A multi-agent meta-controller policy that runs policies for agents within it.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ppo import ppo_policy
from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step

from social_rl.multiagent_tfagents import multiagent_ppo_policy

tfd = tfp.distributions


class AttentionPPOPolicy(ppo_policy.PPOPolicy):
  """A modification of tf_agents PPOPolicy that returns attention info."""

  def _distribution(self, time_step, policy_state, training=False):
    if not policy_state:
      policy_state = {'actor_network_state': (), 'value_network_state': ()}
    else:
      policy_state = policy_state.copy()

    if 'actor_network_state' not in policy_state:
      policy_state['actor_network_state'] = ()
    if 'value_network_state' not in policy_state:
      policy_state['value_network_state'] = ()

    new_policy_state = {'actor_network_state': (), 'value_network_state': ()}

    (distributions, new_policy_state['actor_network_state'], _) = (
        self._apply_actor_network(
            time_step, policy_state['actor_network_state'], training=training))

    if self._collect:
      policy_info = {
          'dist_params':
              ppo_utils.get_distribution_params(
                  distributions,
                  legacy_distribution_network=isinstance(
                      self._actor_network, network.DistributionNetwork))
      }

      if not self._compute_value_and_advantage_in_train:
        # If value_prediction is not computed in agent.train it needs to be
        # computed and saved here.
        (policy_info['value_prediction'],
         new_policy_state['value_network_state']) = self.apply_value_network(
             time_step.observation,
             time_step.step_type,
             value_state=policy_state['value_network_state'],
             training=False)
    else:
      policy_info = ()

    if (not new_policy_state['actor_network_state'] and
        not new_policy_state['value_network_state']):
      new_policy_state = ()
    elif not new_policy_state['value_network_state']:
      del new_policy_state['value_network_state']
    elif not new_policy_state['actor_network_state']:
      del new_policy_state['actor_network_state']

    return policy_step.PolicyStep(distributions, new_policy_state, policy_info)


@gin.configurable
class AttentionMultiagentPPOPolicy(multiagent_ppo_policy.MultiagentPPOPolicy):
  """A modification of MultiagentPPOPolicy that returns attention info."""

  def __init__(
      self,
      *args,
      use_stacks=False,
      **kwargs,
  ):
    """Creates a centralized controller agent that uses joint attention.

    Args:
      *args: See superclass.
      use_stacks: Use ResNet stacks in image encoder (compresses the image).
      **kwargs: See superclass.
    """
    self.use_stacks = use_stacks
    super(AttentionMultiagentPPOPolicy, self).__init__(*args, **kwargs)

  # Building policy out of sub-policies, so pylint:disable=protected-access
  def _make_info_spec(self, time_step_spec):
    # Make multi-agent info spec
    if self._collect:
      info_spec = []
      for p in self._agent_policies:
        agent_info_spec = p.info_spec
        if self.use_stacks:
          image_shape = [
              i // 4 for i in time_step_spec.observation['image'].shape[1:3]
          ]
        else:
          image_shape = time_step_spec.observation['image'].shape[1:3]
        state_spec = tensor_spec.BoundedTensorSpec(
            image_shape, dtype=tf.float32, minimum=0, maximum=1)
        agent_info_spec['attention_weights'] = state_spec
        info_spec.append(agent_info_spec)
      info_spec = tuple(info_spec)
    else:
      info_spec = ()

    return info_spec

  def _apply_actor_network(self, time_step, policy_states, training=False):
    actions = [None] * self.n_agents
    new_states = {'actor_network_state': [None] * self.n_agents}
    attention_weights = [None] * self.n_agents
    for agent_id, policy in enumerate(self._agent_policies):
      # Fixed agents do not act. Used for debugging
      if self.inactive_agent_ids and agent_id in self.inactive_agent_ids:
        actions[agent_id] = tf.ones_like(time_step.discount, dtype=tf.int64) * 6
        new_states['actor_network_state'][agent_id] = policy_states[
            'actor_network_state'][agent_id]
        continue

      agent_time_step = self._get_obs_for_agent(time_step, agent_id)
      if isinstance(policy, greedy_policy.GreedyPolicy):
        policy = policy._wrapped_policy
      agent_policy_state = [
          state[:, agent_id] for state in policy_states['actor_network_state']
      ]
      actions[agent_id], new_states['actor_network_state'][
          agent_id], attention_weights[agent_id] = policy._apply_actor_network(
              agent_time_step, agent_policy_state, training)
    actions = tuple(actions)
    new_states = {
        'actor_network_state': [
            tf.stack(i, axis=1)
            for i in list(zip(*new_states['actor_network_state']))
        ]
    }
    return actions, new_states, attention_weights

  def _distribution(self, time_step, policy_state, training=False):
    # Actor network outputs a list of distributions or actions (one for each
    # agent), and a list of policy states for each agent
    actions_or_distributions, policy_state, attention_weights = self._apply_actor_network(
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
      policy_info = ppo_utils.get_distribution_params(
          distributions,
          False
      )

      # Wrap policy info to be comptabile with new spec
      policy_info = list(policy_info)
      for a in range(len(policy_info)):
        if not self.inactive_agent_ids or a not in self.inactive_agent_ids:
          policy_info[a] = {'dist_params': policy_info[a]}
        policy_info[a].update({'attention_weights': attention_weights[a]})

      # Fake logits for fixed agents.
      if self.inactive_agent_ids and self.learning_agents:
        for a in self.inactive_agent_ids:
          policy_info[a] = {
              'dist_params': {
                  'logits':
                      tf.zeros_like(policy_info[self.learning_agents[0]]
                                    ['dist_params']['logits'])
              }
          }
      policy_info = tuple(policy_info)

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
                           'mode making it incompatible with a greedy policy.')

        return greedy_policy.DeterministicWithLogProb(loc=greedy_action)

      actions = tf.nest.map_structure(dist_fn, distributions)
      step_result = policy_step.PolicyStep(actions, policy_state, ())

    return step_result
