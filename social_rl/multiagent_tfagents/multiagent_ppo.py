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

"""A multi-agent PPO agent. Each agent's policy is independent of the others."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import gin
import numpy as np
import tensorflow as tf  # pylint:disable=g-explicit-tensorflow-version-import
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory as traj_lib

from social_rl.multiagent_tfagents import multiagent_ppo_policy
from social_rl.multiagent_tfagents import multigrid_networks


@gin.configurable
class MultiagentPPO(tf_agent.TFAgent):
  """A PPO Agent implementing the clipped probability ratios."""

  def __init__(
      self,
      time_step_spec,
      action_spec,
      # Specific to multi-agent case
      n_agents,
      learning_rate=1e-4,
      # Specific to multi-grid agents
      actor_fc_layers=(32, 32),
      value_fc_layers=(32, 32),
      lstm_size=(128,),
      conv_filters=8,
      conv_kernel=3,
      direction_fc=5,
      # Modifying agents
      inactive_agent_ids=tuple(),
      non_learning_agents=tuple(),
      # PPO Clip agent params
      importance_ratio_clipping=0.0,
      lambda_value=0.95,
      discount_factor=0.99,
      entropy_regularization=0.05,
      policy_l2_reg=0.0,
      value_function_l2_reg=0.0,
      shared_vars_l2_reg=0.0,
      value_pred_loss_coef=0.5,
      num_epochs=25,
      use_gae=False,
      use_td_lambda_return=False,
      normalize_rewards=True,
      reward_norm_clipping=10.0,
      normalize_observations=True,
      log_prob_clipping=0.0,
      gradient_clipping=None,
      check_numerics=False,
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=None,
      network_build_fn=multigrid_networks.construct_multigrid_networks,
      policy_class=multiagent_ppo_policy.MultiagentPPOPolicy,
      agent_class=ppo_clip_agent.PPOClipAgent,
      name='MultiagentPPO'):
    """Creates a centralized controller agent that creates several PPO Agents.

    Note that all architecture params apply to each of the sub-agents created.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      n_agents: The number of agents in this environment.
      learning_rate: Initial learning rate for all agents.
      actor_fc_layers: Number and size of fully-connected layers in the actor.
      value_fc_layers: Number and size of fully-connected layers in the critic.
      lstm_size: Number of cells in the LSTM in the actor and critic.
      conv_filters: Number of convolutional filters.
      conv_kernel: Size of the convolutional kernel.
      direction_fc: Number of fully-connected neurons connecting the one-hot
        direction to the main LSTM.
      inactive_agent_ids: Integer IDs of agents who will not train or act in the
        environment, but will simply return a no-op action.
      non_learning_agents: Integer IDs of agents who will not train, but still
        act in the environment.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation.
      entropy_regularization: Coefficient for entropy regularization loss term.
      policy_l2_reg: Coefficient for l2 regularization of unshared policy
        weights.
      value_function_l2_reg: Coefficient for l2 regularization of unshared value
        function weights.
      shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
        between the policy and value functions.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss.
      num_epochs: Number of epochs for computing policy updates.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function. (td_lambda_return = gae_advantage +
        value_predictions)
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards.
      reward_norm_clipping: Value above and below to clip normalized reward.
      normalize_observations: If true, keeps moving mean and variance of
        observations and normalizes incoming observations.
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      check_numerics: If true, adds tf.debugging.check_numerics to help find NaN
        / Inf values. For debugging only.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      network_build_fn: Function for constructing agent encoding architecture.
      policy_class: Function for creating individual agent policies.
      agent_class: Function for creating individual agents.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the actor_net is not a DistributionNetwork.
    """
    self.n_agents = n_agents
    self.inactive_agent_ids = inactive_agent_ids
    self.non_learning_agents = non_learning_agents

    # Get single-agent specs
    (single_obs_spec, single_time_step_spec,
     single_action_spec) = self.get_single_agent_specs(time_step_spec,
                                                       action_spec)

    # Make baby agents
    self.agents = [None] * self.n_agents
    self.optimizers = [None] * self.n_agents
    for agent_id in range(self.n_agents):
      with tf.name_scope('agent_' + str(agent_id)):
        self.optimizers[agent_id] = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate)

        # Build actor and critic networks
        actor_net, value_net = network_build_fn(
            single_obs_spec,
            single_action_spec,
            actor_fc_layers=actor_fc_layers,
            value_fc_layers=value_fc_layers,
            lstm_size=lstm_size,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            scalar_fc=direction_fc)

        logging.info('Creating agent %d...', agent_id)
        self.agents[agent_id] = agent_class(
            single_time_step_spec,
            single_action_spec,
            self.optimizers[agent_id],
            actor_net=actor_net,
            value_net=value_net,
            entropy_regularization=entropy_regularization,
            importance_ratio_clipping=0.2,
            normalize_observations=False,
            normalize_rewards=False,
            use_gae=True,
            num_epochs=num_epochs,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            compute_value_and_advantage_in_train=True)
        self.agents[agent_id].initialize()

    with tf.name_scope('meta_agent'):
      # Initialize policies
      self._policies = [self.agents[a].policy for a in range(self.n_agents)]
      policy = policy_class(
          self._policies,
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          clip=False,
          collect=False,
          inactive_agent_ids=inactive_agent_ids)

      self._collect_policies = [
          self.agents[a].collect_policy for a in range(self.n_agents)
      ]
      collect_policy = policy_class(
          self._collect_policies,
          time_step_spec=time_step_spec,
          action_spec=action_spec,
          clip=False,
          collect=True,
          inactive_agent_ids=inactive_agent_ids)

      super(MultiagentPPO, self).__init__(
          time_step_spec,
          action_spec,
          policy,
          collect_policy,
          train_sequence_length=None,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=train_step_counter)

    self._global_step = train_step_counter
    print('Finished constructing multi-agent PPO')

  def get_single_agent_specs(self, time_step_spec, action_spec):
    """Get single agent version of environment specs to feed to baby agents."""

    def make_single_agent_spec(spec):
      if len(spec.shape) == 1:
        shape = 1
      else:
        shape = spec.shape[1:]
      return tensor_spec.BoundedTensorSpec(
          shape=shape,
          name=spec.name,
          minimum=spec.minimum,
          maximum=spec.maximum,
          dtype=spec.dtype)

    single_obs_spec = tf.nest.map_structure(make_single_agent_spec,
                                            time_step_spec.observation)
    single_reward_spec = tensor_spec.TensorSpec(
        shape=(), dtype=time_step_spec.reward.dtype, name='reward')
    single_time_step_spec = ts.TimeStep(time_step_spec.step_type,
                                        single_reward_spec,
                                        time_step_spec.discount,
                                        single_obs_spec)
    single_action_spec = action_spec[0]
    return single_obs_spec, single_time_step_spec, single_action_spec

  def extract_single_agent_trajectory(self, agent_id, trajectory):
    """Pull a single agent's experience out of a trajectory and makes a new one.

    Note: trajectory format (B = batch size, T = # timesteps, N = # agents)
      observation: DictWrapper. Need to pull out agent's observations.
      step_type: (B, T). Agent's is the same as global step type.
      next_step_type: (B, T). Agent's is the same as global step type.
      action: (B, T, N). Need to pull out individual agent's action
      policy_info: ListWrapper of length N containing logits for each agent.
      reward: (B, T, N). Need to pull out individual agent's rewards.
      discount: (B, T).

    Args:
      agent_id: Integer ID for this agent.
      trajectory: As above

    Returns:
      A new trajectory for this agent only
    """
    action = trajectory.action[agent_id]
    reward = trajectory.reward[:, :, agent_id]
    policy_info = trajectory.policy_info[agent_id]

    def get_single_observation(observation):
      if len(observation.shape) == 3:
        # Need at least one additional dim besides batch
        return tf.expand_dims(observation[:, :, agent_id], 2)
      else:
        return observation[:, :, agent_id]

    observation = tf.nest.map_structure(get_single_observation,
                                        trajectory.observation)

    agent_trajectory = traj_lib.Trajectory(
        step_type=trajectory.step_type,
        observation=observation,
        action=action,
        policy_info=policy_info,
        next_step_type=trajectory.next_step_type,
        reward=reward,
        discount=trajectory.discount)

    return agent_trajectory

  def _train(self, experience, weights):
    """Separate experience for each agent, feed to agents, and train each."""
    agent_losses = []
    for a in range(self.n_agents):
      # Fixed agents do not train
      if a in self.inactive_agent_ids:
        continue
      if a in self.non_learning_agents:
        agent_losses.append(tf_agent.LossInfo(loss=0, extra=None))
        continue

      agent_experience = self.extract_single_agent_trajectory(a, experience)
      with tf.name_scope('agent' + str(a) + '_logging/'):
        agent_losses.append(self.agents[a].train(experience=agent_experience))

    total_loss = np.sum([l.loss for l in agent_losses])
    loss_info = tf_agent.LossInfo(loss=total_loss, extra=agent_losses)

    return loss_info
