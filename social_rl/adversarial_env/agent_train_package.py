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
"""Abstracts features common to training adversarial environments and agents."""

import os

from absl import logging

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from social_rl.adversarial_env import adversarial_eval
from social_rl.multiagent_tfagents import multigrid_networks


class AgentTrainPackage(object):
  """Combines an agent with its policy, replay buffer, and checkpointer."""

  def __init__(self,
               env,
               global_step,
               root_dir,
               step_metrics,
               name='Agent',
               is_environment=False,
               use_tf_functions=True,
               max_steps=250,
               replace_reward=True,
               non_negative_regret=False,
               id_num=0,
               block_budget_weight=0.,

               # Architecture hparams
               use_rnn=True,
               learning_rate=1e-4,
               actor_fc_layers=(32, 32),
               value_fc_layers=(32, 32),
               lstm_size=(128,),
               conv_filters=8,
               conv_kernel=3,
               scalar_fc=5,
               entropy_regularization=0.,
               xy_dim=None,

               # Training & logging settings
               num_epochs=25,
               num_eval_episodes=5,
               num_parallel_envs=5,
               replay_buffer_capacity=1001,
               debug_summaries=True,
               summarize_grads_and_vars=True,):
    """Initializes agent, replay buffer, metrics, and checkpointing.

    Args:
      env: An AdversarialTfPyEnvironment with specs and advesary specs.
      global_step: A tf variable tracking the global step.
      root_dir: Path to directory where metrics and checkpoints should be saved.
      step_metrics: A list of tf-agents metrics which represent the x-axis
        during training, such as the number of episodes or the number of
        environment steps.
      name: The name of this agent, e.g. 'Adversary'.
      is_environment: If True, will use the adversary specs from the environment
        and construct a network with additional inputs for the adversary.
      use_tf_functions: If True, will use tf.function to wrap the agent's train
        function.
      max_steps: The maximum number of steps the agent is allowed to interact
        with the environment in every data collection loop.
      replace_reward: If False, will not modify the reward stored in the agent's
        trajectories. This means the agent will be trained with the default
        environment reward rather than regret.
      non_negative_regret: If True, will ensure that the regret reward cannot
        be below 0.
      id_num: The ID number of this agent within the population of agents of the
        same type. I.e. this is adversary agent 3.
      block_budget_weight: Weight to place on the adversary's block budget
        reward. Default is 0 for no block budget.
      use_rnn: If True, will use an RNN within the network architecture.
      learning_rate: The learning rate used to initialize the optimizer for this
        agent.
      actor_fc_layers: The number and size of fully connected layers in the
        policy.
      value_fc_layers: The number and size of fully connected layers in the
        critic / value network.
      lstm_size: The number of LSTM cells in the RNN.
      conv_filters: The number of convolution filters.
      conv_kernel: The width of the convolution kernel.
      scalar_fc: The width of the fully-connected layer which inputs a scalar.
      entropy_regularization: Entropy regularization coefficient.
      xy_dim: Certain adversaries take in the current (x,y) position as a
        one-hot vector. In this case, the maximum value for x or y is required
        to create the one-hot representation.
      num_epochs: Number of epochs for computing PPO policy updates.
      num_eval_episodes: Number of evaluation episodes be eval step, used as
        batch size to initialize eval metrics.
      num_parallel_envs: Number of parallel environments used in trainin, used
        as batch size for training metrics and rewards.
      replay_buffer_capacity: Capacity of this agent's replay buffer.
      debug_summaries: Log additional summaries from the PPO agent.
      summarize_grads_and_vars: If True, logs gradient norms and variances in
        PPO agent.
    """
    self.name = name
    self.id = id_num
    self.max_steps = max_steps
    self.is_environment = is_environment
    self.replace_reward = replace_reward
    self.non_negative_regret = non_negative_regret
    self.block_budget_weight = block_budget_weight

    with tf.name_scope(self.name):
      self.optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate=learning_rate)

      logging.info('\tCalculating specs and building networks...')
      if is_environment:
        self.time_step_spec = env.adversary_time_step_spec
        self.action_spec = env.adversary_action_spec
        self.observation_spec = env.adversary_observation_spec

        (self.actor_net,
         self.value_net) = multigrid_networks.construct_multigrid_networks(
             self.observation_spec, self.action_spec, use_rnns=use_rnn,
             actor_fc_layers=actor_fc_layers, value_fc_layers=value_fc_layers,
             lstm_size=lstm_size, conv_filters=conv_filters,
             conv_kernel=conv_kernel, scalar_fc=scalar_fc,
             scalar_name='time_step',
             scalar_dim=self.observation_spec['time_step'].maximum + 1,
             random_z=True, xy_dim=xy_dim)
      else:
        self.time_step_spec = env.time_step_spec()
        self.action_spec = env.action_spec()
        self.observation_spec = env.observation_spec()

        (self.actor_net,
         self.value_net) = multigrid_networks.construct_multigrid_networks(
             self.observation_spec, self.action_spec, use_rnns=use_rnn,
             actor_fc_layers=actor_fc_layers, value_fc_layers=value_fc_layers,
             lstm_size=lstm_size, conv_filters=conv_filters,
             conv_kernel=conv_kernel, scalar_fc=scalar_fc)

      self.tf_agent = ppo_clip_agent.PPOClipAgent(
          self.time_step_spec,
          self.action_spec,
          self.optimizer,
          actor_net=self.actor_net,
          value_net=self.value_net,
          entropy_regularization=entropy_regularization,
          importance_ratio_clipping=0.2,
          normalize_observations=False,
          normalize_rewards=False,
          use_gae=True,
          num_epochs=num_epochs,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step)
      self.tf_agent.initialize()
      self.eval_policy = self.tf_agent.policy
      self.collect_policy = self.tf_agent.collect_policy

      logging.info('\tAllocating replay buffer ...')
      self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          self.tf_agent.collect_data_spec,
          batch_size=num_parallel_envs,
          max_length=replay_buffer_capacity)
      logging.info('\t\tRB capacity: %i', self.replay_buffer.capacity)
      self.final_reward = tf.zeros(shape=(num_parallel_envs), dtype=tf.float32)
      self.enemy_max = tf.zeros(shape=(num_parallel_envs), dtype=tf.float32)

      # Creates train metrics
      self.step_metrics = step_metrics
      self.train_metrics = step_metrics + [
          tf_metrics.AverageEpisodeLengthMetric(
              batch_size=num_parallel_envs,
              name=name+'_AverageEpisodeLength')
      ]
      self.eval_metrics = [
          tf_metrics.AverageEpisodeLengthMetric(
              batch_size=num_eval_episodes, name=name+'_AverageEpisodeLength')
      ]
      if is_environment:
        self.env_train_metric = adversarial_eval.AdversarialEnvironmentScalar(
            batch_size=num_parallel_envs, name=name + '_AdversaryReward')
        self.env_eval_metric = adversarial_eval.AdversarialEnvironmentScalar(
            batch_size=num_eval_episodes, name=name + '_AdversaryReward')
      else:
        self.train_metrics.append(tf_metrics.AverageReturnMetric(
            batch_size=num_parallel_envs, name=name+'_AverageReturn'))
        self.eval_metrics.append(tf_metrics.AverageReturnMetric(
            batch_size=num_eval_episodes, name=name+'_AverageReturn'))

      self.metrics_group = metric_utils.MetricsGroup(
          self.train_metrics, name + '_train_metrics')
      self.observers = self.train_metrics + [self.replay_buffer.add_batch]

      self.train_dir = os.path.join(root_dir, 'train', name, str(id_num))
      self.eval_dir = os.path.join(root_dir, 'eval', name, str(id_num))
      self.train_checkpointer = common.Checkpointer(
          ckpt_dir=self.train_dir,
          agent=self.tf_agent,
          global_step=global_step,
          metrics=self.metrics_group,
          )
      self.policy_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(self.train_dir, 'policy'),
          policy=self.eval_policy,
          global_step=global_step)
      self.saved_model = policy_saver.PolicySaver(
          self.eval_policy, train_step=global_step)
      self.saved_model_dir = os.path.join(
          root_dir, 'policy_saved_model', name, str(id_num))

      self.train_checkpointer.initialize_or_restore()

      if use_tf_functions:
        self.tf_agent.train = common.function(self.tf_agent.train,
                                              autograph=False)

      self.total_loss = None
      self.extra_loss = None
      self.loss_divergence_counter = 0

  def train_step(self):
    """Collects trajectories and trains the agent on them."""
    trajectories = self.get_trajectories()
    with tf.name_scope(self.name):
      return self.tf_agent.train(experience=trajectories)

  def get_trajectories(self):
    """Retrieves trajectories from replay buffer. Replaces reward if necessary.

    Returns:
      A TF-Agents Trajectory object with all experienced trajectories.
    """
    trajectories = self.replay_buffer.gather_all()

    if not self.replace_reward:
      return trajectories

    if self.is_environment:
      idx_remove = tf.where(trajectories.is_last())
      replaced_reward = tf.sparse.SparseTensor(idx_remove, self.final_reward,
                                               trajectories.reward.shape)
      true_reward = tf.sparse.to_dense(replaced_reward, default_value=0.)
    else:
      ep_len = trajectories.reward.shape[1]
      tiled_enemy_max = tf.tile(
          tf.reshape(self.enemy_max, (-1, 1)), (1, ep_len))
      true_reward = trajectories.reward - (tiled_enemy_max / ep_len)

    return replace_traj_reward(trajectories, true_reward)


def replace_traj_reward(traj, reward):
  return trajectory.Trajectory(
      step_type=traj.step_type,
      observation=traj.observation,
      action=traj.action,
      policy_info=traj.policy_info,
      next_step_type=traj.next_step_type,
      reward=reward,
      discount=traj.discount)
