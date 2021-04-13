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

from typing import Optional, Text
from absl import logging

import gin

import tensorflow as tf  # pylint:disable=g-explicit-tensorflow-version-import
from tf_agents.agents import data_converter
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import tensor_normalizer

from social_rl.multiagent_tfagents import multiagent_ppo

from social_rl.multiagent_tfagents.joint_attention import attention_networks
from social_rl.multiagent_tfagents.joint_attention import attention_ppo_agent
from social_rl.multiagent_tfagents.joint_attention import attention_ppo_policy


@gin.configurable
class MultiagentAttentionPPO(multiagent_ppo.MultiagentPPO):
  """A modification of the multiagent PPO agent that computes joint attention."""

  def __init__(
      self,
      *args,
      attention_bonus_type='kld',
      bonus_ratio=0.0,
      bonus_timescale=1,
      unscaled_bonus_metric=None,
      scaled_bonus_metric=None,
      **kwargs,
  ):
    """Creates a centralized controller agent that uses joint attention.

    Args:
      *args: See superclass.
      attention_bonus_type: Method for computing bonus rewards (kld, jsd).
      bonus_ratio: Final multiplier for bonus rewards.
      bonus_timescale: How to to linearly scale bonus rewards.
      unscaled_bonus_metric: (Optional) MultiagentScalar for raw attention
          bonus rewards.
      scaled_bonus_metric: (Optional) MultiagentScalar for scaled attention
          bonus rewards.
      **kwargs: See superclass.
    """
    self.attention_bonus_type = attention_bonus_type
    self._bonus_ratio = bonus_ratio
    self._bonus_timescale = bonus_timescale
    self._unscaled_bonus_metric = unscaled_bonus_metric
    self._scaled_bonus_metric = scaled_bonus_metric

    super(MultiagentAttentionPPO, self).__init__(
        *args,
        network_build_fn=attention_networks.construct_attention_networks,
        policy_class=attention_ppo_policy.AttentionMultiagentPPOPolicy,
        agent_class=attention_ppo_agent.AttentionPPOAgent,
        **kwargs)

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
    agent_trajectory = super().extract_single_agent_trajectory(agent_id,
                                                               trajectory)
    reward = trajectory.reward[:, :, agent_id]
    policy_info = trajectory.policy_info[agent_id]
    if 'attention_weights' in policy_info:
      p = policy_info['attention_weights']
      eps = tf.keras.backend.epsilon()
      p += eps
      raw_bonus = 0
      score_multiplier = tf.cast(
          tf.clip_by_value(self._global_step, 0, self._bonus_timescale) /
          self._bonus_timescale * self._bonus_ratio, tf.float32)
      for j, other_policy_info in enumerate(trajectory.policy_info):
        if j == agent_id:
          continue
        q = other_policy_info['attention_weights']
        q += eps
        if self.attention_bonus_type.lower() == 'kld':
          raw_bonus -= tf.reduce_sum(p * tf.math.log(p / q), axis=(-1, -2))
        elif self.attention_bonus_type.lower() == 'jsd':
          m = (p + q) / 2
          raw_bonus -= (
              tf.reduce_sum(p * tf.math.log(p / m), axis=(-1, -2)) +
              tf.reduce_sum(q * tf.math.log(q / m), axis=(-1, -2))) / 2
        else:
          raise NotImplementedError
      scaled_bonus = raw_bonus * score_multiplier
      reward += scaled_bonus
      if self._unscaled_bonus_metric:
        self._unscaled_bonus_metric(raw_bonus, agent_id)
      if self._scaled_bonus_metric:
        self._scaled_bonus_metric(scaled_bonus, agent_id)

    agent_trajectory.replace(reward=reward)

    return agent_trajectory


class AttentionPPOAgent(ppo_agent.PPOAgent):
  """A modification of tf_agents PPOAgent that returns attention info."""

  def __init__(
      self,
      time_step_spec,
      action_spec,
      optimizer = None,
      actor_net = None,
      value_net = None,
      importance_ratio_clipping = 0.0,
      lambda_value = 0.95,
      discount_factor = 0.99,
      entropy_regularization = 0.0,
      policy_l2_reg = 0.0,
      value_function_l2_reg = 0.0,
      shared_vars_l2_reg = 0.0,
      value_pred_loss_coef = 0.5,
      num_epochs = 25,
      use_gae = False,
      use_td_lambda_return = False,
      normalize_rewards = True,
      reward_norm_clipping = 10.0,
      normalize_observations = True,
      log_prob_clipping = 0.0,
      kl_cutoff_factor = 0.0,
      kl_cutoff_coef = 0.0,
      initial_adaptive_kl_beta = 0.0,
      adaptive_kl_target = 0.0,
      adaptive_kl_tolerance = 0.0,
      gradient_clipping = None,
      value_clipping = None,
      check_numerics = False,
      # TODO(b/150244758): Change the default to False once we move
      # clients onto Reverb.
      compute_value_and_advantage_in_train = True,
      update_normalizers_in_train = True,
      debug_summaries = False,
      summarize_grads_and_vars = False,
      train_step_counter = None,
      name = 'AttentionPPOAgent'):
    """Creates a PPO Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of `BoundedTensorSpec` representing the actions.
      optimizer: Optimizer to use for the agent, default to using
        `tf.compat.v1.train.AdamOptimizer`.
      actor_net: A `network.DistributionNetwork` which maps observations to
        action distributions. Commonly, it is set to
        `actor_distribution_network.ActorDistributionNetwork`.
      value_net: A `Network` which returns the value prediction for input
        states, with `call(observation, step_type, network_state)`. Commonly, it
        is set to `value_network.ValueNetwork`.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation. Default to `0.99`
        which is the value used for all environments from (Schulman, 2017).
      entropy_regularization: Coefficient for entropy regularization loss term.
        Default to `0.0` because no entropy bonus was used in (Schulman, 2017).
      policy_l2_reg: Coefficient for L2 regularization of unshared actor_net
        weights. Default to `0.0` because no L2 regularization was applied on
        the policy network weights in (Schulman, 2017).
      value_function_l2_reg: Coefficient for l2 regularization of unshared value
        function weights. Default to `0.0` because no L2 regularization was
        applied on the policy network weights in (Schulman, 2017).
      shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
        between actor_net and value_net. Default to `0.0` because no L2
        regularization was applied on the policy network or value network
        weights in (Schulman, 2017).
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss. Default to `0.5`, which was used for all
        environments in the OpenAI baseline implementation. This parameters is
        irrelevant unless you are sharing part of actor_net and value_net. In
        that case, you would want to tune this coeeficient, whose value depends
        on the network architecture of your choice.
      num_epochs: Number of epochs for computing policy updates. (Schulman,2017)
        sets this to 10 for Mujoco, 15 for Roboschool and 3 for Atari.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function; here: `td_lambda_return = gae_advantage +
          value_predictions`. `use_gae` must be set to `True` as well to enable
          TD -lambda returns. If `use_td_lambda_return` is set to True while
          `use_gae` is False, the empirical return will be used and a warning
          will be logged.
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards. While not mentioned directly in (Schulman,
        2017), reward normalization was implemented in OpenAI baselines and
        (Ilyas et al., 2018) pointed out that it largely improves performance.
        You may refer to Figure 1 of https://arxiv.org/pdf/1811.02553.pdf for a
          comparison with and without reward scaling.
      reward_norm_clipping: Value above and below to clip normalized reward.
        Additional optimization proposed in (Ilyas et al., 2018) set to `5` or
        `10`.
      normalize_observations: If `True`, keeps moving mean and variance of
        observations and normalizes incoming observations. Additional
        optimization proposed in (Ilyas et al., 2018). If true, and the
        observation spec is not tf.float32 (such as Atari), please manually
        convert the observation spec received from the environment to tf.float32
        before creating the networks. Otherwise, the normalized input to the
        network (float32) will have a different dtype as what the network
        expects, resulting in a mismatch error.
        Example usage: ```python observation_tensor_spec, action_spec,
          time_step_tensor_spec = ( spec_utils.get_tensor_specs(env))
          normalized_observation_tensor_spec = tf.nest.map_structure(
            lambda s: tf.TensorSpec( dtype=tf.float32, shape=s.shape,
              name=s.name ), observation_tensor_spec )  actor_net =
              actor_distribution_network.ActorDistributionNetwork(
              normalized_observation_tensor_spec, ...) value_net =
              value_network.ValueNetwork( normalized_observation_tensor_spec,
              ...) # Note that the agent still uses the original
              time_step_tensor_spec # from the environment. agent =
              ppo_clip_agent.PPOClipAgent( time_step_tensor_spec, action_spec,
              actor_net, value_net, ...) ```
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      kl_cutoff_factor: Only meaningful when `kl_cutoff_coef > 0.0`. A multipler
        used for calculating the KL cutoff ( = `kl_cutoff_factor *
        adaptive_kl_target`). If policy KL averaged across the batch changes
        more than the cutoff, a squared cutoff loss would be added to the loss
        function.
      kl_cutoff_coef: kl_cutoff_coef and kl_cutoff_factor are additional params
        if one wants to use a KL cutoff loss term in addition to the adaptive KL
        loss term. Default to 0.0 to disable the KL cutoff loss term as this was
        not used in the paper.  kl_cutoff_coef is the coefficient to mulitply by
        the KL cutoff loss term, before adding to the total loss function.
      initial_adaptive_kl_beta: Initial value for beta coefficient of adaptive
        KL penalty. This initial value is not important in practice because the
        algorithm quickly adjusts to it. A common default is 1.0.
      adaptive_kl_target: Desired KL target for policy updates. If actual KL is
        far from this target, adaptive_kl_beta will be updated. You should tune
        this for your environment. 0.01 was found to perform well for Mujoco.
      adaptive_kl_tolerance: A tolerance for adaptive_kl_beta. Mean KL above `(1
        + tol) * adaptive_kl_target`, or below `(1 - tol) * adaptive_kl_target`,
        will cause `adaptive_kl_beta` to be updated. `0.5` was chosen
        heuristically in the paper, but the algorithm is not very sensitive to
        it.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      value_clipping: Difference between new and old value predictions are
        clipped to this threshold. Value clipping could be helpful when training
        very deep networks. Default: no clipping.
      check_numerics: If true, adds `tf.debugging.check_numerics` to help find
        NaN / Inf values. For debugging only.
      compute_value_and_advantage_in_train: A bool to indicate where value
        prediction and advantage calculation happen.  If True, both happen in
        agent.train(). If False, value prediction is computed during data
        collection. This argument must be set to `False` if mini batch learning
        is enabled.
      update_normalizers_in_train: A bool to indicate whether normalizers are
        updated as parts of the `train` method. Set to `False` if mini batch
        learning is enabled, or if `train` is called on multiple iterations of
        the same trajectories. In that case, you would need to use `PPOLearner`
        (which updates all the normalizers outside of the agent). This ensures
        that normalizers are updated in the same way as (Schulman, 2017).
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      TypeError: if `actor_net` or `value_net` is not of type
        `tf_agents.networks.Network`.
    """
    if not isinstance(actor_net, network.Network):
      raise TypeError('actor_net must be an instance of a network.Network.')
    if not isinstance(value_net, network.Network):
      raise TypeError('value_net must be an instance of a network.Network.')

    # PPOPolicy validates these, so we skip validation here.
    actor_net.create_variables(time_step_spec.observation)
    value_net.create_variables(time_step_spec.observation)

    tf.Module.__init__(self, name=name)

    self._optimizer = optimizer
    self._actor_net = actor_net
    self._value_net = value_net
    self._importance_ratio_clipping = importance_ratio_clipping
    self._lambda = lambda_value
    self._discount_factor = discount_factor
    self._entropy_regularization = entropy_regularization
    self._policy_l2_reg = policy_l2_reg
    self._value_function_l2_reg = value_function_l2_reg
    self._shared_vars_l2_reg = shared_vars_l2_reg
    self._value_pred_loss_coef = value_pred_loss_coef
    self._num_epochs = num_epochs
    self._use_gae = use_gae
    self._use_td_lambda_return = use_td_lambda_return
    self._reward_norm_clipping = reward_norm_clipping
    self._log_prob_clipping = log_prob_clipping
    self._kl_cutoff_factor = kl_cutoff_factor
    self._kl_cutoff_coef = kl_cutoff_coef
    self._adaptive_kl_target = adaptive_kl_target
    self._adaptive_kl_tolerance = adaptive_kl_tolerance
    self._gradient_clipping = gradient_clipping or 0.0
    self._value_clipping = value_clipping or 0.0
    self._check_numerics = check_numerics
    self._compute_value_and_advantage_in_train = (
        compute_value_and_advantage_in_train)
    self.update_normalizers_in_train = update_normalizers_in_train
    if not isinstance(self._optimizer, tf.keras.optimizers.Optimizer):
      logging.warning(
          'Only tf.keras.optimizers.Optimizers are well supported, got a '
          'non-TF2 optimizer: %s', self._optimizer)

    self._initial_adaptive_kl_beta = initial_adaptive_kl_beta
    if initial_adaptive_kl_beta > 0.0:
      self._adaptive_kl_beta = common.create_variable(
          'adaptive_kl_beta', initial_adaptive_kl_beta, dtype=tf.float32)
    else:
      self._adaptive_kl_beta = None

    self._reward_normalizer = None
    if normalize_rewards:
      self._reward_normalizer = tensor_normalizer.StreamingTensorNormalizer(
          tensor_spec.TensorSpec([], tf.float32), scope='normalize_reward')

    self._observation_normalizer = None
    if normalize_observations:
      self._observation_normalizer = (
          tensor_normalizer.StreamingTensorNormalizer(
              time_step_spec.observation, scope='normalize_observations'))

    self._advantage_normalizer = tensor_normalizer.StreamingTensorNormalizer(
        tensor_spec.TensorSpec([], tf.float32), scope='normalize_advantages')

    policy = greedy_policy.GreedyPolicy(
        attention_ppo_policy.AttentionPPOPolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
            observation_normalizer=self._observation_normalizer,
            clip=False,
            collect=False))

    collect_policy = attention_ppo_policy.AttentionPPOPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        value_network=value_net,
        observation_normalizer=self._observation_normalizer,
        clip=False,
        collect=True,
        compute_value_and_advantage_in_train=(
            self._compute_value_and_advantage_in_train),
    )

    if isinstance(self._actor_net, network.DistributionNetwork):
      # Legacy behavior
      self._action_distribution_spec = self._actor_net.output_spec
    else:
      self._action_distribution_spec = self._actor_net.create_variables(
          time_step_spec.observation)

    # Set training_data_spec to collect_data_spec with augmented policy info,
    # iff return and normalized advantage are saved in preprocess_sequence.
    if self._compute_value_and_advantage_in_train:
      training_data_spec = None
    else:
      training_policy_info = collect_policy.trajectory_spec.policy_info.copy()
      training_policy_info.update({
          'value_prediction':
              collect_policy.trajectory_spec.policy_info['value_prediction'],
          'return':
              tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
          'advantage':
              tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
      })
      training_data_spec = collect_policy.trajectory_spec.replace(
          policy_info=training_policy_info)

    super(ppo_agent.PPOAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=None,
        training_data_spec=training_data_spec,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

    # This must be built after super() which sets up self.data_context.
    self._collected_as_transition = data_converter.AsTransition(
        self.collect_data_context, squeeze_time_dim=False)

    self._as_trajectory = data_converter.AsTrajectory(
        self.data_context, sequence_length=None)
