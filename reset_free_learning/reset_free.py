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

# Lint as: python2, python3
r"""Train and Eval SAC.

"""
import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import numpy as np
from six.moves import range
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from reset_free_learning import state_distribution_distance

from reset_free_learning.agents_with_value_functions import SacAgent
from reset_free_learning.agents_with_value_functions import Td3Agent

from reset_free_learning.envs import reset_free_wrapper

from reset_free_learning.reset_goal_generator import FixedResetGoal
from reset_free_learning.reset_goal_generator import ResetGoalGenerator
from reset_free_learning.reset_goal_generator import ScheduledResetGoal

from reset_free_learning.utils.env_utils import get_env
from reset_free_learning.utils.other_utils import copy_replay_buffer
from reset_free_learning.utils.other_utils import np_custom_load
from reset_free_learning.utils.other_utils import np_custom_save
from reset_free_learning.utils.other_utils import record_video
from reset_free_learning.utils.other_utils import std_clip_transform

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('offline_dir', None,
                    'Directory for loading replay buffers.')
flags.DEFINE_integer(
    'max_episode_steps', 1000,
    'maximum number of steps in the environment before a full reset is done')
flags.DEFINE_integer('eval_episode_steps', 1000,
                     'maximum number of steps in the evaluation episode')
flags.DEFINE_string('env_name', 'sawyer_push',
                    'name of the environment to be loaded')
flags.DEFINE_integer('random_seed', None, 'random seed')
flags.DEFINE_integer('video_record_interval', 50000, 'video interval')
flags.DEFINE_integer('num_videos_per_interval', 0,
                     'number of videos recording in each video evaluation')
flags.DEFINE_string('reward_type', 'dense', 'reward type for the environment')

# env reset-wrapper properties
flags.DEFINE_integer(
    'num_success_states', 1,
    'Number of successful states in a sequence for the reset to be considered complete'
)
flags.DEFINE_integer(
    'variable_reset_horizon', 0,
    'Number of successful states in a sequence for the reset to be considered complete'
)

# train hyperparameters
flags.DEFINE_integer('num_iterations', 3000000, 'number of iterations')
flags.DEFINE_integer(
    'reset_goal_frequency', 400,
    'virtual episode size, only the goal/task is reset, that is no human intervention is required'
)
flags.DEFINE_integer('initial_collect_steps', 10000, 'number of iterations')
flags.DEFINE_integer('batch_size', 256,
                     'Batch size for updating agent from replay buffer')
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'number of steps collected per iteration')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     ' number of train steps per iteration')

# agent hyperparameters
flags.DEFINE_string('agent_type', 'sac', 'type of agent to use for training')
flags.DEFINE_list('actor_net_size', [256, 256], 'layer size values of actor')
flags.DEFINE_list('critic_net_size', [256, 256],
                  'layer size values for the critic')
flags.DEFINE_float('reward_scale_factor', 0.1, 'reward scale factor, alpha')
flags.DEFINE_float('actor_learning_rate', 3e-4, 'learning rate for the actor')
flags.DEFINE_float('critic_learning_rate', 3e-4, 'learning rate for the critic')
flags.DEFINE_float('discount_factor', 0.99,
                   'discount factor for the reward optimization')
flags.DEFINE_integer('replay_buffer_capacity', int(1e6),
                     'capacity of the replay buffer')

# TD3 hyperparameters
flags.DEFINE_float('exploration_noise_std', 0.1, 'exploration noise')

# SAC hyperparmater
flags.DEFINE_float('alpha_learning_rate', 3e-4,
                   'learning rate for the soft policy parameter')

# reset-free hyperparameters
flags.DEFINE_integer(
    'use_reset_goals', 1,
    """-1-> oracle-reset, terminate on goal success
        0-> reset-free, no reset goals
        1-> reset-free, use reset goals, reset goals can be variable or fixed
        2-> oracle-reset, teleport to reset goal
        3-> reset-free, goal termination of episodes, reset goals can be variable or learned
        4-> reset-free wrapper with goal termination, scheduled reset goals
        5-> oracle-reset, goal termination of episodes, scheduled reset goals
        6-> oracle-reset, the forward goal is variable (do not use)
        7-> oracle-reset, reset goal is variable
    """)

flags.DEFINE_integer(
    'num_action_samples', 1,
    'used for approximating the value function from the critic function')
flags.DEFINE_integer('num_reset_candidates', 1000,
                     'number of candidate states for reset')
flags.DEFINE_float('reset_lagrange_learning_rate', 3e-4,
                   'learning rate for the lagrange_multiplier')
flags.DEFINE_float('value_threshold', 1000, 'value threshold')
flags.DEFINE_float('lagrange_max', np.inf, 'upper bound for lagrange variable')
flags.DEFINE_integer('use_minimum', 1,
                     'the choice of value function used in reverse curriculum')
flags.DEFINE_integer('use_no_entropy_q', 0,
                     'use the other q function to select reset goals')

# relabel data
flags.DEFINE_integer('relabel_goals', 0,
                     'add goal relabelling for optimization')
flags.DEFINE_integer('goal_relabel_type', 0, '0->final, 1-> random future')
flags.DEFINE_integer(
    'num_relabelled_goals', 5,
    'number of relabelled goals per episode, use with random future goal relabelling'
)
flags.DEFINE_integer('relabel_offline_data', 0,
                     'relabel with every intermediate state as a goal')

# point mass environment properties
flags.DEFINE_string('point_mass_env_type', 'default',
                    'environment configuration')
flags.DEFINE_string('playpen_task', 'rc_o', 'environment configuration')
flags.DEFINE_integer('reset_at_goal', 0,
                     'initialize reset-free environment at the goal')

# debug
flags.DEFINE_integer('debug_value_fn_for_reset', 0,
                     'print lagrange variane of reset goal')
flags.DEFINE_integer('num_chunks', 10,
                     'number of chunks to split the offline trajectory')
flags.DEFINE_integer('num_success_for_switch', 10,
                     'number of success to switch to next goal')

FLAGS = flags.FLAGS


def relabel_function(cur_episode, last_step, reward_fn, full_buffer):
  all_data = cur_episode.gather_all()

  # add all actual interaction to the replay buffer
  all_data = nest_utils.unbatch_nested_tensors(all_data)
  for cur_trajectory in nest_utils.unstack_nested_tensors(
      all_data, full_buffer.data_spec):
    # was already added by previous iteration
    if cur_trajectory.step_type.numpy() != 2:
      full_buffer.add_batch(nest_utils.batch_nested_tensors(cur_trajectory))

  last_traj = cur_trajectory._replace(  # pylint: disable=undefined-loop-variable
      step_type=tf.constant(2),
      observation=last_step.observation[0],
      next_step_type=tf.constant(0),
      reward=tf.constant(0.0),
      discount=tf.constant(1., dtype=tf.float32))
  full_buffer.add_batch(nest_utils.batch_nested_tensors(last_traj))

  def _relabel_given_goal(relabel_goal):
    obs_dim = relabel_goal.shape[0]
    all_trajectories = nest_utils.unstack_nested_tensors(
        all_data, full_buffer.data_spec)
    last_traj_idx = len(all_trajectories)
    for traj_idx, cur_trajectory in enumerate(all_trajectories):
      if cur_trajectory.step_type.numpy() != 2:
        new_obs = tf.concat(
            [cur_trajectory.observation[:obs_dim], relabel_goal], axis=0)

        if traj_idx == len(all_trajectories) - 1:
          next_obs = tf.concat(
              [last_step.observation[0, :obs_dim], relabel_goal], axis=0)
        else:
          next_obs = tf.concat([
              all_trajectories[traj_idx + 1].observation[:obs_dim], relabel_goal
          ],
                               axis=0)

        new_reward = tf.constant(reward_fn(obs=next_obs))

        # terminate episode
        if new_reward.numpy() > 0.0:
          new_traj = cur_trajectory._replace(
              observation=new_obs,
              next_step_type=tf.constant(2),
              reward=new_reward,
              discount=tf.constant(0., dtype=tf.float32))
          last_traj_idx = traj_idx + 1
          full_buffer.add_batch(nest_utils.batch_nested_tensors(new_traj))
          break
        else:
          new_traj = cur_trajectory._replace(
              observation=new_obs,
              reward=new_reward,
          )
          full_buffer.add_batch(nest_utils.batch_nested_tensors(new_traj))

    if last_traj_idx == len(all_trajectories):
      last_observation = tf.concat(
          [last_step.observation[0, :obs_dim], relabel_goal], axis=0)
    else:
      last_observation = tf.concat(
          [all_trajectories[last_traj_idx].observation[:obs_dim], relabel_goal],
          axis=0)

    last_traj = cur_trajectory._replace(  # pylint: disable=undefined-loop-variable
        step_type=tf.constant(2),
        observation=last_observation,
        next_step_type=tf.constant(0),
        reward=tf.constant(0.0),
        discount=tf.constant(1., dtype=tf.float32))
    full_buffer.add_batch(nest_utils.batch_nested_tensors(last_traj))

  # relabel with last time step achieved in the episode
  if FLAGS.goal_relabel_type == 0 or (FLAGS.goal_relabel_type == 1 and
                                      last_step.reward.numpy()[0] <= 0.):
    obs_dim = last_step.observation.shape[1] // 2
    _relabel_given_goal(last_step.observation[0, :obs_dim])

  elif FLAGS.goal_relabel_type == 2 and last_step.reward.numpy()[0] <= 0.:
    goals = [
        [1.2, 0., 2.5, 0., -1., -1.],
        [2., 0., 2.4, 0., 0., 0.],
        [0.8, 0., 1.2, 0., 0., 0.],
        [-0.1, -0.3, 0.3, -0.3, 0., 0.],
        [-0.6, -1., -0.2, -1., 0., 0.],
        [-1.8, -1., -1.4, -1., 0., 0.],
        [-2.8, -0.8, -2.4, -1., -1., -1.],
        [-2.4, 0., -2.4, -1., -1., -1.],
        [-1.2, 0., -2.4, -1., -1., -1.],
        [0.0, 0.0, -2.5, -1, -1., -1.],
    ]
    goals = np.stack(goals).astype('float32')
    print('unrelabelled goal:', last_step.observation[0, 6:].numpy())
    relabel_goal_idxs = np.arange(goals.shape[0])
    np.random.shuffle(relabel_goal_idxs)
    obs_dim = last_step.observation.shape[1] // 2

    relabel_count = 0
    for goal_idx in relabel_goal_idxs:
      chosen_goal = goals[goal_idx]
      if (chosen_goal == last_step.observation[0, obs_dim:].numpy()).all():
        continue
      print('goal for relabelling:', chosen_goal)
      _relabel_given_goal(relabel_goal=tf.constant(chosen_goal))

      relabel_count += 1
      if relabel_count >= FLAGS.num_relabelled_goals:
        break

  else:
    print('not adding relabelled trajectories')


# provide data with all the buffer -- make every state a succesful trajectory
def data_multiplier(offline_data, reward_fn):

  def _custom_print(some_traj):  # pylint: disable=unused-variable
    np.set_printoptions(precision=2, suppress=True)
    print('step', some_traj.step_type.numpy(), 'obs',
          some_traj.observation.numpy(), 'action', some_traj.action.numpy(),
          'reward', some_traj.reward.numpy(), 'next_step',
          some_traj.next_step_type.numpy(), 'discount',
          some_traj.discount.numpy())

  all_data = nest_utils.unbatch_nested_tensors(offline_data.gather_all())
  all_trajs = nest_utils.unstack_nested_tensors(all_data,
                                                offline_data.data_spec)

  for idx, traj in enumerate(all_trajs):
    # print('index:', idx)
    if traj.step_type.numpy() == 0:
      ep_start_idx = idx
      # print('new start index:', ep_start_idx)
    # TODO(architsh): remove this and change to else:
    # elif idx in [12, 24, 36, 48, 60, 72, 84, 96, 108]:
    else:
      # print('adding new trajectory')
      obs_dim = traj.observation.shape[0] // 2
      relabel_goal = traj.observation[:obs_dim]
      # print('new goal:', relabel_goal)

      last_traj_idx = len(all_trajs[ep_start_idx:idx + 1])
      for traj_idx, cur_trajectory in enumerate(all_trajs[ep_start_idx:idx +
                                                          1]):
        if cur_trajectory.step_type.numpy() != 2:
          new_obs = tf.concat(
              [cur_trajectory.observation[:obs_dim], relabel_goal], axis=0)

          next_obs = tf.concat([
              all_trajs[ep_start_idx + traj_idx + 1].observation[:obs_dim],
              relabel_goal
          ],
                               axis=0)

          new_reward = tf.constant(reward_fn(obs=next_obs))
          # terminate episode
          if new_reward.numpy() > 0.0:
            new_traj = cur_trajectory._replace(
                observation=new_obs,
                next_step_type=tf.constant(2),
                reward=new_reward,
                discount=tf.constant(0., dtype=tf.float32))
            last_traj_idx = ep_start_idx + traj_idx + 1
            # _custom_print(new_traj)
            offline_data.add_batch(nest_utils.batch_nested_tensors(new_traj))
            break
          else:
            new_traj = cur_trajectory._replace(
                observation=new_obs,
                reward=new_reward,
            )
            # _custom_print(new_traj)
            offline_data.add_batch(nest_utils.batch_nested_tensors(new_traj))

      last_observation = tf.concat(
          [all_trajs[last_traj_idx].observation[:obs_dim], relabel_goal],
          axis=0)
      last_traj = cur_trajectory._replace(  # pylint: disable=undefined-loop-variable
          step_type=tf.constant(2),
          observation=last_observation,
          next_step_type=tf.constant(0),
          reward=tf.constant(0.0),
          discount=tf.constant(1., dtype=tf.float32))
      # _custom_print(last_traj)
      offline_data.add_batch(nest_utils.batch_nested_tensors(last_traj))
      # print('new size:', offline_data.num_frames())


@gin.configurable
def train_eval(
    root_dir,
    offline_dir=None,
    random_seed=None,
    env_name='sawyer_push',
    eval_env_name=None,
    env_load_fn=get_env,
    max_episode_steps=1000,
    eval_episode_steps=1000,
    # The SAC paper reported:
    # Hopper and Cartpole results up to 1000000 iters,
    # Humanoid results up to 10000000 iters,
    # Other mujoco tasks up to 3000000 iters.
    num_iterations=3000000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Params for collect
    # Follow https://github.com/haarnoja/sac/blob/master/examples/variants.py
    # HalfCheetah and Ant take 10000 initial collection steps.
    # Other mujoco tasks take 1000.
    # Different choices roughly keep the initial episodes about the same.
    initial_collect_steps=10000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    reset_goal_frequency=1000,  # virtual episode size for reset-free training
    train_steps_per_iteration=1,
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    # reset-free parameters
    use_minimum=True,
    reset_lagrange_learning_rate=3e-4,
    value_threshold=None,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=0.1,
    # Td3 parameters
    actor_update_period=1,
    exploration_noise_std=0.1,
    target_policy_noise=0.1,
    target_policy_noise_clip=0.1,
    dqda_clipping=None,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=50000,
    # video recording for the environment
    video_record_interval=10000,
    num_videos=0,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):

  start_time = time.time()

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  video_dir = os.path.join(eval_dir, 'videos')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    if random_seed is not None:
      tf.compat.v1.set_random_seed(random_seed)

    if FLAGS.use_reset_goals in [-1]:
      gym_env_wrappers = (functools.partial(
          reset_free_wrapper.GoalTerminalResetWrapper,
          num_success_states=FLAGS.num_success_states,
          full_reset_frequency=max_episode_steps),)
    elif FLAGS.use_reset_goals in [0, 1]:
      gym_env_wrappers = (functools.partial(
          reset_free_wrapper.ResetFreeWrapper,
          reset_goal_frequency=reset_goal_frequency,
          variable_horizon_for_reset=FLAGS.variable_reset_horizon,
          num_success_states=FLAGS.num_success_states,
          full_reset_frequency=max_episode_steps),)
    elif FLAGS.use_reset_goals in [2]:
      gym_env_wrappers = (functools.partial(
          reset_free_wrapper.CustomOracleResetWrapper,
          partial_reset_frequency=reset_goal_frequency,
          episodes_before_full_reset=max_episode_steps //
          reset_goal_frequency),)
    elif FLAGS.use_reset_goals in [3, 4]:
      gym_env_wrappers = (functools.partial(
          reset_free_wrapper.GoalTerminalResetFreeWrapper,
          reset_goal_frequency=reset_goal_frequency,
          num_success_states=FLAGS.num_success_states,
          full_reset_frequency=max_episode_steps),)
    elif FLAGS.use_reset_goals in [5, 7]:
      gym_env_wrappers = (functools.partial(
          reset_free_wrapper.CustomOracleResetGoalTerminalWrapper,
          partial_reset_frequency=reset_goal_frequency,
          episodes_before_full_reset=max_episode_steps //
          reset_goal_frequency),)
    elif FLAGS.use_reset_goals in [6]:
      gym_env_wrappers = (functools.partial(
          reset_free_wrapper.VariableGoalTerminalResetWrapper,
          full_reset_frequency=max_episode_steps),)

    if env_name == 'playpen_reduced':
      train_env_load_fn = functools.partial(
          env_load_fn, reset_at_goal=FLAGS.reset_at_goal)
    else:
      train_env_load_fn = env_load_fn

    env, env_train_metrics, env_eval_metrics, aux_info = train_env_load_fn(
        name=env_name,
        max_episode_steps=None,
        gym_env_wrappers=gym_env_wrappers)

    tf_env = tf_py_environment.TFPyEnvironment(env)
    eval_env_name = eval_env_name or env_name
    eval_tf_env = tf_py_environment.TFPyEnvironment(
        env_load_fn(name=eval_env_name,
                    max_episode_steps=eval_episode_steps)[0])

    eval_metrics += env_eval_metrics

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    if FLAGS.agent_type == 'sac':
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          observation_spec,
          action_spec,
          fc_layer_params=actor_fc_layers,
          continuous_projection_net=functools.partial(
              tanh_normal_projection_network.TanhNormalProjectionNetwork,
              std_transform=std_clip_transform))
      critic_net = critic_network.CriticNetwork(
          (observation_spec, action_spec),
          observation_fc_layer_params=critic_obs_fc_layers,
          action_fc_layer_params=critic_action_fc_layers,
          joint_fc_layer_params=critic_joint_fc_layers,
          kernel_initializer='glorot_uniform',
          last_kernel_initializer='glorot_uniform',
      )

      critic_net_no_entropy = None
      critic_no_entropy_optimizer = None
      if FLAGS.use_no_entropy_q:
        critic_net_no_entropy = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform',
            name='CriticNetworkNoEntropy1')
        critic_no_entropy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate)

      tf_agent = SacAgent(
          time_step_spec,
          action_spec,
          num_action_samples=FLAGS.num_action_samples,
          actor_network=actor_net,
          critic_network=critic_net,
          critic_network_no_entropy=critic_net_no_entropy,
          actor_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=actor_learning_rate),
          critic_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=critic_learning_rate),
          critic_no_entropy_optimizer=critic_no_entropy_optimizer,
          alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=alpha_learning_rate),
          target_update_tau=target_update_tau,
          target_update_period=target_update_period,
          td_errors_loss_fn=td_errors_loss_fn,
          gamma=gamma,
          reward_scale_factor=reward_scale_factor,
          gradient_clipping=gradient_clipping,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step)

    elif FLAGS.agent_type == 'td3':
      actor_net = actor_network.ActorNetwork(
          tf_env.time_step_spec().observation,
          tf_env.action_spec(),
          fc_layer_params=actor_fc_layers,
      )
      critic_net = critic_network.CriticNetwork(
          (observation_spec, action_spec),
          observation_fc_layer_params=critic_obs_fc_layers,
          action_fc_layer_params=critic_action_fc_layers,
          joint_fc_layer_params=critic_joint_fc_layers,
          kernel_initializer='glorot_uniform',
          last_kernel_initializer='glorot_uniform')

      tf_agent = Td3Agent(
          tf_env.time_step_spec(),
          tf_env.action_spec(),
          actor_network=actor_net,
          critic_network=critic_net,
          actor_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=actor_learning_rate),
          critic_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=critic_learning_rate),
          exploration_noise_std=exploration_noise_std,
          target_update_tau=target_update_tau,
          target_update_period=target_update_period,
          actor_update_period=actor_update_period,
          dqda_clipping=dqda_clipping,
          td_errors_loss_fn=td_errors_loss_fn,
          gamma=gamma,
          reward_scale_factor=reward_scale_factor,
          target_policy_noise=target_policy_noise,
          target_policy_noise_clip=target_policy_noise_clip,
          gradient_clipping=gradient_clipping,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step,
      )

    tf_agent.initialize()

    if FLAGS.use_reset_goals > 0:
      if FLAGS.use_reset_goals in [4, 5, 6]:
        reset_goal_generator = ScheduledResetGoal(
            goal_dim=aux_info['reset_state_shape'][0],
            num_success_for_switch=FLAGS.num_success_for_switch,
            num_chunks=FLAGS.num_chunks,
            name='ScheduledResetGoalGenerator')
      else:
        # distance to initial state distribution
        initial_state_distance = state_distribution_distance.L2Distance(
            initial_state_shape=aux_info['reset_state_shape'])
        initial_state_distance.update(
            tf.constant(aux_info['reset_states'], dtype=tf.float32),
            update_type='complete')

        if use_tf_functions:
          initial_state_distance.distance = common.function(
              initial_state_distance.distance)
          tf_agent.compute_value = common.function(tf_agent.compute_value)

        # initialize reset / practice goal proposer
        if reset_lagrange_learning_rate > 0:
          reset_goal_generator = ResetGoalGenerator(
              goal_dim=aux_info['reset_state_shape'][0],
              compute_value_fn=tf_agent.compute_value,
              distance_fn=initial_state_distance,
              use_minimum=use_minimum,
              value_threshold=value_threshold,
              lagrange_variable_max=FLAGS.lagrange_max,
              optimizer=tf.compat.v1.train.AdamOptimizer(
                  learning_rate=reset_lagrange_learning_rate),
              name='reset_goal_generator')
        else:
          reset_goal_generator = FixedResetGoal(
              distance_fn=initial_state_distance)

      # if use_tf_functions:
      #   reset_goal_generator.get_reset_goal = common.function(
      #       reset_goal_generator.get_reset_goal)

      # modify the reset-free wrapper to use the reset goal generator
      tf_env.pyenv.envs[0].set_reset_goal_fn(
          reset_goal_generator.get_reset_goal)

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    if FLAGS.relabel_goals:
      cur_episode_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          data_spec=tf_agent.collect_data_spec,
          batch_size=1,
          scope='CurEpisodeReplayBuffer',
          max_length=int(2 * min(reset_goal_frequency, max_episode_steps)))

      # NOTE: the buffer is replaced because cannot have two buffers.add_batch
      replay_observer = [cur_episode_buffer.add_batch]

    # initialize metrics and observers
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
    ]

    train_metrics += env_train_metrics

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    eval_py_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_agent.policy, use_tf_function=True)

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration)
    if use_tf_functions:
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    if offline_dir is not None:
      offline_data = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          data_spec=tf_agent.collect_data_spec,
          batch_size=1,
          max_length=int(1e5))  # this has to be 100_000
      offline_checkpointer = common.Checkpointer(
          ckpt_dir=offline_dir, max_to_keep=1, replay_buffer=offline_data)
      offline_checkpointer.initialize_or_restore()

      # set the reset candidates to be all the data in offline buffer
      if (FLAGS.use_reset_goals > 0 and
          reset_lagrange_learning_rate > 0) or FLAGS.use_reset_goals in [
              4, 5, 6, 7
          ]:
        tf_env.pyenv.envs[0].set_reset_candidates(
            nest_utils.unbatch_nested_tensors(offline_data.gather_all()))

    if replay_buffer.num_frames() == 0:
      if offline_dir is not None:
        copy_replay_buffer(offline_data, replay_buffer)
        print(replay_buffer.num_frames())

        # multiply offline data
        if FLAGS.relabel_offline_data:
          data_multiplier(replay_buffer,
                          tf_env.pyenv.envs[0].env.compute_reward)
          print('after data multiplication:', replay_buffer.num_frames())

      initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
          tf_env,
          initial_collect_policy,
          observers=replay_observer + train_metrics,
          num_steps=1)
      if use_tf_functions:
        initial_collect_driver.run = common.function(initial_collect_driver.run)

      # Collect initial replay data.
      logging.info(
          'Initializing replay buffer by collecting experience for %d steps with '
          'a random policy.', initial_collect_steps)

      time_step = None
      policy_state = collect_policy.get_initial_state(tf_env.batch_size)

      for iter_idx in range(initial_collect_steps):
        time_step, policy_state = initial_collect_driver.run(
            time_step=time_step, policy_state=policy_state)

        if time_step.is_last() and FLAGS.relabel_goals:
          reward_fn = tf_env.pyenv.envs[0].env.compute_reward
          relabel_function(cur_episode_buffer, time_step, reward_fn,
                           replay_buffer)
          cur_episode_buffer.clear()

        if FLAGS.use_reset_goals > 0 and time_step.is_last(
        ) and FLAGS.num_reset_candidates > 0:
          tf_env.pyenv.envs[0].set_reset_candidates(
              replay_buffer.get_next(
                  sample_batch_size=FLAGS.num_reset_candidates)[0])

    else:
      time_step = None
      policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    if eval_metrics_callback is not None:
      eval_metrics_callback(results, global_step.numpy())
    metric_utils.log_metrics(eval_metrics)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
      return ~trajectories.is_boundary()[0]

    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size, num_steps=2).unbatch().filter(
            _filter_invalid_transition).batch(batch_size).prefetch(5)
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    # manual data save for plotting utils
    np_custom_save(os.path.join(eval_dir, 'eval_interval.npy'), eval_interval)
    try:
      average_eval_return = np_custom_load(
          os.path.join(eval_dir, 'average_eval_return.npy')).tolist()
      average_eval_success = np_custom_load(
          os.path.join(eval_dir, 'average_eval_success.npy')).tolist()
      average_eval_final_success = np_custom_load(
          os.path.join(eval_dir, 'average_eval_final_success.npy')).tolist()
    except:  # pylint: disable=bare-except
      average_eval_return = []
      average_eval_success = []
      average_eval_final_success = []

    print('initialization_time:', time.time() - start_time)
    for iter_idx in range(num_iterations):
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )

      if time_step.is_last() and FLAGS.relabel_goals:
        reward_fn = tf_env.pyenv.envs[0].env.compute_reward
        relabel_function(cur_episode_buffer, time_step, reward_fn,
                         replay_buffer)
        cur_episode_buffer.clear()

      # reset goal generator updates
      if FLAGS.use_reset_goals > 0 and iter_idx % (
          FLAGS.reset_goal_frequency * collect_steps_per_iteration) == 0:
        if FLAGS.num_reset_candidates > 0:
          tf_env.pyenv.envs[0].set_reset_candidates(
              replay_buffer.get_next(
                  sample_batch_size=FLAGS.num_reset_candidates)[0])
        if reset_lagrange_learning_rate > 0:
          reset_goal_generator.update_lagrange_multipliers()

      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
      time_acc += time.time() - start_time

      global_step_val = global_step.numpy()

      if global_step_val % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step_val, train_loss.loss)
        steps_per_sec = (global_step_val - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step_val
        time_acc = 0

      for train_metric in train_metrics:
        if 'Heatmap' in train_metric.name:
          if global_step_val % summary_interval == 0:
            train_metric.tf_summaries(
                train_step=global_step, step_metrics=train_metrics[:2])
        else:
          train_metric.tf_summaries(
              train_step=global_step, step_metrics=train_metrics[:2])

      if global_step_val % summary_interval == 0 and FLAGS.use_reset_goals > 0 and reset_lagrange_learning_rate > 0:
        reset_states, values, initial_state_distance_vals, lagrangian = reset_goal_generator.update_summaries(
            step_counter=global_step)
        for vf_viz_metric in aux_info['value_fn_viz_metrics']:
          vf_viz_metric.tf_summaries(
              reset_states,
              values,
              train_step=global_step,
              step_metrics=train_metrics[:2])

        if FLAGS.debug_value_fn_for_reset:
          num_test_lagrange = 20
          hyp_lagranges = [
              1.0 * increment / num_test_lagrange
              for increment in range(num_test_lagrange + 1)
          ]

          door_pos = reset_states[
              np.argmin(initial_state_distance_vals.numpy() -
                        lagrangian.numpy() * values.numpy())][3:5]
          print('cur lagrange: %.2f, cur reset goal: (%.2f, %.2f)' %
                (lagrangian.numpy(), door_pos[0], door_pos[1]))
          for lagrange in hyp_lagranges:
            door_pos = reset_states[
                np.argmin(initial_state_distance_vals.numpy() -
                          lagrange * values.numpy())][3:5]
            print('test lagrange: %.2f, cur reset goal: (%.2f, %.2f)' %
                  (lagrange, door_pos[0], door_pos[1]))
          print('\n')

      if global_step_val % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step_val)
        metric_utils.log_metrics(eval_metrics)

        # numpy saves for plotting
        if 'AverageReturn' in results.keys():
          average_eval_return.append(results['AverageReturn'].numpy())
        if 'EvalSuccessfulAtAnyStep' in results.keys():
          average_eval_success.append(
              results['EvalSuccessfulAtAnyStep'].numpy())
        if 'EvalSuccessfulEpisodes' in results.keys():
          average_eval_final_success.append(
              results['EvalSuccessfulEpisodes'].numpy())
        elif 'EvalSuccessfulAtLastStep' in results.keys():
          average_eval_final_success.append(
              results['EvalSuccessfulAtLastStep'].numpy())

        if average_eval_return:
          np_custom_save(
              os.path.join(eval_dir, 'average_eval_return.npy'),
              average_eval_return)
        if average_eval_success:
          np_custom_save(
              os.path.join(eval_dir, 'average_eval_success.npy'),
              average_eval_success)
        if average_eval_final_success:
          np_custom_save(
              os.path.join(eval_dir, 'average_eval_final_success.npy'),
              average_eval_final_success)

      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

      if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)

      if global_step_val % video_record_interval == 0:
        for video_idx in range(num_videos):
          video_name = os.path.join(video_dir, str(global_step_val),
                                    'video_' + str(video_idx) + '.mp4')
          record_video(
              lambda: env_load_fn(  # pylint: disable=g-long-lambda
                  name=env_name,
                  max_episode_steps=max_episode_steps)[0],
              video_name,
              eval_py_policy,
              max_episode_length=eval_episode_steps)

    return train_loss


def main(_):
  os.environ['SDL_VIDEODRIVER'] = 'dummy'
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  root_dir = os.path.expanduser(FLAGS.root_dir)

  def _process_list_flag(flag_val):
    if 'xm_parameters' in FLAGS and isinstance(flag_val, str):
      return flag_val.split(',')
    return flag_val

  actor_net_layer_sizes = [
      int(lsize) for lsize in _process_list_flag(FLAGS.actor_net_size)
  ]
  if FLAGS.critic_net_size == ['-1'] or FLAGS.critic_net_size == '-1':
    critic_net_layer_sizes = actor_net_layer_sizes.copy()
  else:
    critic_net_layer_sizes = [
        int(lsize) for lsize in _process_list_flag(FLAGS.critic_net_size)
    ]

  if FLAGS.env_name in ['point_mass', 'point_mass_full_goal']:
    env_load_fn = functools.partial(
        get_env,
        env_type=FLAGS.point_mass_env_type,
        reward_type=FLAGS.reward_type)
  elif FLAGS.env_name == 'sawyer_door':
    env_load_fn = functools.partial(get_env, reward_type=FLAGS.reward_type)
  elif FLAGS.env_name == 'playpen':
    env_load_fn = functools.partial(
        get_env, task_list=FLAGS.playpen_task, reward_type=FLAGS.reward_type)
  elif FLAGS.env_name == 'playpen_reduced':
    env_load_fn = functools.partial(
        get_env, task_list=FLAGS.playpen_task, reward_type=FLAGS.reward_type)
  else:
    env_load_fn = get_env

  if FLAGS.agent_type == 'sac':
    initial_collect_steps = FLAGS.initial_collect_steps
    replay_buffer_capacity = FLAGS.replay_buffer_capacity
    target_update_tau = 0.005
    target_update_period = 1
    td_errors_loss_fn = tf.math.squared_difference

  elif FLAGS.agent_type == 'td3':
    initial_collect_steps = 10000
    replay_buffer_capacity = FLAGS.replay_buffer_capacity
    target_update_tau = 0.05
    target_update_period = 5
    td_errors_loss_fn = tf.compat.v1.losses.huber_loss

  train_eval(
      root_dir,
      offline_dir=FLAGS.offline_dir,
      random_seed=FLAGS.random_seed,
      env_name=FLAGS.env_name,
      env_load_fn=env_load_fn,
      max_episode_steps=FLAGS.max_episode_steps,
      eval_episode_steps=FLAGS.eval_episode_steps,
      video_record_interval=FLAGS.video_record_interval,
      num_videos=FLAGS.num_videos_per_interval,
      num_iterations=FLAGS.num_iterations,
      actor_fc_layers=tuple(actor_net_layer_sizes),
      critic_joint_fc_layers=tuple(critic_net_layer_sizes),
      reward_scale_factor=FLAGS.reward_scale_factor,
      actor_learning_rate=FLAGS.actor_learning_rate,
      critic_learning_rate=FLAGS.critic_learning_rate,
      alpha_learning_rate=FLAGS.alpha_learning_rate,
      use_minimum=bool(FLAGS.use_minimum),
      reset_lagrange_learning_rate=FLAGS.reset_lagrange_learning_rate,
      value_threshold=FLAGS.value_threshold,
      reset_goal_frequency=FLAGS.reset_goal_frequency,
      batch_size=FLAGS.batch_size,
      initial_collect_steps=initial_collect_steps,
      collect_steps_per_iteration=FLAGS.collect_steps_per_iteration,
      train_steps_per_iteration=FLAGS.train_steps_per_iteration,
      replay_buffer_capacity=replay_buffer_capacity,
      exploration_noise_std=FLAGS.exploration_noise_std,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      gamma=FLAGS.discount_factor,
      td_errors_loss_fn=td_errors_loss_fn,
  )


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
