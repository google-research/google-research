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

# python3
r"""Run off-policy evaluation training loop."""

import gc
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
import d4rl  # pylint: disable=unused-import
import gym
from gym.wrappers import time_limit
import numpy as np
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
import tqdm

from policy_eval import utils
from policy_eval.actor import Actor
from policy_eval.behavior_cloning import BehaviorCloning
from policy_eval.dataset import D4rlDataset
from policy_eval.dataset import Dataset
from policy_eval.dual_dice import DualDICE
from policy_eval.model_based import ModelBased
from policy_eval.q_fitter import QFitter

EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Reacher-v2',
                    'Environment for training/evaluation.')
flags.DEFINE_bool('d4rl', False, 'Whether to use D4RL envs and datasets.')
flags.DEFINE_string('d4rl_policy_filename', None,
                    'Path to saved pickle of D4RL policy.')
flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 3e-4, 'Critic learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('behavior_policy_std', None,
                   'Noise scale of behavior policy.')
flags.DEFINE_float('target_policy_std', 0.1, 'Noise scale of target policy.')
flags.DEFINE_integer('num_trajectories', 1000, 'Number of trajectories.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_integer(
    'num_mc_episodes', 256,
    'Number of episodes to unroll to estimate Monte Carlo returns.')
flags.DEFINE_integer('num_updates', 1_000_000, 'Number of updates.')
flags.DEFINE_integer('eval_interval', 10_000, 'Logging interval.')
flags.DEFINE_integer('log_interval', 10_000, 'Logging interval.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_string('save_dir', '/tmp/policy_eval/',
                    'Directory to save results to.')
flags.DEFINE_string(
    'data_dir',
    '/tmp/policy_eval/trajectory_datasets/',
    'Directory with data for evaluation.')
flags.DEFINE_boolean('normalize_states', True, 'Whether to normalize states.')
flags.DEFINE_boolean('normalize_rewards', True, 'Whether to normalize rewards.')
flags.DEFINE_boolean('bootstrap', True,
                     'Whether to generated bootstrap weights.')
flags.DEFINE_enum('algo', 'fqe', ['fqe', 'dual_dice', 'mb', 'iw', 'dr'],
                  'Algorithm for policy evaluation.')
flags.DEFINE_float('noise_scale', 0.25, 'Noise scaling for data augmentation.')


def make_hparam_string(json_parameters=None, **hparam_str_dict):
  if json_parameters:
    for key, value in json.loads(json_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value
  return ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])


def main(_):
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  summary_writer.set_as_default()

  if FLAGS.d4rl:
    d4rl_env = gym.make(FLAGS.env_name)
    gym_spec = gym.spec(FLAGS.env_name)
    if gym_spec.max_episode_steps in [0, None]:  # Add TimeLimit wrapper.
      gym_env = time_limit.TimeLimit(d4rl_env, max_episode_steps=1000)
    else:
      gym_env = d4rl_env
    gym_env.seed(FLAGS.seed)
    env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(gym_env))

    behavior_dataset = D4rlDataset(
        d4rl_env,
        normalize_states=FLAGS.normalize_states,
        normalize_rewards=FLAGS.normalize_rewards,
        noise_scale=FLAGS.noise_scale,
        bootstrap=FLAGS.bootstrap)
  else:
    env = suite_mujoco.load(FLAGS.env_name)
    env.seed(FLAGS.seed)
    env = tf_py_environment.TFPyEnvironment(env)

    data_file_name = os.path.join(FLAGS.data_dir, FLAGS.env_name, '0',
                                  f'dualdice_{FLAGS.behavior_policy_std}.pckl')
    behavior_dataset = Dataset(
        data_file_name,
        FLAGS.num_trajectories,
        normalize_states=FLAGS.normalize_states,
        normalize_rewards=FLAGS.normalize_rewards,
        noise_scale=FLAGS.noise_scale,
        bootstrap=FLAGS.bootstrap)

  tf_dataset = behavior_dataset.with_uniform_sampling(FLAGS.sample_batch_size)
  tf_dataset_iter = iter(tf_dataset)

  if FLAGS.d4rl:
    with tf.io.gfile.GFile(FLAGS.d4rl_policy_filename, 'rb') as f:
      policy_weights = pickle.load(f)
    actor = utils.D4rlActor(env, policy_weights,
                            is_dapg='dapg' in FLAGS.d4rl_policy_filename)
  else:
    actor = Actor(env.observation_spec().shape[0], env.action_spec())
    actor.load_weights(behavior_dataset.model_filename)

  policy_returns = utils.estimate_monte_carlo_returns(env, FLAGS.discount,
                                                      actor,
                                                      FLAGS.target_policy_std,
                                                      FLAGS.num_mc_episodes)
  logging.info('Estimated Per-Step Average Returns=%f', policy_returns)

  if 'fqe' in FLAGS.algo or 'dr' in FLAGS.algo:
    model = QFitter(env.observation_spec().shape[0],
                    env.action_spec().shape[0], FLAGS.lr, FLAGS.weight_decay,
                    FLAGS.tau)
  elif 'mb' in FLAGS.algo:
    model = ModelBased(env.observation_spec().shape[0],
                       env.action_spec().shape[0], learning_rate=FLAGS.lr,
                       weight_decay=FLAGS.weight_decay)
  elif 'dual_dice' in FLAGS.algo:
    model = DualDICE(env.observation_spec().shape[0],
                     env.action_spec().shape[0], FLAGS.weight_decay)
  if 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
    behavior = BehaviorCloning(env.observation_spec().shape[0],
                               env.action_spec(),
                               FLAGS.lr, FLAGS.weight_decay)

  @tf.function
  def get_target_actions(states):
    return actor(
        tf.cast(behavior_dataset.unnormalize_states(states),
                env.observation_spec().dtype),
        std=FLAGS.target_policy_std)[1]

  @tf.function
  def get_target_logprobs(states, actions):
    log_probs = actor(
        tf.cast(behavior_dataset.unnormalize_states(states),
                env.observation_spec().dtype),
        actions=actions,
        std=FLAGS.target_policy_std)[2]
    if tf.rank(log_probs) > 1:
      log_probs = tf.reduce_sum(log_probs, -1)
    return log_probs

  min_reward = tf.reduce_min(behavior_dataset.rewards)
  max_reward = tf.reduce_max(behavior_dataset.rewards)
  min_state = tf.reduce_min(behavior_dataset.states, 0)
  max_state = tf.reduce_max(behavior_dataset.states, 0)

  @tf.function
  def update_step():
    (states, actions, next_states, rewards, masks, weights,
     _) = next(tf_dataset_iter)
    initial_actions = get_target_actions(behavior_dataset.initial_states)
    next_actions = get_target_actions(next_states)

    if 'fqe' in FLAGS.algo or 'dr' in FLAGS.algo:
      model.update(states, actions, next_states, next_actions, rewards, masks,
                   weights, FLAGS.discount, min_reward, max_reward)
    elif 'mb' in FLAGS.algo:
      model.update(states, actions, next_states, rewards, masks,
                   weights)
    elif 'dual_dice' in FLAGS.algo:
      model.update(behavior_dataset.initial_states, initial_actions,
                   behavior_dataset.initial_weights, states, actions,
                   next_states, next_actions, masks, weights, FLAGS.discount)

    if 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
      behavior.update(states, actions, weights)

  gc.collect()

  for i in tqdm.tqdm(range(FLAGS.num_updates), desc='Running Training'):
    update_step()

    if i % FLAGS.eval_interval == 0:
      if 'fqe' in FLAGS.algo:
        pred_returns = model.estimate_returns(behavior_dataset.initial_states,
                                              behavior_dataset.initial_weights,
                                              get_target_actions)
      elif 'mb' in FLAGS.algo:
        pred_returns = model.estimate_returns(behavior_dataset.initial_states,
                                              behavior_dataset.initial_weights,
                                              get_target_actions,
                                              FLAGS.discount,
                                              min_reward, max_reward,
                                              min_state, max_state)
      elif FLAGS.algo in ['dual_dice']:
        pred_returns, pred_ratio = model.estimate_returns(iter(tf_dataset))

        tf.summary.scalar('train/pred ratio', pred_ratio, step=i)
      elif 'iw' in FLAGS.algo or 'dr' in FLAGS.algo:
        discount = FLAGS.discount
        _, behavior_log_probs = behavior(behavior_dataset.states,
                                         behavior_dataset.actions)
        target_log_probs = get_target_logprobs(behavior_dataset.states,
                                               behavior_dataset.actions)
        offset = 0.0
        rewards = behavior_dataset.rewards
        if 'dr' in FLAGS.algo:
          # Doubly-robust is effectively the same as importance-weighting but
          # transforming rewards at (s,a) to r(s,a) + gamma * V^pi(s') -
          # Q^pi(s,a) and adding an offset to each trajectory equal to V^pi(s0).
          offset = model.estimate_returns(behavior_dataset.initial_states,
                                          behavior_dataset.initial_weights,
                                          get_target_actions)
          q_values = (model(behavior_dataset.states, behavior_dataset.actions) /
                      (1 - discount))
          n_samples = 10
          next_actions = [get_target_actions(behavior_dataset.next_states)
                          for _ in range(n_samples)]
          next_q_values = sum(
              [model(behavior_dataset.next_states, next_action) / (1 - discount)
               for next_action in next_actions]) / n_samples
          rewards = rewards + discount * next_q_values - q_values

        # Now we compute the self-normalized importance weights.
        # Self-normalization happens over trajectories per-step, so we
        # restructure the dataset as [num_trajectories, num_steps].
        num_trajectories = len(behavior_dataset.initial_states)
        max_trajectory_length = np.max(behavior_dataset.steps) + 1
        trajectory_weights = behavior_dataset.initial_weights
        trajectory_starts = np.where(np.equal(behavior_dataset.steps, 0))[0]

        batched_rewards = np.zeros([num_trajectories, max_trajectory_length])
        batched_masks = np.zeros([num_trajectories, max_trajectory_length])
        batched_log_probs = np.zeros([num_trajectories, max_trajectory_length])

        for traj_idx, traj_start in enumerate(trajectory_starts):
          traj_end = (trajectory_starts[traj_idx + 1]
                      if traj_idx + 1 < len(trajectory_starts)
                      else len(rewards))
          traj_length = traj_end - traj_start
          batched_rewards[traj_idx, :traj_length] = rewards[traj_start:traj_end]
          batched_masks[traj_idx, :traj_length] = 1.
          batched_log_probs[traj_idx, :traj_length] = (
              -behavior_log_probs[traj_start:traj_end] +
              target_log_probs[traj_start:traj_end])

        batched_weights = (batched_masks *
                           (discount **
                            np.arange(max_trajectory_length))[None, :])

        clipped_log_probs = np.clip(batched_log_probs, -6., 2.)
        cum_log_probs = batched_masks * np.cumsum(clipped_log_probs, axis=1)
        cum_log_probs_offset = np.max(cum_log_probs, axis=0)
        cum_probs = np.exp(cum_log_probs - cum_log_probs_offset[None, :])
        avg_cum_probs = (
            np.sum(cum_probs * trajectory_weights[:, None], axis=0) /
            (1e-10 + np.sum(batched_masks * trajectory_weights[:, None],
                            axis=0)))
        norm_cum_probs = cum_probs / (1e-10 + avg_cum_probs[None, :])

        weighted_rewards = batched_weights * batched_rewards * norm_cum_probs
        trajectory_values = np.sum(weighted_rewards, axis=1)
        avg_trajectory_value = ((1 - discount) *
                                np.sum(trajectory_values * trajectory_weights) /
                                np.sum(trajectory_weights))
        pred_returns = offset + avg_trajectory_value

      pred_returns = behavior_dataset.unnormalize_rewards(pred_returns)

      tf.summary.scalar('train/pred returns', pred_returns, step=i)
      logging.info('pred returns=%f', pred_returns)

      tf.summary.scalar('train/true minus pred returns',
                        policy_returns - pred_returns,
                        step=i)
      logging.info('true minus pred returns=%f', policy_returns - pred_returns)


if __name__ == '__main__':
  app.run(main)
