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

"""Shared utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures
import os
import pickle

from absl import flags
from absl import logging

import gym
import numpy as np
import tensorflow.compat.v1 as tf
from tf_agents.environments import suite_mujoco
from tf_agents.specs import array_spec

flags.DEFINE_integer('checkpoint_iterations', 50, 'Periodicity of checkpoints.')
flags.DEFINE_integer('eval_iterations', 50, 'Periodicity of evaluations.')
flags.DEFINE_integer('num_evals', 10, 'Number of evaluations.')

FLAGS = flags.FLAGS

_CHECKPOINT_FILENAME = 'model.ckpt'


def get_state_and_action_specs(gym_env, action_bounds=None):
  """Returns state and action specs for a Gym environment.

  Args:
    gym_env: gym.core.Env. A Gym environment.
    action_bounds: list of strings. Min and max values in string for action
      variables.

  Returns:
    (BoundedArraySpec, BoundedArraySpec). The first is a state spec and the
    second is a action spec.
  """
  if isinstance(gym_env.observation_space, gym.spaces.Box):
    state_spec = array_spec.BoundedArraySpec(
        shape=gym_env.observation_space.shape,
        dtype=gym_env.observation_space.dtype,
        minimum=gym_env.observation_space.low,
        maximum=gym_env.observation_space.high)
  else:
    raise NotImplementedError(type(gym_env.observation_space))

  if action_bounds:
    assert len(action_bounds) == 2
    action_min = np.tile(float(action_bounds[0]), gym_env.action_space.shape)
    action_max = np.tile(float(action_bounds[1]), gym_env.action_space.shape)
  else:
    action_min = gym_env.action_space.low
    action_max = gym_env.action_space.high

  if isinstance(gym_env.action_space, gym.spaces.Box):
    action_spec = array_spec.BoundedArraySpec(
        shape=gym_env.action_space.shape,
        dtype=gym_env.action_space.dtype,
        minimum=action_min,
        maximum=action_max)
  else:
    raise NotImplementedError(type(gym_env.action_space))
  return state_spec, action_spec


def create_env(env_name):
  """Creates Environment."""
  if env_name == 'Pendulum':
    env = gym.make('Pendulum-v0')
  elif env_name == 'Hopper':
    env = suite_mujoco.load('Hopper-v2')
  elif env_name == 'Walker2D':
    env = suite_mujoco.load('Walker2d-v2')
  elif env_name == 'HalfCheetah':
    env = suite_mujoco.load('HalfCheetah-v2')
  elif env_name == 'Ant':
    env = suite_mujoco.load('Ant-v2')
  elif env_name == 'Humanoid':
    env = suite_mujoco.load('Humanoid-v2')
  else:
    raise ValueError('Unsupported environment: %s' % env_name)
  return env


def _env_reset(env):
  if hasattr(env, 'time_step_spec'):
    return env.reset().observation
  else:
    return env.reset()


def _env_step(env, action):
  if hasattr(env, 'time_step_spec'):
    ts = env.step(action)
    return ts.observation, ts.reward, env.done, env.get_info()
  else:
    return env.step(action)


def warm_up_replay_memory(session, behavior_policy, time_out, discount_factor,
                          replay_memory):
  # The number of events in an epsidoe could be less than the maximum episode
  # length (i.e., time_out) when the environment has a termination state.
  min_replay_memory_size = FLAGS.batch_size * FLAGS.train_steps_per_iteration
  while replay_memory.size < min_replay_memory_size:
    num_events = min_replay_memory_size - replay_memory.size
    num_episodes = int(num_events / time_out) + 1
    collect_experience_parallel(num_episodes, session, behavior_policy,
                                time_out, discount_factor, replay_memory)


def collect_experience_parallel(num_episodes,
                                session,
                                behavior_policy,
                                time_out,
                                discount_factor,
                                replay_memory,
                                collect_init_state_step=False):
  """Executes threads for data collection."""
  old_size = replay_memory.size
  if num_episodes > 1:
    with futures.ThreadPoolExecutor(
        max_workers=FLAGS.collect_experience_parallelism) as executor:
      for _ in range(num_episodes):
        executor.submit(collect_experience, session, behavior_policy, time_out,
                        discount_factor, replay_memory, collect_init_state_step)
  else:
    collect_experience(session, behavior_policy, time_out, discount_factor,
                       replay_memory, collect_init_state_step)
  return replay_memory.size - old_size


def collect_experience(session,
                       behavior_policy,
                       time_out,
                       discount_factor,
                       replay_memory,
                       collect_init_state_step=False):
  """Adds experiences into replay memory.

  Generates an episode, computes Q targets for state and action pairs in the
  episode, and adds them into the replay memory.
  """
  with session.as_default():
    with session.graph.as_default():
      env = create_env(FLAGS.env_name)
      episode, _, _ = _collect_episode(env, time_out, discount_factor,
                                       behavior_policy, collect_init_state_step)
      replay_memory.extend(episode)
      if hasattr(env, 'close'):
        env.close()


def _collect_episode(env, time_out, discount_factor, behavior_policy,
                     collect_init_state_step=False):
  """Collects episodes of trajectories by following a behavior policy."""
  episode = []
  episode_lengths = []
  episode_rewards = []

  state = _env_reset(env)
  init_state = _env_reset(env)
  done = False
  episode_step_count = 0
  e_reward = 0

  for _ in range(time_out):
    # First, sample an action
    action = behavior_policy.action(state, use_action_function=True)
    if action is None:
      break

    next_state, reward, done, info = _env_step(env, action)
    reward = reward if not done else 0.0

    # Save the experience to our buffer
    if collect_init_state_step:
      episode.append([
          init_state, state, action, reward, next_state, episode_step_count,
          done, info
      ])
    else:
      episode.append([state, action, reward, next_state, done, info])

    # update state, e_reward and step count
    state = next_state
    if discount_factor < 1:
      e_reward += (discount_factor**episode_step_count) * reward
    else:
      e_reward += reward

    episode_step_count += 1
    if done:
      break

  if episode_step_count > 0:
    episode_lengths.append(episode_step_count)
    episode_rewards.append(e_reward)
  return (episode, episode_lengths, episode_rewards)


def periodic_updates(iteration,
                     train_step,
                     replay_memories,
                     greedy_policy,
                     saver,
                     sess,
                     time_out,
                     use_action_function=True,
                     tf_summary=None):
  """Evaluates the algorithm."""
  if (FLAGS.checkpoint_dir and FLAGS.checkpoint_iterations and
      iteration % FLAGS.checkpoint_iterations == 0):
    logging.info('Iteration: %d, writing checkpoints..', iteration)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, _CHECKPOINT_FILENAME)
    saver.save(
        sess, checkpoint_file, global_step=train_step, write_meta_graph=False)

    for replay_memory in replay_memories:
      replay_memory.save(FLAGS.checkpoint_dir, delete_old=True)
    logging.info('Iteration: %d, completed writing checkpoints.', iteration)

  if FLAGS.eval_iterations and iteration % FLAGS.eval_iterations == 0:
    logging.info('Iteration: %d, evaluating the model..', iteration)
    scores = []
    action_magnitudes = []
    episode_lens = []
    future_list = []
    with futures.ThreadPoolExecutor(max_workers=FLAGS.num_evals) as executor:
      for _ in range(FLAGS.num_evals):
        future_list.append(
            executor.submit(
                _evaluate_model,
                time_out,
                greedy_policy,
                use_action_function=use_action_function,
                render=False))

      for future in futures.as_completed(future_list):
        score, action_magnitude, episode_len = future.result()
        scores.append(score)
        action_magnitudes.append(action_magnitude)
        episode_lens.append(episode_len)

    avg_score = np.mean(scores)
    avg_action_magitude = np.mean(action_magnitudes)
    avg_episode_len = np.mean(episode_lens)
    logging.info(
        'Iteration: %d, avg_score: %.3f, avg_episode_len: %.3f, '
        'avg_action_magnitude: %.3f', iteration, avg_score, avg_episode_len,
        avg_action_magitude)

    if tf_summary:
      tf_summary.value.extend([
          tf.Summary.Value(tag='avg_score', simple_value=avg_score),
          tf.Summary.Value(
              tag='avg_action_magnitude', simple_value=avg_action_magitude),
          tf.Summary.Value(tag='avg_episode_len', simple_value=avg_episode_len)
      ])


def _evaluate_model(time_out,
                    greedy_policy,
                    use_action_function=False,
                    render=False):
  """Evaluates the model."""
  env = create_env(FLAGS.env_name)
  state = _env_reset(env)
  total_reward = 0.0
  total_action_magnitude = 0.0
  episode_len = 0
  for _ in range(time_out):
    if render:
      env.render()
    action = greedy_policy.action(
        np.reshape(state, [1, -1]), use_action_function)
    if action is None:
      break

    next_state, reward, done, _ = _env_step(env, action)
    state = next_state
    total_reward += reward
    if greedy_policy.continuous_action:
      total_action_magnitude += np.linalg.norm(action, np.inf)
    episode_len += 1
    if done:
      break

  return total_reward, total_action_magnitude / episode_len, episode_len


def save_hparam_config(dict_to_save, config_dir):
  """Saves config file of hparam."""
  filename = os.path.join(config_dir, 'hparam.pickle')
  print('Saving results to %s' % filename)
  if not tf.gfile.Exists(config_dir):
    tf.gfile.MakeDirs(config_dir)
  with tf.gfile.GFile(filename, 'w') as f:
    pickle.dump(dict_to_save, f, protocol=2)


def action_projection(action, action_spec, softmax=False):
  """Projects action tensor onto a bound."""
  if isinstance(action, np.ndarray):
    if softmax:
      e_x = np.exp(action - np.max(action, axis=1))
      return e_x / np.sum(e_x, axis=1)
    else:
      return np.minimum(action_spec.maximum,
                        np.maximum(action_spec.minimum, action))
  else:
    # TF version
    if softmax:
      return tf.nn.softmax(action, axis=1)
    else:
      return tf.minimum(action_spec.maximum,
                        tf.maximum(action_spec.minimum, action))


def create_placeholders_for_q_net(tf_vars):
  """Creates placeholders for feeding values to TF variables.

  Args:
    tf_vars: list. A list of TF variables. These are variables for a neural
      network approximating a Q function.

  Returns:
    dict. A dictionary mapping a string to a tf.placeholder.
  """
  ph_dict = {}
  for var in tf_vars:
    ph_dict['{}_ph'.format(var.name)] = tf.placeholder(
        dtype=var.dtype, shape=var.shape)
  return ph_dict


def build_dummy_q_net(state, action, ph_dict, q_net_vars):
  """Builds a dummy Q network.

  This function builds a neural network where parameters are given by
  placeholders.

  Args:
    state: TF Tensor. State tensor.
    action: TF Tensor. Action tensor.
    ph_dict: dict. A dictionary mapping a TF variable's name to a
      tf.placeholder. There is one placeholder for each variable in
      `q_net_vars`.
    q_net_vars: list. A list of TF variables. The list should have even number
      of variables. One for weights and other for bias for each layer of a
      neural network.

  Returns:
    TF Tensor. Output tensor of a Q network.
  """
  assert bool(q_net_vars) and len(q_net_vars) % 2 == 0
  net = tf.concat([state, action], axis=1)
  # Specific for MLP
  for itr, var in enumerate(q_net_vars):
    if itr % 2 == 0:
      # even itr, multiplicative weights
      net = tf.einsum('ij,jk->ik', net, ph_dict['{}_ph'.format(var.name)])
    else:
      # odd itr, additive weights
      net = tf.nn.bias_add(net, ph_dict['{}_ph'.format(var.name)])

      # Output layer doesn't have an activation function.
      if itr < len(q_net_vars) - 1:
        net = tf.nn.relu(net)
  return net


def make_tf_summary_histogram(values, num_bins=10):
  """Constructs a tf Summary of type histogram from a np array of values.

  Args:
    values: list or np.array.
    num_bins: int. Number of histogram bins.

  Returns:
    tf.HistogramProto.
  """
  values = np.reshape(values, [-1])
  counts, limits = np.histogram(values, bins=num_bins)
  return tf.HistogramProto(
      min=np.amin(values),
      max=np.amax(values),
      num=values.size,
      sum=np.sum(values),
      sum_squares=np.sum(values**2),
      bucket_limit=limits.tolist()[1:],
      bucket=counts.tolist())
