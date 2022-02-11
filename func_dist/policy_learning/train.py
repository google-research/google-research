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

"""Train a goal-conditioned policy to minimize functional distance.
"""
import functools
from typing import Optional, Tuple

from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.agents.jax import sac
from acme.jax.types import PRNGKey
from acme.utils import counting
from acme.utils import loggers
from acme.utils import paths
from acme.utils.google import paths as paths_google
from acme.wrappers import gym_wrapper
from acme.wrappers import single_precision
import flax.linen as nn
import gym
import jax
from jax import numpy as jnp
import ml_collections
from ml_collections import config_flags
import multiworld.envs.mujoco
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.projects.func_dist import pretrain_utils as scenic_pretrain_utils
from scenic.projects.func_dist import train_utils as scenic_train_utils
from scenic.train_lib import train_utils as train_lib_utils
import tensorflow as tf

from func_dist.data_utils import pickle_datasets
from func_dist.dist_learning import train_utils
from func_dist.policy_learning import env_wrappers




# Environment
flags.DEFINE_string(
    'task', 'Image48HumanLikeSawyerPushForwardEnv-v0',
    'Name of the gym environment.')
flags.DEFINE_integer('max_episode_steps', 100, 'Environment step limit.')

# For goal image
flags.DEFINE_string(
    'robot_data_path', None,
    'Path to gzipped pickle file with robot interaction data.')

# Distance function
flags.DEFINE_string(
    'distance_ckpt_to_load', None,
    'Path to flax checkpoint of trained distance model used to define reward.')
flags.DEFINE_enum('distance_ckpt_format', 'flax', ['flax', 'scenic'],
                  'Format of distance_ckpt_to_load.')
config_flags.DEFINE_config_file(
    'scenic_config', None,
    'Path to scenic config (if loading a scenic trained distance model).',
    lock_config=True)
flags.DEFINE_bool('use_true_distance', False,
                  'If True, set reward to true distance between end-effector '
                  'and puck + puck and target.')
flags.DEFINE_list(
    'encoder_conv_filters', [16, 16, 32],
    'Number and sizes of convolutional filters in the embedding network.')  # pytype: disable=wrong-arg-types
flags.DEFINE_integer(
    'encoder_conv_size', 5, 'Convolution kernel size in the embedding network.')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate for training.')

# Reward function
flags.DEFINE_float(
    'baseline_distance', None,
    'If set, considered to be the minimum distance prediction. Its value is '
    'subtracted from all distances (to shift the minimum to be 0). If None, '
    'distances are neither shifted nor lower bounded.')
flags.DEFINE_bool(
    'baseline_distance_from_goal_to_goal', False,
    'If True, use the distance prediction from the goal image to the goal '
    'image as the baseline distance.')
flags.DEFINE_float(
    'distance_reward_weight', 1, 'Multiplier for distance rewards.')
flags.DEFINE_float(
    'environment_reward_weight', 0, 'Multiplier for environment rewards.')

# Policy training
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of steps to train for.')
flags.DEFINE_integer('eval_every', 20_000,
                     'Number of time steps between evaluations.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Number of evaluation episodes.')
flags.DEFINE_integer('num_sgd_steps_per_step', 1,
                     'Number of SGD steps per learner step().')
flags.DEFINE_integer('min_replay_size', 10_000,
                     'Number of time steps to record before training.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('end_on_success', True,
                  'If True, terminate training episodes early if successful.')


# Logging
flags.DEFINE_string('logdir', None, 'Directory for output logs.')

flags.DEFINE_integer(
    'record_episodes_frequency', 100,
    'Interval between recording training episodes (in number of episodes).')
flags.DEFINE_integer(
    'num_episodes_to_record', 3, 'Number of consecutive episodes to record.')


FLAGS = flags.FLAGS


def load_goal_image(dataset_path):
  interaction_data = pickle_datasets.InteractionDataset(dataset_path)
  goal_image = interaction_data.get_goal_image(episode_idx=0)
  return goal_image


def make_environment(
    task,
    end_on_success,
    max_episode_steps,
    distance_fn,
    goal_image,
    baseline_distance = None,
    eval_mode = False,
    logdir = None,
    counter = None,
    record_every = 100,
    num_episodes_to_record = 3):
  """Create the environment and its wrappers."""
  env = gym.make(task)
  env = gym_wrapper.GymWrapper(env)
  if end_on_success:
    env = env_wrappers.EndOnSuccessWrapper(env)
  env = wrappers.StepLimitWrapper(env, max_episode_steps)

  env = env_wrappers.ReshapeImageWrapper(env)
  if distance_fn.history_length > 1:
    env = wrappers.FrameStackingWrapper(env, distance_fn.history_length)
  env = env_wrappers.GoalConditionedWrapper(env, goal_image)
  env = env_wrappers.DistanceModelWrapper(
      env, distance_fn, max_episode_steps, baseline_distance,
      distance_reward_weight=FLAGS.distance_reward_weight,
      environment_reward_weight=FLAGS.environment_reward_weight)
  if FLAGS.use_true_distance:
    env = env_wrappers.RewardWrapper(env)
  if logdir:
    env = env_wrappers.RecordEpisodesWrapper(
        env, counter, logdir, record_every=record_every,
        num_to_record=num_episodes_to_record, eval_mode=eval_mode)
  env = env_wrappers.VisibleStateWrapper(env, eval_mode)

  return single_precision.SinglePrecisionWrapper(env)


def load_distance_fn(
    encoder_conv_filters, encoder_conv_size, key, ckpt_to_load, learning_rate
    ):
  """Load a saved distance model from a checkpoint at ckpt_to_load."""

  def embed_and_predict_distance(
      model_state,
      state,
      goal):
    params = {'params': model_state.distance_optimizer.target}
    batched_state = jnp.expand_dims(state, axis=0)
    batched_goal = jnp.expand_dims(goal, axis=0)
    state_emb = model_state.encoder_fn(params, batched_state)
    goal_emb = model_state.encoder_fn(params, batched_goal)
    emb = jnp.concatenate([state_emb, goal_emb], axis=1)
    emb = jnp.squeeze(emb)
    dist = model_state.distance_fn(params, emb)[0]
    return emb, dist

  model_state = train_utils.restore_or_initialize(
      encoder_conv_filters, encoder_conv_size, key, ckpt_to_load, learning_rate)
  distance_fn = functools.partial(embed_and_predict_distance, model_state)
  distance_fn = env_wrappers.DistanceFn(distance_fn, 1)
  return distance_fn


def load_scenic_distance_fn(
    model,
    train_state,
    config):
  """Load scenic distance function based on XM experiment and worker IDs."""

  def zero_centre(frames):
    return frames * 2.0 - 1.0

  def central_crop_frames(frames):
    _, h, w, _ = frames.shape
    min_dim = min(h, w)
    margin_h = int((h - min_dim) / 2)
    margin_w = int((w - min_dim) / 2)
    cropped_frames = frames[:,
                            margin_h:margin_h + min_dim,
                            margin_w:margin_w + min_dim]
    return cropped_frames

  def embed_and_predict_distance(
      model,
      train_state,
      config,
      state,
      goal,
    ):

    variables = {
        'params': train_state.optimizer['target'],
        **train_state.model_state
    }
    goal = jnp.expand_dims(goal, axis=0)
    # Frame stacking wrapper stacks on the last dimension.
    state = jnp.transpose(state, [3, 0, 1, 2])
    inputs = jnp.concatenate([state, goal], axis=0)
    inputs = central_crop_frames(inputs)
    inputs = resize_fn(inputs)
    if config.dataset_configs.zero_centering:
      inputs = zero_centre(inputs)
    # Add batch dimension.
    inputs = jnp.expand_dims(inputs, axis=0)
    dist, emb = model.flax_model.apply(
        variables, inputs, train=False, mutable=False)
    dist = dist[0][0]
    emb = emb[0]
    return emb, dist

  model.flax_model.return_prelogits = True

  crop_size = config.dataset_configs.crop_size
  input_shape = (crop_size, crop_size, 3)
  resize_fn = jax.vmap(
      functools.partial(jax.image.resize, shape=input_shape, method='bilinear'),
      axis_name='time')
  distance_fn = functools.partial(
      embed_and_predict_distance, model, train_state, config)
  distance_fn = jax.jit(distance_fn)
  distance_fn = env_wrappers.DistanceFn(
      distance_fn, config.dataset_configs.num_frames - 1)
  return distance_fn


def train_and_evaluate(distance_fn, rng):
  """Train a policy on the learned distance function and evaluate task success.

  Args:
    distance_fn: function mapping a (state, goal)-pair to a state embedding and
        a distance estimate used for policy learning.
    rng: random key used to initialize evaluation actor.
  """
  goal_image = load_goal_image(FLAGS.robot_data_path)
  logdir = FLAGS.logdir
  video_dir = paths.process_path(logdir, 'videos')
  print('Writing videos to', video_dir)
  counter = counting.Counter()
  eval_counter = counting.Counter(counter, prefix='eval', time_delta=0.0)
  # Include training episodes and steps and walltime in the first eval logs.
  counter.increment(episodes=0, steps=0, walltime=0)

  environment = make_environment(
      task=FLAGS.task,
      end_on_success=FLAGS.end_on_success,
      max_episode_steps=FLAGS.max_episode_steps,
      distance_fn=distance_fn,
      goal_image=goal_image,
      baseline_distance=FLAGS.baseline_distance,
      logdir=video_dir,
      counter=counter,
      record_every=FLAGS.record_episodes_frequency,
      num_episodes_to_record=FLAGS.num_episodes_to_record)
  environment_spec = specs.make_environment_spec(environment)
  print('Environment spec')
  print(environment_spec)
  agent_networks = sac.make_networks(environment_spec)

  config = sac.SACConfig(
      target_entropy=sac.target_entropy_from_env_spec(environment_spec),
      num_sgd_steps_per_step=FLAGS.num_sgd_steps_per_step,
      min_replay_size=FLAGS.min_replay_size)
  agent = sac.SAC(
      environment_spec, agent_networks, config=config, counter=counter,
      seed=FLAGS.seed)

  env_logger = loggers.CSVLogger(logdir, 'env_loop', flush_every=5)
  eval_env_logger = loggers.CSVLogger(logdir, 'eval_env_loop', flush_every=1)
  train_loop = acme.EnvironmentLoop(
      environment, agent, label='train_loop', logger=env_logger,
      counter=counter)

  eval_actor = agent.builder.make_actor(
      random_key=rng,
      policy_network=sac.apply_policy_and_sample(
          agent_networks, eval_mode=True),
      variable_source=agent)

  eval_video_dir = paths.process_path(logdir, 'eval_videos')
  print('Writing eval videos to', eval_video_dir)
  if FLAGS.baseline_distance_from_goal_to_goal:
    state = goal_image
    if distance_fn.history_length > 1:
      state = np.stack([goal_image] * distance_fn.history_length, axis=-1)
    unused_embeddings, baseline_distance = distance_fn(state, goal_image)
    print('Baseline prediction', baseline_distance)
  else:
    baseline_distance = FLAGS.baseline_distance
  eval_env = make_environment(
      task=FLAGS.task,
      end_on_success=False,
      max_episode_steps=FLAGS.max_episode_steps,
      distance_fn=distance_fn,
      goal_image=goal_image,
      eval_mode=True,
      logdir=eval_video_dir,
      counter=eval_counter,
      record_every=FLAGS.num_eval_episodes,
      num_episodes_to_record=FLAGS.num_eval_episodes,
      baseline_distance=baseline_distance)

  eval_loop = acme.EnvironmentLoop(
      eval_env, eval_actor, label='eval_loop', logger=eval_env_logger,
      counter=eval_counter)

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=FLAGS.num_eval_episodes)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=FLAGS.num_eval_episodes)


def main(_):
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  key = jax.random.PRNGKey(FLAGS.seed)
  key1, key2 = jax.random.split(key)

  multiworld.envs.mujoco.register_goal_example_envs()
  if FLAGS.distance_ckpt_format == 'scenic':
    if FLAGS.scenic_config is not None and FLAGS.distance_ckpt_to_load:
      model, train_state, config = scenic_pretrain_utils.restore_model(
          FLAGS.scenic_config, FLAGS.distance_ckpt_to_load)
    else:
      raise ValueError('Could not locate pretrained model or its config.')
    distance_fn = load_scenic_distance_fn(model, train_state, config)
  else:
    distance_fn = load_distance_fn(
        FLAGS.encoder_conv_filters, FLAGS.encoder_conv_size, key1,
        FLAGS.distance_ckpt_to_load, FLAGS.learning_rate)
  train_and_evaluate(distance_fn, key2)


if __name__ == '__main__':
  app.run(main)
