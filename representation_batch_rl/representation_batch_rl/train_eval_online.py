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

r"""Run training loop.

"""

import os
import random
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs.tensor_spec import TensorSpec
import tqdm

from representation_batch_rl.batch_rl import asac
from representation_batch_rl.batch_rl import awr
from representation_batch_rl.batch_rl import ddpg
from representation_batch_rl.batch_rl import evaluation
from representation_batch_rl.batch_rl import pcl
from representation_batch_rl.batch_rl import sac
from representation_batch_rl.batch_rl import sac_v1
from representation_batch_rl.batch_rl.image_utils import image_aug
from representation_batch_rl.twin_sac import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'pixels-dm-cartpole-swingup',
                    'Environment for training/evaluation.')
flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
flags.DEFINE_float('actor_lr', 3e-4, 'Actor learning rate.')
flags.DEFINE_float('alpha_lr', 3e-4, 'Temperature learning rate.')
flags.DEFINE_float('critic_lr', 3e-4, 'Critic learning rate.')
flags.DEFINE_integer('deployment_batch_size', 1, 'Batch size.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_integer('max_timesteps', 1000_000, 'Max timesteps to train.')
flags.DEFINE_integer('max_length_replay_buffer', 100_000,
                     'Max replay buffer size (image observations use 100k).')
flags.DEFINE_integer('num_random_actions', 10_000,
                     'Fill replay buffer with N random actions.')
flags.DEFINE_integer('start_training_timesteps', 10_000,
                     'Start training when replay buffer contains N timesteps.')
flags.DEFINE_string('save_dir', '/tmp/save/', 'Directory to save results to.')
flags.DEFINE_integer('log_interval', 1_000, 'Log every N timesteps.')
flags.DEFINE_integer('eval_interval', 10_000, 'Evaluate every N timesteps.')
flags.DEFINE_integer('action_repeat', 8,
                     '(optional) action repeat used when instantiating env.')
flags.DEFINE_integer('frame_stack', 0,
                     '(optional) frame stack used when instantiating env.')
flags.DEFINE_enum('algo_name', 'sac', [
    'ddpg',
    'crossnorm_ddpg',
    'sac',
    'pc_sac',
    'pcl',
    'crossnorm_sac',
    'crr',
    'awr',
    'sac_v1',
    'asac',
], 'Algorithm.')
flags.DEFINE_boolean('eager', False, 'Execute functions eagerly.')


class InfiniteTimestep():
  """Timestep override."""

  def __init__(self, timestep):
    self.timestep = timestep
    self.observation = timestep.observation
    self.discount = timestep.discount
    self.reward = timestep.reward
    self.step_type = timestep.step_type

  def is_last(self):
    if hasattr(self.timestep, 'is_last'):
      self.timestep.is_last()
    else:
      return False


def main(_):
  tf.config.experimental_run_functions_eagerly(FLAGS.eager)

  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  random.seed(FLAGS.seed)

  FLAGS.set_default('max_timesteps', FLAGS.max_timesteps // FLAGS.action_repeat)

  if 'pixels-dm' in FLAGS.env_name:
    if 'distractor' in FLAGS.env_name:
      _, _, domain_name, _, _ = FLAGS.env_name.split('-')
    else:
      _, _, domain_name, _ = FLAGS.env_name.split('-')

    if domain_name in ['cartpole']:
      FLAGS.set_default('action_repeat', 8)
    elif domain_name in ['reacher', 'cheetah', 'ball_in_cup', 'hopper']:
      FLAGS.set_default('action_repeat', 4)
    elif domain_name in ['finger', 'walker']:
      FLAGS.set_default('action_repeat', 2)
    print('Loading env')
    env, _ = utils.load_env(FLAGS.env_name, FLAGS.seed, FLAGS.action_repeat,
                            FLAGS.frame_stack)
    eval_env, _ = utils.load_env(FLAGS.env_name, FLAGS.seed,
                                 FLAGS.action_repeat, FLAGS.frame_stack)
    print('Env loaded')
  else:
    raise Exception('Unsupported env')

  is_image_obs = (isinstance(env.observation_spec(), TensorSpec) and
                  len(env.observation_spec().shape) == 3)

  spec = (
      env.observation_spec(),
      env.action_spec(),
      env.reward_spec(),
      env.reward_spec(),  # discount spec
      env.observation_spec()  # next observation spec
  )
  print('Init replay')

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      spec, batch_size=1, max_length=FLAGS.max_length_replay_buffer)
  print('Replay created')
  @tf.function
  def add_to_replay(state, action, reward, discount, next_states):
    replay_buffer.add_batch((state, action, reward, discount, next_states))

  hparam_str = utils.make_hparam_string(
      FLAGS.xm_parameters, seed=FLAGS.seed, env_name=FLAGS.env_name,
      algo_name=FLAGS.algo_name)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  results_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'results', hparam_str))
  print('Init actor')
  if 'ddpg' in FLAGS.algo_name:
    model = ddpg.DDPG(
        env.observation_spec(),
        env.action_spec(),
        cross_norm='crossnorm' in FLAGS.algo_name)
  elif 'crr' in FLAGS.algo_name:
    model = awr.AWR(
        env.observation_spec(),
        env.action_spec(), f='bin_max')
  elif 'awr' in FLAGS.algo_name:
    model = awr.AWR(
        env.observation_spec(),
        env.action_spec(), f='exp_mean')
  elif 'sac_v1' in FLAGS.algo_name:
    model = sac_v1.SAC(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0])
  elif 'asac' in FLAGS.algo_name:
    model = asac.ASAC(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0])
  elif 'sac' in FLAGS.algo_name:
    model = sac.SAC(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0],
        cross_norm='crossnorm' in FLAGS.algo_name,
        pcl_actor_update='pc' in FLAGS.algo_name)
  elif 'pcl' in FLAGS.algo_name:
    model = pcl.PCL(
        env.observation_spec(),
        env.action_spec(),
        target_entropy=-env.action_spec().shape[0])

  print('Init random policy for warmup')
  initial_collect_policy = random_tf_policy.RandomTFPolicy(
      env.time_step_spec(), env.action_spec())
  print('Init replay buffer')
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=tf.data.AUTOTUNE,
      sample_batch_size=FLAGS.sample_batch_size)
  if is_image_obs:
    dataset = dataset.map(image_aug,
                          num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False).prefetch(3)
  else:
    dataset = dataset.prefetch(3)

  def repack(*data):
    return data[0]
  dataset = dataset.map(repack)
  replay_buffer_iter = iter(dataset)

  previous_time = time.time()
  timestep = env.reset()
  episode_return = 0
  episode_timesteps = 0
  step_mult = 1 if FLAGS.action_repeat < 1 else FLAGS.action_repeat

  print('Starting training')
  for i in tqdm.tqdm(range(FLAGS.max_timesteps)):
    if i % FLAGS.deployment_batch_size == 0:
      for _ in range(FLAGS.deployment_batch_size):
        last_timestep = timestep.is_last()
        if last_timestep:

          if episode_timesteps > 0:
            current_time = time.time()
            with summary_writer.as_default():
              tf.summary.scalar(
                  'train/returns',
                  episode_return,
                  step=(i + 1) * step_mult)
              tf.summary.scalar(
                  'train/FPS',
                  episode_timesteps / (current_time - previous_time),
                  step=(i + 1) * step_mult)

          timestep = env.reset()
          episode_return = 0
          episode_timesteps = 0
          previous_time = time.time()

        if (replay_buffer.num_frames() < FLAGS.num_random_actions or
            replay_buffer.num_frames() < FLAGS.deployment_batch_size):
          # Use policy only after the first deployment.
          policy_step = initial_collect_policy.action(timestep)
          action = policy_step.action
        else:
          action = model.actor(timestep.observation, sample=True)

        next_timestep = env.step(action)

        add_to_replay(timestep.observation, action, next_timestep.reward,
                      next_timestep.discount, next_timestep.observation)

        episode_return += next_timestep.reward[0]
        episode_timesteps += 1

        timestep = next_timestep

    if i + 1 >= FLAGS.start_training_timesteps:
      with summary_writer.as_default():
        info_dict = model.update_step(replay_buffer_iter)
      if (i + 1) % FLAGS.log_interval == 0:
        with summary_writer.as_default():
          for k, v in info_dict.items():
            tf.summary.scalar(f'training/{k}', v, step=(i + 1) * step_mult)

    if (i + 1) % FLAGS.eval_interval == 0:
      logging.info('Performing policy eval.')
      average_returns, evaluation_timesteps = evaluation.evaluate(
          eval_env, model)

      with results_writer.as_default():
        tf.summary.scalar(
            'evaluation/returns', average_returns, step=(i + 1) * step_mult)
        tf.summary.scalar(
            'evaluation/length', evaluation_timesteps, step=(i+1) * step_mult)
      logging.info('Eval at %d: ave returns=%f, ave episode length=%f',
                   (i + 1) * step_mult, average_returns, evaluation_timesteps)

    if ((i + 1) * step_mult) % 50_000 == 0:
      model.save_weights(
          os.path.join(FLAGS.save_dir, 'results', FLAGS.env_name + '__' + str(
              (i + 1) * step_mult)))


if __name__ == '__main__':
  app.run(main)
