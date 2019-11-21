# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Evaluation of policies for TD3 and DDPG.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from common import Actor
import gym
import lfd_envs
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from utils import do_rollout

FLAGS = flags.FLAGS
flags.DEFINE_string('env', 'Hopper-v1',
                    'Environment for training/evaluation.')
flags.DEFINE_string('load_dir', '', 'Directory to save models.')
flags.DEFINE_boolean('use_gpu', False,
                     'Directory to write TensorBoard summaries.')
flags.DEFINE_boolean('wrap_for_absorbing', False,
                     'Use the wrapper for absorbing states.')
flags.DEFINE_integer('num_trials', 10, 'Number of evaluation trials to run.')
flags.DEFINE_string('master', 'local', 'Location of the session.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of Parameter Server tasks.')
flags.DEFINE_integer('task_id', 0, 'Id of the current TF task.')


def wait_for_next_checkpoint(log_dir,
                             last_checkpoint=None,
                             seconds_to_sleep=1,
                             timeout=20):
  """Blocking wait until next checkpoint is written to logdir.

  Can timeout at regular intervals to log a timeout warning (a good indicator
  the thread is still alive).

  Args:
    log_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or None if we're expecting a
      checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum amount of time to wait before printing timeout warning
      and checking for a new checkpoint. If left as None, then the thread will
      wait indefinitely.

  Returns:
    next_checkpoint filename.
  """

  while True:
    logging.info('Waiting for next policy checkpoint...')
    next_checkpoint = tf.contrib.training.wait_for_new_checkpoint(
        log_dir,
        last_checkpoint,
        seconds_to_sleep=seconds_to_sleep,
        timeout=timeout)
    if next_checkpoint is None:
      logging.warn('Timeout waiting for checkpoint, trying again...')
    elif next_checkpoint != last_checkpoint:
      # Found a new checkpoint.
      logging.warn('Found a new checkpoint ("%s").', next_checkpoint)
      break
    else:
      logging.warn('No new checkpoint found, trying again...')

  return next_checkpoint


def main(_):
  """Run td3/ddpg evaluation."""
  tfe.enable_eager_execution()

  if FLAGS.use_gpu:
    tf.device('/device:GPU:0').__enter__()

  tf.gfile.MakeDirs(FLAGS.log_dir)
  summary_writer = tf.contrib.summary.create_file_writer(
      FLAGS.log_dir, flush_millis=10000)

  env = gym.make(FLAGS.env)
  if FLAGS.wrap_for_absorbing:
    env = lfd_envs.AbsorbingWrapper(env)

  obs_shape = env.observation_space.shape
  act_shape = env.action_space.shape

  with tf.variable_scope('actor'):
    actor = Actor(obs_shape[0], act_shape[0])

  random_reward, _ = do_rollout(
      env, actor, None, num_trajectories=10, sample_random=True)

  reward_scale = tfe.Variable(1, name='reward_scale')
  saver = tfe.Saver(actor.variables + [reward_scale])

  last_checkpoint = tf.train.latest_checkpoint(FLAGS.load_dir)
  with summary_writer.as_default():
    while True:
      last_checkpoint = wait_for_next_checkpoint(FLAGS.load_dir,
                                                 last_checkpoint)

      total_numsteps = int(last_checkpoint.split('-')[-1])

      saver.restore(last_checkpoint)

      average_reward, average_length = do_rollout(
          env, actor, None, noise_scale=0.0, num_trajectories=FLAGS.num_trials)

      logging.info(
          'Evaluation: average episode length %d, average episode reward %f',
          average_length, average_reward)

      print('Evaluation: average episode length {}, average episode reward {}'.
            format(average_length, average_reward))

      with tf.contrib.summary.always_record_summaries():
        if reward_scale.numpy() != 1.0:
          tf.contrib.summary.scalar(
              'reward/scaled', (average_reward - random_reward) /
              (reward_scale.numpy() - random_reward),
              step=total_numsteps)
        tf.contrib.summary.scalar('reward', average_reward, step=total_numsteps)
        tf.contrib.summary.scalar('length', average_length, step=total_numsteps)


if __name__ == '__main__':
  app.run(main)
