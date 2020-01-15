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

# Lint as: python3
"""Trains and evaluates Safety Critic offline.

Trains and evaluates safety critic on train and test replay buffers, and plots
AUC, Acc, and related metrics.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import os.path as osp

from absl import app
from absl import flags
from absl import logging
from .algorithm import agents
from .algorithm import safe_sac_agent
from .envs import minitaur  # pylint: disable=unused-import
from .envs import point_mass
import gin
import gin.tf
import numpy as np
import tensorflow.compat.v1 as tf
from tf_agents.agents.sac import sac_agent  # pylint: disable=unused-import
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_pybullet
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tqdm import tqdm
from .utils import misc


flags.DEFINE_string('root_dir', '/tmp/safemrl/sc_train/experiments',
                    'root_dir to save tensorboard event files and models')
flags.DEFINE_string('load_base_path', None,
                    'base path for loading experience from')
flags.DEFINE_string(
    'load_dirs', None, 'comma-separated list of directories to load experience'
    ' from')
flags.DEFINE_string('test_dir', None, 'directory to load test experience from')
flags.DEFINE_string(
    'load_env_fn', 'pybullet',
    'type of environment to load (chooses which tf_agent env '
    'loader to use)')
flags.DEFINE_string('env_name', 'MinitaurTargetVelocityEnv-v0',
                    'name of environment to load')
flags.DEFINE_multi_string('gin_file', None, 'gin files')
flags.DEFINE_multi_string('gin_bindings', None, 'gin bindings')

FLAGS = flags.FLAGS

n_epochs = 50
n_trans = 1000000
batch_size = 128
eval_freq, log_freq = 100, 5000
n_parallel = 16


def load_environment():
  env_name = FLAGS.env_name
  if FLAGS.load_env_fn == 'pybullet':
    env_load_fn = suite_pybullet
  elif FLAGS.load_env_fn == 'gym':
    env_load_fn = suite_gym
  elif FLAGS.load_env_fn in ['point_mass', 'pm']:
    env_load_fn = point_mass.env_load_fn
  return env_load_fn(env_name)


def experience_to_transitions(experience):
  not_boundary = np.where(~experience.is_boundary())[1]
  tf.nest.map_structure(lambda x: tf.gather(x, not_boundary, axis=1),
                        experience)
  transitions = trajectory.to_transition(experience)
  time_steps, policy_steps, next_time_steps = transitions
  actions = policy_steps.action
  return time_steps, actions, next_time_steps


@common.function
def train_step(tf_agent, safety_critic, batch, safety_rewards, optimizer):
  """Helper function for creating a train step."""
  rb_data, buf_info = batch
  safe_rew = tf.gather(safety_rewards, buf_info.ids, axis=1)

  time_steps, actions, next_time_steps = tf_agent._experience_to_transitions(  # pylint: disable=protected-access
      rb_data)
  time_steps = time_steps._replace(reward=safe_rew[:, :-1])  # pylint: disable=protected-access
  next_time_steps = next_time_steps._replace(reward=safe_rew[:, 1:])
  fail_pct = safety_rewards.sum() / safety_rewards.shape[1]
  loss_weight = 0.5 / ((next_time_steps.reward) * fail_pct +
                       (1 - next_time_steps.reward) * (1 - fail_pct))
  trainable_safety_variables = safety_critic.trainable_variables
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    assert trainable_safety_variables, ('No trainable safety critic variables'
                                        ' to optimize.')
    tape.watch(trainable_safety_variables)
    loss = safety_critic_loss(
        tf_agent,
        safety_critic,
        time_steps,
        actions,
        next_time_steps,
        safety_rewards=next_time_steps.reward,
        weights=loss_weight)

    tf.debugging.check_numerics(loss, 'Critic loss is inf or nan.')
    safety_critic_grads = tape.gradient(loss, trainable_safety_variables)
    grads_and_vars = list(zip(safety_critic_grads, trainable_safety_variables))
    optimizer.apply_gradients(grads_and_vars)
  return loss


@common.function
def safety_critic_loss(tf_agent,
                       safety_critic,
                       time_steps,
                       actions,
                       next_time_steps,
                       safety_rewards,
                       weights=None):
  """Returns a critic loss with safety."""
  next_actions, next_log_pis = tf_agent._actions_and_log_probs(  # pylint: disable=protected-access
      next_time_steps)
  del next_log_pis
  target_input = (next_time_steps.observation[0], next_actions[0])
  target_q_values, unused_network_state1 = safety_critic(
      target_input, next_time_steps.step_type[0])
  target_q_values = tf.nn.sigmoid(target_q_values)
  safety_rewards = tf.to_float(safety_rewards)

  td_targets = tf.stop_gradient(safety_rewards + (1 - safety_rewards) *
                                next_time_steps.discount * target_q_values)
  td_targets = tf.squeeze(td_targets)

  pred_input = (time_steps.observation[0], actions[0])
  pred_td_targets, unused_network_state1 = safety_critic(
      pred_input, time_steps.step_type[0])
  loss = tf.losses.sigmoid_cross_entropy(td_targets, pred_td_targets)

  if weights is not None:
    loss *= tf.to_float(tf.squeeze(weights))

  # Take the mean across the batch.
  loss = tf.reduce_mean(input_tensor=loss)
  return loss


@common.function
def eval_safety_critic(safety_critic, boundary_states, boundary_actions,
                       safe_rew_boundary, boundary_step_type, metrics):
  """Evaluate safety critic."""
  pred_input = (boundary_states, boundary_actions)
  pred_q, _ = safety_critic(pred_input, boundary_step_type)
  pred_fail = tf.nn.sigmoid(pred_q)
  for m in metrics:
    m.update_state(safe_rew_boundary, pred_fail)
  eval_metrics = [m.result().numpy() for m in metrics]
  for m in metrics:
    m.reset_states()
  return eval_metrics


@common.function
def train_sc(tf_agent, safety_critic, optimizer, replay_buffer, test_agent,
             test_rb, agent_ckpts, rb_ckpts, test_agent_ckpt, test_rb_ckpt,
             sc_checkpointer):
  """Train function."""
  del test_agent
  del test_agent_ckpt
  del test_rb_ckpt
  ckpt_idx = 0
  batch_ds_boundary = n_epochs * n_trans * ckpt_idx // batch_size
  avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

  test_rb_data = test_rb.gather_all()
  test_safe_rew = agents.process_replay_buffer(test_rb, as_tensor=False)

  boundary = np.where(test_rb_data.is_boundary().numpy())[1]
  boundary_states = tf.gather(test_rb_data.observation[0], boundary)
  boundary_actions = tf.gather(test_rb_data.action[0], boundary)
  boundary_step_type = tf.gather(test_rb_data.step_type[0], boundary)
  safe_rew_boundary = tf.gather(test_safe_rew[0], boundary)

  metrics = [
      tf.keras.metrics.BinaryAccuracy(name='acc'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.TruePositives(name='tpr'),
      tf.keras.metrics.FalsePositives(name='fpr'),
      tf.keras.metrics.TrueNegatives(name='tnr'),
      tf.keras.metrics.FalseNegatives(name='fnr')
  ]

  n_datasets = len(rb_ckpts)
  n_batches = n_epochs * n_trans * n_datasets // batch_size

  for b_num in tqdm(range(n_batches)):
    if b_num >= batch_ds_boundary:  # checks if rollover to next dataset
      ckpt_idx += 1
      batch_ds_boundary = n_epochs * n_trans * ckpt_idx // batch_size
      rb_ckpt = rb_ckpts[ckpt_idx]
      train_path = agent_ckpts[ckpt_idx]
      logging.info('loading policy & agent from: %s', train_path)
      tf_agent, global_step = misc.load_agent_ckpt(train_path, tf_agent)
      global_step_val = global_step.numpy()
      assert global_step_val == 500000, ('agent global step was {} instead of '
                                         '500000'.format(global_step_val))

      logging.info('Loading checkpoint: %s', os.path.join(rb_ckpt))
      replay_buffer._clear(clear_all_variables=True)  # pylint: disable=protected-access
      misc.load_rb_ckpt(rb_ckpt, replay_buffer)
      rb_data = replay_buffer.gather_all()  # pylint: disable=unused-variable
      ds = iter(
          replay_buffer.as_dataset(
              sample_batch_size=batch_size,
              num_parallel_calls=n_parallel).prefetch(batch_size))
      safe_rew = agents.process_replay_buffer(replay_buffer, as_tensor=False)

    batch = next(ds)
    loss = train_step(tf_agent, safety_critic, batch, safe_rew, optimizer)
    avg_loss(loss)

    if b_num % log_freq == 0:
      logging.info('safety critic loss: %f', loss.numpy())
      tf.compat.v2.summary.scalar(
          name='safety_critic_loss', data=avg_loss.result(), step=b_num)
      avg_loss.reset_states()
    if b_num % eval_freq == 0:
      eval_metrics = eval_safety_critic(safety_critic, boundary_states,
                                        boundary_actions, safe_rew_boundary,
                                        boundary_step_type, metrics)
      with tf.name_scope('eval'):
        for m, m_val in zip(metrics, eval_metrics):
          tf.compat.v2.summary.scalar(name=m.name, data=m_val, step=b_num)
          m.reset_states()
  sc_checkpointer.save(global_step=n_batches)


def main(argv):
  del argv
  configs = FLAGS.gin_file or []
  bindings = FLAGS.gin_bindings or []

  gin.parse_config_files_and_bindings(configs, bindings, skip_unknown=True)
  load_base_path = FLAGS.load_base_path
  load_dirs = FLAGS.load_dirs.split(',')

  train_ckpts, rb_ckpts = [], []
  for load_dir in load_dirs:
    # lists multiple workers to combine replay buffers from
    logging.info('loading from: %s', osp.join(load_base_path, load_dir))
    train_ckpts.append(osp.join(load_base_path, load_dir, '1/train'))
    rb_ckpts.append(
        osp.join(load_base_path, load_dir, '1/train', 'replay_buffer'))

  env = load_environment()
  tf_env = tf_py_environment.TFPyEnvironment(env)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  tf_agent = safe_sac_agent.SafeSacAgent(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      train_step_counter=global_step)
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      tf_agent.collect_data_spec)
  test_agent = safe_sac_agent.SafeSacAgent(  # or use sac_agent.SacAgent
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      train_step_counter=global_step)
  test_rb = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      tf_agent.collect_data_spec)

  test_agent_ckpt = osp.join(FLAGS.load_base_path, FLAGS.test_dir, '1/train')
  test_rb_ckpt = osp.join(FLAGS.load_base_path, FLAGS.test_dir,
                          '1/train/replay_buffer')

  # loads and copies test replay buffer for validation
  misc.load_rb_ckpt(test_rb_ckpt, replay_buffer)
  misc.copy_rb(replay_buffer, test_rb)

  observation_spec, action_spec = (tf_env.observation_spec(),  # pylint: disable=unused-variable
                                   tf_env.action_spec())

  writer, train_path = misc.create_default_writer_and_save_dir(  # pylint: disable=unused-variable
      FLAGS.root_dir)

  safety_critic = tf_agent._safety_critic_network  # pylint: disable=protected-access
  safety_critic.create_variables()
  sc_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_path, 'safety_critic'),
      safety_critic=safety_critic,
      max_to_keep=5)
  sc_checkpointer.initialize_or_restore()

  optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

  train_sc(tf_agent, safety_critic, optimizer, replay_buffer, test_agent,
           test_rb, train_ckpts, rb_ckpts, test_agent_ckpt, test_rb_ckpt,
           sc_checkpointer)


if __name__ == '__main__':
  app.run(main)
