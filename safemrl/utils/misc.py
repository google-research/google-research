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
"""Miscellanious utils for loading TF-Agent objects and env visualization.

Convenience methods to enable lightweight usage of TF-Agents library, env
visualization, and related uses.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import datetime
import os.path as osp
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from tf_agents.utils import common
from tensorflow.contrib import summary as contrib_summary


def load_rb_ckpt(ckpt_dir, replay_buffer, ckpt_step=None):
  rb_checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, max_to_keep=5, replay_buffer=replay_buffer)
  if ckpt_step is None:
    rb_checkpointer.initialize_or_restore().assert_existing_objects_matched()
  else:
    rb_checkpointer._checkpoint.restore(  # pylint: disable=protected-access
        osp.join(ckpt_dir, 'ckpt-{}'.format(ckpt_step)))
    rb_checkpointer._load_status.assert_existing_objects_matched()  # pylint: disable=protected-access
  return replay_buffer


def load_agent_ckpt(ckpt_dir, tf_agent, global_step=None):
  if global_step is None:
    global_step = tf.compat.v1.train.get_or_create_global_step()
  train_checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, agent=tf_agent, global_step=global_step)
  train_checkpointer.initialize_or_restore().assert_existing_objects_matched()
  return tf_agent, global_step


def cleanup_checkpoints(checkpoint_dir):
  checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint_state is None:
    return
  for checkpoint_path in checkpoint_state.all_model_checkpoint_paths:
    tf.compat.v1.train.remove_checkpoint(checkpoint_path)
  return


def copy_rb(rb_s, rb_t):
  for x1, x2 in zip(rb_s.variables(), rb_t.variables()):
    x2.assign(x1)
  return rb_t


def load_pi_ckpt(ckpt_dir, agent):
  train_checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, max_to_keep=1, agent=agent)
  train_checkpointer.initialize_or_restore().assert_existing_objects_matched()
  return agent.policy


def load_policies(agent, base_path, independent_runs):
  pi_loaded = []
  for run in independent_runs:
    pi_ckpt_path = osp.join(base_path, run, 'train/policies/')
    pi_loaded.append(load_pi_ckpt(pi_ckpt_path, agent))
  return pi_loaded


def create_default_writer_and_save_dir(root_dir):
  """Creates default directories."""
  base_dir = osp.expanduser(root_dir)
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  tag = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  tb_logdir = osp.join(base_dir, tag, 'tb')
  save_dir = osp.join(base_dir, tag, 'train')
  tf.io.gfile.makedirs(tb_logdir)
  tf.io.gfile.makedirs(save_dir)
  writer = contrib_summary.create_file_writer(tb_logdir)
  writer.set_as_default()
  return writer, save_dir


def record_episode_vis_summary(tf_env, tf_policy, savepath=None):
  """Records summaries."""
  time_step = tf_env.reset()
  policy_state = tf_policy.get_initial_state()
  states, actions = [], []
  while not time_step.is_last():
    action_step = tf_policy.action(time_step, policy_state)
    a = action_step.action.numpy()[0]
    actions.append(a)
    s = time_step.observation['observation'].numpy()[0]
    states.append(s)
    policy_state = action_step.state
    time_step = tf_env.step(action_step.action)

  wall_mat = tf_env._env.envs[0].walls.copy()  # pylint: disable=protected-access
  gx, gy = tf_env._env.envs[0]._goal  # pylint: disable=protected-access
  wall_mat[gx, gy] = 3
  w, h = wall_mat.shape
  f, ax = plt.subplots(figsize=(w * .8, h * .8))
  ax.matshow(wall_mat)
  ax.plot(states, c='r')
  ax.set_xticks([])
  ax.set_yticks([])
  if savepath:
    f.savefig(savepath)
    f.close()
