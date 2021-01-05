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

"""Test script."""

import os
import time

from gym.wrappers.monitor import Monitor  # pylint: disable=unused-import

import numpy as np
import tensorflow as tf

from tf_agents.environments import tf_py_environment
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common, nest_utils  # pylint: disable=g-multiple-import

from reset_free_learning.envs import kitchen
from reset_free_learning.envs import playpen
from reset_free_learning.envs import playpen_reduced
from reset_free_learning.envs import point_mass
from reset_free_learning.envs import point_mass_full_goal
from reset_free_learning.envs import pusher2d_simple
from reset_free_learning.envs import sawyer_door_close
from reset_free_learning.envs import sawyer_object
from reset_free_learning.reset_free_wrapper import GoalTerminalResetFreeWrapper  # pylint: disable=unused-import
from reset_free_learning.reset_free_wrapper import GoalTerminalResetWrapper
from reset_free_learning.reset_free_wrapper import ResetFreeWrapper  # pylint: disable=unused-import


def print_initial_state(step):
  obs_strings = [str(step.observation[idx]) for idx in range(74 // 2)]
  obs_to_string = '[' + ','.joing(obs_strings) + ']'
  print(obs_to_string)


def get_env(name='sawyer_object', **env_kwargs):
  if name == 'sawyer_object':
    env = sawyer_object.SawyerObject(  # pylint: disable=redefined-outer-name
        random_init=True,
        task_type='push',
        obs_type='with_goal',
        goal_low=(-0.1, 0.8, 0.05),
        goal_high=(0.1, 0.9, 0.3),
        liftThresh=0.04,
        sampleMode='equal',
        rewMode='orig',
        rotMode='fixed')
    env.set_camera_view(view='topview')
    env.set_max_path_length(int(1e8))
  if name == 'pusher2d_simple':
    env = pusher2d_simple.PusherEnv()
  if name == 'point_mass':
    env = point_mass.PointMassEnv(**env_kwargs)
  if name == 'point_mass_full_goal':
    env = point_mass_full_goal.PointMassEnv(**env_kwargs)
  if name == 'sawyer_door':
    env = sawyer_door_close.SawyerDoor(random_init=True, obs_type='with_goal')
    env.set_camera_view(view='topview')
    env.set_max_path_length(int(1e8))
  if name == 'kitchen':
    env = kitchen.Kitchen()
  if name == 'playpen':
    env = playpen.ContinuousPlayPen(**env_kwargs)
  if name == 'playpen_reduced':
    env = playpen_reduced.ContinuousPlayPen(**env_kwargs)
  return env


def copy_replay_buffer(small_buffer, big_buffer):
  """Copy small buffer into the big buffer."""
  all_data = nest_utils.unbatch_nested_tensors(small_buffer.gather_all())
  for trajectory in nest_utils.unstack_nested_tensors(  # pylint: disable=redefined-outer-name
      all_data, big_buffer.data_spec):
    big_buffer.add_batch(trajectory)


def data_multiplier(offline_data, reward_fn):  # pylint: disable=redefined-outer-name
  """Offline data multiplication."""
  np.set_printoptions(precision=2, suppress=True)

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
    print('index:', idx)
    if traj.step_type.numpy() == 0:
      ep_start_idx = idx
      print('\n\n\nnew start index:', ep_start_idx)
    elif idx in [12, 24, 36, 48, 60, 72, 84, 96, 108]:
      print('adding new trajectory')
      obs_dim = traj.observation.shape[0] // 2
      relabel_goal = traj.observation[:obs_dim]
      print('new goal:', repr(relabel_goal.numpy()))

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
            offline_data.add_batch(new_traj)
            break
          else:
            new_traj = cur_trajectory._replace(
                observation=new_obs,
                reward=new_reward,
            )
            # _custom_print(new_traj)
            offline_data.add_batch(new_traj)

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
      offline_data.add_batch(last_traj)
      print('new size:', offline_data.num_frames())


if __name__ == '__main__':
  max_episode_steps = 5000000
  # env = get_env(name='point_mass_full_goal', env_type='y', reward_type='sparse')
  # env = get_env(name='kitchen')
  env = get_env(name='playpen_reduced', task_list='rc_o', reward_type='sparse')

  base_dir = os.path.abspath('experiments/env_logs/playpen_reduced/symmetric/')
  env_log_dir = os.path.join(base_dir, 'rc_o/traj1/')
  # env = ResetFreeWrapper(env, reset_goal_frequency=500, full_reset_frequency=max_episode_steps)
  env = GoalTerminalResetWrapper(
      env,
      episodes_before_full_reset=max_episode_steps // 500,
      goal_reset_frequency=500)
  # env = Monitor(env, env_log_dir, video_callable=lambda x: x % 1 == 0, force=True)

  env = wrap_env(env)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  tf_env.render = env.render
  time_step_spec = tf_env.time_step_spec()
  action_spec = tf_env.action_spec()
  policy = random_tf_policy.RandomTFPolicy(
      action_spec=action_spec, time_step_spec=time_step_spec)
  collect_data_spec = trajectory.Trajectory(
      step_type=time_step_spec.step_type,
      observation=time_step_spec.observation,
      action=action_spec,
      policy_info=policy.info_spec,
      next_step_type=time_step_spec.step_type,
      reward=time_step_spec.reward,
      discount=time_step_spec.discount)
  offline_data = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=collect_data_spec, batch_size=1, max_length=int(1e5))
  rb_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(env_log_dir, 'replay_buffer'),
      max_to_keep=10_000,
      replay_buffer=offline_data)
  rb_checkpointer.initialize_or_restore()

  # replay buffer copy magic
  do_a_copy = False
  if do_a_copy:
    buffer_list = [
        os.path.join(base_dir, 'rc_o/combined/replay_buffer'),
        os.path.join(base_dir, 'rc_k/combined/replay_buffer'),
        os.path.join(base_dir, 'rc_p/combined/replay_buffer'),
        os.path.join(base_dir, 'rc_b/combined/replay_buffer'),
    ]

    for buffer_dir in buffer_list:
      loaded_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          data_spec=collect_data_spec, batch_size=1, max_length=int(1e5))
      cur_checkpointer = common.Checkpointer(
          ckpt_dir=buffer_dir, max_to_keep=10_000, replay_buffer=loaded_buffer)
      print(loaded_buffer.num_frames())
      copy_replay_buffer(loaded_buffer, offline_data)

    rb_checkpointer.save(global_step=0)

  start_time = time.time()
  # env.do_custom_reset(pos=np.array([0, 8, 1.57]))
  time_step = tf_env.reset()
  # print_initial_state(time_step)
  # print(time_step.observation)

  step_size = 0.5
  command_to_action_map = {
      'd': [step_size, 0],
      'a': [-step_size, 0],
      'w': [0, step_size],
      's': [0, -step_size],
      'x': [0, 0]
  }
  pick_drop_map = {'p': [1], 'l': [-1]}

  print(offline_data.num_frames())
  # data_multiplier(offline_data, tf_env.pyenv.envs[0].env.compute_reward)
  # print(offline_data.num_frames())
  # print(offline_data.gather_all())
  # exit()
  rb_checkpoint_idx = 0

  for i in range(1, 2000):
    tf_env.render(mode='human')
    print(time_step)
    action_step = policy.action(time_step)

    # get action from user
    command = input('action:')
    if len(command) > 1:
      action = np.concatenate(
          [command_to_action_map[command[0]], pick_drop_map[command[1]]])
    else:
      action = np.concatenate([command_to_action_map[command[0]], [1]])

    # add noise to action
    action[:2] += np.random.uniform(low=-0.1, high=0.1, size=2)
    action_step = action_step._replace(
        action=tf.constant([action], dtype=tf.float32))

    next_time_step = tf_env.step(action_step.action)
    print('reward:', next_time_step.reward)
    offline_data.add_batch(
        trajectory.from_transition(time_step, action_step, next_time_step))

    if next_time_step.is_last():
      # print(i, env.get_info())
      time_step = next_time_step
      print('last step:', time_step)
      next_time_step = tf_env.step(action_step.action)  # dummy action for reset
      offline_data.add_batch(
          trajectory.from_transition(time_step, action_step, next_time_step))

      command = input('save offline data?')
      if command == 'y':
        print('saving data')
        rb_checkpointer.save(global_step=rb_checkpoint_idx)
        rb_checkpoint_idx += 1
      elif command == 'c':
        print('clearing data')
        offline_data.clear()
      else:
        print('not saving data')

    time_step = next_time_step

  print('time:', time.time() - start_time)


# dummy stuff to store plotting code
else:
  import tensorflow as tf  # tf  # pylint: disable=g-import-not-at-top

  import matplotlib  # pylint: disable=g-import-not-at-top, unused-import
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top, unused-import

  import numpy as np  # pylint: disable=g-import-not-at-top, reimported
  import seaborn as sns  # pylint: disable=g-import-not-at-top, unused-import
  import re  # pylint: disable=g-import-not-at-top, unused-import

  import pickle as pkl  # pylint: disable=g-import-not-at-top, unused-import
  import os  # pylint: disable=g-import-not-at-top, reimported

  max_index = int(1e7)

  def smooth(x, alpha):
    if isinstance(x, list):
      size = len(x)
    else:
      size = x.shape[0]
    for idx in range(1, size):
      x[idx] = (1 - alpha) * x[idx] + alpha * x[idx - 1]
    return x

  def make_graph_with_variance(vals, x_interval):
    data_x = []
    data_y = []
    global max_index

    for y_coords, eval_interval in zip(vals, x_interval):
      data_y.append(smooth(y_coords, 0.95))
      x_coords = [eval_interval * idx for idx in range(len(y_coords))]
      data_x.append(x_coords)

    plot_dict = {}
    cur_max_index = max_index
    # for cur_x, cur_y in zip(data_x, data_y):
    #   cur_max_index = min(cur_max_index, cur_x[-1])
    # print(cur_max_index)

    for cur_x, cur_y in zip(data_x, data_y):
      for x, y in zip(cur_x, cur_y):
        if x <= cur_max_index:
          if x in plot_dict.keys():
            plot_dict[x].append(y)
          else:
            plot_dict[x] = [y]

    index, means, stds = [], [], []
    for key in sorted(plot_dict.keys()):  # pylint: disable=g-builtin-op
      index.append(key)
      means.append(np.mean(plot_dict[key]))
      stds.append(np.std(plot_dict[key]))
    means = np.array(smooth(means, 0.9))
    stds = np.array(smooth(stds, 0.8))
    return index, means, stds

  def np_custom_load(fname):
    with tf.gfile.Open(fname, 'rb') as f:
      load_file = np.load(f).astype(np.float32)
    return load_file

  color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
  style_map = []
  for line_style in ['-', '--', '-.', ':']:
    style_map += [color + line_style for color in color_map]

  def plot_call(job_id,
                worker_ids,
                legend_label,
                plot_style,
                file_path,
                y_plot='return'):
    """Outermost function for plotting graphs with variance."""
    print(worker_ids)
    job_id = str(job_id)
    if y_plot == 'return':
      y_coords = [
          np_custom_load(file_path +  # pylint: disable=g-complex-comprehension
                         job_id + '/' + worker_id +
                         '/eval/average_eval_return.npy')
          for worker_id in worker_ids
      ]
    elif y_plot == 'success':
      y_coords = [
          np_custom_load(file_path +  # pylint: disable=g-complex-comprehension
                         job_id + '/' + worker_id +
                         '/eval/average_eval_success.npy')
          for worker_id in worker_ids
      ]
    eval_interval = [
        np_custom_load('/home/architsh/brain/reset_free/reset_free/' + job_id +
                       '/' + worker_id + '/eval/eval_interval.npy')
        for worker_id in worker_ids
    ]
    index, means, stds = make_graph_with_variance(y_coords, eval_interval)
    plt.plot(index, means, plot_style, label=legend_label)
    cur_color = plot_style[0]
    plt.fill_between(
        index, means - stds, means + stds, color=cur_color, alpha=0.2)
