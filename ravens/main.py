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

#!/usr/bin/env python
"""Ravens main training script."""

import argparse
import datetime
import os

import pickle
import numpy as np

from ravens import agents
from ravens import Dataset
from ravens import Environment
from ravens import tasks

import tensorflow as tf


def rollout(agent, env, task):
  """Standard gym environment rollout.

  We do a slight hack to save the last observation and last info, which we
  need for goal-conditioning Transporters, and also inspection of data to
  check what the final images look like after the last action.

  For inspecting performance in more detail, it is recommended to print
  `info['extras']` at each time step, with --disp turned on.

  Args:
    agent: Agent to sample
    env: Ravens environemnt.
    task: Ravens task.

  Returns:
    total_reward: scalar reward signal from the episode, usually 1.0 for
      demonstrators of standard ravens tasks.
    episode: a list of (obs,act,info) tuples used to add to the dataset,
      which formats it for sampling later in training.
    t: time steps (i.e., actions takens) for this episode. If t=0 then
      something bad happened and we shouldn't count this episode.
    last_obs_info: tuple of the (obs,info) at the final time step, the
      one that doesn't have an action.
  """
  episode = []
  total_reward = 0
  obs = env.reset(task)
  info = env.info
  for t in range(task.max_steps):
    act = agent.act(obs, info)
    if obs and act['primitive']:
      episode.append((obs, act, info))
    (obs, reward, done, info) = env.step(act)
    total_reward += reward
    last_obs_info = (obs, info)
    if done:
      break
  return total_reward, episode, t, last_obs_info


def skip_testing_during_training(task):
  """Filter to determine if we should be running test-time evaluation.

  In cloth and bag tasks, we need `--disp` (at least with PyBullet 2.8.4),
  and that causes problems if instantiating multiple `Environment`s, as in
  standard testing. Furthermore, all 'Deformable Ravens' tasks have finer
  grained evaluation criteria, and it is easier to have a dedicated script,
  `load.py`, which can process more information. We filter for all these,
  while being careful to avoid filtering 'cable'.

  Args:
    task: String representing the task name from the argument parser.

  Returns:
    bool indicating test-time evaluation.
  """
  return ('cable-' in task) or ('cloth' in task) or ('bag' in task)


def ignore_this_demo(args, total_reward, t, last_info):
  """Check to see if we should filter out a scripted demo episode.

  Several filters: if t==0, that means the initial state was a success. For
  the bag tasks, we can exit gracefully if reaching a failure state, which
  speeds up computation.

  Args:
    args: main arguments.
    total_reward: demonstration cumulative reward.
    t: num time steps.
    last_info: last observation info (last_obs_info).

  Returns:
    bool indicating if the demo should be ignored.
  """
  last_extras = last_info['extras']

  # For bag tasks.
  if 'exit_gracefully' in last_extras:
    assert last_extras['exit_gracefully']
    return True

  # Easier bag-items. We can get 0.5 reward by placing the cube only (bad).
  if args.task == 'bag-items-easy' and total_reward <= 0.5:
    return True

  # Harder: ignore if (a) no beads in zone, OR, (b) didn't get both items.
  if args.task == 'bag-items-hard':
    return (last_extras['zone_items_rew'] < 0.5 or
            last_extras['zone_beads_rew'] == 0)

  return t == 0


def main():
  # Parse command line arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default='0')
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--task', default='hanoi')
  parser.add_argument('--agent', default='transporter')
  parser.add_argument('--hz', default=240.0, type=float)
  parser.add_argument('--num_demos', default='100')
  parser.add_argument('--num_rots', default=36, type=int)
  parser.add_argument('--gpu_mem_limit', default=None)
  parser.add_argument('--subsamp_g', action='store_true')
  parser.add_argument('--crop_bef_q', default=1, type=int)
  args = parser.parse_args()

  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[int(args.gpu)], 'GPU')

  # Configure how much GPU to use.
  if args.gpu_mem_limit is not None:
    mem_limit = 1024 * int(args.gpu_mem_limit)
    print(args.gpu_mem_limit)
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

  # Initialize environment and task.
  env = Environment(args.disp, hz=args.hz)
  task = tasks.names[args.task]()
  dataset = Dataset(os.path.join('data', args.task))
  if args.subsamp_g:
    dataset.subsample_goals = True

  # Collect training data from oracle demonstrations.
  max_order = 3
  max_demos = 10**max_order
  task.mode = 'train'
  seed_toadd_train = 0

  while dataset.num_episodes < max_demos:
    seed = dataset.num_episodes + seed_toadd_train
    np.random.seed(seed)
    print(f'Demonstration: {dataset.num_episodes + 1}/{max_demos}, seed {seed}')
    total_reward, episode, t, last_obs_info = rollout(
        task.oracle(env), env, task)

    # Check if episode should be added, if not, then add seed offset.
    _, last_info = last_obs_info
    if ignore_this_demo(args, total_reward, t, last_info):
      seed_toadd_train += 1
      li = last_info['extras']
      print(f'Ignoring demo. {li}, seed_toadd: {seed_toadd_train}')
    else:
      dataset.add(episode, last_obs_info)

  # Collect validation dataset with different random seeds.
  validation_dataset = Dataset(os.path.join('validation_data', args.task))
  num_validation = 100
  seed_tosub_valid = 0

  while validation_dataset.num_episodes < num_validation:
    seed = 2**32 - 1 - validation_dataset.num_episodes - seed_tosub_valid
    np.random.seed(seed)
    print(
        f'Validation Demonstration: {validation_dataset.num_episodes + 1}/{num_validation}, seed {seed}'
    )
    total_reward, episode, t, last_obs_info = rollout(
        task.oracle(env), env, task)

    # Check if episode should be added, if not, then subtract seed offset.
    _, last_info = last_obs_info
    if ignore_this_demo(args, total_reward, t, last_info):
      seed_tosub_valid += 1
      li = last_info['extras']
      print(f'Ignoring demo. {li}, seed_tosub: {seed_tosub_valid}')
    else:
      validation_dataset.add(episode, last_obs_info)

  env.stop()
  del env

  # Evaluate on increasing orders of magnitude of demonstrations.
  num_train_runs = 1  # 3+ to measure variance over random initialization
  num_train_iters = 40000
  test_interval = 2000
  num_test_episodes = 20

  # there are a few seeds that the oracle
  # can't complete either, skip these
  # TODO(peteflorence): compute this automatically for each task
  oracle_cant_complete_seed = []
  if args.task == 'insertion-sixdof':
    oracle_cant_complete_seed.append(3)
  num_test_episodes += len(oracle_cant_complete_seed)

  # Do multiple training runs from scratch.
  for train_run in range(num_train_runs):

    # Set up tensorboard logger.
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = os.path.join('logs', args.agent, args.task, current_time,
                                 'train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Set the beginning of the agent name. We may add more to it.
    name = f'{args.task}-{args.agent}-{args.num_demos}-{train_run}'

    # Initialize agent and limit random dataset sampling to fixed set.
    tf.random.set_seed(train_run)
    if args.agent == 'transporter':
      name = f'{name}-rots-{args.num_rots}-crop_bef_q-{args.crop_bef_q}'
      agent = agents.names[args.agent](
          name,
          args.task,
          num_rotations=args.num_rots,
          crop_bef_q=(args.crop_bef_q == 1))
    elif 'transporter-goal' in args.agent:
      # For transporter-goal and transporter-goal-naive agents.
      name = f'{name}-rots-{args.num_rots}'
      if args.subsamp_g:
        name += '-sub_g'
      else:
        name += '-fin_g'
      agent = agents.names[args.agent](
          name, args.task, num_rotations=args.num_rots)
    else:
      agent = agents.names[args.agent](name, args.task)
    np.random.seed(train_run)
    num_demos = int(args.num_demos)
    train_episodes = np.random.choice(range(max_demos), num_demos, False)
    dataset.set(train_episodes)
    # agent.load(10000)

    performance = []
    while agent.total_iter < num_train_iters:

      # Train agent.
      tf.keras.backend.set_learning_phase(1)
      agent.train(
          dataset,
          num_iter=test_interval,
          writer=train_summary_writer,
          validation_dataset=validation_dataset)
      tf.keras.backend.set_learning_phase(0)

      # Skip evaluation depending on the task or if it's a goal-based agent.
      if (skip_testing_during_training(args.task) or
          'transporter-goal' in args.agent):
        continue

      # Evaluate agent.
      task.mode = 'test'
      env = Environment(args.disp, hz=args.hz)
      for episode in range(num_test_episodes):
        if episode in oracle_cant_complete_seed:
          continue
        np.random.seed(10**max_order + episode)
        total_reward, _, _, _ = rollout(agent, env, task)
        print(f'Test: {episode} Total Reward: {total_reward:.2f}')
        performance.append((agent.total_iter, total_reward))
      env.stop()
      del env

      # Save results.
      pickle.dump(performance, open(f'{name}.pkl', 'wb'))


if __name__ == '__main__':
  main()
