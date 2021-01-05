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

#!/usr/bin/env python
"""Ravens main training script."""

import argparse
import os
import pickle

import numpy as np
from ravens import agents
from ravens import Dataset
from ravens import Environment
from ravens import tasks
import tensorflow as tf


def main():

  # Parse command line arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--task', default='insertion')
  parser.add_argument('--agent', default='transporter')
  parser.add_argument('--n_demos', default=100, type=int)
  parser.add_argument('--n_steps', default=40000, type=int)
  parser.add_argument('--n_runs', default=1, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--gpu_limit', default=None, type=int)
  args = parser.parse_args()

  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[args.gpu], 'GPU')

  # Configure how much GPU to use (in Gigabytes).
  if args.gpu_limit is not None:
    mem_limit = 1024 * args.gpu_limit
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

  # Initialize environment and task.
  env = Environment(args.disp, hz=480)
  task = tasks.names[args.task]()
  task.mode = 'test'

  # Load test dataset.
  dataset = Dataset(os.path.join('data', f'{args.task}-test'))

  # Run testing for each training run.
  for train_run in range(args.n_runs):
    name = f'{args.task}-{args.agent}-{args.n_demos}-{train_run}'

    # Initialize agent.
    np.random.seed(train_run)
    tf.random.set_seed(train_run)
    agent = agents.names[args.agent](name, args.task)

    # # Run testing every interval.
    # for train_step in range(0, args.n_steps + 1, args.interval):

    # Load trained agent.
    if args.n_steps > 0:
      agent.load(args.n_steps)

    # Run testing and save total rewards with last transition info.
    results = []
    for i in range(dataset.n_episodes):
      print(f'Test: {i + 1}/{dataset.n_episodes}')
      episode, seed = dataset.load(i)
      goal = episode[-1]
      total_reward = 0
      np.random.seed(seed)
      obs, reward, _, info = env.reset(task)
      for _ in range(task.max_steps):
        act = agent.act(obs, info, goal)
        obs, reward, done, info = env.step(act)
        total_reward += reward
        print(f'{done} {total_reward}')
        if done:
          break
      results.append((total_reward, info))

      # Save results.
      pickle.dump(results, open(f'{name}-{args.n_steps}.pkl', 'wb'))

if __name__ == '__main__':
  main()
