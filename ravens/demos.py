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
"""Data collection script."""

import argparse
import os

import numpy as np

from ravens import Dataset
from ravens import Environment
from ravens import tasks


def main():

  # Parse command line arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--task', default='insertion')
  parser.add_argument('--mode', default='train')
  parser.add_argument('--n', default=1000, type=int)
  args = parser.parse_args()

  # Initialize environment and task.
  env = Environment(args.disp, hz=480)
  task = tasks.names[args.task]()
  task.mode = args.mode

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env)
  dataset = Dataset(os.path.join('data', f'{args.task}-{task.mode}'))

  # Train seeds are even and test seeds are odd.
  seed = dataset.max_seed
  if seed < 0:
    seed = -1 if (task.mode == 'test') else -2

  # Collect training data from oracle demonstrations.
  while dataset.n_episodes < args.n:
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{args.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    obs, reward, _, info = env.reset(task)
    for _ in range(task.max_steps):
      act = agent.act(obs, info)
      episode.append((obs, act, reward, info))
      obs, reward, done, info = env.step(act)
      total_reward += reward
      print(f'{done} {total_reward}')
      if done:
        break
    episode.append((obs, None, reward, info))

    # Only save completed demonstrations.
    # TODO(andyzeng): add back deformable logic.
    if total_reward > 0.99:
      dataset.add(seed, episode)

if __name__ == '__main__':
  main()
