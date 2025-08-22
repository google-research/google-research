# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Data generation script.

Generates data for a given task using parallel workers.
"""

import functools
import multiprocessing as mp
import os
import random

from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
import hydra
import numpy as np
import tasks


def run_expert_policy(seed, task, env, agent):
  """Run expert policy on a given episode."""
  # Set seeds.
  np.random.seed(seed)
  random.seed(seed)

  env.set_task(task)
  obs = env.reset()
  info = env.info
  reward = 0
  episode = []
  total_reward = 0

  if not task.lang_goals:
    return [], 0

  for _ in range(task.max_steps):
    act = agent.act(obs, info)
    episode.append((obs, act, reward, info))
    lang_goal = info['lang_goal']
    obs, reward, done, info = env.step(act)
    total_reward += reward
    print(
        f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}'
    )
    if done:
      break

  # Add the final observation to the episode data.
  episode.append((obs, None, reward, info))
  return episode, total_reward


def build_dataset(cfg, indices):
  """Generates multiple episodes.

  Args:
    cfg: Hydra config containing configuration for the episode.
    indices: List of indices for which episode will be created.

  Raises:
    Exception: Invalid mode. Valid options are train, val and test.
    Exception: Seeds for val and test overlap
  """
  # Initialize environment and task.
  env = Environment(
      cfg['assets_root'],
      disp=cfg['disp'],
      shared_memory=cfg['shared_memory'],
      hz=480,
      record_cfg=cfg['record'])
  task = tasks.names[cfg['task']]()
  task.mode = cfg['mode']
  save_data = cfg['save_data']

  # Initialize scripted oracle agent and dataset.
  agent = task.oracle(env)
  data_path = os.path.join(
      cfg['data_dir'], '{}-{}-{}'.format(cfg['task'], task.mode,
                                         cfg['version']))
  dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
  print(f'Saving to: {data_path}')
  print(f'Mode: {task.mode}')

  # Train seeds are even and val/test seeds are odd.
  # Test seeds are offset by 10000
  seed = dataset.max_seed
  if seed < 0:
    if task.mode == 'train':
      seed = -2
    elif task.mode == 'val':  # NOTE: beware of increasing val set to >100
      seed = -1
    elif task.mode == 'test':
      seed = -1 + 100000
    else:
      raise Exception('Invalid mode. Valid options: train, val, test')

  if cfg['seed_addnm'] > 0:
    seed += cfg['seed_addnm']

  if indices:
    seed += 2 * (indices[0])

  # Collect training data from oracle demonstrations.
  for ep_id in indices:
    while True:
      seed += 2

      episode, total_reward = run_expert_policy(seed, task, env, agent)

      if not episode:
        print('Could not create episode!')
        # If the episode failed, increase the seed and try again.
        # increase the seed by 1000, +2 happens at the beginning of loop.
        seed += 998
        continue

      # Only save completed demonstrations.
      if save_data and total_reward > 0.99:
        print(seed, ep_id)
        dataset.add(seed, episode)
      else:
        # If the episode failed, increase the seed and try again.
        # increase the seed by 1000, +2 happens at the beginning of loop.
        seed += 998
        continue

      # If episode generated successfully, break and generate next.
      break


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
  pool = mp.Pool(cfg['p'])
  no_of_episodes = cfg['n']
  p_ids = np.array_split(np.arange(no_of_episodes), cfg['p'])
  par = functools.partial(build_dataset, cfg)
  pool.map(par, p_ids)


if __name__ == '__main__':
  main()  # pylint: disable=no-value-for-parameter
