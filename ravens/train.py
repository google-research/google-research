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

import numpy as np
from ravens import agents
from ravens import Dataset
import tensorflow as tf


def main():

  # Parse command line arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', default='insertion')
  parser.add_argument('--agent', default='transporter')
  parser.add_argument('--n_demos', default=100, type=int)
  parser.add_argument('--n_steps', default=40000, type=int)
  parser.add_argument('--n_runs', default=1, type=int)
  parser.add_argument('--interval', default=1000, type=int)
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

  # Load train and test datasets.
  train_dataset = Dataset(os.path.join('data', f'{args.task}-train'))
  test_dataset = Dataset(os.path.join('data', f'{args.task}-test'))

  # Run training from scratch multiple times.
  for train_run in range(args.n_runs):
    name = f'{args.task}-{args.agent}-{args.n_demos}-{train_run}'

    # Set up tensorboard logger.
    curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('logs', args.agent, args.task, curr_time, 'train')
    writer = tf.summary.create_file_writer(log_dir)

    # Initialize agent.
    np.random.seed(train_run)
    tf.random.set_seed(train_run)
    agent = agents.names[args.agent](name, args.task)

    # Limit random sampling during training to a fixed dataset.
    max_demos = train_dataset.n_episodes
    episodes = np.random.choice(range(max_demos), args.n_demos, False)
    train_dataset.set(episodes)

    # Train agent and save snapshots.
    while agent.total_steps < args.n_steps:
      for _ in range(args.interval):
        agent.train(train_dataset, writer)
      agent.validate(test_dataset, writer)
      agent.save()

if __name__ == '__main__':
  main()
