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

#!/usr/bin/env python
"""Script to plot training results."""

import argparse
import os

import pickle
import numpy as np

from ravens import utils


def main():

  # Parse command line arguments.
  parser = argparse.ArgumentParser()
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--task', default='insertion')
  parser.add_argument('--agent', default='transporter')
  parser.add_argument('--n_demos', default=100, type=int)
  args = parser.parse_args()

  name = f'{args.task}-{args.agent}-{args.n_demos}-'
  print(name)

  # Load and print results to console.
  path = os.path.join('.')
  curve = []
  for fname in sorted(os.listdir(path)):
    if name in fname and '.pkl' in fname:
      n_steps = int(fname[(fname.rfind('-') + 1):-4])
      data = pickle.load(open(fname, 'rb'))
      rewards = []
      for reward, _ in data:
        rewards.append(reward)
      rewards = np.array(rewards) * 100
      score = np.mean(rewards)
      std = np.std(rewards)
      print(f'  {n_steps} steps:\t{score:.1f}%\t± {std:.1f}%')
      curve.append((n_steps, score, std))

  # Plot results over training steps.
  title = f'{args.agent} on {args.task} w/ {args.n_demos} demos'
  ylabel = 'Testing Task Success (%)'
  xlabel = '# of Training Steps'
  if args.disp:
    logs = {}
    curve = np.array(curve)
    logs[name] = (curve[:, 0], curve[:, 1], curve[:, 2])
    fname = f'{name}-plot.png'
    utils.plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
    print(f'Done. Plot image saved to: {fname}')

if __name__ == '__main__':
  main()
