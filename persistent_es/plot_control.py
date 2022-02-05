"""Plot the results of continuous control experiments.

Example:
--------
python plot_control.py
"""
import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')

import plot_utils


figure_dir = 'figures/control'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)


exps = [
    ('ES K=1000', 'saves/control/es-K1000/es-Swimmer-v2-lr:0.3-sigma:0.3-N:10-T:1000-K:1000-c:0-d:0'),
    ('ES K=100', 'saves/control/es-K100/es-Swimmer-v2-lr:0.1-sigma:0.1-N:10-T:1000-K:100-c:0-d:0'),
    ('PES K=100', 'saves/control/pes-K100/pes-Swimmer-v2-lr:0.1-sigma:0.3-N:10-T:1000-K:100-c:0-d:0'),
]

for (label, exp_dir) in exps:
  steps = []
  rewards = []

  min_num_rewards = 1e9
  for seed_dir in os.listdir(exp_dir):
    log = plot_utils.load_log(os.path.join(exp_dir, seed_dir), fname='iteration.csv')
    steps.append(log['total_steps'])
    rewards.append(log['reward_mean'])

    if len(log['reward_mean']) < min_num_rewards:
      min_num_rewards = len(log['reward_mean'])

  for list_idx in range(len(rewards)):
    rewards[list_idx] = rewards[list_idx][:min_num_rewards]

  means = np.array(rewards).mean(axis=0)
  stds = np.array(rewards).std(axis=0)

  steps = steps[0][:min_num_rewards]
  plt.plot(steps, means, linewidth=2, label=label)
  plt.fill_between(steps, means - stds, means+stds, linewidth=2, alpha=0.3)

plt.title('Swimmer-v2', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Total Environment Steps', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.ylim(0, 400)
plt.xlim(-1000, 5e5)
plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
plt.grid()
plt.gca().xaxis.set_major_formatter(
    tick.FuncFormatter(plot_utils.reformat_large_tick_values)
)
sns.despine()

plt.savefig(os.path.join(figure_dir, 'swimmer.pdf'),
            bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(figure_dir, 'swimmer.png'),
            bbox_inches='tight', pad_inches=0)
