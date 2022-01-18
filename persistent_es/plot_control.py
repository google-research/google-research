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

"""Plot the results of continuous control experiments.

Example:
--------
python plot_control.py
"""
import os
import csv
import ipdb
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns

sns.set_style('white')
sns.set_palette('bright')

def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as
    4500 into 4.5K and also appropriately turns 4000 into 4K
    (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since
    # that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]

    return new_tick_format

def load_log(exp_dir, log_filename='iteration.csv'):
    result_dict = defaultdict(list)
    with open(os.path.join(exp_dir, log_filename), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row:
                try:
                    if key in ['global_iteration', 'iteration', 'epoch']:
                        result_dict[key].append(int(row[key]))
                    else:
                        result_dict[key].append(float(row[key]))
                except:
                    pass
    return result_dict


exps = [('ES K=1000', 'es-K1000/es-Swimmer-v2-lr:0.1-sigma:0.1-N:10-T:1000-K:1000-c:0-d:0'),
        ('ES K=100', 'es-K100/es-Swimmer-v2-lr:0.1-sigma:0.1-N:10-T:1000-K:100-c:0-d:0'),
        ('PES K=100', 'pes-K100/pes-Swimmer-v2-lr:0.1-sigma:0.3-N:10-T:1000-K:100-n:0-c:0-d:1'),
       ]

for (label, exp_dir) in exps:
    steps = []
    rewards = []

    min_num_rewards = 1e9
    for seed_dir in os.listdir(exp_dir):
        log = load_log(os.path.join(exp_dir, seed_dir))
        steps.append(log['total_steps'])
        rewards.append(log['reward'])

        if len(log['reward']) < min_num_rewards:
            min_num_rewards = len(log['reward'])

    for list_idx in range(len(rewards)):
        rewards[list_idx] = rewards[list_idx][:min_num_rewards]

    means = np.array(rewards).mean(axis=0)
    mins = np.array(rewards).min(axis=0)
    maxes = np.array(rewards).max(axis=0)

    steps = steps[0][:min_num_rewards]
    plt.plot(steps, means, linewidth=2, label=label)
    plt.fill_between(steps, mins, maxes, linewidth=2, alpha=0.3)

plt.title('Swimmer-v2', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Total Environment Steps', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.ylim(0, 400)
plt.xlim(-1000, 5e5)
plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
plt.grid()
plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
sns.despine()
plt.savefig('control.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('control.png', bbox_inches='tight', pad_inches=0, dpi=300)
