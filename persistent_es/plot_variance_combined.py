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

"""This script plots the empirical variance measurements for different

scenarios of the toy RNN task.

Examples:
---------
First, you must run the RNN variance measurement experiments as follows:

python rnn_variance.py --scenario=real
python rnn_variance.py --scenario=random
python rnn_variance.py --scenario=repeat


Then, the results of these runs can be plotted using:

python plot_variance_combined.py
"""
import os
import pickle as pkl

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
sns.set_palette('bright')

# To plot the variance for PES
fname_real = 'saves/rnn_variance/pes_lstm_variance_dict_real.pkl'
fname_random = 'saves/rnn_variance/pes_lstm_variance_dict_random.pkl'
fname_repeat = 'saves/rnn_variance/pes_lstm_variance_dict_repeat.pkl'

# To plot the variance for PES-Analytic
# fname_real = 'saves/rnn_variance_analytic/pes_lstm_variance_dict_real.pkl'
# fname_random = 'saves/rnn_variance_analytic/pes_lstm_variance_dict_random.pkl'
# fname_repeat = 'saves/rnn_variance_analytic/pes_lstm_variance_dict_repeat.pkl'

figure_dir = 'figures/variance'
# figure_dir = 'figures/variance_analytic'

with open(fname_real, 'rb') as f:
  var_real = pkl.load(f)

with open(fname_random, 'rb') as f:
  var_random = pkl.load(f)

with open(fname_repeat, 'rb') as f:
  var_repeat = pkl.load(f)

colors = ['#E00072', '#00830B', '#2B1A7F', '#E06111', '#4F4C4B', '#02D4F9']
sns.set_palette(colors)

plt.figure(figsize=(7, 5))
for i, N_pert in enumerate([1000]):
  num_unrolls_list = []
  variances_real = []
  variances_random = []
  variances_repeat = []

  for (K, num_unrolls, N) in var_real['variance_dict']:
    if N == N_pert:
      var = var_real['variance_dict'][(K, num_unrolls,
                                       N)] / var_real['total_grad_norm']
      num_unrolls_list.append(num_unrolls)
      variances_real.append(var)

  for (K, num_unrolls, N) in var_random['variance_dict']:
    if N == N_pert:
      var = var_random['variance_dict'][(K, num_unrolls,
                                         N)] / var_random['total_grad_norm']
      variances_random.append(var)

  for (K, num_unrolls, N) in var_repeat['variance_dict']:
    if N == N_pert:
      var = var_repeat['variance_dict'][(K, num_unrolls,
                                         N)] / var_repeat['total_grad_norm']
      variances_repeat.append(var)

  plt.plot(
      num_unrolls_list,
      variances_real,
      linewidth=3,
      color=colors[0],
      marker='o',
      label='Real Seq.')
  plt.plot(
      num_unrolls_list,
      variances_random,
      linewidth=3,
      color=colors[1],
      marker='o',
      label='Random Seq.')
  plt.plot(
      num_unrolls_list,
      variances_repeat,
      linewidth=3,
      color=colors[2],
      marker='o',
      label='Repeated Seq.')

plt.xlabel('# Unrolls', fontsize=22)
plt.ylabel('Variance', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20, fancybox=True, framealpha=0.3, loc='upper left', ncol=1)
sns.despine()

if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

figure_fname = 'pes_combined_variance_real_random_repeat'
plt.savefig(
    os.path.join(figure_dir, '{}.pdf'.format(figure_fname)),
    bbox_inches='tight',
    pad_inches=0)
plt.savefig(
    os.path.join(figure_dir, '{}.png'.format(figure_fname)),
    bbox_inches='tight',
    pad_inches=0)
