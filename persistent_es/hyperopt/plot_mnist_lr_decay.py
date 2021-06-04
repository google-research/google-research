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

"""Plot meta-optimization trajectories from saved CSV files for the MNIST LR decay experiment.

Example:
--------
python plot_mnist_lr_decay.py
"""
import os
import sys
import csv
import ipdb
import pickle as pkl
from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')

# Local imports
import plot_utils


flatui = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#02D4F9", "#4F4C4B",]
sns.set_palette(flatui)
sns.palplot(sns.color_palette())
dark_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

flatui = ["#FF349B", "#18DF29", "#674DEA", "#FF8031", "#02D4F9", "#4F4C4B",]
sns.set_palette(flatui)
sns.palplot(sns.color_palette())
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

if not os.path.exists('figures'):
    os.makedirs('figures')

plot_utils.plot_heatmap('saves/mnist_grid/grid_mnist_mlp_lr:inverse-time-decay_sgdm_T_5000_N_100/seed_3/result.pkl',
                        key='train_sum_loss',
                        xlabel='Decay',
                        ylabel='Initial LR',
                        cmap=plt.cm.Purples_r,
                        levels=10,
                        figsize=(8,6),
                       )

plt.xticks([-2, -1, 0, 1, 2], fontsize=18)
plt.yticks([-3, -2, -1, 0, 1], fontsize=18)

es_K100 = plot_utils.load_log('saves/mnist_lr_decay/train_sum_loss/es-mnist-mlp-obj:train_sum_loss-tune:lr:itd-T:5000-K:100-N:100-sigma:0.1-olr:0.1-seed:3', fname='frequent.csv')
es_K10 = plot_utils.load_log('saves/mnist_lr_decay/train_sum_loss/es-mnist-mlp-obj:train_sum_loss-tune:lr:itd-T:5000-K:10-N:100-sigma:0.1-olr:0.01-seed:3', fname='frequent.csv')
es_K1 = plot_utils.load_log('saves/mnist_lr_decay/train_sum_loss/es-mnist-mlp-obj:train_sum_loss-tune:lr:itd-T:5000-K:1-N:100-sigma:0.1-olr:0.001-seed:3', fname='frequent.csv')

pes_K100 = plot_utils.load_log('saves/mnist_lr_decay/train_sum_loss/pes-mnist-mlp-obj:train_sum_loss-tune:lr:itd-T:5000-K:100-N:100-sigma:0.1-olr:0.03-seed:3', fname='frequent.csv')
pes_K10 = plot_utils.load_log('saves/mnist_lr_decay/train_sum_loss/pes-mnist-mlp-obj:train_sum_loss-tune:lr:itd-T:5000-K:10-N:100-sigma:0.1-olr:0.001-seed:3', fname='frequent.csv')
pes_K1 = plot_utils.load_log('saves/mnist_lr_decay/train_sum_loss/pes-mnist-mlp-obj:train_sum_loss-tune:lr:itd-T:5000-K:1-N:100-sigma:0.1-olr:0.001-seed:3', fname='frequent.csv')

plt.plot(np.log10(np.exp(pes_K10['lr_1'])),
         np.log10(np.exp(pes_K10['lr_0'])),
         color=colors[1], linewidth=2, label='PES K=10')

plt.plot(np.log10(np.exp(es_K100['lr_1'])),
         np.log10(np.exp(es_K100['lr_0'])),
         linewidth=2, color=colors[0], linestyle='--', label='ES K=100')

plt.plot(np.log10(np.exp(es_K10['lr_1'])),
         np.log10(np.exp(es_K10['lr_0'])),
         linewidth=2, color=colors[1], linestyle='--', label='ES K=10')

plt.xlim(-2.2, 2.2)

plt.savefig('figures/mnist_lr_decay_es_pes.pdf', bbox_inches='tight', pad_inches=0)


# ================================================
# Plotting validation accuracy
# ================================================
plot_heatmap('saves/mnist_grid/grid_mnist_mlp_lr:inverse-time-decay_sgdm_T_5000_N_100/seed_3/result.pkl',
             key='val_acc',
             xlabel='Decay',
             ylabel='Initial LR',
             cmap=plt.cm.Purples_r,
             levels=10,
             figsize=(8,6),
            )

plt.xticks([-2, -1, 0, 1, 2], fontsize=18)
plt.yticks([-3, -2, -1, 0, 1], fontsize=18)

es_K10 = plot_utils.load_log('saves/mnist_lr_decay/val_sum_fixed_acc/es-mnist-mlp-obj:val_sum_fixed_acc-tune:lr:itd-T:5000-K:10-N:100-sigma:0.1-olr:0.01-seed:3', fname='frequent.csv')
pes_K10 = plot_utils.load_log('saves/mnist_lr_decay/val_sum_fixed_acc/pes-mnist-mlp-obj:val_sum_fixed_acc-tune:lr:itd-T:5000-K:10-N:100-sigma:0.1-olr:0.003-seed:3', fname='frequent.csv')

pes_K100 = plot_utils.load_log('saves/mnist_lr_decay/val_sum_fixed_acc/pes-mnist-mlp-obj:val_sum_fixed_acc-tune:lr:itd-T:5000-K:100-N:100-sigma:1.0-olr:0.01-seed:3', fname='frequent.csv')
es_K100 = plot_utils.load_log('saves/mnist_lr_decay/val_sum_fixed_acc/es-mnist-mlp-obj:val_sum_fixed_acc-tune:lr:itd-T:5000-K:100-N:100-sigma:1.0-olr:0.01-seed:3', fname='frequent.csv')

plt.plot(np.log10(np.exp(es_K10['lr_1'])),
         np.log10(np.exp(es_K10['lr_0'])),
         linewidth=2, color=colors[1], linestyle='--', label='ES K=10')

plt.plot(np.log10(np.exp(pes_K10['lr_1'])),
         np.log10(np.exp(pes_K10['lr_0'])),
         linewidth=2, color=colors[1], label='PES K=10')

plt.plot(np.log10(np.exp(es_K100['lr_1'])),
         np.log10(np.exp(es_K100['lr_0'])),
         linewidth=2, color=colors[0], linestyle='--', label='ES K=100')

plt.plot(np.log10(np.exp(pes_K100['lr_1'])),
         np.log10(np.exp(pes_K100['lr_0'])),
         linewidth=2, color=colors[0], label='PES K=100')

plt.savefig('figures/mnist_lr_decay_es_pes_val_acc.pdf', bbox_inches='tight', pad_inches=0)
