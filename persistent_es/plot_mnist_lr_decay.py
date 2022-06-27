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

"""Plot meta-optimization trajectories from saved CSV files for the MNIST LR

decay experiment.

Example:
--------
python plot_mnist_lr_decay.py
"""
import os
import pdb
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')

# Local imports
import plot_utils


mycolors = [
    '#FF349B',
    '#18DF29',
    '#674DEA',
    '#FF8031',
    '#02D4F9',
    '#4F4C4B',
]
sns.set_palette(mycolors)
sns.palplot(sns.color_palette())
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

figure_dir = 'figures/mnist_heatmaps'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

# Plot train loss heatmap
# ---------------------------------------------------------------------------
plot_utils.plot_heatmap(
    'saves/mnist_grid_sgdm/grid_mnist_mlp_lr:inverse-time-decay_sgdm_T_5000_N_40/seed_3/result.pkl',
    key='unroll_obj',
    xlabel='Log LR Decay',
    ylabel='Log Init LR',
    cmap=plt.cm.Purples_r,
    levels=30,
    sigma=1.0,
    use_smoothing=True,
    show_contours=True,
    contour_alpha=0.2,
    figsize=(8, 6),
)

es_K10 = plot_utils.load_log(
    'saves/mnist_lr_decay/train_sum_loss/es-mnist-mlp-obj:train_sum_loss-tune:lr:inverse-time-decay-T:5000-K:10-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')
pes_K10 = plot_utils.load_log(
    'saves/mnist_lr_decay/train_sum_loss/pes-mnist-mlp-obj:train_sum_loss-tune:lr:inverse-time-decay-T:5000-K:10-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')

es_K100 = plot_utils.load_log(
    'saves/mnist_lr_decay/train_sum_loss/es-mnist-mlp-obj:train_sum_loss-tune:lr:inverse-time-decay-T:5000-K:100-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')
pes_K100 = plot_utils.load_log(
    'saves/mnist_lr_decay/train_sum_loss/pes-mnist-mlp-obj:train_sum_loss-tune:lr:inverse-time-decay-T:5000-K:100-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')

plt.plot(
    np.log10(np.exp(pes_K10['lr_1']))[:2000],
    np.log10(np.exp(pes_K10['lr_0']))[:2000],
    color=colors[1],
    linewidth=3,
    label='PES K=10')

plt.plot(
    np.log10(np.exp(es_K10['lr_1'])),
    np.log10(np.exp(es_K10['lr_0'])),
    linewidth=3,
    color=colors[1],
    linestyle='--',
    label='ES K=10')

plt.plot(
    np.log10(np.exp(pes_K100['lr_1'])),
    np.log10(np.exp(pes_K100['lr_0'])),
    linewidth=3,
    color=colors[0],
    label='PES K=100')

plt.plot(
    np.log10(np.exp(es_K100['lr_1'])),
    np.log10(np.exp(es_K100['lr_0'])),
    linewidth=3,
    color=colors[0],
    linestyle='--',
    label='ES K=100')

plt.text(
    -1.6,
    -1.55,
    'PES K=10',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[1]
    })
plt.text(
    1.0,
    -1.4,
    'ES K=10',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[1]
    })

plt.text(
    -0.7,
    -1.05,
    'PES K=100',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[0]
    })
plt.text(
    0.9,
    -1,
    'ES K=100',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[0]
    })

plt.plot([np.log10(30)], [np.log10(0.003)],
         marker='*',
         markersize=15,
         color='k')
plt.text(
    1.3, -2.75, 'Init', fontdict={
        'size': 20,
        'weight': 'normal',
        'color': 'k'
    })

plt.xlim(-2.2, 2.2)
plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=18)
plt.yticks(fontsize=18)

plt.xlabel('Log LR Decay', fontsize=22)
plt.ylabel('Log Init LR', fontsize=22)

plt.savefig(
    os.path.join(figure_dir, 'mnist_train_loss.pdf'),
    bbox_inches='tight',
    pad_inches=0)
plt.savefig(
    os.path.join(figure_dir, 'mnist_train_loss.png'),
    bbox_inches='tight',
    pad_inches=0)
# ---------------------------------------------------------------------------

# Plot validation accuracy heatmap
# ---------------------------------------------------------------------------
plot_utils.plot_heatmap(
    'saves/mnist_grid_sgdm/grid_mnist_mlp_lr:inverse-time-decay_sgdm_T_5000_N_40/seed_3/result.pkl',
    key='val_acc',
    xlabel='Log LR Decay',
    ylabel='Log Init LR',
    cmap=plt.cm.Blues,
    levels=30,
    sigma=1.0,
    use_smoothing=True,
    show_contours=True,
    contour_alpha=0.4,
    figsize=(8, 6),
)

es_K10 = plot_utils.load_log(
    'saves/mnist_lr_decay_val_acc/val_sum_acc/es-mnist-mlp-obj:val_sum_acc-tune:lr:inverse-time-decay-T:5000-K:10-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')
pes_K10 = plot_utils.load_log(
    'saves/mnist_lr_decay_val_acc/val_sum_acc/pes-mnist-mlp-obj:val_sum_acc-tune:lr:inverse-time-decay-T:5000-K:10-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')

es_K100 = plot_utils.load_log(
    'saves/mnist_lr_decay_val_acc/val_sum_acc/es-mnist-mlp-obj:val_sum_acc-tune:lr:inverse-time-decay-T:5000-K:100-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')
pes_K100 = plot_utils.load_log(
    'saves/mnist_lr_decay_val_acc/val_sum_acc/pes-mnist-mlp-obj:val_sum_acc-tune:lr:inverse-time-decay-T:5000-K:100-nc:1-npc:1000-sigma:0.1-olr:0.01-ob1:0.9-ob2:0.999-ic:-1-oc:-1-seed:3',
    fname='frequent.csv')

plt.plot(
    np.log10(np.exp(pes_K10['lr_1']))[:1000],
    np.log10(np.exp(pes_K10['lr_0']))[:1000],
    color=colors[1],
    linewidth=4,
    label='PES K=10')

plt.plot(
    np.log10(np.exp(es_K10['lr_1'])),
    np.log10(np.exp(es_K10['lr_0'])),
    linewidth=4,
    color=colors[1],
    linestyle='--',
    label='ES K=10')

plt.plot(
    np.log10(np.exp(pes_K100['lr_1'])),
    np.log10(np.exp(pes_K100['lr_0'])),
    linewidth=3,
    color=colors[0],
    label='PES K=100')

plt.plot(
    np.log10(np.exp(es_K100['lr_1'])),
    np.log10(np.exp(es_K100['lr_0'])),
    linewidth=3,
    color=colors[0],
    linestyle='--',
    label='ES K=100')

plt.text(
    -1.3,
    -1.8,
    'PES K=10',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[1]
    })
plt.text(
    1.1,
    -1.5,
    'ES K=10',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[1]
    })

plt.text(
    -0.8,
    -1.1,
    'PES K=100',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[0]
    })
plt.text(
    0.9,
    -0.9,
    'ES K=100',
    fontdict={
        'size': 20,
        'weight': 'normal',
        'color': colors[0]
    })

plt.xlim(-2.2, 2.2)
plt.ylim(-3, -0.5)

plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=18)
plt.yticks(fontsize=18)

plt.xlabel('Log LR Decay', fontsize=22)
plt.ylabel('Log Init LR', fontsize=22)

plt.plot([np.log10(30)], [np.log10(0.003)],
         marker='*',
         markersize=15,
         color='k')
plt.text(
    1.3, -2.75, 'Init', fontdict={
        'size': 20,
        'weight': 'normal',
        'color': 'k'
    })

plt.savefig(
    os.path.join(figure_dir, 'mnist_val_acc.pdf'),
    bbox_inches='tight',
    pad_inches=0)
plt.savefig(
    os.path.join(figure_dir, 'mnist_val_acc.png'),
    bbox_inches='tight',
    pad_inches=0)
# ---------------------------------------------------------------------------
