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

"""Plot loss curves from saved CSV files for the toy regression experiment.

Example:
--------
python plot_toy_regression.py
"""
import os
import csv
import ipdb
import pickle as pkl
from collections import defaultdict

import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')

# Darker colors
flatui = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#02D4F9", "#4F4C4B",]
sns.set_palette(flatui)
sns.palplot(sns.color_palette())

# Plotting from saved CSV files
def load_log(exp_dir, log_filename='train_log.csv'):
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

def plot_heatmap(pkl_path,
                 xlabel,
                 ylabel,
                 smoothed=False,
                 sigma=5.0,
                 cmap=plt.cm.viridis,
                 colorbar=True,
                 figsize=(10,8)):
    with open(pkl_path, 'rb') as f:
        heatmap_data = pkl.load(f)

    if smoothed:
        smoothed_F_grid = scipy.ndimage.gaussian_filter(heatmap_data['L_grid'], sigma=sigma)
        best_smoothed_theta = np.unravel_index(smoothed_F_grid.argmin(), smoothed_F_grid.shape)
        best_smoothed_x = heatmap_data['xv'][best_smoothed_theta]
        best_smoothed_y = heatmap_data['yv'][best_smoothed_theta]

        plt.figure(figsize=figsize)
        plt.pcolormesh(heatmap_data['xv'], heatmap_data['yv'], smoothed_F_grid, norm=colors.LogNorm(), cmap=cmap)
        if colorbar:
            plt.colorbar()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel(ylabel, fontsize=22)
    else:
        plt.figure(figsize=figsize)
        plt.pcolormesh(heatmap_data['xv'], heatmap_data['yv'], heatmap_data['L_grid'], norm=colors.LogNorm(), cmap=cmap)
        if colorbar:
            plt.colorbar()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(xlabel, fontsize=22)
        plt.ylabel(ylabel, fontsize=22)


if not os.path.exists('figures'):
    os.makedirs('figures')

tbptt_k10 = load_log('saves/toy_regression/tbptt-s:linear-optim:adam-lr:0.01-T:100-K:10-N:100-sigma:1.0-seed:1', 'iteration.csv')
rtrl_k10 = load_log('saves/toy_regression/rtrl-s:linear-optim:adam-lr:0.01-T:100-K:10-N:100-sigma:1.0-seed:1', 'iteration.csv')
uoro_k10 = load_log('saves/toy_regression/uoro-s:linear-optim:adam-lr:0.01-T:100-K:10-N:100-sigma:1.0-seed:1', 'iteration.csv')
es_k10 = load_log('saves/toy_regression/es-s:linear-optim:adam-lr:0.01-T:100-K:10-N:100-sigma:1.0-seed:1', 'iteration.csv')
pes_k10 = load_log('saves/toy_regression/pes-s:linear-optim:adam-lr:0.01-T:100-K:10-N:100-sigma:1.0-seed:1', 'iteration.csv')

plot_heatmap('saves/toy_regression/sgd_lr:linear_sum_T_100_N_400_grid.pkl',
             xlabel='Initial LR',
             ylabel='Final LR',
             smoothed=False,
             cmap=plt.cm.Purples_r,
             colorbar=False,
             figsize=(7,5))

plt.plot(np.array(tbptt_k10['theta0']), np.array(tbptt_k10['theta1']), linewidth=3, label='TBPTT')
plt.plot(np.array(uoro_k10['theta0']), np.array(uoro_k10['theta1']), linewidth=3, label='UORO')
plt.plot(np.array(rtrl_k10['theta0']), np.array(rtrl_k10['theta1']), linewidth=3, label='RTRL')
plt.plot(np.array(es_k10['theta0']), np.array(es_k10['theta1']), linewidth=3, label='ES')
plt.plot(np.array(pes_k10['theta0']), np.array(pes_k10['theta1']), linewidth=3, label='PES')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Initial LR', fontsize=24)
plt.ylabel('Final LR', fontsize=24)
plt.legend(fontsize=20, fancybox=True, framealpha=0.7)
plt.savefig('figures/toy_regression_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=300)

# ================================================================================================

plt.figure(figsize=(6,4))
plt.plot(tbptt_k10['inner_problem_steps'], tbptt_k10['L'], linewidth=3, label='TBPTT')
plt.plot(uoro_k10['inner_problem_steps'], uoro_k10['L'], linewidth=3, label='UORO')
plt.plot(rtrl_k10['inner_problem_steps'], rtrl_k10['L'], linewidth=3, label='RTRL')
plt.plot(es_k10['inner_problem_steps'], es_k10['L'], linewidth=3, label='ES')
plt.plot(pes_k10['inner_problem_steps'], pes_k10['L'], linewidth=3, label='PES')

plt.xscale('log')
plt.xticks(fontsize=18)
plt.yticks([500, 1000, 1500, 2000, 2500], fontsize=18)
plt.xlabel('Inner Iterations', fontsize=20)
plt.ylabel('Meta Objective', fontsize=20)
plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
sns.despine()
plt.savefig('figures/toy_regression_meta_obj.pdf', bbox_inches='tight', pad_inches=0)
