"""Plot loss curves from saved CSV files for the influence balancing experiment.

Example:
--------
python plot_toy_regression.py
"""
import os
import csv
import pdb
import pickle as pkl
from collections import defaultdict

import numpy as np
import scipy.ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')

# Lighter colors
mycolors = ["#FF349B", "#18DF29", "#674DEA", "#FF8031", "#02D4F9", "#4F4C4B",]
sns.set_palette(mycolors)
sns.palplot(sns.color_palette())

figure_dir = 'figures/toy_regression'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

# Plotting from saved CSV files
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

def plot_heatmap(pkl_path, xlabel, ylabel, smoothed=False, sigma=5.0,
                 cmap=plt.cm.viridis, colorbar=True, figsize=(10,8)):
  with open(pkl_path, 'rb') as f:
      heatmap_data = pkl.load(f)

  if smoothed:
    smoothed_F_grid = scipy.ndimage.gaussian_filter(heatmap_data['L_grid'],
                                                    sigma=sigma)
    best_smoothed_theta = np.unravel_index(smoothed_F_grid.argmin(),
                                           smoothed_F_grid.shape)
    best_smoothed_x = heatmap_data['xv'][best_smoothed_theta]
    best_smoothed_y = heatmap_data['yv'][best_smoothed_theta]

    plt.figure(figsize=figsize)
    cmesh = plt.pcolormesh(
        heatmap_data['xv'], heatmap_data['yv'], heatmap_data['L_grid'],
        vmin=5e2, vmax=1e4, norm=colors.LogNorm(), cmap=cmap
    )
    if colorbar:
      cbar = plt.colorbar(cmesh)
      cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
  else:
    plt.figure(figsize=figsize)
    cmesh = plt.pcolormesh(
        heatmap_data['xv'], heatmap_data['yv'], heatmap_data['L_grid'],
        vmin=5e2, vmax=1e4, norm=colors.LogNorm(), cmap=cmap,
        linewidth=0, rasterized=True
    )
    if colorbar:
      cbar = plt.colorbar(cmesh)
      cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)


tbptt_k10 = load_log('saves/toy_regression/tbptt-sum-lr:linear--4.0,-4.0-optim:adam-lr:0.01-T:100-K:10-Nc:1-Npc:100-sigma:0.3-seed:3')
rtrl_k10 = load_log('saves/toy_regression/rtrl-sum-lr:linear--4.0,-4.0-optim:adam-lr:0.01-T:100-K:10-Nc:1-Npc:100-sigma:0.3-seed:3')
uoro_k10 = load_log('saves/toy_regression/uoro-sum-lr:linear--4.0,-4.0-optim:adam-lr:0.01-T:100-K:10-Nc:1-Npc:100-sigma:0.3-seed:3')
es_k10 = load_log('saves/toy_regression/es-sum-lr:linear--4.0,-4.0-optim:adam-lr:0.01-T:100-K:10-Nc:1-Npc:100-sigma:1.0-seed:3')
pes_k10 = load_log('saves/toy_regression/pes-sum-lr:linear--4.0,-4.0-optim:adam-lr:0.01-T:100-K:10-Nc:1-Npc:100-sigma:0.3-seed:3')


plot_heatmap(
    'saves/toy_regression/sgd_lr:linear_sum_T_100_N_2000_grid.pkl',
    xlabel='Initial Log-LR',
    ylabel='Final Log-LR',
    smoothed=False,
    cmap=plt.cm.Purples_r,
    colorbar=True,
    figsize=(8,5)
)

plt.plot(np.array(tbptt_k10['theta0']), np.array(tbptt_k10['theta1']),
         linewidth=3, label='TBPTT')
plt.plot(np.array(uoro_k10['theta0']), np.array(uoro_k10['theta1']),
         linewidth=3, label='UORO')
plt.plot(np.array(rtrl_k10['theta0']), np.array(rtrl_k10['theta1']),
         linewidth=3, label='RTRL')
plt.plot(np.array(es_k10['theta0']), np.array(es_k10['theta1']),
         linewidth=3, label='ES')
plt.plot(np.array(pes_k10['theta0']), np.array(pes_k10['theta1']),
         linewidth=3, label='PES')

plt.legend(fontsize=18, fancybox=True, framealpha=0.3, loc='upper left')
plt.savefig(os.path.join(figure_dir, 'toy_regression_heatmap.png'),
            bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig(os.path.join(figure_dir, 'toy_regression_heatmap.pdf'),
            bbox_inches='tight', pad_inches=0)

# =============================================================================

plt.figure(figsize=(6,4))
plt.plot(tbptt_k10['inner_problem_steps'], tbptt_k10['L'],
         linewidth=3, label='TBPTT')
plt.plot(uoro_k10['inner_problem_steps'], uoro_k10['L'],
         linewidth=3, label='UORO')
plt.plot(rtrl_k10['inner_problem_steps'], rtrl_k10['L'],
         linewidth=3, label='RTRL')
plt.plot(es_k10['inner_problem_steps'], es_k10['L'],
         linewidth=3, label='ES')
plt.plot(pes_k10['inner_problem_steps'], pes_k10['L'],
         linewidth=3, label='PES')

plt.xscale('log')
plt.xticks(fontsize=18)
plt.yticks([500, 1000, 1500, 2000, 2500], fontsize=18)
plt.xlabel('Inner Iterations', fontsize=20)
plt.ylabel('Meta Objective', fontsize=20)
plt.legend(fontsize=18, fancybox=True, framealpha=0.3)
sns.despine()
plt.savefig(os.path.join(figure_dir, 'toy_regression_meta_obj.png'),
            bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(figure_dir, 'toy_regression_meta_obj.pdf'),
            bbox_inches='tight', pad_inches=0)
