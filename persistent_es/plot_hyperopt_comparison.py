"""Plot the hyperopt results, comparing random search, ES, and PES.

Example:
--------
python plot_hyperopt_comparison.py
"""
import os
import csv
import pdb
import pickle as pkl

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.colors as colors
import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')

# Local imports
import plot_utils


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mycolors = ["#FF349B", "#18DF29", "#674DEA", "#FF8031", "#02D4F9", "#4F4C4B",]
sns.set_palette(mycolors)
sns.palplot(sns.color_palette())


figure_dir = 'figures/hyperopt'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)


# Load the random search results
# ==================================================================
dirname = 'saves/many_hparams/random/random_fashion_mnist_mlp_mask:fixed,lr:fixed-pl,mom:fixed-pl_sgdm_T_1000_N_10000'
inner_problem_len = 1000

all_results = []
for subdir in os.listdir(dirname):
  loaded = False
  while not loaded:
    try:
      with open(os.path.join(dirname, subdir, 'result.pkl'), 'rb') as f:
        result = pkl.load(f)
      loaded = True
    except:
      pass
  all_results.append(result)

all_running_mins = []
for res in all_results:
  rs_values = np.nan_to_num(res['unroll_obj'], nan=1e6)
  rs_running_min = np.minimum.accumulate(rs_values)
  all_running_mins.append(rs_running_min)

shortest_len = min([len(arr) for arr in all_running_mins])
rs_iterations = [inner_problem_len * i for i in range(shortest_len)]
stacked_running_mins = np.stack([arr[:shortest_len] for arr in all_running_mins])
rs_mean = np.mean(stacked_running_mins, axis=0)
rs_min = np.min(stacked_running_mins, axis=0)
rs_max = np.max(stacked_running_mins, axis=0)
# ==================================================================


# Load the vanilla ES results
# ==================================================================
base_dir = 'saves/many_hparams/es/val_sum_loss'
metric_key = 'perf/unroll_obj'

all_es_running_mins = []
all_es_iterations = []
for exp_dir in os.listdir(base_dir):
  try:
    log = plot_utils.load_log(os.path.join(base_dir, exp_dir), 'iteration.csv')
    running_min = np.minimum.accumulate(log[metric_key])
    all_es_running_mins.append(running_min)
    all_es_iterations.append(log['perf/total_inner_iterations'])
  except:
    pass

shortest_len = min([len(arr) for arr in all_es_running_mins])
es_iterations = all_es_iterations[0][:shortest_len]
es_stacked_running_mins = np.stack([arr[:shortest_len] for arr in all_es_running_mins])
es_mean = np.mean(es_stacked_running_mins, axis=0)
es_min = np.min(es_stacked_running_mins, axis=0)
es_max = np.max(es_stacked_running_mins, axis=0)
# ==================================================================


# Load the PES results
# ==================================================================
metric_key = 'perf/unroll_obj'
base_dir = 'saves/many_hparams/pes/val_sum_loss/'

all_pes_running_mins = []
all_pes_iterations = []
for exp_dir in os.listdir(base_dir):
  try:
    log = plot_utils.load_log(os.path.join(base_dir, exp_dir), 'iteration.csv')
    running_min = np.minimum.accumulate(log[metric_key])
    all_pes_running_mins.append(running_min)
    all_pes_iterations.append(log['perf/total_inner_iterations'])
  except:
    pass

shortest_len = min([len(arr) for arr in all_pes_running_mins])
pes_iterations = all_pes_iterations[0][:shortest_len]
pes_stacked_running_mins = np.stack([arr[:shortest_len] for arr in all_pes_running_mins])
pes_mean = np.mean(pes_stacked_running_mins, axis=0)
pes_min = np.min(pes_stacked_running_mins, axis=0)
pes_max = np.max(pes_stacked_running_mins, axis=0)
# ==================================================================


# Plot the results
# ==================================================================
plt.figure(figsize=(6,4))

plt.fill_between(pes_iterations, pes_min, pes_max, alpha=0.3)
plt.plot(pes_iterations, pes_mean, linewidth=3, label='PES')

plt.fill_between(es_iterations, es_min, es_max, alpha=0.3)
plt.plot(es_iterations, es_mean, linewidth=3, label='ES')

plt.fill_between(rs_iterations, rs_min, rs_max, alpha=0.3)
plt.plot(rs_iterations, rs_mean, linewidth=3, label='Random')

plt.ylim(420, 540)
plt.xlim(-1e5, 3e6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Total Compute', fontsize=18)
plt.ylabel('Best Meta-Obj Value', fontsize=18)
sns.despine()
plt.gca().xaxis.set_major_formatter(
    tick.FuncFormatter(plot_utils.reformat_large_tick_values)
)
plt.legend(fontsize=18, fancybox=True, framealpha=0.3, loc='upper right')
plt.tight_layout()

plt.savefig(os.path.join(figure_dir, 'hyperopt_comparison_val.pdf'),
            bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(figure_dir, 'hyperopt_comparison_val.png'),
            bbox_inches='tight', pad_inches=0, dpi=300)
