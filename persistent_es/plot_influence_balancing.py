"""Plot loss curves from saved CSV files for the influence balancing experiment.

Example:
--------
python plot_influence_balancing.py
"""
import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_palette('bright')
colors = sns.color_palette('bright')


# Load data from saved CSV files
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

tbptt_K1 = load_log('saves/influence/tbptt-lr:0.0001-K:1-sigma:0.1-N:1000/')
tbptt_K10 = load_log('saves/influence/tbptt-lr:0.0001-K:10-sigma:0.1-N:1000/')
tbptt_K100 = load_log('saves/influence/tbptt-lr:0.0001-K:100-sigma:0.1-N:1000/')
es_K1 = load_log('saves/influence/es-lr:0.0001-K:1-sigma:0.1-N:1000/')
pes_K1 = load_log('saves/influence/pes-lr:0.0001-K:1-sigma:0.1-N:1000/')
rtrl_K1 = load_log('saves/influence/rtrl-lr:0.0001-K:1-sigma:0.1-N:1000/')
uoro_K1 = load_log('saves/influence/uoro-lr:1e-05-K:1-sigma:0.1-N:1000/')

plt.figure(figsize=(6,4.8))
plt.plot(tbptt_K1['iteration'], tbptt_K1['loss'],
         linewidth=3, label='TBPTT 1')
plt.plot(tbptt_K10['iteration'], tbptt_K10['loss'],
         linewidth=3, label='TBPTT 10')
plt.plot(tbptt_K100['iteration'], tbptt_K100['loss'],
         linewidth=3, label='TBPTT 100')
plt.plot(es_K1['iteration'], es_K1['loss'],
         linewidth=3, linestyle=':', label='ES')
plt.plot(pes_K1['iteration'], pes_K1['loss'],
         linewidth=4, color=colors[-1], label='PES')
plt.plot(uoro_K1['iteration'], uoro_K1['loss'],
         linewidth=3, label='UORO')
plt.plot(rtrl_K1['iteration'], rtrl_K1['loss'],
         linewidth=2, color='k', linestyle='--', label='RTRL')

plt.xlim(0, 3000)
plt.ylim(1e-13, 1e14)
plt.yscale('log')
plt.xticks([0, 1000, 2000, 3000], fontsize=18)
plt.yticks([1e-13, 1e-5, 1e3, 1e11], fontsize=18)
plt.xlabel('Iteration', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.legend(fontsize=18, ncol=2, loc='upper center',
           fancybox=True, framealpha=0.5)
sns.despine()
plt.tight_layout()

figure_dir = 'figures/influence_balancing'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

plt.savefig(os.path.join(figure_dir, 'influence_balancing.pdf'),
            bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(figure_dir, 'influence_balancing.png'),
            bbox_inches='tight', pad_inches=0, dpi=300)
