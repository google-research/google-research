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

"""Plot UCI L2 regularization experiments.

Example:
--------
python plot_uci.py
"""
import os
import csv
import ipdb
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import jetpack


def load_log(exp_dir, fname='train_log.csv'):
    result_dict = defaultdict(list)
    with open(os.path.join(exp_dir, fname), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row:
                try:
                    if key in ['global_iteration', 'iteration', 'epoch']:
                        result_dict[key].append(int(row[key]))
                    else:
                        result_dict[key].append(float(row[key]))
                except:
                    result_dict[key].append(eval(row[key])[0])
                    pass
    return result_dict

# Paths for PES and ES results
# ----------------------------
pes_5 = 'saves_uci/ob1:0.99,ob2:0.9999,theta:5-yacht-pes-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
pes_3 = 'saves_uci/ob1:0.99,ob2:0.9999,theta:3-yacht-pes-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
pes_1 = 'saves_uci/ob1:0.99,ob2:0.9999,theta:1-yacht-pes-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
pes_n1 = 'saves_uci/ob1:0.99,ob2:0.9999,theta:-1-yacht-pes-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
pes_n3 = 'saves_uci/ob1:0.99,ob2:0.9999,theta:-3-yacht-pes-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
pes_n5 = 'saves_uci/ob1:0.99,ob2:0.9999,theta:-5-yacht-pes-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'

es_5 = 'saves_uci/ob1:0.99,ob2:0.99,theta:5-yacht-es-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
es_3 = 'saves_uci/ob1:0.99,ob2:0.99,theta:3-yacht-es-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
es_1 = 'saves_uci/ob1:0.99,ob2:0.99,theta:1-yacht-es-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
es_n1 = 'saves_uci/ob1:0.99,ob2:0.99,theta:-1-yacht-es-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
es_n3 = 'saves_uci/ob1:0.99,ob2:0.99,theta:-3-yacht-es-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
es_n5 = 'saves_uci/ob1:0.99,ob2:0.99,theta:-5-yacht-es-lr:0.001-olr:0.003-K:1-N:4-sig:0.01-s:3'
# ----------------------------

# Load PES and ES results from CSV files
# --------------------------------------
log_pes_5 = load_log(pes_5, fname='iteration.csv')
log_pes_3 = load_log(pes_3, fname='iteration.csv')
log_pes_1 = load_log(pes_1, fname='iteration.csv')
log_pes_n1 = load_log(pes_n1, fname='iteration.csv')
log_pes_n3 = load_log(pes_n3, fname='iteration.csv')
log_pes_n5 = load_log(pes_n5, fname='iteration.csv')

log_es_5 = load_log(es_5, fname='iteration.csv')
log_es_3 = load_log(es_3, fname='iteration.csv')
log_es_1 = load_log(es_1, fname='iteration.csv')
log_es_n1 = load_log(es_n1, fname='iteration.csv')
log_es_n3 = load_log(es_n3, fname='iteration.csv')
log_es_n5 = load_log(es_n5, fname='iteration.csv')
# --------------------------------------


# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#4F4C4B", "#02D4F9"]

plt.plot(log_es_5['total_inner_iterations'], log_es_5['theta'], linewidth=3, color=colors[1], label='ES')
plt.plot(log_es_3['total_inner_iterations'], log_es_3['theta'], linewidth=3, color=colors[1])
plt.plot(log_es_1['total_inner_iterations'], log_es_1['theta'], linewidth=3, color=colors[1])
plt.plot(log_es_n1['total_inner_iterations'], log_es_n1['theta'], linewidth=3, color=colors[1])
plt.plot(log_es_n3['total_inner_iterations'], log_es_n3['theta'], linewidth=3, color=colors[1])
plt.plot(log_es_n5['total_inner_iterations'], log_es_n5['theta'], linewidth=3, color=colors[1])

plt.plot(log_pes_5['total_inner_iterations'], log_pes_5['theta'], linewidth=3, color=colors[0], label='PES')
plt.plot(log_pes_3['total_inner_iterations'], log_pes_3['theta'], linewidth=3, color=colors[0])
plt.plot(log_pes_1['total_inner_iterations'], log_pes_1['theta'], linewidth=3, color=colors[0])
plt.plot(log_pes_n1['total_inner_iterations'], log_pes_n1['theta'], linewidth=3, color=colors[0])
plt.plot(log_pes_n3['total_inner_iterations'], log_pes_n3['theta'], linewidth=3, color=colors[0])
plt.plot(log_pes_n5['total_inner_iterations'], log_pes_n5['theta'], linewidth=3, color=colors[0])

plt.axhline(y=1.33, linestyle='--', linewidth=4, color='black', label='Optimal', alpha=0.9)

plt.xticks([0, 100000, 200000, 300000, 400000], ['0', '100k', '200k', '300k', '400k'], fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Total Compute', fontsize=20)
plt.ylabel('Log L2 Coefficient', fontsize=20)
plt.xlim(0, 300000)
plt.legend(fontsize=20, loc=5)

jetpack.breathe()
plt.savefig('figures/UCI_L2_trajectories.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('figures/UCI_L2_trajectories.png', bbox_inches='tight', pad_inches=0)


plt.plot(log_es_5['total_inner_iterations'], log_es_5['val_loss'], linewidth=3, color=colors[1], label='ES')
plt.plot(log_es_3['total_inner_iterations'], log_es_3['val_loss'], linewidth=3, color=colors[1])
plt.plot(log_es_1['total_inner_iterations'], log_es_1['val_loss'], linewidth=3, color=colors[1])
plt.plot(log_es_n1['total_inner_iterations'], log_es_n1['val_loss'], linewidth=3, color=colors[1])
plt.plot(log_es_n3['total_inner_iterations'], log_es_n3['val_loss'], linewidth=3, color=colors[1])
plt.plot(log_es_n5['total_inner_iterations'], log_es_n5['val_loss'], linewidth=3, color=colors[1])

plt.plot(log_pes_5['total_inner_iterations'], log_pes_5['val_loss'], linewidth=3, color=colors[0], label='PES')
plt.plot(log_pes_3['total_inner_iterations'], log_pes_3['val_loss'], linewidth=3, color=colors[0])
plt.plot(log_pes_1['total_inner_iterations'], log_pes_1['val_loss'], linewidth=3, color=colors[0])
plt.plot(log_pes_n1['total_inner_iterations'], log_pes_n1['val_loss'], linewidth=3, color=colors[0])
plt.plot(log_pes_n3['total_inner_iterations'], log_pes_n3['val_loss'], linewidth=3, color=colors[0])
plt.plot(log_pes_n5['total_inner_iterations'], log_pes_n5['val_loss'], linewidth=3, color=colors[0])

plt.xticks([0, 100000, 200000, 300000, 400000], ['0', '100k', '200k', '300k', '400k'], fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Total Compute', fontsize=20)
plt.ylabel('Val Loss', fontsize=20)
plt.legend(fontsize=20)

plt.xlim(0, 300000)
plt.ylim(14, 15)

jetpack.breathe()
plt.savefig('figures/UCI_L2_val_loss.pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('figures/UCI_L2_val_loss.png', bbox_inches='tight', pad_inches=0)
