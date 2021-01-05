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

r"""Compile metric gaps from runs on random MDPs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import app
from absl import flags
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import tensorflow.compat.v1 as tf

from rl_metrics_aaai2021 import utils

flags.DEFINE_string('base_dir', None, 'Base directory where stats are stored.')

FLAGS = flags.FLAGS


MDPStats = utils.MDPStats


Y_LABELS = {
    'time': 'Time to convergence',
    'num_iterations': 'Number of iterations to convergence',
    'min_gap': r'$\min (d(s, t) - |V^*(s) - V^*(t)|)$',
    'avg_gap': r'$avg (d(s, t) - |V^*(s) - V^*(t)|)$',
    'max_gap': r'$\max (d(s, t) - |V^*(s) - V^*(t)|)$',
}


PRETTY_NAMES = {
    'bisimulation': r'$d^{\sim}$',
    'lax_bisimulation': r'$d^{\sim}_L$',
    'd_delta': r'$d_{\Delta}$',
    'd_delta_star': r'$d_{\Delta^*}$',
    'uniform': r'$d_{Unif}$',
}


def main(_):
  flags.mark_flags_as_required(['base_dir'])
  data = {'Metric': [], 'num_states': [], 'num_actions': [],
          'run': [], 'time': [], 'num_iterations': [],
          'min_gap': [], 'avg_gap': [], 'max_gap': []}
  directories = tf.io.gfile.listdir(FLAGS.base_dir)
  total_num_actions = set()
  for directory in directories:
    if not tf.io.gfile.isdir(os.path.join(FLAGS.base_dir, directory)):
      continue
    num_states, num_actions = directory.split('_')
    num_states = int(num_states)
    num_actions = int(num_actions)
    total_num_actions.add(num_actions)
    statistics_file = os.path.join(FLAGS.base_dir, directory, 'mdp_stats.pkl')
    if not tf.io.gfile.exists(statistics_file):
      continue
    with tf.io.gfile.GFile(statistics_file, 'rb') as f:
      statistics = pickle.load(f)
    for metric in statistics:
      if metric not in PRETTY_NAMES:
        continue
      for i, run in enumerate(statistics[metric]):
        data['Metric'].append(PRETTY_NAMES[metric])
        data['num_states'].append(num_states)
        data['num_actions'].append(num_actions)
        data['run'].append(i)
        data['time'].append(run.time)
        data['num_iterations'].append(run.num_iterations)
        data['min_gap'].append(run.min_gap)
        data['avg_gap'].append(run.avg_gap)
        data['max_gap'].append(run.max_gap)
  df = pd.DataFrame(data=data)
  for na in total_num_actions:
    for value in ['time', 'num_iterations', 'min_gap', 'avg_gap', 'max_gap']:
      fig = plt.figure(figsize=(18, 12))
      ax = fig.add_subplot(111)
      tmp_df = df[df.num_actions == na]
      sns.lineplot(x='num_states', y='{}'.format(value),
                   hue='Metric', data=tmp_df, ci=99, lw=3)
      ax.set_xlabel('Number of States', fontsize=24)
      ax.set_ylabel(Y_LABELS[value], fontsize=24)
      plt.setp(ax.get_legend().get_texts(), fontsize='20')
      plt.setp(ax.get_legend().get_title(), fontsize='24')
      pdf_file = os.path.join(
          FLAGS.base_dir, '{}_{}.pdf'.format(value, na))
      with tf.io.gfile.GFile(pdf_file, 'w') as f:
        plt.savefig(f, format='pdf', dpi=300, bbox_inches='tight')
      plt.clf()
      plt.close('all')

if __name__ == '__main__':
  app.run(main)
