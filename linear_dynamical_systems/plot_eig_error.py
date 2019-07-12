# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Script to plot results on learning eigenvalues."""

from matplotlib import pylab
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')

# df_2d2c = pd.read_csv('may19_2d_2c/results.csv')
# df_2d3c = pd.read_csv('may19_2d_3c/results.csv')
# df_2d2c['num_clusters'] = 2
# df_2d3c['num_clusters'] = 3
# df = pd.concat([df_2d2c, df_2d3c])
y_name = 't_secs'
x_name = 'seq_len'
x_name = 'inv_sqrt_seq_len'
y_name = 'l2_a_error'
df_2d = pd.concat([
    pd.read_csv('may20_nondiag_learn_eig_2d_v2/eig_results.csv'),
    pd.read_csv('may20_nondiag_learn_eig_2d_v3/eig_results.csv'),
    pd.read_csv('may20_nondiag_learn_eig_2d_v4/eig_results.csv')
])
df_3d = pd.concat([
    pd.read_csv('may20_nondiag_learn_eig_3d_v2/eig_results.csv'),
    pd.read_csv('may20_nondiag_learn_eig_3d_v3/eig_results.csv'),
    pd.read_csv('may20_nondiag_learn_eig_3d_v4/eig_results.csv')
])
df_2d['hidden_dim'] = 2
df_3d['hidden_dim'] = 3
df = pd.concat([df_2d, df_3d])
df = df[df.method != 'true']
method_name_mapping = {
    'AR': 'AR',
    'ARMA_RLS': 'ARMA',
    'LDS_GIBBS': 'LDS',
    'kshape': 'k-Shape',
    'PCA': 'PCA',
    # 'dtw_km': 'DTW',
    # 'ARMA_MLE': 'ARMA_MLE',
    # 'raw_output': 'raw_outputs'
}
df['method'] = df.method.map(method_name_mapping)
df = df[~df.method.isnull()]
df['inv_sqrt_seq_len'] = df.seq_len.apply(lambda x: round(1.0 / np.sqrt(x), 3))
df = df[df.inv_sqrt_seq_len <= 0.05]
hue_order = ['AR', 'ARMA', 'LDS', 'ARMA_MLE', 'k-Shape', 'PCA']
hue_order = hue_order[:3]
g = sns.catplot(
    x=x_name,
    y=y_name,
    hue='method',
    sharex=False,
    col='hidden_dim',
    data=df,
    kind='point',
    capsize=.2,
    palette=sns.color_palette('Set2', 6),
    height=4,
    aspect=1.0,
    hue_order=hue_order,
    ci=95,
    scale=1.0,
    markers=['x', 'v', '>', '.', 'o', '+', '<', '1', '2', '3', '4'],
    linestyles=['--', '-.', '-', '--', '-.', '-'],
    join=True)
g.set_axis_labels('1/sqrt(sequence length)',
                  'Absolute l-2 error in eigenvalue estimation')
if x_name == 'inv_sqrt_seq_len':
  g.set_xticklabels(rotation=30)
  ax = g.facet_axis(0, 0)
  ax.set_xlim(auto=True)
  ax.set_xlim(ax.get_xlim()[::-1])
  ax = g.facet_axis(0, 1)
  ax.set_xlim(auto=True)
  ax.set_xlim(ax.get_xlim()[::-1])
if y_name == 't_secs':
  ax = g.facet_axis(0, 0)
  ax.set(yscale='log')
  ax = g.facet_axis(0, 1)
  ax.set(yscale='log')
g.despine(left=True)
pylab.gcf().subplots_adjust(bottom=0.20)
pylab.show()
