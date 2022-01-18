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

"""Script to plot results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pylab
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')

df_2d2c = pd.concat([
    pd.read_csv('may20_2d_2c_nondiag_b/results.csv'),
    pd.read_csv('may20_2d_2c_nondiag_c/results.csv')
])
df_2d3c = pd.concat([
    pd.read_csv('may20_2d_3c_nondiag_b/results.csv'),
    pd.read_csv('may20_2d_3c_nondiag_c/results.csv')
])
df_2d5c = pd.concat([
    pd.read_csv('may24_2d_5c_nondiag/results.csv'),
    pd.read_csv('may24_2d_5c_kshape_nondiag/results.csv')
])
df_2d10c = pd.read_csv('may24_2d_10c_1000_nondiag/results.csv')
df_2d2c['num_clusters'] = 2
df_2d2c['hidden_dim'] = 2
df_2d3c['num_clusters'] = 3
df_2d3c['hidden_dim'] = 2
df_2d5c['num_clusters'] = 5
df_2d5c['hidden_dim'] = 2
df_2d10c['hidden_dim'] = 2
df_2d10c['num_clusters'] = 10
df_3d2c = pd.read_csv('may20_3d_2c_nondiag_b/results.csv')
df_3d2c['hidden_dim'] = 3
df_3d2c['num_clusters'] = 2
df = pd.concat([df_2d2c, df_2d3c, df_2d5c, df_2d10c])
# df = pd.concat([df_2d2c, df_2d3c, df_2d5c, df_2d10c])
df = df[df.method != 'true']
df = df[df.method != 'ARMA_OLS']
df = df[df.method != 'AR_OLS']
df = df[df.method != 'raw_output']
df = df[df.seq_len == 1000]

metric_names = [
    'adj_mutual_info', 'adj_rand_score', 'v_measure', 't_secs', 'failed_ratio'
]
metric_names = ['adj_mutual_info', 'adj_rand_score', 'v_measure', 't_secs']
stats_list = []
for metric in metric_names:
  stats = df.groupby(['hidden_dim', 'num_clusters', 'seq_len',
                      'method'])[metric].agg(['mean', 'count', 'std'])
  ci95_hi = []
  ci95_lo = []
  mean_w_ci = []
  for i in stats.index:
    m, c, s = stats.loc[i]
    ci95_hi.append(m + 1.96 * s / np.sqrt(c))
    ci95_lo.append(m - 1.96 * s / np.sqrt(c))
    mean_w_ci.append('%.2f (%.2f-%.2f)' %
                     (m, m - 1.96 * s / np.sqrt(c), m + 1.96 * s / np.sqrt(c)))
  stats['ci95_hi'] = ci95_hi
  stats['ci95_lo'] = ci95_lo
  stats['mean_w_ci'] = mean_w_ci
  stats['metric'] = metric
  stats = stats.reset_index()
  stats = stats.reset_index().set_index(['method', 'num_clusters', 'metric'])
  print(stats[['mean_w_ci']])
  stats.to_csv(metric + '_agg_2d_nondiag.csv')
  stats_list.append(stats['mean_w_ci'])
agg_df = pd.DataFrame(data={'val': pd.concat(stats_list)})
agg_df = agg_df.pivot_table(
    index=['num_clusters', 'method'],
    columns=['metric'],
    values='val',
    aggfunc=lambda x: ''.join(str(v) for v in x))
agg_df = agg_df[metric_names]
print(agg_df.to_latex())

metric = 'adj_mutual_info'
# metric = 't_secs'
method_name_mapping = {
    'AR': 'AR',
    'ARMA': 'ARMA',
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
hue_order = ['AR', 'ARMA', 'LDS', 'k-Shape', 'PCA']
g = sns.catplot(
    x='seq_len',
    y=metric,
    hue='method',
    col='num_clusters',
    data=df,
    kind='point',
    capsize=.2,
    palette=sns.color_palette('Set2', 6),
    scale=1.0,
    height=6,
    aspect=0.75,
    hue_order=hue_order,
    ci=90,
    join=False,
    markers=['x', 'v', '>', '.', 'o', '+', '<', '1', '2', '3', '4'],
    linestyles=':')
g.set_axis_labels('Sequence length', 'Adj. Mutual Information')
if metric == 't_secs':
  ax = g.facet_axis(0, 0)
  ax.set(yscale='log')
  ax = g.facet_axis(0, 1)
  ax.set(yscale='log')
g.despine(left=True)
pylab.show()
