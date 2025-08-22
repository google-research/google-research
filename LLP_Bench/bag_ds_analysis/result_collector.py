# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Collects all analysis results, all training results, performs K-means."""
from collections.abc import Sequence
import json

from absl import app
import analysis_constants
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans


def main(argv):
  """Main function."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # generating metrics df

  saving_dir = '../results/metric_dicts/'
  agg_list = []
  for m in range(1, 27):
    c = analysis_constants.C + str(m)
    with open(saving_dir + c + '.json', 'r') as fp:
      agg_list.append(json.load(fp))
  metrics_df1 = pd.DataFrame(agg_list)

  agg_list = []
  for m in range(1, 27):
    for n in range(m + 1, 27):
      c1 = analysis_constants.C + str(m)
      c2 = analysis_constants.C + str(n)
      with open(saving_dir + c1 + '_' + c2 + '.json', 'r') as fp:
        agg_list.append(json.load(fp))
  metrics_df2 = pd.DataFrame(agg_list)

  metrics_df = pd.concat([metrics_df1, metrics_df2])

  saving_dir = '../results/dist_dicts/'
  agg_list = []
  for m in range(1, 27):
    c = analysis_constants.C + str(m)
    with open(saving_dir + c + '.json', 'r') as fp:
      agg_list.append(json.load(fp))
  bag_dist_df1 = pd.DataFrame(agg_list)

  agg_list = []
  for m in range(1, 27):
    for n in range(m + 1, 27):
      c1 = analysis_constants.C + str(m)
      c2 = analysis_constants.C + str(n)
      with open(saving_dir + c1 + '_' + c2 + '.json', 'r') as fp:
        agg_list.append(json.load(fp))
  bag_dist_df2 = pd.DataFrame(agg_list)

  bag_dist_df = pd.concat([bag_dist_df1, bag_dist_df2])

  metrics_df = metrics_df.merge(bag_dist_df, on=['c1', 'c2'], how='inner')
  metrics_df.to_csv('../results/metrics_df.csv', index=False)

  saving_dir = '../results/metric_dicts/sscl_'
  agg_list = []
  for m in range(1, 18):
    c = analysis_constants.C + str(m)
    with open(saving_dir + c + '.json', 'r') as fp:
      agg_list.append(json.load(fp))
  metrics_df1 = pd.DataFrame(agg_list)

  agg_list = []
  for m in range(1, 18):
    for n in range(m + 1, 18):
      c1 = analysis_constants.C + str(m)
      c2 = analysis_constants.C + str(n)
      with open(saving_dir + c1 + '_' + c2 + '.json', 'r') as fp:
        agg_list.append(json.load(fp))
  metrics_df2 = pd.DataFrame(agg_list)

  metrics_df_sscl = pd.concat([metrics_df1, metrics_df2])
  metrics_df_sscl.to_csv('../results/label_metrics_df_sscl.csv', index=False)

  c1_c2_list = metrics_df_sscl[
      metrics_df_sscl['percentage_instances_left'] >= 30
  ][['c1', 'c2']].tolist()

  saving_dir = '../results/dist_dicts/sscl_'

  agg_list = []
  for c1, c2 in c1_c2_list:
    with open(saving_dir + c1 + '_' + c2 + '.json', 'r') as fp:
      agg_list.append(json.load(fp))
  bag_dist_df = pd.DataFrame(agg_list)

  metrics_df_sscl = metrics_df.merge(bag_dist_df, on=['c1', 'c2'], how='inner')
  metrics_df_sscl.to_csv('../results/metrics_df_sscl.csv', index=False)

  # filtering and clustering

  metrics_df = metrics_df[metrics_df['percentage_instances_left'] >= 30]

  x_train = preprocessing.StandardScaler().fit_transform(
      np.array(
          metrics_df[[
              'bag_size_50_percentile',
              'bag_size_70_percentile',
              'bag_size_85_percentile',
              'bag_size_95_percentile',
          ]]
      )
  )
  clusters = KMeans(n_clusters=4, random_state=42).fit_predict(x_train)
  metrics_df['bag_size_clusters'] = clusters
  cluster_dict = {
      i: metrics_df[metrics_df['bag_size_clusters'] == i][
          'bag_size_85_percentile'
      ].mean()
      for i in range(4)
  }
  name_dict = zip(
      np.sort(cluster_dict.values()),
      ['Very Short-tailed', 'Short-tailed', 'Long-tailed', 'Very Long-tailed'],
  )
  metrics_df['bag_size_clusters'] = metrics_df['bag_size_clusters'].apply(
      lambda x: name_dict[cluster_dict[x]]
  )

  x_train = preprocessing.StandardScaler().fit_transform(
      np.array(metrics_df[['ratio_of_means']])
  )
  clusters = KMeans(n_clusters=4, random_state=42).fit_predict(x_train)
  metrics_df['ratio_clusters'] = clusters
  cluster_dict = {
      i: metrics_df[metrics_df['ratio_clusters'] == i]['ratio_of_means'].mean()
      for i in range(4)
  }
  name_dict = zip(
      np.sort(cluster_dict.values()),
      ['Less-separated', 'Medium-separated', 'Well-separated', 'Far-separated'],
  )
  metrics_df['ratio_clusters'] = metrics_df['ratio_clusters'].apply(
      lambda x: name_dict[cluster_dict[x]]
  )

  x_train = preprocessing.StandardScaler().fit_transform(
      np.array(metrics_df[['std_label_prop']])
  )
  clusters = KMeans(n_clusters=4, random_state=42).fit_predict(x_train)
  metrics_df['std_label_prop_clusters'] = clusters
  cluster_dict = {
      i: metrics_df[metrics_df['std_label_prop_clusters'] == i][
          'std_label_prop'
      ].mean()
      for i in range(4)
  }
  name_dict = zip(
      np.sort(cluster_dict.values()),
      ['Very Low', 'Low', 'Medium', 'High'],
  )
  metrics_df['std_label_prop_clusters'] = metrics_df[
      'std_label_prop_clusters'
  ].apply(lambda x: name_dict[cluster_dict[x]])

  metrics_df.to_csv(
      '../results/filtered_metrics_df_with_clusters.csv', index=False
  )

  x_train = preprocessing.StandardScaler().fit_transform(
      np.array(
          metrics_df_sscl[[
              'bag_size_50_percentile',
              'bag_size_70_percentile',
              'bag_size_85_percentile',
              'bag_size_95_percentile',
          ]]
      )
  )
  clusters = KMeans(n_clusters=3, random_state=42).fit_predict(x_train)
  metrics_df_sscl['bag_size_clusters'] = clusters
  cluster_dict = {
      i: metrics_df_sscl[metrics_df_sscl['bag_size_clusters'] == i][
          'bag_size_85_percentile'
      ].mean()
      for i in range(3)
  }
  name_dict = zip(
      np.sort(cluster_dict.values()),
      ['Short-tailed', 'Medium-tailed', 'Long-tailed'],
  )
  metrics_df_sscl['bag_size_clusters'] = metrics_df_sscl[
      'bag_size_clusters'
  ].apply(lambda x: name_dict[cluster_dict[x]])

  x_train = preprocessing.StandardScaler().fit_transform(
      np.array(metrics_df_sscl[['ratio_of_means']])
  )
  clusters = KMeans(n_clusters=3, random_state=42).fit_predict(x_train)
  metrics_df_sscl['ratio_clusters'] = clusters
  cluster_dict = {
      i: metrics_df_sscl[metrics_df_sscl['ratio_clusters'] == i][
          'ratio_of_means'
      ].mean()
      for i in range(3)
  }
  name_dict = zip(
      np.sort(cluster_dict.values()),
      ['Less-separated', 'Medium-separated', 'Well-separated'],
  )
  metrics_df_sscl['ratio_clusters'] = metrics_df_sscl['ratio_clusters'].apply(
      lambda x: name_dict[cluster_dict[x]]
  )

  x_train = preprocessing.StandardScaler().fit_transform(
      np.array(metrics_df_sscl[['std_label_prop']])
  )
  clusters = KMeans(n_clusters=3, random_state=42).fit_predict(x_train)
  metrics_df_sscl['std_label_prop_clusters'] = clusters
  cluster_dict = {
      i: metrics_df_sscl[metrics_df_sscl['std_label_prop_clusters'] == i][
          'std_label_prop'
      ].mean()
      for i in range(3)
  }
  name_dict = zip(
      np.sort(cluster_dict.values()),
      ['Low', 'Medium', 'High'],
  )
  metrics_df_sscl['std_label_prop_clusters'] = metrics_df_sscl[
      'std_label_prop_clusters'
  ].apply(lambda x: name_dict[cluster_dict[x]])

  metrics_df_sscl.to_csv(
      '../results/filtered_metrics_df_sscl_with_clusters.csv', index=False
  )

  # generating auc df

  method_list = analysis_constants.METHODS_LIST
  c1_c2_list = analysis_constants.C1_C2_LIST

  ans_df = None
  for method in method_list:
    result_dir = '../results/training_dicts/feature_bags_ds/' + method + '/'
    dict_list = []
    for c1, c2 in c1_c2_list:
      for trial in range(5):
        file_name = (
            result_dir + str(trial) + '_C' + str(c1) + '_C' + str(c2) + '.json'
        )
        with open(file_name, 'r') as fp:
          dict_list.append(json.load(fp))
    fin_df = pd.DataFrame(dict_list)
    result_df = fin_df[['c1', 'c2', 'auc']]
    if ans_df is None:
      ans_df = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: (100 * x['auc']).mean().round(2))
          .reset_index()
      )
      ans_df.rename(columns={0: method}, inplace=True)
      ans_df[method + '_std'] = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: (100 * x['auc']).std().round(2))
          .reset_index()[0]
      )
    else:
      ans_df[method] = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: (100 * x['auc']).mean().round(2))
          .reset_index()[0]
      )
      ans_df[method + '_std'] = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: (100 * x['auc']).std().round(2))
          .reset_index()[0]
      )

  feature_bag_df = metrics_df.merge(ans_df, on=['c1', 'c2'], how='inner')
  feature_bag_df.to_csv(
      '../results/feature_bag_ds_auc_with_metrics_and_clusters.csv', index=False
  )

  random_auc_df = None
  for method in method_list:
    result_dir = '../results/training_dicts/random_bags_ds/' + method + '/'
    dict_list = []
    for bag_size in [64, 128, 256, 512]:
      for trial in range(5):
        file_name = result_dir + str(trial) + '_bs-' + str(bag_size) + '.json'
        with open(file_name, 'r') as fp:
          dict_list.append(json.load(fp))
    fin_df = pd.DataFrame(dict_list)
    result_df = fin_df[['bag_size', 'auc']]
    if random_auc_df is None:
      random_auc_df = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: (100 * x['auc']).mean().round(2))
          .reset_index()
      )
      random_auc_df.rename(columns={0: method}, inplace=True)
      random_auc_df[method + '_std'] = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: (100 * x['auc']).std().round(2))
          .reset_index()[0]
      )
    else:
      random_auc_df[method] = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: (100 * x['auc']).mean().round(2))
          .reset_index()[0]
      )
      random_auc_df[method + '_std'] = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: (100 * x['auc']).std().round(2))
          .reset_index()[0]
      )

  random_auc_df.to_csv('../results/random_bags_ds_auc.csv', index=False)

  res_dict_list = []
  for method in method_list:
    for bag_size in [64, 128, 256, 512]:
      for split in [0, 1, 2, 3, 4]:
        for c1, c2 in c1_c2_list:
          with open(
              '../results/training_dicts/fixed_size_feature_bags_ds/'
              + method
              + '/'
              + bag_size
              + '/'
              + str(split)
              + '_bs-'
              + bag_size
              + '_C'
              + str(c1)
              + '_C'
              + str(c2)
              + '.json',
              'r',
          ) as fp:
            res_dict = json.load(fp)
          res_dict_list.append(res_dict)
  feat_rand_res_df = pd.DataFrame(res_dict_list)

  feat_rand_auc_df = None
  for method in method_list:
    temp_df = feat_rand_res_df[feat_rand_res_df['method'] == method][
        ['c1', 'c2', 'bag_size', 'auc']
    ]
    if feat_rand_auc_df is None:
      feat_rand_auc_df = (
          temp_df.groupby(['c1', 'c2', 'bag_size'])
          .apply(lambda x: (100 * x['auc']).mean().round(2))
          .reset_index()
      )
      feat_rand_auc_df.rename(columns={0: method}, inplace=True)
      feat_rand_auc_df[method + '_std'] = (
          temp_df.groupby(by=['c1', 'c2', 'bag_size'])
          .apply(lambda x: (100 * x['auc']).std().round(2))
          .reset_index()[0]
      )
    else:
      feat_rand_auc_df[method] = (
          temp_df.groupby(['c1', 'c2', 'bag_size'])
          .apply(lambda x: (100 * x['auc']).mean().round(2))
          .reset_index()[0]
      )
      feat_rand_auc_df[method + '_std'] = (
          temp_df.groupby(['c1', 'c2', 'bag_size'])
          .apply(lambda x: (100 * x['auc']).std().round(2))
          .reset_index()[0]
      )

  feat_rand_auc_df.to_csv(
      '../results/fixed_size_feature_bags_ds_auc.csv', index=False
  )

  method_list = analysis_constants.SSCL_METHODS_LIST
  c1_c2_list = analysis_constants.SSCL_C1_C2_LIST

  ans_df = None
  for method in method_list:
    result_dir = (
        '../results/training_dicts/feature_bags_ds/' + method + '_sscl/'
    )
    dict_list = []
    for c1, c2 in c1_c2_list:
      for trial in range(5):
        file_name = (
            result_dir + str(trial) + '_C' + str(c1) + '_C' + str(c2) + '.json'
        )
        with open(file_name, 'r') as fp:
          dict_list.append(json.load(fp))
    fin_df = pd.DataFrame(dict_list)
    result_df = fin_df[['c1', 'c2', 'best_mse']]
    if ans_df is None:
      ans_df = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: x['best_mse'].mean().round(2))
          .reset_index()
      )
      ans_df.rename(columns={0: method}, inplace=True)
      ans_df[method + '_std'] = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: x['best_mse'].std().round(2))
          .reset_index()[0]
      )
    else:
      ans_df[method] = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: x['best_mse'].mean().round(2))
          .reset_index()[0]
      )
      ans_df[method + '_std'] = (
          result_df.groupby(by=['c1', 'c2'])
          .apply(lambda x: x['best_mse'].std().round(2))
          .reset_index()[0]
      )

  feature_bag_df_sscl = metrics_df_sscl.merge(
      ans_df, on=['c1', 'c2'], how='inner'
  )
  feature_bag_df_sscl.to_csv(
      '../results/sscl_feature_bag_ds_auc_with_metrics_and_clusters.csv',
      index=False,
  )

  random_auc_df = None
  for method in method_list:
    result_dir = '../results/training_dicts/random_bags_ds/' + method + '_sscl/'
    dict_list = []
    for bag_size in [64, 128, 256, 512]:
      for trial in range(5):
        file_name = result_dir + str(trial) + '_bs-' + str(bag_size) + '.json'
        with open(file_name, 'r') as fp:
          dict_list.append(json.load(fp))
    fin_df = pd.DataFrame(dict_list)
    result_df = fin_df[['bag_size', 'best_mse']]
    if random_auc_df is None:
      random_auc_df = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: x['best_mse'].mean().round(2))
          .reset_index()
      )
      random_auc_df.rename(columns={0: method}, inplace=True)
      random_auc_df[method + '_std'] = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: x['best_mse'].std().round(2))
          .reset_index()[0]
      )
    else:
      random_auc_df[method] = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: x['best_mse'].mean().round(2))
          .reset_index()[0]
      )
      random_auc_df[method + '_std'] = (
          result_df.groupby(by=['bag_size'])
          .apply(lambda x: x['best_mse'].std().round(2))
          .reset_index()[0]
      )

  random_auc_df.to_csv('../results/sscl_random_bags_ds_auc.csv', index=False)


if __name__ == '__main__':
  app.run(main)
