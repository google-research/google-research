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

#!/usr/bin/python
#
# Copyright 2021 The On Combining Bags to Better Learn from
# Label Proportions Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset Preprocessing and feature bucketing."""
import pathlib
import pickle

import numpy as np
import pandas as pd

list_of_cols = []

colnames = ['label']

for i in range(1, 14):
  colnames.append('N' + str(i))

for i in range(1, 27):
  colnames.append('C' + str(i))

data_dir = (pathlib.Path(__file__).parent / 'Dataset/').resolve()

df_full = pd.read_csv(
    str(data_dir) + '/train.txt', header=None, delimiter='\t', names=colnames)

extended_cuts = [1.5**j - 0.51 for j in range(40)]

extended_cuts.insert(0, float('-inf'))

extended_cuts.append(float('inf'))

for i in range(1, 14):
  colname = 'N' + str(i)
  df_full[colname] = pd.cut(df_full[colname], labels=False, bins=extended_cuts)
  print(str(i) + 'th column processed.')
  print(df_full.head(10).to_string())
  df_full[colname] = df_full[colname].fillna(df_full[colname].mean())
  print(str(i) + 'th column processed with mean')
  print(df_full.head(10).to_string())

for i in range(1, 27):
  colname = 'C' + str(i)
  df_array, _ = pd.factorize(df_full[colname])
  df_full[colname] = pd.Series(df_array)
  print(str(i) + 'th column ordinalled')
  print(df_full.head(10).to_string())
  mean_colname = df_full[df_full[colname] >= 0][colname].mean()
  df_full[colname].replace(-1, int(mean_colname), inplace=True)
  print(str(i) + 'th column processed with mean')
  print(df_full.head(10).to_string())

df_full = df_full.astype(int)

df_full.to_csv(str(data_dir) + '/train-processed-ints.csv', index=False)

df_C15_index_freq = df_full['C15'].value_counts(
        sort=False).to_frame().reset_index().sort_values(
            by=['C15', 'index'],
            ascending=False,
            kind='mergesort').reset_index()

list_of_list_of_indices = []

list_of_cumufreqs = []

for i in range(50):
  list_of_list_of_indices.append([])
  list_of_cumufreqs.append(0)


def find_min_pos(input_list):
  """Function to find index with minimum value."""
  length = len(input_list)
  if length == 0:
    return -1

  curr_min = input_list[0]
  curr_min_index = 0

  for j in range(1, length):
    if curr_min > input_list[j]:
      curr_min = input_list[j]
      curr_min_index = j

  return curr_min_index


len_df = len(df_C15_index_freq.index)

for i in range(len_df):
  row = df_C15_index_freq.iloc[i]
  set_to_add = find_min_pos(list_of_cumufreqs)
  list_of_list_of_indices[set_to_add].append(row['index'])
  list_of_cumufreqs[set_to_add] = list_of_cumufreqs[set_to_add] + row['C15']

print('list_of_cumufreqs')
print(list_of_cumufreqs)
print()
print('max(list_of_cumufreqs)')
print(max(list_of_cumufreqs))
print()
print()
print('list_of_list_of_indices')
print(list_of_list_of_indices)

df_full['C15_bucket_index'] = -1

for i in range(len(list_of_list_of_indices)):
  df_full.loc[df_full['C15'].isin(list_of_list_of_indices[i]),
              'C15_bucket_index'] = i

print(df_full['C15_bucket_index'].value_counts().to_string())

df_C14_C7_bag_sizes = df_full[['C7', 'C14'
                              ]].groupby(['C7', 'C14'
                                         ]).size().reset_index(name='bag_size')

df_C14_C7_bag_sizes = df_C14_C7_bag_sizes[
    (df_C14_C7_bag_sizes['bag_size'] <= 3125)
    & (df_C14_C7_bag_sizes['bag_size'] >= 63)]

df_C14_numbags = df_C14_C7_bag_sizes[['C7', 'C14']].groupby(
    ['C14']).size().reset_index(name='num_bags')

df_to_bucket = df_C14_numbags.sort_values(
    by='num_bags', ascending=False).reset_index(drop=True)

print(df_to_bucket.to_string())

list_of_list_of_indices = []

list_of_cumufreqs = []

for i in range(5):
  list_of_list_of_indices.append([])
  list_of_cumufreqs.append(0)


len_df = len(df_to_bucket.index)

for i in range(len_df):
  row = df_to_bucket.iloc[i]
  set_to_add = find_min_pos(list_of_cumufreqs)
  list_of_list_of_indices[set_to_add].append(row['C14'])
  list_of_cumufreqs[
      set_to_add] = list_of_cumufreqs[set_to_add] + row['num_bags']

print('list_of_cumufreqs')
print(list_of_cumufreqs)
print()
print('max(list_of_cumufreqs)')
print(max(list_of_cumufreqs))
print()
print()
print('list_of_list_of_indices')
print(list_of_list_of_indices)

df_full['C14_bucket_index'] = -1

for i in range(len(list_of_list_of_indices)):
  df_full.loc[df_full['C14'].isin(list_of_list_of_indices[i]),
              'C14_bucket_index'] = i

df_full['C14_bucket_index'] = -1

for i in range(len(list_of_list_of_indices)):
  df_full.loc[df_full['C14'].isin(list_of_list_of_indices[i]),
              'C14_bucket_index'] = i

print(df_full['C14_bucket_index'].value_counts().to_string())

df_C7_C14_C15_C14_bucket_index = df_full[[
    'C7', 'C14', 'C15_bucket_index', 'C14_bucket_index'
]]

df_temp = df_C7_C14_C15_C14_bucket_index.copy()

df_temp = pd.get_dummies(
    df_temp, columns=['C15_bucket_index'], prefix='C15_bucket_index_onehot')

df_temp['bag_size'] = 1

df_temp_grouped = df_temp.groupby(['C7', 'C14',
                                   'C14_bucket_index']).sum().reset_index()

df_temp_grouped = df_temp_grouped[(df_temp_grouped['bag_size'] <= 3125)
                                  & (df_temp_grouped['bag_size'] >= 63)]

list_of_count_cols = df_temp_grouped.columns[3:53].to_list()

df_temp_grouped['appended_list'] = df_temp_grouped[
    list_of_count_cols].values.tolist()

list_of_corr_matrices = []

for i in range(5):
  ithlist_of_arrays = df_temp_grouped[df_temp_grouped['C14_bucket_index'] ==
                                      i]['appended_list'].to_list()
  sum_outer_prod = np.zeros((50, 50))

  for arr in ithlist_of_arrays:
    sum_outer_prod = sum_outer_prod + np.outer(arr, arr)

  print(sum_outer_prod / len(ithlist_of_arrays))
  list_of_corr_matrices.append(sum_outer_prod / len(ithlist_of_arrays))

print(list_of_corr_matrices)

file_to_write = open(str(data_dir) + '/corr_matrices_C14_bucket', 'wb')
pickle.dump(list_of_corr_matrices, file_to_write)
file_to_write.close()

list_of_mean_vecs = []

for i in range(5):
  ithlist_of_arrays = df_temp_grouped[df_temp_grouped['C14_bucket_index'] ==
                                      i]['appended_list'].to_list()
  sum_vecs = np.zeros((50,))

  for arr in ithlist_of_arrays:
    sum_vecs = sum_vecs + arr

  print(sum_vecs / len(ithlist_of_arrays))
  list_of_mean_vecs.append(sum_vecs / len(ithlist_of_arrays))

print(list_of_mean_vecs)

file_to_write = open(str(data_dir) + '/mean_vecs_C14_bucket', 'wb')

pickle.dump(list_of_mean_vecs, file_to_write)
file_to_write.close()

list_of_cols_with_bucket_indices_C7_C14 = ([
    'label', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11',
    'N12', 'N13', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C17',
    'C18', 'C19', 'C20', 'C22', 'C23', 'C25', 'C15_bucket_index',
    'C14_bucket_index', 'C7', 'C14'
])
df_selected_cols = df_full[list_of_cols_with_bucket_indices_C7_C14]

list_of_cols = ([
    'label', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11',
    'N12', 'N13', 'C1', 'C2', 'C5', 'C6', 'C8', 'C9', 'C11', 'C13', 'C17',
    'C18', 'C19', 'C20', 'C22', 'C23', 'C25'
])

offsets = ([
    0, 22, 53, 81, 98, 138, 170, 197, 219, 245, 252, 266, 287, 309, 1768, 2350,
    2654, 2676, 3308, 3310, 8992, 12185, 12194, 17845, 20016, 20018, 20034,
    20048
])

no_of_cols = len(list_of_cols)

for i in range(1, no_of_cols):
  colname = list_of_cols[i]
  offset = offsets[i - 1]

  df_selected_cols[colname] = df_selected_cols[colname] + offset + i - 1

df_selected_cols.to_csv(
    str(data_dir) +
    '/train-processed-allints_selected_cols_C15_C14_bucket_index_C7_C14_offsets.csv',
    index=False)
