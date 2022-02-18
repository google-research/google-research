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

"""Preprocessing Movielens-20m dataset."""
import datetime
import pathlib
import pickle

import numpy as np
import pandas as pd

np.random.seed(527492)

data_dir = (pathlib.Path(__file__).parent / "Dataset/").resolve()

df_genome_scores = pd.read_csv(str(data_dir) + "/genome-scores.csv")

df_movie_genome_tags_rowwise = df_genome_scores.pivot(
    index="movieId", columns="tagId", values="relevance").reset_index()

df_movie_genome_tags_rowwise = df_movie_genome_tags_rowwise.rename_axis(
    None, axis=1)

df_movie_genome_tags_rowwise.to_csv(
    str(data_dir) + "/movie_genome_tags_rowwise.csv", index=False)

df_all_ratings = pd.read_csv(str(data_dir) + "/ratings.csv")

df_filtered_ratings = pd.merge(
    df_all_ratings, df_movie_genome_tags_rowwise[["movieId"]], on="movieId")

df_filtered_ratings = df_filtered_ratings.sort_values(
    by=["userId", "movieId"]).reset_index()

df_filtered_ratings["date"] = df_filtered_ratings.apply(
    lambda x: datetime.datetime.fromtimestamp(x["timestamp"]).strftime("%Y%m%d"
                                                                      ),
    axis=1)

df_filtered_ratings["date"] = pd.to_numeric(df_filtered_ratings["date"])

df_genres = pd.read_csv(str(data_dir) + "/movies.csv")

df_filtered_ratings = pd.merge(
    df_filtered_ratings, df_genres[["movieId", "genres"]], on="movieId")

df_filtered_ratings["randcol"] = np.random.randint(1, 6,
                                                   df_filtered_ratings.shape[0])

df_filtered_ratings["genres_subdivided"] = df_filtered_ratings.apply(
    lambda x: x["genres"] + "_" + str(x["randcol"]), axis=1)

df_genres_subdivided_index_freq = df_filtered_ratings[
    "genres_subdivided"].value_counts(
        sort=False).to_frame().reset_index().sort_values(
            by=["genres_subdivided", "index"],
            ascending=False,
            kind="mergesort").reset_index()

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


len_df = len(df_genres_subdivided_index_freq.index)

for i in range(len_df):
  row = df_genres_subdivided_index_freq.iloc[i]
  set_to_add = find_min_pos(list_of_cumufreqs)
  list_of_list_of_indices[set_to_add].append(row["index"])
  list_of_cumufreqs[
      set_to_add] = list_of_cumufreqs[set_to_add] + row["genres_subdivided"]

df_filtered_ratings["genres_subdivided_bucket_index"] = -1

for i in range(len(list_of_list_of_indices)):
  df_filtered_ratings.loc[
      df_filtered_ratings["genres_subdivided"].isin(list_of_list_of_indices[i]),
      "genres_subdivided_bucket_index"] = i

df_filtered_ratings["month"] = df_filtered_ratings.apply(
    lambda x: datetime.datetime.fromtimestamp(x["timestamp"]).strftime("%m"),
    axis=1)

df_filtered_ratings["month"] = df_filtered_ratings["month"].astype(int)

df_filtered_ratings["ts_mod5"] = df_filtered_ratings["timestamp"] % 5

df_selected = df_filtered_ratings[[
    "month", "date", "ts_mod5", "genres_subdivided_bucket_index"
]]

df_temp = pd.get_dummies(
    df_selected,
    columns=["genres_subdivided_bucket_index"],
    prefix="genres_subdivided_bucket_index_onehot")

df_temp["bag_size"] = 1

df_temp_grouped = df_temp.groupby(["month", "date",
                                   "ts_mod5"]).sum().reset_index()

df_temp_grouped = df_temp_grouped[(df_temp_grouped["bag_size"] <= 3125)
                                  & (df_temp_grouped["bag_size"] >= 63)]

list_of_count_cols = df_temp_grouped.columns[3:53].to_list()

df_temp_grouped["appended_list"] = df_temp_grouped[
    list_of_count_cols].values.tolist()

list_of_corr_matrices = []

for i in range(1, 13):
  ithlist_of_arrays = df_temp_grouped[df_temp_grouped["month"] ==
                                      i]["appended_list"].to_list()
  sum_outer_prod = np.zeros((50, 50))

  for arr in ithlist_of_arrays:
    sum_outer_prod = sum_outer_prod + np.outer(arr, arr)

  list_of_corr_matrices.append(sum_outer_prod / len(ithlist_of_arrays))

file_to_write = open(str(data_dir) + "/corr_matrices", "wb")

pickle.dump(list_of_corr_matrices, file_to_write)
file_to_write.close()

list_of_mean_vecs = []

for i in range(1, 13):
  ithlist_of_arrays = df_temp_grouped[df_temp_grouped["month"] ==
                                      i]["appended_list"].to_list()
  sum_vecs = np.zeros((50,))

  for arr in ithlist_of_arrays:
    sum_vecs = sum_vecs + arr

  list_of_mean_vecs.append(sum_vecs / len(ithlist_of_arrays))

file_to_write = open(str(data_dir) + "/mean_vecs", "wb")

pickle.dump(list_of_mean_vecs, file_to_write)
file_to_write.close()

df_movie_genome_tags_rowwise.set_index(
    keys="movieId", inplace=True, verify_integrity=True)

df_filtered_ratings["label"] = df_filtered_ratings.apply(
    lambda x: 0 if x["rating"] < 4.0 else 1, axis=1)

df_filtered_ratings.to_csv(str(data_dir) + "/filtered_ratings.csv", index=False)
