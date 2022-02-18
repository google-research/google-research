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

"""Procesing results from Mutual Contamination model code."""
import pathlib
import pandas as pd

path_to_root_results_dir = (pathlib.Path(__file__).parent /
                            "../../Results/").resolve()

mutcon_exp_results_dir = str(path_to_root_results_dir) + "/Raw_Results/"

dataset_name_list = ["Heart", "Ionosphere", "Australian"]

param_list = ["1e0", "1e-1", "1e-2", "1e-3", "1e-4"]

for dataset_name in dataset_name_list:

  dataset_df = pd.DataFrame()

  dataset_outfile = str(
      path_to_root_results_dir) + "/" + dataset_name + "MutConStats"

  param_cols = pd.DataFrame()

  for cluster_bags_method in range(1, 8):

    for param in param_list:

      cluster_bags_methodoutfile_param = (mutcon_exp_results_dir +
                                          dataset_name + "_" +
                                          param + "_MutConOutputClusterBags_"
                                          + str(cluster_bags_method))

      param_df = pd.read_csv(cluster_bags_methodoutfile_param, header=None)

      assert len(param_df.index) == 25

      print(param_df.head(5).to_string())

      print(param_df[param_df.columns[4]].to_string())

      param_cols[param] = param_df[param_df.columns[4]].copy()

      print("param cols")

    mean_series = param_cols.mean(axis=0)

    best_col = mean_series.idxmax()

    print("Best param for Dataset ", dataset_name, " and Method ",
          cluster_bags_method, " is ", best_col)

    best_param_df_this_dataset_method = pd.DataFrame()

    best_param_df_this_dataset_method["MutCon"] = param_cols[best_col]

    best_param_df_this_dataset_method = best_param_df_this_dataset_method * 100

    stats_df = best_param_df_this_dataset_method.describe()

    stats_df.insert(0, "stats", stats_df.index)

    stats_df.insert(0, "method", cluster_bags_method)

    stats_df = stats_df.loc[["mean", "std"]]

    stats_df = stats_df.round(decimals=2)

    print(stats_df.to_string())

    dataset_df = dataset_df.append(stats_df, ignore_index=True)

    print(dataset_df.to_string())

  dataset_df.insert(0, "DataSet", dataset_name)

  print(dataset_df.to_string())

  dataset_df.to_csv(dataset_outfile, index=False)
