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

"""Processing results from R Code."""
import pathlib
import pandas as pd

path_to_root_results_dir = (pathlib.Path(__file__).parent /
                            "../../Results/").resolve()

r_exp_results_dir = str(path_to_root_results_dir) + "/Raw_Results/"

dataset_name_list = ["Heart", "Ionosphere", "Australian"]

algo_list = ["LR", "MM", "LMM(v(G,s))", "AMM(MM)", "AMM(LMM(v(G,s)))"]

lambda_list_size = 4
gamma_list_size = 3
sigma_list_size = 3

# LR on lambda_list
# MM on lambda_list
## LMM(v(G,s)) on all 3
## AMM_MM on lambda_list
## AMM_LMM(v(G,s)) on lambda_list, gamma_list
no_of_cols_map = {
    "LR": lambda_list_size,
    "MM": lambda_list_size,
    "LMM(v(G,s))": lambda_list_size * gamma_list_size * gamma_list_size,
    "AMM(MM)": lambda_list_size,
    "AMM(LMM(v(G,s)))": lambda_list_size * gamma_list_size
}

final_columns = [
    "name", "method", "LR", "MM", "LMM(v(G,s))", "AMM(MM)", "AMM(LMM(v(G,s)))"
]

best_param_df = pd.DataFrame(columns=final_columns)

for dataset_name in dataset_name_list:

  dataset_df = pd.DataFrame()

  dataset_outfile = str(
      path_to_root_results_dir) + "/" + dataset_name + "RexpStats"

  for cluster_bags_method in range(1, 8):

    cluster_bags_methodoutfile = r_exp_results_dir + dataset_name + "RexpOutputClusterBags_" + str(
        cluster_bags_method)

    df = pd.read_csv(cluster_bags_methodoutfile, header=None)

    best_param_df_this_dataset_method = pd.DataFrame(index=df.index)

    # best_param_df_this_dataset_method["name"] = dataset_name

    # best_param_df_this_dataset_method["method"] = cluster_bags_method

    print(best_param_df_this_dataset_method)

    print()
    print("*******")
    print("DatasetName ", dataset_name)
    print("ClusterBagsMethod ", cluster_bags_method)

    print("Index length ", len(df.index))

    assert len(df.index) == 25

    offset = 4
    for algo in algo_list:
      df_algo = df[df.columns[offset:offset + no_of_cols_map[algo]]]
      # find the best column
      mean_series = df_algo.mean(axis=0)
      best_col = mean_series.idxmax()
      print("The range of columsn for ", algo, " are ", offset, " and ",
            offset + no_of_cols_map[algo])
      print("The best column number for ", algo, "is ", best_col)
      assert best_col in range(offset, offset + no_of_cols_map[algo])

      print(df[df.columns[best_col]])

      best_param_df_this_dataset_method[algo] = df[df.columns[best_col]]

      offset = offset + no_of_cols_map[algo]

    assert len(best_param_df_this_dataset_method.index) == 25

    print(best_param_df_this_dataset_method.head(5).to_string())

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
