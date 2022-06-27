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

"""Processing Tensorflow Code Results for (Generalized) bags."""
import pathlib
import pandas as pd

path_to_root_results_dir = (pathlib.Path(__file__).parent /
                            "../../Results/").resolve()

tf_exp_results_dir = str(path_to_root_results_dir) + "/Raw_Results/"

dataset_name_list = ["Heart", "Ionosphere", "Australian"]

columns = [
    "name", "method", "split", "fold", "LIN(l2)S", "LIN(l1)S", "LIN(l2)R",
    "LIN(l1)R", "LIN(l2)U", "LIN(l1)U", "LIN(KL)U"
]

for dataset_name in dataset_name_list:

  dataset_df = pd.DataFrame()

  dataset_outfile = str(
      path_to_root_results_dir) + "/" + dataset_name + "TFexpStats"

  for cluster_bags_method in range(1, 8):

    cluster_bags_methodoutfile = (tf_exp_results_dir + dataset_name +
                                  "_TFexpOutputClusterBags_" +
                                  str(cluster_bags_method))

    df = pd.read_csv(cluster_bags_methodoutfile, names=columns)

    df = df.drop(["name", "method", "split", "fold"], axis=1)

    print(df.to_string())

    df = df * 100

    new_df = df.describe()

    new_df.insert(0, "stats", new_df.index)

    new_df.insert(0, "method", cluster_bags_method)

    new_df = new_df.loc[["mean", "std"]]

    new_df = new_df.round(decimals=2)

    print(new_df.to_string())

    dataset_df = dataset_df.append(new_df, ignore_index=True)

    # new_df = pd.DataFrame()

    # print(df.mean(axis=1).to_string())

    # new_df["mean"] = df.mean(axis=0).to_frame()

    print(dataset_df.to_string())

  dataset_df.insert(0, "DataSet", dataset_name)

  print(dataset_df.to_string())

  dataset_df.to_csv(dataset_outfile, index=False)
