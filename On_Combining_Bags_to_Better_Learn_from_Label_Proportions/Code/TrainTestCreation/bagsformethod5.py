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
# Copyright 2021 The On Combining Bags to Better Learn
# from Label Proportions Authors. All Rights Reserved.
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

"""Generating Training Bags for Scenario V."""
import pickle
import random
import numpy as np
import pandas as pd
from scipy.stats import *  # pylint: disable=wildcard-import


def makeonlybags(n_clusters,
                 head_inclusion_powerlaw,
                 tail_inclusion_powerlaw,
                 p_law_param,
                 n_head,
                 n_tail,
                 cluster_bias,
                 trainfile,
                 cluster_dir,
                 n_tot_features,
                 option,
                 directory_to_read,
                 random_seed=None,
                 numpy_seed=None):
  # pylint: disable=unused-argument
  """Generating Training Bags for Scenario V."""

  random.seed(random_seed)

  np.random.seed(numpy_seed)

  train_df = pd.read_csv(trainfile)

  train_df.reset_index(drop=True, inplace=True)

  # ###Reading instead of creating
  cluster_to_indices_list = pickle.load(
      open(directory_to_read + "cluster_indices", "rb"))

  for cluster_label in range(n_clusters):
    print("size of cluster ", cluster_label, " is ",
          len(cluster_to_indices_list[cluster_label]))

  # create the head bags
  cluster_label_to_head_bags_list = []

  # All Bags
  all_bags_list = []

  for cluster_label in range(n_clusters):
    this_cluster_head_bags = []
    no_of_indices = len(cluster_to_indices_list[cluster_label])
    for _ in range(n_head * cluster_bias[cluster_label]):

      # pylint: disable=undefined-variable
      head_inclusion_prob = powerlaw.rvs(
          head_inclusion_powerlaw[cluster_label], size=1)[0]  ###
      print("Sampled head incl prob = ", head_inclusion_prob)  ###

      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=head_inclusion_prob)
      this_bag = random.sample(cluster_to_indices_list[cluster_label],
                               no_of_sampled_indices)
      this_bag.sort()
      this_cluster_head_bags.append(this_bag)
      all_bags_list.append(this_bag)
    cluster_label_to_head_bags_list.append(this_cluster_head_bags)
    # write the head bags for this cluster
    head_bags_file = cluster_dir + "head_bags_" + str(cluster_label)
    with open(head_bags_file, "wb") as writing_to_head_bags_file:
      pickle.dump(this_cluster_head_bags, writing_to_head_bags_file)

  # create the tail bags
  cluster_label_to_tail_bags_list = []

  for cluster_label in range(n_clusters):
    this_cluster_tail_bags = []
    no_of_indices = len(cluster_to_indices_list[cluster_label])
    for _ in range(n_tail * cluster_bias[cluster_label]):

      # pylint: disable=undefined-variable
      tail_inclusion_prob = powerlaw.rvs(
          tail_inclusion_powerlaw[cluster_label], size=1)[0]  ###
      print("Sampled tail incl prob = ", tail_inclusion_prob)  ###

      no_of_sampled_indices = np.random.binomial(
          n=no_of_indices, p=tail_inclusion_prob)
      this_bag = random.sample(cluster_to_indices_list[cluster_label],
                               no_of_sampled_indices)
      this_bag.sort()
      this_cluster_tail_bags.append(this_bag)
      all_bags_list.append(this_bag)
    cluster_label_to_tail_bags_list.append(this_cluster_tail_bags)
    # write the head bags for this cluster
    tail_bags_file = cluster_dir + "tail_bags_" + str(cluster_label)
    # print("tail_bags_file: ", tail_bags_file)
    with open(tail_bags_file, "wb") as writing_to_tail_bags_file:
      pickle.dump(this_cluster_tail_bags, writing_to_tail_bags_file)

  # write all bags
  all_bags_file = cluster_dir + "all_bags"
  with open(all_bags_file, "wb") as writing_to_all_bags_file:
    pickle.dump(all_bags_list, writing_to_all_bags_file)

  # create the raw training set using all bags

  new_train_df = pd.DataFrame()

  bag_no = 1

  for bag_list in all_bags_list:
    if not bag_list:
      continue
    this_bag_df = train_df.iloc[bag_list].copy()
    this_bag_df["bag"] = bag_no
    new_train_df = new_train_df.append(this_bag_df, ignore_index=True)
    bag_no = bag_no + 1

  new_train_df = new_train_df.sample(frac=1)
  print(new_train_df.head(10).to_string())
  new_train_df.to_csv(cluster_dir + "full_train.csv", index=False)
