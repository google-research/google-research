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

"""Library for handling communities in temporal graphs."""

from absl import logging
import numpy as np
import pandas as pd


_MAX_ZERO_DENSITY_SAMPLES = 100


def _sample_community_cluster(
    community_node_map,
    community_names,
    init_community_p,
    target_community_cluster_size,
):
  """Samples a cluster of communities given a target size.

  Args:
    community_node_map: A map from community name to the set of nodes in the
      community.
    community_names: A list of community names.
    init_community_p: A list of floats representing the initial probability of
      sampling each community.
    target_community_cluster_size: The target size of the community cluster.

  Returns:
    A set of nodes in the sampled community cluster and the updated probability
    of sampling each community.
  """
  community_p = np.array(init_community_p) / np.sum(init_community_p)
  community_cluster_names = []
  community_cluster_size = 0
  community_cluster = set()
  while community_cluster_size < target_community_cluster_size:
    sampled_community = np.random.choice(community_names, p=community_p)
    community_cluster_names.append(sampled_community)
    community_cluster.update(community_node_map[sampled_community])
    community_p[community_names.index(sampled_community)] = 0.0
    community_p /= np.sum(community_p)
    community_cluster_size += len(community_node_map[sampled_community])
  return community_cluster, community_p


def _get_edges_in_time_interval(
    df, community_cluster, min_ts, max_ts
):
  """Helper to get number of community cluster edges in a time interval."""
  return len(
      df[
          df['source'].isin(community_cluster)
          & df['target'].isin(community_cluster)
          & (df['ts'] >= min_ts)
          & (df['ts'] < max_ts)
      ]
  )


def _enough_edges_in_interval(
    df,
    community_cluster,
    min_ts,
    max_ts,
    edge_density_full_interval,
    warmstart_quantile_delta,
):
  """Helper to check if enough edges are in a time interval."""
  num_edges_in_interval = _get_edges_in_time_interval(
      df, community_cluster, min_ts, max_ts
  )
  edge_density_in_interval = num_edges_in_interval / (
      len(community_cluster) * (max_ts - min_ts)
  )
  if (
      edge_density_in_interval
      < warmstart_quantile_delta * edge_density_full_interval
  ):
    return False
  return True


def reindex_communities(
    community_node_map,
    node_index_dict,
):
  """Reindexes a community map to a new node index dict."""
  for community_name in community_node_map:
    reindexed_node_set = set()
    for original_node_index in community_node_map[community_name]:
      reindexed_node_set.add(node_index_dict[original_node_index])
    community_node_map[community_name] = reindexed_node_set


def get_community_cluster_nodes(
    edgelist_df,
    community_node_map,
    time_interval_balance_factor,
    val_time,
    test_time,
    target_community_cluster_size,
    warmstart_quantiles_to_check = [0.10, 0.25, 0.50],
    warmstart_quantiles_delta = 0.50,
):
  """Samples train and test community clusters as node sets.

  Specifically, this function samples two community clusters that are approx.
  `target_community_cluster_size` in size. The first "train" cc will be used
  for train & val, whereas the second "test" cc will be used for test only:

    |            community cluster 1               | community cluster 2 |
  min_t [ train period ] val_time [val period] test_time [test period] max_t

  The `time_interval_balance_factor` parameter ensures that the sampling does
  not choose communities that are too density-imbalanced across the train / val
  / test splits. Specifically, define the edge-density of a community cluster C
  on a split (time interval) S as D_C[S] = |E_C[S]| / (|S| * (|C| * (|C| - 1))).
  This function will sample C1 and C2 such that:

    D_C1[train] / time_interval_balance_factor <= D_C1[val] <= D_C1[train] *
    time_interval_balance_factor    (1)
  and
    D_C1[train] / time_interval_balance_factor <= D_C2[test] <= D_C1[train] *
    time_interval_balance_factor   (2)

  ...effectively ensuring that the edge-density of the val and test graphs are
  within a `time_interval_balance_factor` of the edge-density of the test graph.

  >> Val / Test temporal density balancing: this function also checks that the
  first k sub-intervals of the val and test intervals have above some threshold
  of edge density. In particular, this function accepts a vector of temporal
  quantiles (e.g. q = [0.05, 0.10, 0.15, ...]) and ensures that for every i,
  the edge density on the range [q[i], q[i+1]] is above some delta% of the edge
  density on the full interval. This check is important to ensure that warm-
  start techniques on the resulting datasets are successful.

  The actual sampling is performed as follows:

  1. Train cc: sample communities without replacement with probability
     proportional to the community sizes. Stop when the cc size surpasses
     `target_community_cluster_size`. Re-sample if condition (1) above is not
     met.
  2. Test cc: sample communities (from the remainder after sampling train cc)
     without replacement with probability proportional to the community sizes.
     Stop when the cc size surpasses `target_community_cluster_size`. Re-sample
     if condition (2) above is not met.

  Args:
    edgelist_df: A dataframe containing the edgelist of the graph and associated
      timestamps.
    community_node_map: A map from community name to the set of nodes in the
      community.
    time_interval_balance_factor: This number controls the level of
      (multiplicative) imbalance between the train and val community clusters
      and also between the val and test community clusters.
    val_time: An integer representing the timestamp marking the beginning of the
      val time interval.
    test_time: An integer representing the timestamp marking the beginning of
      the test time interval.
    target_community_cluster_size: The target size of each community cluster.
    warmstart_quantiles_to_check: A list of quantiles to check for warmstart
      edge density.
    warmstart_quantiles_delta: The minimum fraction of the full interval density
      that must be present in each warmstart sub-interval.

  Returns:
    A set of nodes in the sampled train community cluster and a set of nodes in
    the sampled test community cluster.
  """

  warmstart_quantiles_to_check = sorted(warmstart_quantiles_to_check)
  if (
      warmstart_quantiles_to_check[0] <= 0.0
      or warmstart_quantiles_to_check[-1] > 0.5
  ):
    raise ValueError(
        'warmstart_quantiles_to_check must be a list of quantiles in (0, 0.5].'
    )

  community_names = []
  community_sizes = []
  for community in community_node_map:
    community_names.append(community)
    community_sizes.append(len(community_node_map[community]))

  init_community_p = np.array(community_sizes) / np.sum(community_sizes)

  # Sample train cluster
  min_timestamp = edgelist_df['ts'].min()
  max_timestamp = edgelist_df['ts'].max()
  train_community_p = []
  train_community_cluster = set()
  num_tries = 0
  train_edge_density = 0.0
  train_edge_density_zero_count = 0
  val_edge_density_zero_count = 0
  test_edge_density_zero_count = 0
  while True:
    num_tries += 1
    logging.info('train_community_cluster sampling trial %d', num_tries)
    train_community_cluster, train_community_p = _sample_community_cluster(
        community_node_map,
        community_names,
        init_community_p,
        target_community_cluster_size,
    )
    num_edges_in_train = _get_edges_in_time_interval(
        edgelist_df, train_community_cluster, min_timestamp, val_time
    )
    num_edges_in_val = _get_edges_in_time_interval(
        edgelist_df, train_community_cluster, val_time, test_time
    )
    train_edge_density = num_edges_in_train / (
        (val_time - min_timestamp) * len(train_community_cluster)
    )
    val_edge_density = num_edges_in_val / (
        (test_time - val_time) * len(train_community_cluster)
    )
    if train_edge_density == 0.0 or val_edge_density == 0.0:
      if train_edge_density == 0.0:
        logging.info(' >> train_edge_density == 0.0')
        train_edge_density_zero_count += 1
        if train_edge_density_zero_count > _MAX_ZERO_DENSITY_SAMPLES:
          raise ValueError(
              'train_edge_density == 0.0 for %d samples.'
              % train_edge_density_zero_count
          )
      if val_edge_density == 0.0:
        logging.info(' >> val_edge_density == 0.0')
        val_edge_density_zero_count += 1
        if val_edge_density_zero_count > _MAX_ZERO_DENSITY_SAMPLES:
          raise ValueError(
              'val_edge_density == 0.0 for %d samples.'
              % val_edge_density_zero_count
          )
      continue
    if (
        train_edge_density / time_interval_balance_factor <= val_edge_density
        and val_edge_density
        <= train_edge_density * time_interval_balance_factor
    ):
      val_interval_length = test_time - val_time
      density_diagnostics = []
      quantile_vec = [0.0] + warmstart_quantiles_to_check
      for i in range(len(warmstart_quantiles_to_check)):
        density_diagnostics.append(
            _enough_edges_in_interval(
                edgelist_df,
                train_community_cluster,
                int(val_time + quantile_vec[i] * val_interval_length),
                int(val_time + quantile_vec[i + 1] * val_interval_length),
                val_edge_density,
                warmstart_quantiles_delta,
            )
        )
      if all(density_diagnostics):
        break
      else:
        logging.info(' >> bad warm-start density found in val')
    else:
      logging.info(' >> bad total density found in val')

  # Sample test cluster
  num_tries = 0
  test_community_cluster = set()
  while True:
    num_tries += 1
    logging.info('test_community_cluster sampling trial %d', num_tries)
    test_community_cluster, _ = _sample_community_cluster(
        community_node_map,
        community_names,
        train_community_p,
        target_community_cluster_size,
    )
    num_edges_in_test = _get_edges_in_time_interval(
        edgelist_df, test_community_cluster, val_time, test_time
    )
    test_edge_density = num_edges_in_test / (
        len(test_community_cluster) * (test_time - val_time)
    )
    if test_edge_density == 0.0:
      logging.info(' >> test_edge_density == 0.0')
      test_edge_density_zero_count += 1
      if test_edge_density_zero_count > _MAX_ZERO_DENSITY_SAMPLES:
        raise ValueError(
            'test_edge_density == 0.0 for %d samples.'
            % test_edge_density_zero_count
        )
    if (
        test_edge_density / time_interval_balance_factor <= val_edge_density
        and test_edge_density
        <= train_edge_density * time_interval_balance_factor
    ):
      test_interval_length = max_timestamp - test_time
      density_diagnostics = []
      quantile_vec = [0.0] + warmstart_quantiles_to_check
      for i in range(len(warmstart_quantiles_to_check)):
        density_diagnostics.append(
            _enough_edges_in_interval(
                edgelist_df,
                test_community_cluster,
                int(test_time + quantile_vec[i] * test_interval_length),
                int(test_time + quantile_vec[i + 1] * test_interval_length),
                test_edge_density,
                warmstart_quantiles_delta,
            )
        )
      if all(density_diagnostics):
        break
      else:
        logging.info(' >> bad warm-start density found in test')
    else:
      logging.info(' >> bad total density found in test')

  return train_community_cluster, test_community_cluster
