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

"""Sample and Generate negative edges that are going to be used for evaluation of a dynamic graph learning model

Negative samples are generated and saved to files ONLY once;
    other times, they should be loaded from file with instances of the
    `negative_sampler.py`.
"""

import os

import numpy as np
import torch
import torch_geometric
import tqdm

from fm4tlp.utils import utils


class NegativeEdgeGenerator(object):

  def __init__(
      self,
      dataset_name,
      first_dst_id,
      last_dst_id,
      num_neg_e = 100,  # number of negative edges sampled per positive edges --> make it constant => 1000
      strategy = "rnd",
      rnd_seed = 123,
      hist_ratio = 0.5,
      historical_data = None,
  ):
    r"""Negative Edge Sampler class

    this is a class for generating negative samples for a specific datasets
    the set of the positive samples are provided, the negative samples are
    generated with specific strategies
    and are saved for consistent evaluation across different methods
    negative edges are sampled with 'oen_vs_many' strategy.
    it is assumed that the destination nodes are indexed sequentially with
    'first_dst_id'
    and 'last_dst_id' being the first and last index, respectively.

    Parameters:
        dataset_name: name of the dataset
        first_dst_id: identity of the first destination node
        last_dst_id: indentity of the last destination node
        num_neg_e: number of negative edges being generated per each positive
        edge
        strategy: how to generate negative edges; can be 'rnd' or 'hist_rnd'
        rnd_seed: random seed for consistency
        hist_ratio: if the startegy is 'hist_rnd', how much of the negatives are
        historical
        historical_data: previous records of the positive edges

    Returns:
        None
    """
    self.rnd_seed = rnd_seed
    np.random.seed(self.rnd_seed)
    self.dataset_name = dataset_name

    self.first_dst_id = first_dst_id
    self.last_dst_id = last_dst_id
    self.num_neg_e = num_neg_e
    assert strategy in [
        "rnd",
        "hist_rnd",
    ], "The supported strategies are `rnd` or `hist_rnd`!"
    self.strategy = strategy
    if self.strategy == "hist_rnd":
      assert (
          historical_data != None
      ), "Train data should be passed when `hist_rnd` strategy is selected."
      self.hist_ratio = hist_ratio
      self.historical_data = historical_data

  def generate_negative_samples(
      self,
      data,
      split_mode,
      partial_path,
  ):
    r"""Generate negative samples

    Parameters:
        data: an object containing positive edges information
        split_mode: specifies whether to generate negative edges for
        'validation' or 'test' splits
        partial_path: in which directory save the generated negatives
    """
    # file name for saving or loading...
    filename = (
        partial_path
        + "/"
        + self.dataset_name
        + "_"
        + split_mode
        + "_"
        + "ns"
        + ".pkl"
    )

    if self.strategy == "rnd":
      self.generate_negative_samples_rnd(data, split_mode, filename)
    elif self.strategy == "hist_rnd":
      self.generate_negative_samples_hist_rnd(
          self.historical_data, data, split_mode, filename
      )
    else:
      raise ValueError("Unsupported negative sample generation strategy!")

  def generate_negative_samples_rnd(
      self,
      data,
      split_mode,
      filename,
  ):
    r"""Generate negative samples based on the `HIST-RND` strategy:

        - for each positive edge, sample a batch of negative edges from all
        possible edges with the same source node
        - filter actual positive edges

    Parameters:
        data: an object containing positive edges information
        split_mode: specifies whether to generate negative edges for
        'validation' or 'test' splits
        filename: name of the file containing the generated negative edges
    """
    print(
        f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split:"
        f" {split_mode}"
    )
    assert split_mode in [
        "val",
        "test",
    ], "Invalid split-mode! It should be `val` or `test`!"

    if os.path.exists(filename):
      print(
          f"INFO: negative samples for '{split_mode}' evaluation are already"
          " generated!"
      )
    else:
      print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
      # retrieve the information from the batch
      pos_src, pos_dst, pos_timestamp = (
          data.src.cpu().numpy(),
          data.dst.cpu().numpy(),
          data.t.cpu().numpy(),
      )

      # all possible destinations
      all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

      evaluation_set = {}
      # generate a list of negative destinations for each positive edge
      pos_edge_tqdm = tqdm.tqdm(
          zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
      )
      for (
          pos_s,
          pos_d,
          pos_t,
      ) in pos_edge_tqdm:
        t_mask = pos_timestamp == pos_t
        src_mask = pos_src == pos_s
        fn_mask = np.logical_and(t_mask, src_mask)
        pos_e_dst_same_src = pos_dst[fn_mask]
        filtered_all_dst = np.setdiff1d(all_dst, pos_e_dst_same_src)

        """
                when num_neg_e is larger than all possible destinations simple return all possible destinations
                """
        if self.num_neg_e > len(filtered_all_dst):
          neg_d_arr = filtered_all_dst
        else:
          neg_d_arr = np.random.choice(
              filtered_all_dst, self.num_neg_e, replace=False
          )  # never replace negatives

        evaluation_set[(pos_s, pos_d, pos_t)] = neg_d_arr

      # save the generated evaluation set to disk
      utils.save_pkl(evaluation_set, filename)

  def generate_historical_edge_set(
      self,
      historical_data,
  ):
    r"""Generate the set of edges seen durign training or validation

    ONLY `train_data` should be passed as historical data; i.e., the HISTORICAL
    negative edges should be selected from training data only.

    Parameters:
        historical_data: contains the positive edges observed previously

    Returns:
        historical_edges: distict historical positive edges
        hist_edge_set_per_node: historical edges observed for each node
    """
    sources = historical_data.src.cpu().numpy()
    destinations = historical_data.dst.cpu().numpy()
    historical_edges = {}
    hist_e_per_node = {}
    for src, dst in zip(sources, destinations):
      # edge-centric
      if (src, dst) not in historical_edges:
        historical_edges[(src, dst)] = 1

      # node-centric
      if src not in hist_e_per_node:
        hist_e_per_node[src] = [dst]
      else:
        hist_e_per_node[src].append(dst)

    hist_edge_set_per_node = {}
    for src, dst_list in hist_e_per_node.items():
      hist_edge_set_per_node[src] = np.array(list(set(dst_list)))

    return historical_edges, hist_edge_set_per_node

  def generate_negative_samples_hist_rnd(
      self,
      historical_data,
      data,
      split_mode,
      filename,
  ):
    r"""Generate negative samples based on the `HIST-RND` strategy:

        - up to 50% of the negative samples are selected from the set of edges
        seen during the training with the same source node.
        - the rest of the negative edges are randomly sampled with the fixed
        source node.

    Parameters:
        historical_data: contains the history of the observed positive edges
        including
                        distinct positive edges and edges observed for each
                        positive node
        data: an object containing positive edges information
        split_mode: specifies whether to generate negative edges for
        'validation' or 'test' splits
        filename: name of the file to save generated negative edges

    Returns:
        None
    """
    print(
        f"INFO: Negative Sampling Strategy: {self.strategy}, Data Split:"
        f" {split_mode}"
    )
    assert split_mode in [
        "val",
        "test",
    ], "Invalid split-mode! It should be `val` or `test`!"

    if os.path.exists(filename):
      print(
          f"INFO: negative samples for '{split_mode}' evaluation are already"
          " generated!"
      )
    else:
      print(f"INFO: Generating negative samples for '{split_mode}' evaluation!")
      # retrieve the information from the batch
      pos_src, pos_dst, pos_timestamp = (
          data.src.cpu().numpy(),
          data.dst.cpu().numpy(),
          data.t.cpu().numpy(),
      )

      pos_ts_edge_dict = {}  # {ts: {src: [dsts]}}
      pos_edge_tqdm = tqdm.tqdm(
          zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
      )
      for (
          pos_s,
          pos_d,
          pos_t,
      ) in pos_edge_tqdm:
        if pos_t not in pos_ts_edge_dict:
          pos_ts_edge_dict[pos_t] = {pos_s: [pos_d]}
        else:
          if pos_s not in pos_ts_edge_dict[pos_t]:
            pos_ts_edge_dict[pos_t][pos_s] = [pos_d]
          else:
            pos_ts_edge_dict[pos_t][pos_s].append(pos_d)

      # all possible destinations
      all_dst = np.arange(self.first_dst_id, self.last_dst_id + 1)

      # get seen edge history
      (
          historical_edges,
          hist_edge_set_per_node,
      ) = self.generate_historical_edge_set(historical_data)

      # sample historical edges
      max_num_hist_neg_e = int(self.num_neg_e * self.hist_ratio)

      evaluation_set = {}
      # generate a list of negative destinations for each positive edge
      pos_edge_tqdm = tqdm.tqdm(
          zip(pos_src, pos_dst, pos_timestamp), total=len(pos_src)
      )
      for (
          pos_s,
          pos_d,
          pos_t,
      ) in pos_edge_tqdm:
        pos_e_dst_same_src = np.array(pos_ts_edge_dict[pos_t][pos_s])

        # sample historical edges
        num_hist_neg_e = 0
        neg_hist_dsts = np.array([])
        seen_dst = []
        if pos_s in hist_edge_set_per_node:
          seen_dst = hist_edge_set_per_node[pos_s]
          if len(seen_dst) >= 1:
            filtered_all_seen_dst = np.setdiff1d(seen_dst, pos_e_dst_same_src)
            # filtered_all_seen_dst = seen_dst #! no collision check
            num_hist_neg_e = (
                max_num_hist_neg_e
                if max_num_hist_neg_e <= len(filtered_all_seen_dst)
                else len(filtered_all_seen_dst)
            )
            neg_hist_dsts = np.random.choice(
                filtered_all_seen_dst, num_hist_neg_e, replace=False
            )

        # sample random edges
        if len(seen_dst) >= 1:
          invalid_dst = np.concatenate((np.array(pos_e_dst_same_src), seen_dst))
        else:
          invalid_dst = np.array(pos_e_dst_same_src)
        filtered_all_rnd_dst = np.setdiff1d(all_dst, invalid_dst)

        num_rnd_neg_e = self.num_neg_e - num_hist_neg_e
        """
                when num_neg_e is larger than all possible destinations simple return all possible destinations
                """
        if num_rnd_neg_e > len(filtered_all_rnd_dst):
          neg_rnd_dsts = filtered_all_rnd_dst
        else:
          neg_rnd_dsts = np.random.choice(
              filtered_all_rnd_dst, num_rnd_neg_e, replace=False
          )
        # concatenate the two sets: historical and random
        neg_dst_arr = np.concatenate((neg_hist_dsts, neg_rnd_dsts))
        evaluation_set[(pos_s, pos_d, pos_t)] = neg_dst_arr

      # save the generated evaluation set to disk
      utils.save_pkl(evaluation_set, filename)
