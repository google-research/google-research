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

import os.path as osp
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from fm4tlp.utils import negative_sampler
from fm4tlp.utils import pre_process


class LinkPropPredDataset(object):

  def __init__(
      self,
      name,
      group,
      mode,
      root = "./data",
      meta_dict = None,
      preprocess = True,
  ):
    r"""Dataset class for link prediction dataset.

    Stores meta information about each dataset such as evaluation metrics etc.
    also automatically pre-processes the dataset.

    Args:
        name: name of the dataset
        data: Continent/community
        mode: Train, validation, or test
        root: root directory to store the dataset folder
        meta_dict: dictionary containing meta information about the dataset,
          should contain key 'dir_name' which is the name of the dataset folder
        preprocess: whether to pre-process the dataset
    """
    self.root = root
    self.name = name
    self.data = self.name + "_" + group + "_" + mode
    self.metric = "mrr"

    self.dir_name = self.name + "_" + group + "_" + mode

    if meta_dict is None:
      self.dir_name = self.name + "_" + group + "_" + mode
      meta_dict = {"dir_name": self.dir_name}
    else:
      self.dir_name = meta_dict["dir_name"]

    self.meta_dict = meta_dict

    if "fname" not in self.meta_dict:
      self.meta_dict["fname"] = self.root + "/" + self.data + "_edgelist.csv"
      self.meta_dict["nodefile"] = None

    if mode == "val":
      self.meta_dict["val_ns"] = (
          self.root + "/" + self.name + "_" + group + "_val_ns.pkl"
      )
    elif mode == "test":
      self.meta_dict["test_ns"] = (
          self.root + "/" + self.name + "_" + group + "_test_ns.pkl"
      )

    # initialize
    self._node_feat = None
    self._edge_feat = None
    self._full_data = None

    if preprocess:
      self.pre_process()

    self.ns_sampler = negative_sampler.NegativeEdgeSampler(
        dataset_name="-".join(self.name.split("_")), strategy="hist_rnd"
    )

  def generate_processed_files(
      self,
  ):
    """Generates processed files for the dataset.

    Returns:
        df: pandas data frame
    """
    node_feat = None

    if self.meta_dict["nodefile"] is not None:
      if not osp.exists(self.meta_dict["nodefile"]):
        raise FileNotFoundError(
            f"File not found at {self.meta_dict['nodefile']}"
        )

    if self.name in ("tgbl_flight"):
      pd_data = pre_process.csv_to_pd_data(self.meta_dict["fname"])
    elif self.name == "tgbl_comment":
      pd_data = pre_process.csv_to_pd_data_rc(self.meta_dict["fname"])
    elif self.name in ("tgbl_coin", "tgbl_review"):
      pd_data = pre_process.csv_to_pd_data_sc(self.meta_dict["fname"])
    elif self.name in ("tgbl_wiki"):
      pd_data = pre_process.load_edgelist_wiki(self.meta_dict["fname"])
    else:
      raise ValueError(f"Dataset {self.name} not supported")

    df = pd_data[0]
    edge_feat = pd_data[1]
    node_ids = pd_data[2]

    if self.meta_dict["nodefile"] is not None:
      node_feat = pre_process.process_node_feat(
          self.meta_dict["nodefile"], node_ids
      )

    return df, edge_feat, node_feat, node_ids

  def pre_process(self):
    """Pre-process the dataset, must be run before accessing dataset properties."""

    df, edge_feat, node_feat, unused_node_ids = self.generate_processed_files()
    sources = np.array(df["u"])
    destinations = np.array(df["i"])
    timestamps = np.array(df["ts"])
    edge_idxs = np.array(df["idx"])
    weights = np.array(df["w"])

    edge_label = np.ones(len(df))  ## should be 1 for all pos edges
    self._edge_feat = edge_feat
    self._node_feat = node_feat

    full_data = {
        "sources": sources,
        "destinations": destinations,
        "timestamps": timestamps,
        "edge_idxs": edge_idxs,
        "edge_feat": edge_feat,
        "w": weights,
        "edge_label": edge_label,
    }
    self._full_data = full_data

  @property
  def eval_metric(self):
    """Evaluation metric for the dataset, loaded from info.py.

    Returns:
        eval_metric: str, the evaluation metric
    """
    return self.metric

  @property
  def negative_sampler(self):
    """Negative sampler of the dataset, will load negative samples from disc.

    Returns:
        negative_sampler: NegativeEdgeSampler
    """
    return self.ns_sampler

  def load_val_ns(self):
    """Load the negative samples for the validation set."""
    self.ns_sampler.load_eval_set(
        fname=self.meta_dict["val_ns"], split_mode="val"
    )

  def load_test_ns(self):
    """Load the negative samples for the test set."""
    self.ns_sampler.load_eval_set(
        fname=self.meta_dict["test_ns"], split_mode="test"
    )

  @property
  def node_feat(self):
    r"""Returns the node features of the dataset with dim [N, feat_dim].

    Returns:
        node_feat: np.ndarray, [N, feat_dim] or None if there is no node feature
    """
    return self._node_feat

  @property
  def edge_feat(self):
    r"""Returns the edge features of the dataset with dim [E, feat_dim].

    Returns:
        edge_feat: np.ndarray, [E, feat_dim] or None if there is no edge feature
    """
    return self._edge_feat

  @property
  def full_data(self):
    r"""The full data of the dataset as a dictionary.

    Dictionary has keys:
     * 'sources'
     * 'destinations'
     * 'timestamps'
     * 'edge_idxs'
     * 'edge_feat'
     * 'w'
     * 'edge_label'

    Returns:
        full_data: Dict[str, Any]
    """
    if self._full_data is None:
      raise ValueError(
          "dataset has not been processed yet, please call pre_process() first"
      )
    return self._full_data
