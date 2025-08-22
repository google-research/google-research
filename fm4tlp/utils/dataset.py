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

"""Dataset class for link prediction datasets."""
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

from fm4tlp.utils import negative_sampler
from fm4tlp.utils import utils


_PROCESSED_DATASETS = ["comment"]


class LinkPropPredDataset(object):

  def __init__(
      self,
      name,
      root = None,
  ):
    r"""Dataset class for link prediction dataset.

    Stores meta information about each dataset such as evaluation metrics etc.

    Args:
        name: Name of the dataset.
        root: Root directory to store the dataset folder. This directory should
          contain subfolders, one of which is equivalent to `name`.

    Raises:
      FileNotFoundError: If there is no `name` subfolder in `root`.
    """
    self.name = name  ## original name
    self.dir_name = os.path.join(root, name)
    if not tf.io.gfile.isdir(self.dir_name):
      raise FileNotFoundError(
          f"Processed data folder for {name} not found in {root}"
      )

    self.metric = "mrr"
    self.meta_dict = {"dir_name": self.dir_name}
    self.meta_dict["val_ns"] = os.path.join(
        self.dir_name, "tgbl-" + self.name + "_val_ns.pkl"
    )
    self.meta_dict["test_ns"] = os.path.join(
        self.dir_name, "tgbl-" + self.name + "_test_ns.pkl"
    )

    # initialize
    self._node_feat = None
    self._edge_feat = None
    self._full_data = None
    self._train_data = None
    self._val_data = None
    self._test_data = None

    self.ns_sampler = negative_sampler.NegativeEdgeSampler(
        dataset_name=self.name, strategy="hist_rnd"
    )
    self.pre_process()

  def load_processed_files(self):
    r"""Turns raw data .csv file into a Pandas dataframe, stored on disc.

    Returns:
        df: pandas data frame

    Raises:
        FileNotFoundError: If the raw data .csv file is not found.
    """
    out_df = self.dir_name + "/" + "ml_tgbl-{}.pkl".format(self.name)
    out_edge_feat = (
        self.dir_name + "/" + "ml_tgbl-{}.pkl".format(self.name + "_edge")
    )
    out_node_feat = (
        self.dir_name + "/" + "ml_tgbl-{}.pkl".format(self.name + "_node")
    )

    if tf.io.gfile.exists(out_df):
      print("loading processed file")
      df = pd.read_pickle(out_df)
      edge_feat = utils.load_pkl(out_edge_feat)
    else:
      raise FileNotFoundError(f"processed file {out_df} not found")
    node_feat = None
    if tf.io.gfile.exists(out_node_feat):
      node_feat = utils.load_pkl(out_node_feat)

    return df, edge_feat, node_feat

  def pre_process(self):
    """Pre-processes the dataset and generates the splits.

    Generates the edge data and different train, val, test splits. Must be run
    before dataset properties can be accessed.
    """
    # TO-DO for link prediction, y = 1 because these are all true edges,
    # edge feat = weight + edge feat.

    # check if path to file is valid
    df, edge_feat, node_feat = self.load_processed_files()
    sources = np.array(df["u"])
    destinations = np.array(df["i"])
    timestamps = np.array(df["ts"])
    edge_idxs = np.array(df["idx"])
    weights = np.array(df["w"])

    edge_label = np.ones(len(df))  # should be 1 for all pos edges
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
    train_mask, val_mask, test_mask = self.generate_splits(full_data)
    self._train_mask = train_mask
    self._val_mask = val_mask
    self._test_mask = test_mask

  def generate_splits(
      self,
      full_data,
      val_ratio = 0.15,
      test_ratio = 0.15,
  ):
    r"""Generates train, validation, and test splits from the full dataset.

    Args:
        full_data: dictionary containing the full dataset
        val_ratio: ratio of validation data
        test_ratio: ratio of test data

    Returns:
        train_data: dictionary containing the training dataset
        val_data: dictionary containing the validation dataset
        test_data: dictionary containing the test dataset
    """
    val_time, test_time = list(
        np.quantile(
            full_data["timestamps"],
            [(1 - val_ratio - test_ratio), (1 - test_ratio)],
        )
    )
    timestamps = full_data["timestamps"]

    train_mask = timestamps <= val_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    return train_mask, val_mask, test_mask

  @property
  def eval_metric(self):
    """The official evaluation metric for the dataset, loaded from info.py.

    Returns:
        eval_metric: str, the evaluation metric
    """
    return self.metric

  @property
  def negative_sampler(self):
    r"""Returns the dataset negative sampler, loads negative samples from disc.

    Returns:
        negative_sampler: NegativeEdgeSampler
    """
    return self.ns_sampler

  def load_val_ns(self):
    r"""Loads the negative samples for the validation set."""
    self.ns_sampler.load_eval_set(
        fname=self.meta_dict["val_ns"], split_mode="val"
    )

  def load_test_ns(self):
    r"""Loads the negative samples for the test set."""
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
      edge_feat: np.ndarray, [E, feat_dim] or None if there is no edge feature.
    """
    return self._edge_feat

  @property
  def full_data(self):
    r"""The full data of the dataset as a dictionary.

    The keys correspond to: 'sources', 'destinations', 'timestamps',
    'edge_idxs', 'edge_feat', 'w', 'edge_label'.

    Returns:
      full_data: Dict[str, Any]
    """
    if self._full_data is None:
      raise ValueError(
          "dataset has not been processed yet, please call pre_process() first"
      )
    return self._full_data

  @property
  def train_mask(self):
    r"""Returns the train mask of the dataset.

    Returns:
      train_mask: training masks.
    """
    if self._train_mask is None:
      raise ValueError("training split hasn't been loaded")
    return self._train_mask

  @property
  def val_mask(self):
    r"""Returns the validation mask of the dataset.

    Returns:
      val_mask: Dict[str, Any]
    """
    if self._val_mask is None:
      raise ValueError("validation split hasn't been loaded")
    return self._val_mask

  @property
  def test_mask(self):
    r"""Returns the test mask of the dataset.

    Returns:
        test_mask: Dict[str, Any]
    """
    if self._test_mask is None:
      raise ValueError("test split hasn't been loaded")
    return self._test_mask
