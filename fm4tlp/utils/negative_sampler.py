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

"""Sample negative edges for evaluation of dynamic link prediction.

Load already generated negative edges from file, batch them based on the
positive edge, and return the evaluation set
"""

import random

import numpy as np
import tensorflow.compat.v1 as tf
import torch

from fm4tlp.utils import utils


class NegativeEdgeSampler(object):

  def __init__(
      self,
      dataset_name,
      strategy = "hist_rnd",
  ):
    r"""Negative Edge Sampler.

        Loads and query the negative batches based on the positive batches
        provided.
    constructor for the negative edge sampler class

    Args:
      dataset_name: Name of the dataset.
      strategy: Specifies which set of negatives should be loaded; can be 'rnd'
        or 'hist_rnd'.
    """
    self.dataset_name = dataset_name
    assert strategy in [
        "rnd",
        "hist_rnd",
    ], "The supported strategies are `rnd` or `hist_rnd`!"
    self.strategy = strategy
    self.eval_set = {}

  def load_eval_set(
      self,
      fname,
      split_mode = "val",
  ):
    r"""Load the evaluation set from disk, can be either val or test ns samples.

    Args:
      fname: The file name of the evaluation ns on disk.
      split_mode: The split mode of the evaluation set, can be either `val` or
        `test`.

    Raises:
        FileNotFoundError: If the file with the input name is not found.
    """
    assert split_mode in [
        "val",
        "test",
    ], "Invalid split-mode! It should be `val`, `test`"
    if not tf.io.gfile.exists(fname):
      raise FileNotFoundError(f"File not found at {fname}")
    self.eval_set[split_mode] = utils.load_pkl(fname)

  def reset_eval_set(
      self,
      split_mode = "test",
  ):
    r"""Resets evaluation set.

    Args:
      split_mode: Specifies whether to generate negative edges for 'validation'
        or 'test' splits.
    """
    assert split_mode in [
        "val",
        "test",
    ], "Invalid split-mode! It should be `val`, `test`!"
    self.eval_set[split_mode] = None

  def query_batch(
      self,
      pos_src,
      pos_dst,
      pos_timestamp,
      split_mode,
  ):
    r"""For each positive edge in `pos_batch` returns a list of negative edges.

    `split_mode` specifies whether the valiation or test evaluation set should
    be retrieved.

    Args:
      pos_src: list of positive source nodes
      pos_dst: list of positive destination nodes
      pos_timestamp: list of timestamps of the positive edges
      split_mode: specifies whether to generate negative edges for 'validation'
        or 'test' splits

    Returns:
      neg_samples: a list of list; each internal list contains the set of
        negative edges that should be evaluated against each positive edge.

    Raises:
      RuntimeError: If NumPy arrays cannot be successfully extracted from (any
      of) the input tensors.
    """
    assert split_mode in [
        "val",
        "test",
    ], "Invalid split-mode! It should be `val`, `test`!"
    if self.eval_set[split_mode] is None:
      raise ValueError(
          f"Evaluation set is None! You should load the {split_mode} evaluation"
          " set first!"
      )

    # check the argument types...
    if torch is not None and isinstance(pos_src, torch.Tensor):
      pos_src = pos_src.detach().cpu().numpy()
    if torch is not None and isinstance(pos_dst, torch.Tensor):
      pos_dst = pos_dst.detach().cpu().numpy()
    if torch is not None and isinstance(pos_timestamp, torch.Tensor):
      pos_timestamp = pos_timestamp.detach().cpu().numpy()

    if (
        not isinstance(pos_src, np.ndarray)
        or not isinstance(pos_dst, np.ndarray)
        or not isinstance(pos_timestamp, np.ndarray)
    ):
      raise RuntimeError(
          "pos_src, pos_dst, and pos_timestamp need to be either numpy ndarray"
          " or torch tensor!"
      )

    neg_samples = []
    for pos_s, pos_d, pos_t in zip(pos_src, pos_dst, pos_timestamp):
      if (pos_s, pos_d, pos_t) not in self.eval_set[split_mode]:
        raise ValueError(
            f"The edge ({pos_s}, {pos_d}, {pos_t}) is not in the '{split_mode}'"
            " evaluation set! Please check the implementation."
        )
      else:
        neg_samples.append([
            int(neg_dst)
            for neg_dst in self.eval_set[split_mode][(pos_s, pos_d, pos_t)]
        ])

    return neg_samples


def get_negatives(
    historical_neighbor_set,
    all_nodes,
    num_neg,
    historical_frac = 0.5,
):
  """Generates negative samples from historical neighbor set + random nodes.

  Args:
    historical_neighbor_set: Set of historical neighbors of a source node.
    all_nodes: Set of all nodes in the graph.
    num_neg: Number of negative samples to generate.
    historical_frac: Fraction of negative samples to be historical.

  Returns:
    Array of negative samples.
  """
  historical_neg_num = min(
      int(num_neg * historical_frac), len(historical_neighbor_set)
  )
  historical_negative_nodes = random.sample(
      list(historical_neighbor_set), historical_neg_num
  )

  random_neg_num = num_neg - historical_neg_num
  random_negative_node_set = set()
  while len(random_negative_node_set) < random_neg_num:
    random_negative_node_set.update(
        set(random.sample(all_nodes, random_neg_num)).difference(
            historical_neighbor_set
        )
    )

  random_negative_nodes = list(random_negative_node_set)
  random.shuffle(random_negative_nodes)
  random_negative_nodes = random_negative_nodes[:random_neg_num]

  return np.array(historical_negative_nodes + random_negative_nodes)
