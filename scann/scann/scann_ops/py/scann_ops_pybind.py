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

"""Wrapper around pybind module that provides convenience functions for instantiating ScaNN searchers."""

# pylint: disable=g-import-not-at-top,g-bad-import-order,unused-import
import os
import sys

import numpy as np

# needed because of C++ dependency on TF headers
import tensorflow as _tf
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cc/python"))
import scann_pybind
from scann.scann_ops.py import scann_builder


class ScannSearcher(object):
  """Wrapper class around pybind module that provides a cleaner interface."""

  def __init__(self, searcher):
    self.searcher = searcher

  def search(self,
             q,
             final_num_neighbors=None,
             pre_reorder_num_neighbors=None,
             leaves_to_search=None):
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = -1 if pre_reorder_num_neighbors is None else pre_reorder_num_neighbors
    leaves = -1 if leaves_to_search is None else leaves_to_search
    return self.searcher.search(q, final_nn, pre_nn, leaves)

  def search_batched(self,
                     queries,
                     final_num_neighbors=None,
                     pre_reorder_num_neighbors=None,
                     leaves_to_search=None):
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = -1 if pre_reorder_num_neighbors is None else pre_reorder_num_neighbors
    leaves = -1 if leaves_to_search is None else leaves_to_search
    return self.searcher.search_batched(queries, final_nn, pre_nn, leaves,
                                        False)

  def search_batched_parallel(self,
                              queries,
                              final_num_neighbors=None,
                              pre_reorder_num_neighbors=None,
                              leaves_to_search=None):
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = -1 if pre_reorder_num_neighbors is None else pre_reorder_num_neighbors
    leaves = -1 if leaves_to_search is None else leaves_to_search
    return self.searcher.search_batched(queries, final_nn, pre_nn, leaves, True)

  def serialize(self, artifacts_dir):
    self.searcher.serialize(artifacts_dir)


def builder(db, num_neighbors, distance_measure):
  """pybind analogue of builder() in scann_ops.py; see docstring there."""

  def builder_lambda(db, config, training_threads, **kwargs):
    return create_searcher(db, config, training_threads, **kwargs)

  return scann_builder.ScannBuilder(
      db, num_neighbors, distance_measure).set_builder_lambda(builder_lambda)


def create_searcher(db, scann_config, training_threads=0):
  return ScannSearcher(
      scann_pybind.ScannNumpy(db, scann_config, training_threads))


def load_searcher(artifacts_dir):
  """Loads searcher assets from artifacts_dir and returns a ScaNN searcher."""

  def load_if_exists(filename):
    path = os.path.join(artifacts_dir, filename)
    return np.load(path) if os.path.isfile(path) else None

  db = load_if_exists("dataset.npy")
  tokenization = load_if_exists("datapoint_to_token.npy")
  hashed_db = load_if_exists("hashed_dataset.npy")
  int8_db = load_if_exists("int8_dataset.npy")
  int8_multipliers = load_if_exists("int8_multipliers.npy")
  db_norms = load_if_exists("dp_norms.npy")

  return ScannSearcher(
      scann_pybind.ScannNumpy(db, tokenization, hashed_db, int8_db,
                              int8_multipliers, db_norms, artifacts_dir))
