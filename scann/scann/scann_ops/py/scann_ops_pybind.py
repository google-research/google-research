# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2020 Google LLC
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


def create_searcher(db, scann_config, training_threads=0):
  return ScannSearcher(
      scann_pybind.ScannNumpy(db, scann_config, training_threads))


def load_searcher(db, artifacts_dir):
  tokenization_path = os.path.join(artifacts_dir, "datapoint_to_token.npy")
  hashed_db_path = os.path.join(artifacts_dir, "hashed_dataset.npy")

  tokenization = np.load(tokenization_path) if os.path.isfile(
      tokenization_path) else None
  hashed_db = np.load(hashed_db_path) if os.path.isfile(
      hashed_db_path) else None
  return ScannSearcher(
      scann_pybind.ScannNumpy(db, tokenization, hashed_db, artifacts_dir))
