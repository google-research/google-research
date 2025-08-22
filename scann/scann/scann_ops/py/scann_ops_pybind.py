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

"""Wrapper around pybind module that provides convenience functions for instantiating ScaNN searchers."""

import os
import pickle as pkl
import numpy as np
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cc/python"))
import scann_pybind
from scann.scann_ops.py import scann_builder
from scann.scann_ops.py import scann_ops_pybind_backcompat


def _open(path, mode):
  return open(path, mode)


class ScannSearcher(object):
  """Wrapper class around pybind module that provides a cleaner interface."""

  def __init__(self, searcher, docids=None):
    self.searcher = searcher
    # Simple docid mapping.
    self.docids = docids
    if docids is not None:
      self.docid_to_id = {docid: id for id, docid in enumerate(docids)}
      if len(docids) != len(self.docid_to_id):
        raise ValueError("Duplicates found in docids.")

  def search(
      self,
      q,
      final_num_neighbors=-1,
      pre_reorder_num_neighbors=-1,
      leaves_to_search=-1,
  ):
    """Single-query search; -1 for a param uses the searcher's default value."""
    idx, dist = self.searcher.search(q, final_num_neighbors,
                                     pre_reorder_num_neighbors,
                                     leaves_to_search)
    idx = idx if self.docids is None else [self.docids[j] for j in idx]
    return idx, dist

  def search_batched(
      self,
      queries,
      final_num_neighbors=None,
      pre_reorder_num_neighbors=None,
      leaves_to_search=None,
  ):
    """Search method for multiple queries."""
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = (-1 if pre_reorder_num_neighbors is None else
              pre_reorder_num_neighbors)
    leaves = -1 if leaves_to_search is None else leaves_to_search
    idx, dist = self.searcher.search_batched(
        queries,
        final_nn,
        pre_nn,
        leaves,
        False,
        0,  # Ignored when parallel=False.
    )
    idx = (
        idx
        if self.docids is None else [[self.docids[j] for j in i] for i in idx])
    return idx, dist

  def search_batched_parallel(
      self,
      queries,
      final_num_neighbors=None,
      pre_reorder_num_neighbors=None,
      leaves_to_search=None,
      batch_size=256,
  ):
    """Search method for multiple queries with multiple threads."""
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = (-1 if pre_reorder_num_neighbors is None else
              pre_reorder_num_neighbors)
    leaves = -1 if leaves_to_search is None else leaves_to_search
    idx, dist = self.searcher.search_batched(queries, final_nn, pre_nn, leaves,
                                             True, batch_size)
    idx = (
        idx
        if self.docids is None else [[self.docids[j] for j in i] for i in idx])
    return idx, dist

  def serialize(self, artifacts_dir, relative_path=False):
    self.searcher.serialize(artifacts_dir, relative_path)
    docids_fn = os.path.join(artifacts_dir, "scann_docids.pkl")

    if self.docids is not None:
      pkl.dump(self.docids, _open(docids_fn, "wb"))

  def get_health_stats(self):
    return self.searcher.get_health_stats()

  def initialize_health_stats(self):
    return self.searcher.initialize_health_stats()

  def upsert(self, docids, database, batch_size=1):
    """Insert or update datapoints into the searcher."""
    if not isinstance(docids, list):
      docids = [docids]
    if not isinstance(database, np.ndarray):
      database = np.array(database)
    if database.ndim == 1:
      database = np.expand_dims(database, 0)
    if len(docids) != database.shape[0]:
      raise ValueError(
          "Number of items mismatch in docids and database vectors:"
          f" {len(docids)} != {database.shape[0]}")
    if self.docids is None:
      raise ValueError("Cannot upsert because docids have not been specified "
                       "when initializing.")
    indices = [self.docid_to_id.get(docid) for docid in docids]

    for idx, docid in zip(indices, docids):
      if idx is not None:
        self.docids[idx] = docid
      else:
        self.docids.append(docid)
        self.docid_to_id[docid] = len(self.docids) - 1
    _ = self.searcher.upsert(indices, database, batch_size)

  def delete(self, docids):
    """Delete datapoints from searcher."""
    if not isinstance(docids, list):
      docids = [docids]
    indices = []
    for docid in docids:
      if docid not in self.docid_to_id:
        raise KeyError(f"Docid not found: {docid} ")
      idx = self.docid_to_id[docid]
      indices.append(idx)
      old_idx = len(self.docids) - 1
      if idx != old_idx:
        old_docid = self.docids[old_idx]
        self.docids[idx] = old_docid
        self.docid_to_id[old_docid] = idx
      self.docids.pop()
      self.docid_to_id.pop(docid)
    _ = self.searcher.delete(indices)

  def rebalance(self, config=None):
    """Rebalances the searcher."""
    # TODO(guorq): currently, this performs a full retrain based on the initial
    # config.
    config = "" if config is None else config
    return self.searcher.rebalance(config)

  def reserve(self, num_datapoints):
    return self.searcher.reserve(num_datapoints)

  def size(self):
    return self.searcher.size()

  def set_num_threads(self, num_threads):
    self.searcher.set_num_threads(num_threads)

  def config(self):
    """Returns the config."""
    return self.searcher.config()


def builder(db, num_neighbors, distance_measure):
  """pybind analogue of builder() in scann_ops.py; see docstring there."""

  class ScannBuilder(scann_builder.ScannBuilder):

    def create_config(self):
      """Create a config with autopilot rewrite without actually building the searcher."""
      config = super().create_config()
      if self.params.get("autopilot") is not None:
        config = scann_pybind.ScannNumpy.suggest_autopilot(
            config, self.db.shape[0], self.db.shape[1])
      return config

  def builder_lambda(db, config, training_threads, **kwargs):
    return create_searcher(db, config, training_threads, **kwargs)

  return ScannBuilder(db, num_neighbors,
                      distance_measure).set_builder_lambda(builder_lambda)


def create_searcher(db,
                    scann_config,
                    training_threads=0,
                    docids=None,
                    **unused_kwargs):
  """Creates a searcher object wrapping a ScannNumpy object."""
  if docids is not None:
    if len(docids) != db.shape[0]:
      raise ValueError(
          f"docid and database size mismatch: {len(docids)} != {db.shape[0]}.")
  if isinstance(db, np.ndarray) and db.shape[0] == 0:
    scann_config += f"""
      input_output {{
        pure_dynamic_config {{
          vector_type: DENSE
          dimensionality: {db.shape[1]}
        }}
      }}
    """
  return ScannSearcher(
      scann_pybind.ScannNumpy(db, scann_config, training_threads),
      docids=docids)


def load_searcher(artifacts_dir, assets_backcompat_shim=True):
  """Loads searcher assets from artifacts_dir and returns a ScaNN searcher."""
  is_dir = os.path.isdir(artifacts_dir)
  if not is_dir:
    raise ValueError(f"{artifacts_dir} is not a directory.")

  assets_pbtxt = os.path.join(artifacts_dir, "scann_assets.pbtxt")
  if not scann_ops_pybind_backcompat.path_exists(assets_pbtxt):
    if not assets_backcompat_shim:
      raise ValueError("No scann_assets.pbtxt found.")
    print("No scann_assets.pbtxt found. ScaNN assumes this searcher was from an"
          " earlier release, and is calling `populate_and_save_assets_proto`"
          "from `scann_ops_pybind_backcompat` to create a scann_assets.pbtxt. "
          "Note this compatibility shim may be removed in the future.")
    scann_ops_pybind_backcompat.populate_and_save_assets_proto(artifacts_dir)

  docids_path = os.path.join(artifacts_dir, "scann_docids.pkl")
  exists = os.path.isfile(docids_path)
  docid = pkl.load(_open(docids_path, "rb")) if exists else None

  with _open(assets_pbtxt, "r") as f:
    return ScannSearcher(
        scann_pybind.ScannNumpy(artifacts_dir, f.read()), docid)
