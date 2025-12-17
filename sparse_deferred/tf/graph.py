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

"""Provides tf functions for `GraphStruct`."""

import functools
import tensorflow as tf

from sparse_deferred.structs import graph_struct
import sparse_deferred.tf as sdtf

tf = tf.compat.v2

GraphStruct = graph_struct.GraphStruct


def batch_dataset(
    ds, batch_size, stack_edges = False
):
  """Create a single graph consisting of multiple subgraphs.

  It is strongly recommended that the caller has called add_pooling to the input
  graph structs as batching works by adding a `virtual` node set named `g` (with
  edges connected to all nodes) and the client can then readout the values over
  the `g` node set.

  Args:
    ds: A tf.data.Dataset
    batch_size: The batch_size
    stack_edges: If False (default) the new edges will have format (source,
      target). If true, the edges will be a 2D Tensor with format like
      [[source], [target]].

  Returns:
    A tf.data.Dataset instance
  """
  datasets = tuple([ds.shard(batch_size, i) for i in range(batch_size)])
  return tf.data.Dataset.zip(tuple(datasets)).map(
      functools.partial(
          graph_struct.combine_graph_structs,
          sdtf.engine,
          stack_edges=stack_edges,
      )
  )


def db_to_tf_dataset(db):
  return tf.data.Dataset.range(db.size).map(
      functools.partial(db.get_item_with_engine, sdtf.engine))


class InMemoryDB(graph_struct.InMemoryDB):
  """InMemoryDB with get_item() yields GraphStruct w/ tf.Tensor feats&edges."""

  def as_tf_dataset(self):
    return db_to_tf_dataset(self)

  def finalize(self, to_device_fn=tf.convert_to_tensor):
    return super().finalize(to_device_fn=to_device_fn)

  @classmethod
  def from_file(cls, filename):
    db = InMemoryDB()
    db.load_from_file(filename, to_device_fn=tf.convert_to_tensor)
    return db
