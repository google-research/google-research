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

"""Python API for ScaNN - single machine, dense vector similarity search."""

import hashlib
import os
import tensorflow as tf

_scann_ops_so = tf.load_op_library(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cc/_scann_ops.so"))
scann_create_searcher = _scann_ops_so.addons_scann_create_searcher
scann_search = _scann_ops_so.addons_scann_search
scann_search_batched = _scann_ops_so.addons_scann_search_batched
scann_to_tensors = _scann_ops_so.addons_scann_to_tensors
tensors_to_scann = _scann_ops_so.addons_tensors_to_scann


def searcher_from_module(module, db):
  return ScannSearcher(module.recreate_handle(db))


class ScannState(tf.Module):
  """Class that wraps ScaNN searcher assets for object-based checkpointing."""

  def __init__(self, tensors):
    super(ScannState, self).__init__()
    scann_config, serialized_partitioner, datapoint_to_token, ah_codebook, hashed_dataset = tensors

    def make_var(v):
      with tf.compat.v1.variable_scope(
          tf.compat.v1.VariableScope(use_resource=True, reuse=False)):
        return tf.Variable(v, validate_shape=False)

    self.scann_config = make_var(scann_config)
    self.serialized_partitioner = make_var(serialized_partitioner)
    self.datapoint_to_token = make_var(datapoint_to_token)
    self.ah_codebook = make_var(ah_codebook)
    self.hashed_dataset = make_var(hashed_dataset)

  @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
  def recreate_handle(self, db):
    """Creates resource handle to searcher from ScaNN searcher assets."""
    return tensors_to_scann(db, self.scann_config, self.serialized_partitioner,
                            self.datapoint_to_token, self.ah_codebook,
                            self.hashed_dataset)


class ScannSearcher(object):
  """Wrapper class that holds the ScaNN searcher resource handle."""

  def __init__(self, searcher_handle):
    self.searcher_handle = searcher_handle

  def search(self,
             q,
             final_num_neighbors=None,
             pre_reorder_num_neighbors=None,
             leaves_to_search=None):
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = -1 if pre_reorder_num_neighbors is None else pre_reorder_num_neighbors
    leaves = -1 if leaves_to_search is None else leaves_to_search
    return scann_search(self.searcher_handle, q, final_nn, pre_nn, leaves)

  def search_batched(self,
                     q,
                     final_num_neighbors=None,
                     pre_reorder_num_neighbors=None,
                     leaves_to_search=None):
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = -1 if pre_reorder_num_neighbors is None else pre_reorder_num_neighbors
    leaves = -1 if leaves_to_search is None else leaves_to_search
    return scann_search_batched(self.searcher_handle, q, final_nn, pre_nn,
                                leaves, False)

  def search_batched_parallel(self,
                              q,
                              final_num_neighbors=None,
                              pre_reorder_num_neighbors=None,
                              leaves_to_search=None):
    final_nn = -1 if final_num_neighbors is None else final_num_neighbors
    pre_nn = -1 if pre_reorder_num_neighbors is None else pre_reorder_num_neighbors
    leaves = -1 if leaves_to_search is None else leaves_to_search
    return scann_search_batched(self.searcher_handle, q, final_nn, pre_nn,
                                leaves, True)

  def serialize_to_module(self):
    return ScannState(scann_to_tensors(self.searcher_handle))


def create_searcher(db,
                    scann_config,
                    training_threads=0,
                    container="",
                    shared_name=None):
  """Create a ScaNN searcher given a dataset and text config proto."""
  if shared_name is None:
    config_hash = hashlib.sha256(scann_config.encode("utf-8")).hexdigest()
    shared_name = "scann-" + config_hash[:16]
  return ScannSearcher(
      scann_create_searcher(
          x=db,
          scann_config=scann_config,
          training_threads=training_threads,
          container=container,
          shared_name=shared_name))
