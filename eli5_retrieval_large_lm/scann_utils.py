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

"""ScaNN utils.

Originally from the //third_party/py/language/orqa codebase, but converted to
TF 2 and modified for the needs of the project.
"""
import scann
import tensorflow as tf

builder = scann.scann_ops_pybind.builder


def load_scann_searcher(var_name,
                        checkpoint_path,
                        num_neighbors,
                        dimensions_per_block=2,
                        num_leaves=1000,
                        num_leaves_to_search=100,
                        training_sample_size=100000,
                        reordering_num_neighbors=0):
  """Load scann searcher from checkpoint."""
  with tf.device("/cpu:0"):
    ckpt = tf.train.load_checkpoint(checkpoint_path)
    try:
      np_db = ckpt.get_tensor(var_name)
    except tf.errors.NotFoundError:
      np_db = ckpt.get_tensor(var_name + "/.ATTRIBUTES/VARIABLE_VALUE")

    builder_intance = builder(
        db=np_db,
        num_neighbors=num_neighbors,
        distance_measure="dot_product")
    builder_intance = builder_intance.tree(
        num_leaves=num_leaves,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=training_sample_size)
    builder_intance = builder_intance.score_ah(
        dimensions_per_block=dimensions_per_block)
    if reordering_num_neighbors:
      builder_intance = builder_intance.reorder(
          reordering_num_neighbors=reordering_num_neighbors)

    searcher = builder_intance.build()
  return np_db, searcher
