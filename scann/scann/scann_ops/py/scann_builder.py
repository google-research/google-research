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

"""Builder to create ScaNN searchers of various configurations."""
from scann.scann_ops.py import scann_ops
from scann.scann_ops.py import scann_ops_pybind


def _factory_decorator(key):
  """Wraps a function that produces a portion of the ScaNN config proto."""

  def func_taker(f):
    """Captures arguments to function and saves them to params for later."""

    def inner(self, *args, **kwargs):
      if key in self.params:
        raise Exception("{} has already been configured".format(key))
      kwargs.update(zip(f.__code__.co_varnames[1:], args))
      self.params[key] = kwargs
      return self

    inner.proto_maker = f
    return inner

  return func_taker


class ScannBuilder(object):
  """Builder class."""

  def __init__(self, db, num_neighbors, distance_measure):
    self.params = {}
    self.training_threads = 0
    self.db = db
    self.num_neighbors = num_neighbors
    self.distance_measure = distance_measure

  def set_n_training_threads(self, threads):
    self.training_threads = threads
    return self

  @_factory_decorator("tree")
  def tree(
      self,
      num_leaves,
      num_leaves_to_search,
      training_sample_size=100000,
      min_partition_size=50,
      training_iterations=10,
      spherical=False,
      quantize_centroids=False,
      # the following are set automatically
      distance_measure=None):
    """Configure partitioning. If not called, no partitioning is performed."""
    return """
      partitioning {{
        num_children: {}
        min_cluster_size: {}
        max_clustering_iterations: {}
        partitioning_distance {{
          distance_measure: "SquaredL2Distance"
        }}
        query_spilling {{
          spilling_type: FIXED_NUMBER_OF_CENTERS
          max_spill_centers: {}
        }}
        expected_sample_size: {}
        query_tokenization_distance_override {}
        partitioning_type: {}
        query_tokenization_type: {}
      }}
    """.format(num_leaves, min_partition_size, training_iterations,
               num_leaves_to_search, training_sample_size, distance_measure,
               "SPHERICAL" if spherical else "GENERIC",
               "FIXED_POINT_INT8" if quantize_centroids else "FLOAT")

  @_factory_decorator("score_ah")
  def score_ah(
      self,
      dimensions_per_block,
      anisotropic_quantization_threshold=float("nan"),
      training_sample_size=100000,
      min_cluster_size=100,
      hash_type="lut16",
      training_iterations=10,
      # the following are set automatically
      residual_quantization=None,
      n_dims=None):
    """Configure asymmetric hashing. Must call this or score_brute_force."""
    hash_types = ["lut16", "lut256"]
    if hash_type == hash_types[0]:
      clusters_per_block = 16
      lookup_type = "INT8_LUT16"
    elif hash_type == hash_types[1]:
      clusters_per_block = 256
      lookup_type = "INT8"
    else:
      raise ValueError("hash_type must be one of {}".format(hash_types))
    if n_dims % dimensions_per_block == 0:
      proj_config = """
        projection_type: CHUNK
        num_blocks: {}
        num_dims_per_block: {}
      """.format(n_dims // dimensions_per_block, dimensions_per_block)
    else:
      proj_config = """
        projection_type: VARIABLE_CHUNK
        variable_blocks {{
          num_blocks: {}
          num_dims_per_block: {}
        }}
        variable_blocks {{
          num_blocks: {}
          num_dims_per_block: {}
        }}
      """.format(n_dims // dimensions_per_block, dimensions_per_block, 1,
                 n_dims % dimensions_per_block)
    return """
      hash {{
        asymmetric_hash {{
          lookup_type: {}
          use_residual_quantization: {}
          quantization_distance {{
            distance_measure: "SquaredL2Distance"
          }}
          num_clusters_per_block: {}
          projection {{
            input_dim: {}
            {}
          }}
          noise_shaping_threshold: {}
          expected_sample_size: {}
          min_cluster_size: {}
          max_clustering_iterations: {}
        }}
      }} """.format(lookup_type, residual_quantization, clusters_per_block,
                    n_dims, proj_config, anisotropic_quantization_threshold,
                    training_sample_size, min_cluster_size, training_iterations)

  @_factory_decorator("score_bf")
  def score_brute_force(self, quantize=False):
    return """
      brute_force {{
        fixed_point {{
          enabled: {}
        }}
      }}
    """.format(quantize)

  @_factory_decorator("reorder")
  def reorder(self, reordering_num_neighbors, quantize=False):
    return """
      exact_reordering {{
        approx_num_neighbors: {}
        fixed_point {{
          enabled: {}
        }}
      }}
    """.format(reordering_num_neighbors, quantize)

  def create_config(self):
    """Returns a text ScaNN config matching the specification in self.params."""
    allowed_measures = {
        "dot_product": '{distance_measure: "DotProductDistance"}',
        "squared_l2": '{distance_measure: "SquaredL2Distance"}',
    }
    distance_measure = allowed_measures.get(self.distance_measure)
    if distance_measure is None:
      raise ValueError("distance_measure must be one of {}".format(
          list(allowed_measures.keys())))
    config = """
      num_neighbors: {}
      distance_measure {}
    """.format(self.num_neighbors, distance_measure)

    tree_params = self.params.get("tree")
    if tree_params is not None:
      tree_params["distance_measure"] = distance_measure
      config += self.tree.proto_maker(self, **tree_params)

    ah = self.params.get("score_ah")
    bf = self.params.get("score_bf")
    if ah is not None and bf is None:
      ah["residual_quantization"] = tree_params is not None and self.distance_measure == "dot_product"
      ah["n_dims"] = self.db.shape[1]
      config += self.score_ah.proto_maker(self, **ah)
    elif bf is not None and ah is None:
      config += self.score_brute_force.proto_maker(self, **bf)
    else:
      raise Exception(
          "Exactly one of score_ah or score_brute_force must be used")

    reorder_params = self.params.get("reorder")
    if reorder_params is not None:
      config += self.reorder.proto_maker(self, **reorder_params)
    return config

  def create_tf(self, **kwargs):
    config = self.create_config()
    return scann_ops.create_searcher(self.db, config, self.training_threads,
                                     **kwargs)

  def create_pybind(self):
    config = self.create_config()
    return scann_ops_pybind.create_searcher(self.db, config,
                                            self.training_threads)
