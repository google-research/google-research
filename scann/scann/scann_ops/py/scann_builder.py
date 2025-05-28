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

"""Builder to create ScaNN searchers of various configurations."""

import enum
from typing import Optional


def _factory_decorator(key):
  """Wraps a function that produces a portion of the ScaNN config proto."""

  def func_taker(f):
    """Captures arguments to function and saves them to params for later."""

    def inner(self, *args, **kwargs):
      if key in self.params:
        raise Exception(f"{key} has already been configured")
      kwargs.update(zip(f.__code__.co_varnames[1:], args))
      self.params[key] = kwargs
      return self

    inner.proto_maker = f
    return inner

  return func_taker


class ReorderType(enum.Enum):
  FLOAT32 = 1
  INT8 = 2
  BFLOAT16 = 3


class IncrementalMode(enum.Enum):
  NONE = 1
  ONLINE = 2
  ONLINE_INCREMENTAL = 3


class ScannBuilder(object):
  """Builder class."""

  def __init__(self, db, num_neighbors, distance_measure):
    self.params = {}
    self.training_threads = 0
    self.builder_lambda = None
    self.db = db
    self.num_neighbors = num_neighbors
    self.distance_measure = distance_measure

  def set_n_training_threads(self, threads):
    self.training_threads = threads
    return self

  def set_builder_lambda(self, builder_lambda):
    """Sets builder_lambda, which creates a ScaNN searcher upon calling build().

    Args:
      builder_lambda: a function that takes a dataset, ScaNN config text proto,
        number of training threads, and optional kwargs as arguments, and
        returns a ScaNN searcher.

    Returns:
      The builder object itself, as expected from the builder pattern.
    """
    self.builder_lambda = builder_lambda
    return self

  @_factory_decorator("pca")
  def pca(
      self,
      reduction_dim = None,
      pca_significance_threshold = 0.80,
      pca_truncation_threshold = 0.6,
  ):
    """Configure PCA, when dimensionality reduction is possible."""
    dim = self.db.shape[1]
    pca_stanza = ""
    if reduction_dim is not None and pca_significance_threshold is None:
      pca_stanza = f"""num_dims_per_block: {reduction_dim}"""
    elif pca_significance_threshold is not None and reduction_dim is None:
      pca_stanza = f"""
          pca_significance_threshold: {pca_significance_threshold}
          pca_truncation_threshold: {pca_truncation_threshold}
      """
    else:
      raise ValueError("pca must be called with either reduction_dim or "
                       "pca_significance_threshold")
    return f"""
      projection: {{
        projection_type: PCA
        input_dim: {dim}
        {pca_stanza}
      }}"""

  @_factory_decorator("truncate")
  def truncate(self, reduction_dim):
    """Configure truncation of the input dimension, useful in MRL embeddings."""
    dim = self.db.shape[1]
    if reduction_dim >= dim:
      raise ValueError(f"reduction_dim must be less than {dim}")
    return f"""
      projection: {{
          projection_type: TRUNCATE
          num_dims_per_block: {reduction_dim}
          input_dim: {dim}
      }}"""

  @_factory_decorator("upper_tree")
  def upper_tree(
      self,
      num_leaves,
      num_leaves_to_search,
      avq = float("nan"),
      soar_lambda = None,
      overretrieve_factor=None,
      scoring_mode = ReorderType.INT8,
      anisotropic_quantization_threshold=float("nan"),
  ):
    """Configure an additional tree layer; REQUIRES tree() call in builder."""
    scoring_mode = {
        ReorderType.INT8: "FIXED8",
        ReorderType.BFLOAT16: "BFLOAT16",
        ReorderType.FLOAT32: "FLOAT32",
    }[scoring_mode]
    return f"""
    enabled: true
    num_centroids: {num_leaves}
    num_centroids_to_search: {num_leaves_to_search}
    avq: {avq}
    soar: {{
      enabled: {soar_lambda is not None}
      lambda: {soar_lambda or 1.5}
      overretrieve_factor: {overretrieve_factor or 2.0}
    }}
    quantization: {scoring_mode}
    noise_shaping_threshold: {anisotropic_quantization_threshold}
    """

  @_factory_decorator("tree")
  def tree(
      self,
      num_leaves,
      num_leaves_to_search,
      training_sample_size=100000,
      min_partition_size=50,
      training_iterations=12,
      spherical=False,
      quantize_centroids=False,
      random_init=True,
      incremental_threshold=None,
      avq=None,
      soar_lambda=None,
      overretrieve_factor=None,
      # the following are set automatically
      distance_measure=None,
      projection=None,
      upper_tree=None,
  ):
    """Configure partitioning. If not called, no partitioning is performed."""
    incremental_stanza = ""
    if isinstance(incremental_threshold, int):
      incremental_stanza = ("incremental_training_config { "
                            f"number_of_datapoints: {incremental_threshold} }}")
    elif isinstance(incremental_threshold, float):
      incremental_stanza = (
          f"incremental_training_config {{ fraction: {incremental_threshold} }}"
      )

    avq_stanza = f"avq: {avq}" if avq is not None else ""
    if avq is not None and self.distance_measure != "dot_product":
      raise ValueError("AVQ only applies to dot product distance.")

    soar_stanza = ""
    if soar_lambda is not None:
      if self.distance_measure != "dot_product":
        raise ValueError("SOAR requires dot product distance.")
      overretrieve_factor_stanza = (
          f"overretrieve_factor: {overretrieve_factor}"
          if overretrieve_factor is not None else "")
      soar_stanza = f"""database_spilling {{
        spilling_type: TWO_CENTER_ORTHOGONALITY_AMPLIFIED
        orthogonality_amplification_lambda: {soar_lambda}
        {overretrieve_factor_stanza}
      }}"""
    if upper_tree is not None:
      upper_tree = f"bottom_up_top_level_partitioner {{ {upper_tree} }}"

    return f"""
      partitioning {{
        num_children: {num_leaves}
        min_cluster_size: {min_partition_size}
        max_clustering_iterations: {training_iterations}
        single_machine_center_initialization: {
            "RANDOM_INITIALIZATION" if random_init else "DEFAULT_KMEANS_PLUS_PLUS"
        }
        partitioning_distance {{
          distance_measure: "SquaredL2Distance"
        }}
        query_spilling {{
          spilling_type: FIXED_NUMBER_OF_CENTERS
          max_spill_centers: {num_leaves_to_search}
        }}
        expected_sample_size: {training_sample_size}
        query_tokenization_distance_override {distance_measure}
        partitioning_type: {"SPHERICAL" if spherical else "GENERIC"}
        query_tokenization_type: {"FIXED_POINT_INT8" if quantize_centroids else "FLOAT"}
        {incremental_stanza}
        {avq_stanza}
        {soar_stanza}
        {projection or ""}
        {upper_tree or ""}
      }}
    """

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
      n_dims=None,
      projection=None,
  ):
    """Configure asymmetric hashing. Must call this or score_brute_force."""
    del min_cluster_size  # Deprecated field.
    hash_types = ["lut16", "lut256"]
    if hash_type == hash_types[0]:
      clusters_per_block = 16
      lookup_type = "INT8_LUT16"
    elif hash_type == hash_types[1]:
      clusters_per_block = 256
      lookup_type = "INT8"
    else:
      raise ValueError(f"hash_type must be one of {hash_types}")
    full_blocks, partial_block_dims = divmod(n_dims, dimensions_per_block)  # pytype: disable=wrong-arg-types
    if projection is not None:
      # Projection will be set by C++ logic.
      proj_config = f"""
        projection_type: CHUNK
        num_dims_per_block: {dimensions_per_block}
      """
    elif partial_block_dims == 0:
      proj_config = f"""
        input_dim: {n_dims}
        projection_type: CHUNK
        num_blocks: {full_blocks}
        num_dims_per_block: {dimensions_per_block}
      """
    else:
      proj_config = f"""
        input_dim: {n_dims}
        projection_type: VARIABLE_CHUNK
        variable_blocks {{
          num_blocks: {full_blocks}
          num_dims_per_block: {dimensions_per_block}
        }}
        variable_blocks {{
          num_blocks: 1
          num_dims_per_block: {partial_block_dims}
        }}
      """
    # global top-N requires:
    # (1) LUT16, (2) int16 accumulators, and (3) residual quantization.
    global_topn = (
        hash_type == hash_types[0] and
        (full_blocks + (partial_block_dims > 0)) <= 256 and
        residual_quantization)
    return f"""
      hash {{
        asymmetric_hash {{
          lookup_type: {lookup_type}
          use_residual_quantization: {residual_quantization}
          use_global_topn: {global_topn}
          quantization_distance {{
            distance_measure: "SquaredL2Distance"
          }}
          num_clusters_per_block: {clusters_per_block}
          projection {{
            {proj_config}
          }}
          fixed_point_lut_conversion_options {{
            float_to_int_conversion_method: ROUND
          }}
          noise_shaping_threshold: {anisotropic_quantization_threshold}
          expected_sample_size: {training_sample_size}
          max_clustering_iterations: {training_iterations}
        }}
      }} """

  @_factory_decorator("score_bf")
  def score_brute_force(self, quantize=ReorderType.FLOAT32):
    # Backwards-compatibility shims for when quantize was a bool parameter.
    if quantize is True:  # pylint: disable=g-bool-id-comparison
      quantize = ReorderType.INT8
    elif quantize is False:  # pylint: disable=g-bool-id-comparison
      quantize = ReorderType.FLOAT32
    return f"""
      brute_force {{
        {"bfloat16" if quantize == ReorderType.BFLOAT16 else "fixed_point"} {{
          enabled: {quantize != ReorderType.FLOAT32}
        }}
      }}
    """

  @_factory_decorator("reorder")
  def reorder(
      self,
      reordering_num_neighbors,
      quantize = ReorderType.FLOAT32,
      anisotropic_quantization_threshold = float("nan"),
  ):
    """Configure reordering (more accurate scoring after AH scoring)."""
    # Backwards-compatibility shims for when quantize was a bool parameter.
    if quantize is True:  # pylint: disable=g-bool-id-comparison
      quantize = ReorderType.INT8
    elif quantize is False:  # pylint: disable=g-bool-id-comparison
      quantize = ReorderType.FLOAT32
    return f"""
      exact_reordering {{
        approx_num_neighbors: {reordering_num_neighbors}
        {"bfloat16" if quantize == ReorderType.BFLOAT16 else "fixed_point"} {{
          enabled: {quantize != ReorderType.FLOAT32}
          noise_shaping_threshold: {anisotropic_quantization_threshold}
        }}
      }}
    """

  @_factory_decorator("autopilot")
  def autopilot(self, mode=IncrementalMode.NONE, quantize=ReorderType.FLOAT32):
    """Configure autopilot."""
    mode_string = {
        IncrementalMode.NONE: "NONE",
        IncrementalMode.ONLINE: "ONLINE",
        IncrementalMode.ONLINE_INCREMENTAL: "ONLINE_INCREMENTAL",
    }
    reorder_string = {
        ReorderType.FLOAT32: "FLOAT32",
        ReorderType.INT8: "INT8",
        ReorderType.BFLOAT16: "BFLOAT16",
    }
    return f"""
    autopilot {{
      tree_ah {{
        incremental_mode: {mode_string[mode]}
        reordering_dtype: {reorder_string[quantize]}
      }}
    }}
  """

  def create_config(self):
    """Returns a text ScaNN config matching the specification in self.params."""
    allowed_measures = {
        "dot_product": '{distance_measure: "DotProductDistance"}',
        "squared_l2": '{distance_measure: "SquaredL2Distance"}',
    }
    distance_measure = allowed_measures.get(self.distance_measure)
    if distance_measure is None:
      allowed = list(allowed_measures.keys())
      raise ValueError(f"distance_measure must be one of {allowed}")
    config = f"""
      num_neighbors: {self.num_neighbors}
      distance_measure {distance_measure}
    """

    autopilot_params = self.params.get("autopilot")
    if autopilot_params is not None:
      config += self.autopilot.proto_maker(self, **autopilot_params)
      # We will return return the config directly since the autopilot stanza
      # is all we need.
      return config

    pca_params = self.params.get("pca")
    truncate_params = self.params.get("truncate")
    projection = None
    if pca_params is not None and truncate_params is None:
      projection = self.pca.proto_maker(self, **pca_params)
    elif pca_params is None and truncate_params is not None:
      projection = self.truncate.proto_maker(self, **truncate_params)
    elif pca_params is not None and truncate_params is not None:
      raise ValueError("Exactly 1 of pca or truncate must be set")

    tree_params = self.params.get("tree")
    if tree_params is not None:
      tree_params["distance_measure"] = distance_measure
      upper_tree = self.params.get("upper_tree")
      if upper_tree is not None:
        upper_tree = self.upper_tree.proto_maker(self, **upper_tree)
      config += self.tree.proto_maker(
          self, **tree_params, projection=projection, upper_tree=upper_tree)

    ah = self.params.get("score_ah")
    bf = self.params.get("score_bf")
    if ah is not None and bf is None:
      if "residual_quantization" not in ah:
        ah["residual_quantization"] = (
            tree_params is not None and self.distance_measure == "dot_product")
      ah["n_dims"] = self.db.shape[1]
      config += self.score_ah.proto_maker(self, **ah, projection=projection)
    elif bf is not None and ah is None:
      config += self.score_brute_force.proto_maker(self, **bf)
    else:
      raise ValueError("Exactly 1 of score_ah or score_brute_force must be set")

    reorder_params = self.params.get("reorder")
    if reorder_params is not None:
      config += self.reorder.proto_maker(self, **reorder_params)
    return config

  def build(self, docids=None, **kwargs):
    """Calls builder_lambda to return a ScaNN searcher with the built config.

    The type of the returned searcher (pybind or TensorFlow) is configured
    through builder_lambda; see builder() in scann_ops.py and
    scann_ops_pybind.py for their respective builder_lambda's.

    Args:
      docids: if docids are present, they must be a list of strings with the
        same size as the database. docids are required in order to update the
        searcher (add, remove and delete).  In addition, docids must not be
        duplicated and should uniquely indicate the datapoint.
      **kwargs: additional, optional parameters to pass to builder_lambda.

    Returns:
      The returned value from builder_lambda(), which is a ScaNN searcher.
    Raises:
      Exception: if no builder_lambda was set.
    """
    if self.builder_lambda is None:
      raise Exception("build() called but no builder lambda was set.")

    config = self.create_config()
    return self.builder_lambda(
        self.db, config, self.training_threads, docids=docids, **kwargs)
