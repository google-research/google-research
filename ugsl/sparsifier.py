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

"""Sparsifier module of the GSL Layer.

The sparsifier module contains multiple sparsifiers tried in existing models.
"""
import tensorflow as tf

from ugsl import graph_structure


@tf.keras.utils.register_keras_serializable(package="GSL")
class TopK(tf.keras.layers.Layer):
  """Sparsifies a full parameter adjacency by returning the top `k` edges."""

  def __init__(
      self,
      number_of_nodes,
      k,
      **kwargs,
  ):
    super().__init__()
    self._number_of_nodes = number_of_nodes
    self._k = k

  def build(self, input_shape):
    """Checks the input shape."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    shape = tuple(input_shape.as_list())
    if shape and shape[0] is None:
      shape = shape[1:]
    if len(shape) != 2:
      raise ValueError(f"Expected tensor of rank 2 (got {len(shape)})")
    if None in shape:
      raise ValueError(f"Expected defined inner dimensions (got {shape})")
    if shape != (self._number_of_nodes, self._number_of_nodes):
      raise ValueError(
          f"Expected tensor of shape ({self._number_of_nodes},"
          f" {self._number_of_nodes}) (got {shape})"
      )

  def call(self, inputs):
    similarities = inputs
    vals, inds = tf.math.top_k(similarities, k=self._k + 1)
    weights = tf.reshape(vals, [-1])
    targets = tf.reshape(inds, [-1])
    sources = tf.repeat(tf.range(0, self._number_of_nodes), self._k + 1)
    return graph_structure.GraphStructure(sources, targets, weights)

  def get_config(self):
    return dict(
        number_of_nodes=self._number_of_nodes,
        k=self._k,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class TopKEpsilonNN(tf.keras.layers.Layer):
  """Sparsifies a full parameter adjacency by returning the top `k` edges."""

  def __init__(
      self,
      number_of_nodes,
      k,
      epsilon,
      **kwargs,
  ):
    super().__init__()
    self._number_of_nodes = number_of_nodes
    self._k = k
    self._epsilon = epsilon

  def build(self, input_shape):
    """Checks the input shape."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    shape = tuple(input_shape.as_list())
    if shape and shape[0] is None:
      shape = shape[1:]
    if len(shape) != 2:
      raise ValueError(f"Expected tensor of rank 2 (got {len(shape)})")
    if None in shape:
      raise ValueError(f"Expected defined inner dimensions (got {shape})")
    if shape != (self._number_of_nodes, self._number_of_nodes):
      raise ValueError(
          f"Expected tensor of shape ({self._number_of_nodes},"
          f" {self._number_of_nodes}) (got {shape})"
      )

  def call(self, inputs):
    similarities = inputs
    # kNN
    vals, inds = tf.math.top_k(similarities, k=self._k + 1)
    weights = tf.reshape(vals, [-1])
    targets = tf.reshape(inds, [-1])
    sources = tf.repeat(tf.range(0, self._number_of_nodes), self._k + 1)
    # epsilonNN
    indices = tf.where(weights > self._epsilon)
    sources = tf.gather_nd(sources, indices)
    targets = tf.gather_nd(targets, indices)
    weights = tf.gather_nd(weights, indices)
    return graph_structure.GraphStructure(sources, targets, weights)

  def get_config(self):
    return dict(
        number_of_nodes=self._number_of_nodes,
        k=self._k,
        epsilon=self._epsilon,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class EpsilonNN(tf.keras.layers.Layer):
  """Sparsifies an adjacency by returning edges with a score > epsilon."""

  def __init__(
      self,
      epsilon,
      **kwargs,
  ):
    super().__init__()
    self._epsilon = epsilon

  def build(self, input_shape):
    """Checks the input shape."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    shape = input_shape.as_list()
    if shape and shape[0] is None:
      shape = shape[1:]
    if len(shape) != 2:
      raise ValueError(f"Expected tensor of rank 2 (got {len(shape)})")
    if None in shape:
      raise ValueError(f"Expected defined inner dimensions (got {shape})")
    if shape[0] != shape[1]:
      raise ValueError(f"Expected square tensor (got {shape})")

  def call(self, inputs):
    similarities = inputs
    indices = tf.where(similarities > self._epsilon)
    # Source and target indices are the last two dimensions.
    sources, targets = indices[:, -2], indices[:, -1]
    weights = tf.gather_nd(similarities, indices)
    return graph_structure.GraphStructure(sources, targets, weights)

  def get_config(self):
    return dict(
        epsilon=self._epsilon,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class Bernoulli(tf.keras.layers.Layer):
  """Randomly samples the adjacency according to given probabilities."""

  def __init__(
      self,
      *,
      do_sigmoid,
      soft_version,
      epsilon,
      temperature,
      seed = None,
      **kwargs,
  ):
    super().__init__()
    self._do_sigmoid = do_sigmoid
    self._seed = seed
    self._soft_version = soft_version
    self._temperature = temperature
    self._epsilon = epsilon
    self._enn_layer = EpsilonNN(epsilon=epsilon)
    if self._seed:
      tf.random.set_seed(self._seed)

  def build(self, input_shape):
    """Checks the input shape."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    shape = input_shape.as_list()
    if shape and shape[0] is None:
      shape = shape[1:]
    if len(shape) != 2:
      raise ValueError(f"Expected tensor of rank 2 (got {len(shape)})")
    if None in shape:
      raise ValueError(f"Expected defined inner dimensions (got {shape})")
    if shape[0] != shape[1]:
      raise ValueError(f"Expected square tensor (got {shape})")

  def call(self, inputs):
    similarities = inputs
    if self._do_sigmoid:
      similarities = tf.sigmoid(similarities)
    else:
      # As suggested in the VIB-GSL paper.
      similarities = tf.clip_by_value(similarities, 0.01, 0.99)
    if self._soft_version:
      noise = tf.random.uniform(shape=[], maxval=1.0, seed=self._seed)
      similarities = tf.math.log(
          tf.math.divide_no_nan(similarities, 1 - similarities)
      ) + tf.math.log(tf.math.divide_no_nan(noise, 1 - noise))

      similarities = tf.sigmoid(
          tf.math.divide_no_nan(similarities, self._temperature)
      )
      return self._enn_layer(similarities)
    outputs = tf.floor(
        tf.random.uniform(similarities.shape, maxval=1.0, seed=self._seed)
        + similarities
    )
    indices = tf.where(outputs > 0)
    rows, cols = indices[:, -2], indices[:, -1]
    values = tf.gather_nd(similarities, indices)
    print(rows, cols, values)
    return graph_structure.GraphStructure(rows, cols, values)

  def get_config(self):
    return dict(
        do_sigmoid=self._do_sigmoid,
        seed=self._seed,
        soft_version=self._soft_version,
        temperature=self._temperature,
        epsilon=self._epsilon,
        **super().get_config(),
    )


@tf.keras.utils.register_keras_serializable(package="GSL")
class DilatedTopK(tf.keras.layers.Layer):
  """Sparsifies a full parameter adjacency with dilation.

  Returns top `k` edges that are dilated by `d`. See
  https://sites.google.com/corp/view/deep-gcns for reference.
  """

  def __init__(
      self,
      number_of_nodes,
      k,
      d,
      random_dilation = False,
      seed = None,
      **kwargs,
  ):
    super().__init__()
    self._number_of_nodes = number_of_nodes
    self._k = k
    self._d = d
    self._random_dilation = random_dilation
    self._seed = seed

  def build(self, input_shape):
    """Checks the input shape."""
    if not isinstance(input_shape, tf.TensorShape):
      raise ValueError(f"Expected `TensorShape` (got {type(input_shape)})")
    shape = tuple(input_shape.as_list())
    if shape and shape[0] is None:
      shape = shape[1:]
    if len(shape) != 2:
      raise ValueError(f"Expected tensor of rank 2 (got {len(shape)})")
    if None in shape:
      raise ValueError(f"Expected defined inner dimensions (got {shape})")
    if shape != (self._number_of_nodes, self._number_of_nodes):
      raise ValueError(
          f"Expected tensor of shape ({self._number_of_nodes},"
          f" {self._number_of_nodes}) (got {shape})"
      )

  def call(self, inputs):
    similarities = inputs
    top_k = self._k * self._d if self._k != 1 else self._k + 1
    vals, inds = tf.math.top_k(similarities, k=top_k)
    if self._random_dilation:
      possible_indices = tf.random.shuffle(tf.range(1, top_k), seed=self._seed)[
          : self._k
      ]
      dilation_indices = tf.concat([tf.constant([0]), possible_indices], 0)
    else:
      dilation_indices = tf.concat(
          [tf.constant([0]), tf.range(1, top_k, self._d)], 0
      )
    vals = tf.gather(vals, dilation_indices, axis=-1)
    inds = tf.gather(inds, dilation_indices, axis=-1)
    values = tf.reshape(vals, [-1])
    cols = tf.reshape(inds, [-1])
    rows = tf.repeat(tf.range(0, self._number_of_nodes), self._k + 1)
    return graph_structure.GraphStructure(rows, cols, values)

  def get_config(self):
    return dict(
        number_of_nodes=self._number_of_nodes,
        k=self._k,
        d=self._d,
        random_dilation=self._random_dilation,
        seed=self._seed,
        **super().get_config(),
    )


def get_sparsifier(
    number_of_nodes, name, **kwargs
):
  """Get the corresponding sparsifier given the name in the input.

  Args:
    number_of_nodes: number of nodes in the graph.
    name: name of the sparsifier.
    **kwargs: rest of arguments.

  Returns:
    The sparsifier to be used in the gsl framework.
  Raises:
    ValueError: if the sparsifier is not defined.
  """
  if name == "epsilon":
    return EpsilonNN(**kwargs)
  elif name == "knn":
    return TopK(number_of_nodes=number_of_nodes, **kwargs)
  elif name == "dilated-knn":
    return DilatedTopK(number_of_nodes=number_of_nodes, **kwargs)
  elif name == "bernoulli":
    return Bernoulli(**kwargs)
  elif name == "knn-epsilon":
    return TopKEpsilonNN(number_of_nodes=number_of_nodes, **kwargs)
  else:
    raise ValueError(f"Sparsifier {name} is not defined.")
