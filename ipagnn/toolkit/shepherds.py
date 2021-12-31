# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Helper classes for processing sparse data through the full data pipeline.

There are many cases where we want to maintain sparse mappings. Some examples:
- Mappings of nodes to graphs.
- Mappings of nodes to per-graph candidate groups.
- Mappings of output targets to input nodes that the output could have been
  copied from.
- Mapping of source nodes to dest nodes for message passing or attention.

For each of these mappings, we need a common set of operations:
- Create representations of the mappings to store in tf.Examples.
- Dynamically batch together multiple mappings.
- Render the mapping out in a convenient form for the TensorFlow operations that
  use the mapping to implement some function of interest.

The classes in this module are meant to encapsulate all of this functionality,
so that all the logic for processing the data is in one place, and to make it
easier to have problems with many sparse mappings.

--------------------
Creating tf.Examples
--------------------

At tf.Example creation time, we need to create these relations. For graph
problems, this would happen in the implementation of
_add_{graph,target}_features in /l/b/r/ps/toolkit/graph_problem.GraphInstance.
Example code might look like the following:

  def _add_graph_features(self, features):
    out_to_in_pairs = ...   # Problem-specific way of getting out-to-in pairs.
    shepherd = SparseTensorShepherd("out_to_in")
    features.update(shepherd.tf_example_features(out_to_in_pairs, dense_shape))


--------------------
Loading tf.Examples
--------------------

Next we'll need to load the data using the tf.data.Dataset API. For this, we
need metadata about the set and types of features that were added to the
tf.Example. This is implemented in `feature_descriptions`. We can then call

  tf.parse_single_example(serialized_example, shepherd.feature_loading_spec()))

to get back a dictionary of tensors from the tf.Examples.


----------------
Dynamic batching
----------------

We want to batch by adding as many graphs as we can to a batch until we hit
a maximum number of nodes or edges. We want this to be done in Tensorflow so
that it can be used in a tf.data pipeline, and  don't want to have to write this
logic separately for every combination of sparse mappings that we might want to
use in a model. See the `batching_loop_*` methods in this class and the
tf.while_loop in sparse_shepherds_test.py::test_dynamic_batching for a design
that let's us write a generic tf.while_loop for the dynamic batching while
plugging in whatever shepherd-specific logic we need for batching together the
sparse mappings.

The basic idea is that the shepherd takes ownership of a subset of loop vars
in the batching tf.while_loop, and it's also allowed to see tensors describing
the current instance. It then updates its loop vars in response. In this way,
several shepherds can update their state within a single loop without having to
change the Tensorflow code. See `test_dynamic_batching`.


-------------------
Using in TensorFlow
-------------------

Finally, after the data has been loaded, batched, and written to the batched
tensors dict, we can ask the shepherd for a sparse tensor representation of the
batched mapping using `get_tensor`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six
import tensorflow.compat.v1 as tf
from ipagnn.toolkit import problem

# Loop variables used in the shepherd-specific component of tf.while_loop
# that does dynamic batching.
# Attributes:
#   source_offset: An integer keeping track of the size of the source domain.
#   dest_offset: An integer keeping track of the size of the dest domain.
#   source_indices_ta: A tf.TensorArray where element i stores a variable-sized
#     tensor of source indices for batch element i.
#   dest_indices_ta: A tf.TensorArray where element i stores a variable-sized
#     tensor of dest indices for batch element i.
_SparseTensorLoopVars = (
    collections.namedtuple('_SparseTensorLoopVars',
                           ['source_offset',
                            'dest_offset',
                            'source_indices_ta',
                            'dest_indices_ta']))

_SparseTensorWithValuesLoopVars = (
    collections.namedtuple('_SparseTensorWithValuesLoopVars',
                           ['source_offset',
                            'dest_offset',
                            'source_indices_ta',
                            'dest_indices_ta',
                            'values_ta']))

_TENSORARRAY_INITIAL_SIZE = 0


def apply_mod_padding(tensor, mod_paddings):
  """Pads `tensor` such that its shape is 0 mod `mod_paddings`."""
  shape = tf.shape(tensor)
  def _amount_to_pad(dim, mod_padding):
    m = tf.mod(dim, mod_padding)
    return tf.cond(tf.math.equal(m, 0), lambda: 0, lambda: mod_padding - m)
  paddings = [
      (0, _amount_to_pad(shape[i], mod_padding))
      for i, mod_padding in enumerate(mod_paddings)
  ]
  return tf.pad(tensor, paddings)


class WithKeyRenaming(object):
  """Adapter class to rename the output key in the batching loop of a shepherd.

  This is useful if you want to do two different transformations on the
  same field in the tf.Example, and have the results preserved
  in different fields of the output superbatch.

  Currently, this adapter will only work for shepherds that output a single key,
  whose name is returned by the key() method.

  All methods of this class are delegated to the given shepherd,
  and the only method whose result is changed is `batching_loop_results`.
  """

  # If in the future we wish to support shepherds that have multiple keys,
  # this should be easy: (a) change FROM_KEY and TO_KEY to be a list,
  # and (b) Change the delegating methods so that all of the key methods
  # of self.shepherd are correctly delegated. Right now, we only delegate
  # a method called `key`.

  def __init__(self, shepherd, from_key, to_key):
    """Creates a new key-renaming shepherd.

    Args:
      shepherd: Shepherd to delegate to.
      from_key: Key that appears in the output of the shepherd batching loop.
      to_key: Key to rename from_key to.
    """
    self.shepherd = shepherd
    self.from_key = from_key
    self.to_key = to_key

  def batching_loop_results(self, loop_vars):
    """Returns the result of the batching loop for this shepherd.

    This will be the same as would have been returned by self.shepherd,
    except that if self.from_key appears, it will be renamed to self.to_key.

    Args:
      loop_vars: Loop vars to pass to the shepherd.
    """
    original_results = self.shepherd.batching_loop_results(loop_vars)
    results = dict()
    for old_key, result_value in six.iteritems(original_results):
      new_key = self.to_key if old_key == self.from_key else old_key
      results[new_key] = result_value
    return results

  # The below methods are all delegates to `self.shepherd`.

  def key(self):
    return self.shepherd.key()

  def feature_types(self):
    return self.shepherd.feature_types()

  def feature_shapes(self):
    return self.shepherd.feature_shapes()

  def feature_loading_spec(self):
    return self.shepherd.feature_types()

  def batching_loop_initial_values(self):
    return self.shepherd.batching_loop_initial_values()

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    return self.shepherd.batching_loop_body(i, tensor_dict, loop_vars)

  def get_tensor(self):
    return self.shepherd.get()


class DenseTensorShepherd(object):
  """Shepherds dense data through generation, saving, loading, and batching.

  Tensors from different instances can vary in shape in the first dimension,
  which is assumed to be the "batch" dimension. The shape in the other
  dimensions must be constant across instances but can otherwise be arbitrary.
  """

  def __init__(self, name, dtype, element_shape=None, expand_dims=None,
               mod_paddings=None):
    """Creates a shepherd for dense tensor data.

    Args:
      name: String name to be used for loading data from tf.Examples.
      dtype: TensorFlow type of the tensor (e.g., tf.float32).
      element_shape: The shape of each element. Can be not fully defined, in
        which case VarLenFeatures will be used.
      expand_dims: If set, expand these dims on the result of the shepherd.
      mod_paddings: If set, pad so dimensions are 0 mod these values.
    """
    self._name = name
    self._element_shape = tf.TensorShape(element_shape)
    self._dtype = dtype
    self._expand_dims = expand_dims
    self._mod_paddings = mod_paddings

  def key(self):
    return self._name

  def feature_types(self):
    """Returns a dictionary mapping key names to Tensorflow types."""
    return {
        self.key(): self._dtype
    }

  def feature_shapes(self):
    """Returns a dictionary mapping key names to Tensorflow shapes."""
    return {
        self.key(): self._element_shape
    }

  def feature_loading_spec(self):
    """Returns a dictionary with information needed to deserialize tf.Examples.

    Returns:
      A dictionary mapping feature names to example-reading specs.
    """
    if self._element_shape.is_fully_defined():
      feature_description = tf.FixedLenFeature(self._element_shape, self._dtype)
    else:
      feature_description = tf.VarLenFeature(self._dtype)

    return {
        self.key(): feature_description
    }

  def batching_loop_initial_values(self):
    """Initial values for all loop variables this shepherd is responsible for.

    This is method 1 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called before the start of the
    tf.while_loop that implements the batching.

    Returns:
      A list of TensorFlow Variables defining the tf.while_loop state managed
      by the shepherd.
    """
    tensor_array = tf.TensorArray(self._dtype,
                                  size=_TENSORARRAY_INITIAL_SIZE,
                                  dynamic_size=True,
                                  element_shape=None,
                                  infer_shape=False)
    return [tensor_array]

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Adds an instance to the cumulative state that the shepherd manages.

    This is method 2 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called during the body of the tf.while_loop
    that implements the batching.

    Args:
      i: A tensor-valued loop counter keeping track of how many instances we've
        already added to the batch.
      tensor_dict: A dictionary mapping keys to Tensors representing all the
        data for the current instance. The shepherd accesses the fields it needs
        by getting the fields of `tensor_dict` with keys determined by the
        shepherd's name.
      loop_vars: A list of TensorFlow Variables that have accumulated previous
        batch instances, but not the current one yet.

    Returns:
      A list of Tensor-valued loop variables that have incorporated previous
      batch instances and the instance represented by `tensor_dict`.
    """
    tensor_array = loop_vars[0]
    return [
        tensor_array.write(i, tf.cast(tensor_dict[self.key()], self._dtype))
    ]

  def batching_loop_results(self, loop_vars):
    """Converts the final loop variables into a batched representation.

    This is method 3 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called after termination of the
    tf.while_loop that implements the batching.

    Args:
      loop_vars: A list of TensorFlow Variables that have accumulated all
        batch instances.

    Returns:
      A dictionary mapping the shepherd's key to a final batched representation
      of all the instances that were added to the batch. This will be merged
      with dictionaries from other shepherds to build up a dictionary
      representing batched versions of all the data in the instances.
    """
    tensor_array = loop_vars[0]
    if self._element_shape.rank == 0:
      # Scalars should be stacked.
      result = tensor_array.stack(name=self._name + '/results_stack')
    else:
      result = tensor_array.concat(name=self._name + '/results_concat')
      result.set_shape(self._element_shape)

    if self._expand_dims is not None:
      result = tf.expand_dims(result, axis=self._expand_dims)
    if self._mod_paddings is not None:
      result = apply_mod_padding(result, self._mod_paddings)

    return {
        self.key(): result
    }

  def get_tensor(self, tensor_dict):
    return tensor_dict[self.key()]


class SparseTensorShepherd(object):
  """Shepherds sparse data through generation, batching, and loading."""

  def __init__(self, name):
    """Creates a shepherd.

    Args:
      name: String name to be used as a prefix for tf.Example keys.
    """
    self._name = name

  def source_indices_key(self):
    return '{}/source_indices'.format(self._name)

  def dest_indices_key(self):
    return '{}/dest_indices'.format(self._name)

  def dense_shape_key(self):
    return '{}/dense_shape'.format(self._name)

  def feature_types(self):
    """Returns a dictionary mapping key names to Tensorflow types."""
    return {
        self.dense_shape_key(): tf.int64,
        self.source_indices_key(): tf.int64,
        self.dest_indices_key(): tf.int64,
    }

  def feature_loading_spec(self):
    """Returns a dictionary with information needed to deserialize tf.Examples.

    Returns:
      A dictionary mapping feature names to example-reading specs.
    """
    return {
        # TODO(dtarlow): Make dense_shape a tf.FixedLenFeature after figuring
        # out the right way to load these (without tf.sparse_tensor_to_dense) in
        # shepherds_test._make_test_dataset.
        feature_key: tf.VarLenFeature(feature_type)
        for feature_key, feature_type in six.iteritems(self.feature_types())
    }

  def feature_shapes(self):
    """Returns a dictionary mapping key names to Tensorflow shapes."""

    return {
        # A pair of elements defining the dense shape of the sparse tensor.
        self.dense_shape_key(): tf.TensorShape([2]),
        # 1D lists of indices of unknown length.
        self.source_indices_key(): tf.TensorShape([None]),
        self.dest_indices_key(): tf.TensorShape([None]),
    }

  def tf_example_features(self, pairs, dense_shape):
    """Converts a list of pairs to tf.Example features.

    This is called during tf.Example creation time and doesn't need to use
    Tensorflow ops.

    Args:
      pairs: A list of (i, j) pairs such that the relation holds from i to j.
      dense_shape: A pair of integers with the dense shape of the SparseTensor.

    Returns:
      A dictionary mapping features names to tf.Features. This will be modified
      in-place to have tf.Features representing the relation.
    """
    if pairs:
      source_indices, dest_indices = zip(*pairs)
    else:
      source_indices, dest_indices = [], []
    return {
        self.dense_shape_key(): problem.int_feature(dense_shape),
        self.source_indices_key(): problem.int_feature(source_indices),
        self.dest_indices_key(): problem.int_feature(dest_indices)
    }

  def batching_loop_initial_values(self):
    """Returns a list of initial values for the batching tf.while_loop.

    The return value here defines the shepherd-specific loop vars for a
    tf.while_loop that does dynamic batching. Updated values for these loop vars
    will get passed in to `batching_loop_body` and `batching_loop_results`.

    Returns:
      A possibly nested list of tensors.
    """
    source_offset = tf.constant(0, dtype=tf.int64)
    dest_offset = tf.constant(0, dtype=tf.int64)
    source_indices_ta = tf.TensorArray(tf.int64,
                                       size=_TENSORARRAY_INITIAL_SIZE,
                                       dynamic_size=True,
                                       element_shape=None,
                                       infer_shape=False)
    dest_indices_ta = tf.TensorArray(tf.int64,
                                     size=_TENSORARRAY_INITIAL_SIZE,
                                     dynamic_size=True,
                                     element_shape=None,
                                     infer_shape=False)
    return _SparseTensorLoopVars(source_offset, dest_offset,
                                 source_indices_ta, dest_indices_ta)

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Shepherd-specific body function for a tf.while_loop.

    Args:
      i: The index into the batch of the current instance.
      tensor_dict: A dictionary mapping tensor names to tensors for the current
        instance.
      loop_vars: A list of shepherd-specific loop vars. Must correspond to those
        returned by `batching_loop_initial_values`.

    Returns:
      A list of `loop_vars` updated to have added the current instance to the
      batch.
    """
    source_offset, dest_offset, source_indices_ta, dest_indices_ta = loop_vars

    dense_shape = tensor_dict[self.dense_shape_key()]
    source_indices = tensor_dict[self.source_indices_key()] + source_offset
    dest_indices = tensor_dict[self.dest_indices_key()] + dest_offset

    source_indices_ta = source_indices_ta.write(i, source_indices)
    dest_indices_ta = dest_indices_ta.write(i, dest_indices)
    source_offset += dense_shape[0]
    dest_offset += dense_shape[1]

    return _SparseTensorLoopVars(source_offset, dest_offset,
                                 source_indices_ta, dest_indices_ta)

  def batching_loop_results(self, loop_vars):
    """Convert the final shepherd-specific loop vars to batched data.

    Args:
      loop_vars: A list of shepherd-specific loop vars final values when the
        tf.while_loop for batching data finishes.

    Returns:
      A dictionary mapping feature names to tensors of batched values.
    """
    source_offset, dest_offset, source_indices_ta, dest_indices_ta = loop_vars

    return {
        self.dense_shape_key(): tf.stack([source_offset, dest_offset]),
        # Has tf.TensorShape([source_offset]).
        self.source_indices_key(): source_indices_ta.concat(
            name=(self.source_indices_key() + '/results_concat')),
        # Has tf.TensorShape([dest_offset]).
        self.dest_indices_key(): dest_indices_ta.concat(
            name=(self.dest_indices_key() + '/results_concat')),
    }

  def get_indices(self, tensor_dict):
    source_indices = tensor_dict[self.source_indices_key()]
    dest_indices = tensor_dict[self.dest_indices_key()]

    # 'indices' is a [source_indices.shape[0], 2] with each row representing
    # a (row, column) pair where there is a nonzero entry in the SparseTensor.
    indices = tf.concat([tf.expand_dims(source_indices, 1),
                         tf.expand_dims(dest_indices, 1)], axis=1)
    return indices

  def get_tensor(self, tensor_dict):
    """Construct a SparseTensor representation of the relation from tensors.

    This can either be called with a decoded tf.Example, in which case this
    returns a sparse tensor for an individual instance, or with the result
    of `batching_loop_results`, in which case it returns a sparse tensor for
    the whole batch.

    Args:
      tensor_dict: A dictionary mapping key names to tensors.

    Returns:
      A SparseTensor representation of the shepherd's data.
    """
    dense_shape = tensor_dict[self.dense_shape_key()]
    source_indices = tensor_dict[self.source_indices_key()]
    dest_indices = tensor_dict[self.dest_indices_key()]

    # 'indices' is a [source_indices.shape[0], 2] with each row representing
    # a (row, column) pair where there is a nonzero entry in the SparseTensor.
    indices = tf.concat([tf.expand_dims(source_indices, 1),
                         tf.expand_dims(dest_indices, 1)], axis=1)
    values = tf.ones_like(source_indices, dtype=tf.float32)

    return tf.sparse_reorder(
        tf.SparseTensor(
            indices=indices, values=values, dense_shape=dense_shape))


class SparseTensorWithValuesShepherd(SparseTensorShepherd):
  """SparseTensorShepherd that supports values as well.

  TODO(dtarlow): Consider renaming this to SparseTensorShepherd and
  SparseTensorShepherd to SparseIndicatorShepherd or something similar, since
  it would be natural to expect that SparseTensors have values.
  """

  def values_key(self):
    return '{}/values'.format(self._name)

  def feature_types(self):
    """Returns a dictionary mapping key names to Tensorflow types."""
    feature_types = super(SparseTensorWithValuesShepherd, self).feature_types()
    feature_types[self.values_key()] = tf.float32
    return feature_types

  def feature_shapes(self):
    """Returns a dictionary mapping key names to Tensorflow shapes."""
    shapes = super(SparseTensorWithValuesShepherd, self).feature_shapes()
    shapes[self.values_key()] = tf.TensorShape([None])
    return shapes

  def tf_example_features(self, pairs, values, dense_shape):
    """Converts a list of pairs to tf.Example features.

    This is called during tf.Example creation time and doesn't need to use
    Tensorflow ops.

    Args:
      pairs: A list of (i, j) pairs such that the relation holds from i to j.
      values: A list of the same length as `pairs` with values of the sparse
        tensor at the index specified by the pair.
      dense_shape: A pair of integers with the dense shape of the SparseTensor.

    Returns:
      A dictionary mapping features names to tf.Features. This will be modified
      in-place to have tf.Features representing the relation.
    """
    if len(pairs) != len(values):
      raise AssertionError(
          'Length of `pairs`: {} not equal to `values`: {}'.format(
              len(pairs), len(values)))
    features = (
        super(SparseTensorWithValuesShepherd, self).tf_example_features(
            pairs, dense_shape))

    features[self.values_key()] = problem.float32_feature(values)
    return features

  def batching_loop_initial_values(self):
    """Returns a list of initial values for the batching tf.while_loop.

    The return value here defines the shepherd-specific loop vars for a
    tf.while_loop that does dynamic batching. Updated values for these loop vars
    will get passed in to `batching_loop_body` and `batching_loop_results`.

    Returns:
      A possibly nested list of tensors.
    """
    sparse_tensor_vars = (super(SparseTensorWithValuesShepherd, self)
                          .batching_loop_initial_values())
    values_ta = tf.TensorArray(self.feature_types()[self.values_key()],
                               size=_TENSORARRAY_INITIAL_SIZE,
                               dynamic_size=True,
                               element_shape=None,
                               infer_shape=False)
    return _SparseTensorWithValuesLoopVars(sparse_tensor_vars.source_offset,
                                           sparse_tensor_vars.dest_offset,
                                           sparse_tensor_vars.source_indices_ta,
                                           sparse_tensor_vars.dest_indices_ta,
                                           values_ta)

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Shepherd-specific body function for a tf.while_loop.

    Args:
      i: The index into the batch of the current instance.
      tensor_dict: A dictionary mapping tensor names to tensors for the current
        instance.
      loop_vars: A list of shepherd-specific loop vars. Must correspond to those
        returned by `batching_loop_initial_values`.

    Returns:
      A list of `loop_vars` updated to have added the current instance to the
      batch.
    """
    sparse_tensor_vars = _SparseTensorLoopVars(loop_vars.source_offset,
                                               loop_vars.dest_offset,
                                               loop_vars.source_indices_ta,
                                               loop_vars.dest_indices_ta)
    super_result_vars = (
        super(SparseTensorWithValuesShepherd, self)
        .batching_loop_body(i, tensor_dict, sparse_tensor_vars))
    result_values_ta = (
        loop_vars.values_ta.write(i, tensor_dict[self.values_key()]))

    return _SparseTensorWithValuesLoopVars(super_result_vars.source_offset,
                                           super_result_vars.dest_offset,
                                           super_result_vars.source_indices_ta,
                                           super_result_vars.dest_indices_ta,
                                           result_values_ta)

  def batching_loop_results(self, loop_vars):
    """Convert the final shepherd-specific loop vars to batched data.

    Args:
      loop_vars: A list of shepherd-specific loop vars final values when the
        tf.while_loop for batching data finishes.

    Returns:
      A dictionary mapping feature names to tensors of batched values.
    """
    super_loop_vars = _SparseTensorLoopVars(loop_vars.source_offset,
                                            loop_vars.dest_offset,
                                            loop_vars.source_indices_ta,
                                            loop_vars.dest_indices_ta)

    super_results = (super(SparseTensorWithValuesShepherd, self)
                     .batching_loop_results(super_loop_vars))
    super_results[self.values_key()] = loop_vars.values_ta.concat()
    return super_results

  def get_tensor(self, tensor_dict):
    """Construct a SparseTensor representation of the relation from tensors.

    This can either be called with a decoded tf.Example, in which case this
    returns a sparse tensor for an individual instance, or with the result
    of `batching_loop_results`, in which case it returns a sparse tensor for
    the whole batch.

    Args:
      tensor_dict: A dictionary mapping key names to tensors.

    Returns:
      A SparseTensor representation of the shepherd's data.
    """
    dense_shape = tensor_dict[self.dense_shape_key()]
    source_indices = tensor_dict[self.source_indices_key()]
    dest_indices = tensor_dict[self.dest_indices_key()]
    values = tensor_dict[self.values_key()]

    # 'indices' is a [source_indices.shape[0], 2] with each row representing
    # a (row, column) pair where there is a nonzero entry in the SparseTensor.
    indices = tf.concat([tf.expand_dims(source_indices, 1),
                         tf.expand_dims(dest_indices, 1)], axis=1)

    return tf.sparse_reorder(
        tf.SparseTensor(indices=indices,
                        values=values,
                        dense_shape=dense_shape))


class NodeIndexShepherd(object):
  """Shepherds a single node index per instance through the data pipeline.

  TODO(dtarlow): This is actually a more general shepherd than just one to
  handle node indices. Generalize it appropriately.
  """

  def __init__(self, name, node_count_key, dtype):
    """Creates a shepherd for data containing a node index.

    Args:
      name: String name to be used as a prefix for tf.Example keys.
      node_count_key: Name of node count field, which is assumed to be present
        in the per-instance tf.Examples.
      dtype: Type of node data.
    """
    self._name = name
    self._node_count_key = node_count_key
    self._dtype = dtype
    self._shape = tf.TensorShape([1])

  def key(self):
    return self._name

  def feature_types(self):
    """Returns a dictionary mapping key names to Tensorflow types."""
    return {
        self.key(): self._dtype
    }

  def feature_loading_spec(self):
    """Returns a dictionary with information needed to deserialize tf.Examples.

    Returns:
      A dictionary mapping feature names to example-reading specs.
    """
    return {
        self.key(): tf.FixedLenFeature(self._shape, self._dtype)
    }

  def batching_loop_initial_values(self):
    """Initial values for all loop variables this shepherd is responsible for.

    This is method 1 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called before the start of the
    tf.while_loop that implements the batching.

    Returns:
      A list of TensorFlow Variables defining the tf.while_loop state managed
      by the shepherd.
    """
    tensor_array = tf.TensorArray(self._dtype,
                                  size=0,
                                  dynamic_size=True,
                                  element_shape=None,
                                  infer_shape=False)
    # Total number of nodes in the batch so far.
    batch_num_nodes = tf.constant(0, dtype=tf.int64)
    return [batch_num_nodes, tensor_array]

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Adds an instance to the cumulative state that the shepherd manages.

    This is method 2 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called during the body of the tf.while_loop
    that implements the batching.

    Args:
      i: A tensor-valued loop counter keeping track of how many instances we've
        already added to the batch.
      tensor_dict: A dictionary mapping keys to Tensors representing all the
        data for the current instance. The shepherd accesses the fields it needs
        by getting the fields of `tensor_dict` with keys determined by the
        shepherd's name.
      loop_vars: A list of TensorFlow Variables that have accumulated previous
        batch instances, but not the current one yet.

    Returns:
      A list of Tensor-valued loop variables that have incorporated previous
      batch instances and the instance represented by `tensor_dict`.
    """
    batch_num_nodes, tensor_array = loop_vars
    local_node_indices = tensor_dict[self.key()]

    batch_node_indices = local_node_indices + tf.cast(batch_num_nodes,
                                                      self._dtype)
    batch_num_nodes += tf.cast(tensor_dict[self._node_count_key], tf.int64)

    return [batch_num_nodes, tensor_array.write(i, batch_node_indices)]

  def batching_loop_results(self, loop_vars):
    """Converts the final loop variables into a batched representation.

    This is method 3 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called after termination of the
    tf.while_loop that implements the batching.

    Args:
      loop_vars: A list of TensorFlow Variables that have accumulated all
        batch instances.

    Returns:
      A dictionary mapping the shepherd's key to a final batched representation
      of all the instances that were added to the batch. This will be merged
      with dictionaries from other shepherds to build up a dictionary
      representing batched versions of all the data in the instances.
    """
    tensor_array = loop_vars[1]
    result = tensor_array.stack()
    return {
        self.key(): result
    }

  def get_tensor(self, tensor_dict):
    return tensor_dict[self.key()]

  def get_mask(self, tensor_dict, node_count=None):
    if node_count is None:
      node_count = tensor_dict[self._node_count_key]

    return tf.reduce_sum(
        tf.one_hot(self.get_tensor(tensor_dict), depth=node_count),
        axis=0)


class NodeIndicesShepherd(NodeIndexShepherd):
  """Shepherds node indices through the data pipeline.

  TODO(dtarlow): This is actually a more general shepherd than just one to
  handle node indices. Generalize it appropriately.
  """

  def __init__(self, name, node_count_key, dtype, shape, mod_paddings=None):
    """Creates a shepherd for data containing node indices.

    Args:
      name: String name to be used as a prefix for tf.Example keys.
      node_count_key: Name of node count field, which is assumed to be present
        in the per-instance tf.Examples.
      dtype: Type of node data.
      shape: The shape.
      mod_paddings: If set, pad so dimensions are 0 mod these values.
    """
    super(NodeIndicesShepherd, self).__init__(name, node_count_key, dtype)
    self._shape = tf.TensorShape(shape)
    self._mod_paddings = mod_paddings

  def feature_loading_spec(self):
    """Returns a dictionary with information needed to deserialize tf.Examples.

    Returns:
      A dictionary mapping feature names to example-reading specs.
    """
    return {
        self.key(): tf.VarLenFeature(self._dtype)
    }

  def batching_loop_results(self, loop_vars):
    """Converts the final loop variables into a batched representation.

    This is method 3 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called after termination of the
    tf.while_loop that implements the batching.

    Args:
      loop_vars: A list of TensorFlow Variables that have accumulated all
        batch instances.

    Returns:
      A dictionary mapping the shepherd's key to a final batched representation
      of all the instances that were added to the batch. This will be merged
      with dictionaries from other shepherds to build up a dictionary
      representing batched versions of all the data in the instances.
    """
    tensor_array = loop_vars[1]
    result = tensor_array.concat()
    if self._mod_paddings is not None:
      result = apply_mod_padding(result, self._mod_paddings)
    return {
        self.key(): result
    }


class BatchHotNodeIndexShepherd(NodeIndexShepherd):
  """Shepherds a single node index per instance through the data pipeline.

  The final result is similar to a multi-hot encoding of the indices, but
  rather than having 1's in positions indicating the index, this shepherd will
  put in the id of the instance plus one. E.g., if the first instance has
  4 nodes and has index 2 and the second instance has 3 nodes and index 0, then
  the "batch-hot" encoding of the data would be [0, 0, 1, 0, 2, 0, 0].
  """

  def batching_loop_initial_values(self):
    """Initial values for all loop variables this shepherd is responsible for.

    This is method 1 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called before the start of the
    tf.while_loop that implements the batching.

    Returns:
      A list of TensorFlow Variables defining the tf.while_loop state managed
      by the shepherd.
    """
    tensor_array = tf.TensorArray(self._dtype,
                                  size=0,
                                  dynamic_size=True,
                                  element_shape=None,
                                  infer_shape=False)
    return [tensor_array]

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Adds an instance to the cumulative state that the shepherd manages.

    This is method 2 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called during the body of the tf.while_loop
    that implements the batching.

    Args:
      i: A tensor-valued loop counter keeping track of how many instances we've
        already added to the batch.
      tensor_dict: A dictionary mapping keys to Tensors representing all the
        data for the current instance. The shepherd accesses the fields it needs
        by getting the fields of `tensor_dict` with keys determined by the
        shepherd's name.
      loop_vars: A list of TensorFlow Variables that have accumulated previous
        batch instances, but not the current one yet.

    Returns:
      A list of Tensor-valued loop variables that have incorporated previous
      batch instances and the instance represented by `tensor_dict`.
    """
    tensor_array = loop_vars[0]
    local_node_indices = tensor_dict[self.key()]
    local_num_nodes = tf.cast(tensor_dict[self._node_count_key], dtype=tf.int32)

    on_value = tf.cast(i + 1, dtype=self._dtype)
    off_value = tf.cast(0, dtype=self._dtype)
    batch_hot = tf.one_hot(local_node_indices,
                           on_value=on_value,
                           off_value=off_value,
                           depth=local_num_nodes,
                           dtype=self._dtype)

    return [tensor_array.write(i, batch_hot)]

  def batching_loop_results(self, loop_vars):
    """Converts the final loop variables into a batched representation.

    This is method 3 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called after termination of the
    tf.while_loop that implements the batching.

    Args:
      loop_vars: A list of TensorFlow Variables that have accumulated all
        batch instances.

    Returns:
      A dictionary mapping the shepherd's key to a final batched representation
      of all the instances that were added to the batch. This will be merged
      with dictionaries from other shepherds to build up a dictionary
      representing batched versions of all the data in the instances.
    """
    tensor_array = loop_vars[0]
    result = tensor_array.concat()
    return {
        self.key(): result
    }

  def get_tensor(self, tensor_dict):
    return tensor_dict[self.key()]

  def get_mask(self, tensor_dict, node_count=None):
    tensor = self.get_tensor(tensor_dict)
    if node_count is not None:
      padding = [[0, node_count - tf.shape(tensor)[0]]]
      tensor = tf.pad(tensor, padding)
    return tf.cast(tensor, tf.float32)


class BatchHotNodeIndicesShepherd(NodeIndicesShepherd):
  """Shepherds indices through the data pipeline.

  Rather than producing a 1-hot mask at the end, indicating nonzero indices
  like in NodeIndicesShepherd, we replace the 1's with (one plus) the identity
  of the graph that the candidate is from. See `BatchHotNodeIndicesShepherd`
  docstring for an example. This shepherd is similar but allows for more than
  one index per instance to be active.
  """

  def batching_loop_initial_values(self):
    """Initial values for all loop variables this shepherd is responsible for.

    This is method 1 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called before the start of the
    tf.while_loop that implements the batching.

    Returns:
      A list of TensorFlow Variables defining the tf.while_loop state managed
      by the shepherd.
    """
    tensor_array = tf.TensorArray(self._dtype,
                                  size=0,
                                  dynamic_size=True,
                                  element_shape=None,
                                  infer_shape=False)
    return [tensor_array]

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Adds an instance to the cumulative state that the shepherd manages.

    This is method 2 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called during the body of the tf.while_loop
    that implements the batching.

    Args:
      i: A tensor-valued loop counter keeping track of how many instances we've
        already added to the batch.
      tensor_dict: A dictionary mapping keys to Tensors representing all the
        data for the current instance. The shepherd accesses the fields it needs
        by getting the fields of `tensor_dict` with keys determined by the
        shepherd's name.
      loop_vars: A list of TensorFlow Variables that have accumulated previous
        batch instances, but not the current one yet.

    Returns:
      A list of Tensor-valued loop variables that have incorporated previous
      batch instances and the instance represented by `tensor_dict`.
    """
    tensor_array = loop_vars[0]
    local_node_indices = tensor_dict[self.key()]
    local_num_nodes = tf.cast(tensor_dict[self._node_count_key], dtype=tf.int32)

    # Give all the "one-hot" entries for the indices the value of batch
    # index + 1. This will allow us to encode candidates and the graph that
    # they came from in the final "batch-hot" vectors.
    on_value = tf.cast(i + 1, dtype=self._dtype)
    off_value = tf.cast(0, dtype=self._dtype)
    multi_hot = tf.reduce_sum(tf.one_hot(local_node_indices,
                                         on_value=on_value,
                                         off_value=off_value,
                                         depth=local_num_nodes,
                                         dtype=self._dtype),
                              axis=0)

    return [tensor_array.write(i, multi_hot)]

  def batching_loop_results(self, loop_vars):
    """Converts the final loop variables into a batched representation.

    This is method 3 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called after termination of the
    tf.while_loop that implements the batching.

    Args:
      loop_vars: A list of TensorFlow Variables that have accumulated all
        batch instances.

    Returns:
      A dictionary mapping the shepherd's key to a final batched representation
      of all the instances that were added to the batch. This will be merged
      with dictionaries from other shepherds to build up a dictionary
      representing batched versions of all the data in the instances.
    """
    tensor_array = loop_vars[0]
    result = tensor_array.concat()
    return {
        self.key(): result
    }

  def get_tensor(self, tensor_dict):
    return tensor_dict[self.key()]

  def get_mask(self, tensor_dict, node_count=None):
    tensor = self.get_tensor(tensor_dict)
    if node_count is not None:
      padding = [[0, node_count - tf.shape(tensor)[0]]]
      tensor = tf.pad(tensor, padding)
    return tf.cast(tensor, tf.float32)


class StringShepherd(object):
  """A shepherd for data of type tf.string.

  Generally we convert string data to integer tensors or embeddings for
  passing to tf, so this is probably most useful for metadata.
  """

  def __init__(self, name):
    """Creates a shepherd for dense tensor data.

    Args:
      name: String name to be used as a prefix for tf.Example keys.
    """
    self._name = name

  def key(self):
    return self._name

  def feature_types(self):
    """Returns a dictionary mapping key names to Tensorflow types."""
    return {
        self.key(): tf.string,
    }

  def feature_shapes(self):
    """Returns a dictionary mapping key names to Tensorflow shapes."""
    return {
        self.key(): [None],
    }

  def feature_loading_spec(self):
    """Returns a dictionary with information needed to deserialize tf.Examples.

    Returns:
      A dictionary mapping feature names to example-reading specs.
    """
    return {
        self.key(): tf.FixedLenFeature([], tf.string)
    }

  def batching_loop_initial_values(self):
    """Initial values for all loop variables this shepherd is responsible for.

    This is method 1 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called before the start of the
    tf.while_loop that implements the batching.

    Returns:
      A list of TensorFlow Variables defining the tf.while_loop state managed
      by the shepherd.
    """
    # pylint: disable=protected-access
    tensor_array = tf.TensorArray(tf.string,
                                  size=_TENSORARRAY_INITIAL_SIZE,
                                  dynamic_size=True,
                                  element_shape=None,
                                  infer_shape=False)
    return [tensor_array]

  def batching_loop_body(self, i, tensor_dict, loop_vars):
    """Adds an instance to the cumulative state that the shepherd manages.

    This is method 2 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called during the body of the tf.while_loop
    that implements the batching.

    Args:
      i: A tensor-valued loop counter keeping track of how many instances we've
        already added to the batch.
      tensor_dict: A dictionary mapping keys to Tensors representing all the
        data for the current instance. The shepherd accesses the fields it needs
        by getting the fields of `tensor_dict` with keys determined by the
        shepherd's name.
      loop_vars: A list of TensorFlow Variables that have accumulated previous
        batch instances, but not the current one yet.

    Returns:
      A list of Tensor-valued loop variables that have incorporated previous
      batch instances and the instance represented by `tensor_dict`.
    """
    tensor_array = loop_vars[0]
    return [tensor_array.write(i, tensor_dict[self.key()])]

  def batching_loop_results(self, loop_vars):
    """Converts the final loop variables into a batched representation.

    This is method 3 of 3 that shepherds need to implement in order to be able
    to dynamically batch data. It is called after termination of the
    tf.while_loop that implements the batching.

    Args:
      loop_vars: A list of TensorFlow Variables that have accumulated all
        batch instances.

    Returns:
      A dictionary mapping the shepherd's key to a final batched representation
      of all the instances that were added to the batch. This will be merged
      with dictionaries from other shepherds to build up a dictionary
      representing batched versions of all the data in the instances.
    """
    tensor_array = loop_vars[0]
    return {
        self.key(): tensor_array.stack()
    }

  def get_tensor(self, tensor_dict):
    return tensor_dict[self.key()]
