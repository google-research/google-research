# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utilities for improving explainability of neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf


def AddIntegratedGradientsOps(graph,
                              attribution_tensors,
                              output_tensor,
                              num_evals,
                              attribution_dims_map,
                              zero_baseline_tensors=None,
                              new_output_scope='attribution',
                              baseline_scope='baseline',
                              tensors_to_keep=None):
  """Modify graph to create ops for computing integrated gradients.

  Function to modify a tensorflow graph by adding ops for attributing the change
  in value of a given output tensor, to different input 'attribution_tensors'
  (see arxiv.org/abs/1703.01365).

  The first dimension of each attribution_tensor and output_tensor is assumed
  to be the batch dimension. That is, if we create multiple input values for the
  attribution tensors, we should be able to concatenate them along the first
  dimension, and the resulting output tensor should have corresponding values
  for different values of its first dimension.

  The attribution works by interpolating between a given input, and a given
  baseline, to create multiple (num_evals) interpolated inputs. At each
  interpolated input, we compute the gradient of the output tensor with respect
  to each attribution tensor. The gradients for each attribution tensor are
  averaged over all interpolated inputs, to get an attribution score for it.

  Example Usage: attribution_feed_dict = AddIntegratedGradientsOps(...)
  Then to get attribution for a given input (specificed by input_feed_dict,
  relative to a baseline given be baseline_feed_dict):
  combined_feed_dict = attribution_feed_dict['create_combined_feed_dict'](
      input_feed_dict, baseline_feed_dict)
  with graph.as_default(), sess.as_default():
    attributions = sess.run(
        attribution_feed_dict['mean_grads'], combined_feed_dict)
  for tensor, attribution in zip(attribution_tensors, attributions):
    print('Attribution for %s: %s' % (tensor.op.name, attribution))

  Warning: This function is not compatible with tf.cond. If there is a tf.cond
  in the graph path between the attribution tensors and the output tensor, the
  attribution ops may not work.
  # TODO(manasrj): Make attribution ops compatible with tf.cond.

  Args:
    graph: The tf.Graph to add attribution ops to.
    attribution_tensors: Tensors for which to compute attribution scores. The
      tensors must satisfy two properties: (1) The output tensor must
      be computable given values for attribution tensors. (2) Each
      attribution tensor must be computationally independent of the
      others, i.e., it should not be the case that one of the
      attribution tensor's value is completely determined by the
      values of the other attribution tensors. Properties (1) and (2) ensure
      the attribution tensors form an input-output cut in the computation
      graph.
    output_tensor: Tensor for whose value we are performing attribution.
    num_evals: Integer scalar. Number of interpolated points at which to
      evaluate gradients. Higher values of this parameter increase computation
      time, but also increase accuracy of attributions.
    attribution_dims_map: Dict mapping attribution tensors to lists of integers.
      For each attribution_tensor, we compute a separate gradient value for each
      slice along the dims in the list. For example, if we have a rank 3
      attribution tensor T that consists of embeddings lookups, with the first
      dimension being the batch dimension, and the second dimension being the
      sparse ids, then setting attribution_dims_map[T] = [1] will give us a
      separate gradient for each sparse id. If an attribution_tensor has no
      entry in attribution_dims_map, then the list defaults to [].
    zero_baseline_tensors: Set of attribution tensors. For each tensor T in this
      set, we compute gradients with respect to T for all interpolated values of
      T between the value computed from the input feed, and zero. For each
      tensor U not in zero_baseline_tensors, we compute gradients for
      interpolated values between the one derived from the input feed, and the
      one derived from the baseline feed.
    new_output_scope: String. New ops needed for computing the output tensor at
      different interpolated values are created under this scope name.
    baseline_scope: String. New ops needed for computing attribution tensor
      interpolated values are created under this scope name.
    tensors_to_keep: Set of tensors. By default, tensors in the graph between
      the output_tensor and attribution tensors are copied to a different part
      of the graph, and evaluated separately for each interpolation. If we want
      a value to be fixed (only computed for the main input instead of each
      interpolation), it should be put in tensors_to_keep.

  Returns:
    attribution_hooks: Dict with the following keys (among others):
      mean_grads: List of attribution scores (aligned with attribution_tensors).
      create_combined_feed_dict: A Function that takes an input feed dict, and
        optionally, a baseline feed dict, and creates a combined feed dict to
        pass to sess.run to get attributions.
  """
  ops_to_tensors = lambda ops: [op.outputs[0] for op in ops]
  attribution_hooks = {}
  if tensors_to_keep is None:
    tensors_to_keep = []
  else:
    tensors_to_keep = list(tensors_to_keep)
  if zero_baseline_tensors is None:
    zero_baseline_tensors = []
  with graph.as_default():
    # Compute parts of graph and check correctness.
    all_ops = graph.get_operations()
    constant_ops = tf.contrib.graph_editor.select.select_ops(
        all_ops, positive_filter=lambda x: x.type == 'Const')
    placeholder_ops = tf.contrib.graph_editor.select.select_ops(
        all_ops, positive_filter=lambda x: x.type == 'Placeholder')
    var_read_ops = tf.contrib.graph_editor.select.select_ops(
        '/read$', graph=graph)
    attr_ops = [t.op for t in attribution_tensors]
    required_ops = set(
        tf.contrib.graph_editor.select.get_backward_walk_ops(
            output_tensor.op,
            stop_at_ts=(tensors_to_keep + list(attribution_tensors) +
                        ops_to_tensors(var_read_ops) +
                        ops_to_tensors(placeholder_ops))))

    # Check that attribution tensors are sufficient to compute output_tensor.
    forward_ops = set(
        tf.contrib.graph_editor.select.get_forward_walk_ops(
            attr_ops + var_read_ops + constant_ops))
    assert required_ops.issubset(forward_ops)

    required_sgv = tf.contrib.graph_editor.subgraph.make_view(required_ops)
    attribution_subgraph, attribution_transform_info = (
        tf.contrib.graph_editor.transform.copy_with_input_replacements(
            required_sgv, {}, graph, new_output_scope))
    attribution_hooks['attribution_subgraph'] = attribution_subgraph
    attribution_hooks['attribution_transform_info'] = attribution_transform_info

    # Copy feed to attribution part of graph so we can have one part for
    # baseline and one for input.
    backward_ops = tf.contrib.graph_editor.select.get_backward_walk_ops(
        attr_ops, stop_at_ts=ops_to_tensors(var_read_ops))
    backward_sgv = tf.contrib.graph_editor.subgraph.make_view(backward_ops)
    _, baseline_transform_info = (
        tf.contrib.graph_editor.transform.copy_with_input_replacements(
            backward_sgv, {}, graph, baseline_scope))
    attribution_hooks['baseline_transform_info'] = baseline_transform_info

    # Function to compute combined feed dict. The default setting of
    # baseline_transform_info is to get around python's late binding.
    def CreateCombinedFeedDict(input_feed_dict,
                               baseline_feed_dict=None,
                               baseline_transform_info=baseline_transform_info):
      """Combine baseline and input feed dicts into a common feed dict."""
      combined_feed_dict = input_feed_dict.copy()
      if baseline_feed_dict is None:
        baseline_feed_dict = input_feed_dict
      for tensor, feed_value in baseline_feed_dict.items():
        if isinstance(tensor, tf.Tensor):
          combined_feed_dict[baseline_transform_info.transformed(tensor)] = (
              feed_value)
        elif isinstance(tensor, tf.SparseTensor):
          sparse_transformed_tensor = tf.SparseTensor(
              baseline_transform_info.transformed(tensor.indices),
              baseline_transform_info.transformed(tensor.values),
              baseline_transform_info.transformed(tensor.dense_shape))
          combined_feed_dict[sparse_transformed_tensor] = feed_value
        else:
          raise ValueError('Invalid Entry %s in Feed Dict.' % tensor)
      return combined_feed_dict

    attribution_hooks['create_combined_feed_dict'] = CreateCombinedFeedDict

    # Create new tensors with the multipliers to insert after previous ones.
    attribution_hooks['multipliers'] = []
    attribution_hooks['weighted_attribution_tensors'] = []
    for attribution_tensor in attribution_tensors:
      with tf.control_dependencies(
          [tf.assert_equal(tf.shape(attribution_tensor)[0], 1)]):
        attribution_dims = (
            attribution_dims_map[attribution_tensor]
            if attribution_tensor in attribution_dims_map else [])
        vocab_size = len(attribution_tensor.get_shape())
        attribution_dim_cond = tf.sparse_to_indicator(
            tf.SparseTensor(
                tf.expand_dims(
                    tf.range(len(attribution_dims), dtype=tf.int64), 1),
                attribution_dims, [vocab_size]), vocab_size)
        base_multiplier_shape = tf.concat([
            tf.expand_dims(num_evals, 0),
            tf.ones_like(tf.shape(attribution_tensor))[1:]
        ], 0)
        tile_dims = tf.where(attribution_dim_cond, tf.shape(attribution_tensor),
                             tf.ones_like(tf.shape(attribution_tensor)))
        pre_tile_multiplier = tf.reshape(
            tf.range(tf.to_float(num_evals)) / tf.to_float(num_evals - 1),
            base_multiplier_shape)
        multiplier = tf.tile(pre_tile_multiplier, tile_dims)
        if attribution_tensor in zero_baseline_tensors:
          weighted_attribution_tensor = multiplier * attribution_tensor
        else:
          base_attribution_tensor = baseline_transform_info.transformed(
              attribution_tensor)
          weighted_attribution_tensor = (
              multiplier * attribution_tensor +
              (1 - multiplier) * base_attribution_tensor)
        attribution_hooks['weighted_attribution_tensors'].append(
            weighted_attribution_tensor)
        attribution_hooks['multipliers'].append(multiplier)

    tf.contrib.graph_editor.reroute_ts(
        attribution_hooks['weighted_attribution_tensors'],
        attribution_tensors,
        can_modify=attribution_subgraph.ops)
    g = tf.gradients(
        attribution_transform_info.transformed(output_tensor),
        attribution_hooks['multipliers'])
    attribution_hooks['mean_grads'] = [tf.reduce_mean(grad, 0) for grad in g]
  return attribution_hooks


def AddBOWIntegratedGradientsOps(graph,
                                 embedding_lookup_list,
                                 embedding_list,
                                 other_attribution_tensors,
                                 output_tensor,
                                 new_output_scope='attribution',
                                 baseline_scope='baseline',
                                 tensors_to_keep=None):
  """Attribution for bag of words.

  Attributions assume sum as embedding_combiner. For other combiners, just
  divide the attribution by the appropriate amount. e.g. for 'mean', divide by
  sum of weights.

  Args:
    graph: The tf.Graph to add attribution ops to.
    embedding_lookup_list: List of embedding_lookup tensors. Each tensor must
      have rank 3, with the first dimension being batch dimension, second
      dimension being the object in the example being embedded, and third
      dimension being values of the embedding itself.
    embedding_list: List of embedding tensors. Must be aligned with
      embedding_lookup_list. That is, the two must have the same length, and
      embedding_list[k] must be the value obtained by summing the entries in
      embedding_lookup_list[k] over axis 1, for each k. Thus each entry in
      embedding_list must have rank two, with the first dimension being batch
      dimension and second one representing the summed embedding.
    other_attribution_tensors: List of Tensors (in addition to embeddings) for
      which to compute attribution scores.
    output_tensor: Tensor for whose value we are performing attribution.
    new_output_scope: String. New ops needed for computing the output tensor at
      different interpolated values are created under this scope name.
    baseline_scope: String. New ops needed for computing attribution tensor
      interpolated values are created under this scope name.
    tensors_to_keep: Set of tensors. By default, tensors in the graph between
      the output_tensor and attribution tensors are copied to a different part
      of the graph, and evaluated separately for each interpolation. If we want
      a value to be fixed (only computed for the main input instead of each
      interpolation), it should be put in tensors_to_keep.

  Returns:
    attribution_hooks: Dict with the following keys (in addition to keys of
      AddIntegratedGradientsOps)
      bow_attributions: List of attribution scores for embedding features,
        aligned with embedding_lookup_list and embedding_list.
      num_evals: placeholder with default value 50. Gets passed to
        AddIntegratedGradientsOps.
  """
  for embedding_lookup, embedding in zip(embedding_lookup_list, embedding_list):
    assert len(embedding_lookup.get_shape()) == 3
    assert len(embedding.get_shape()) == 2
  with graph.as_default():
    num_evals = tf.placeholder_with_default(
        tf.constant(50, name='num_evals'), shape=())
    attribution_dims_map = {embedding: [1] for embedding in embedding_list}
    attribution_hooks = AddIntegratedGradientsOps(
        graph=graph,
        attribution_tensors=embedding_list + other_attribution_tensors,
        output_tensor=output_tensor,
        num_evals=num_evals,
        attribution_dims_map=attribution_dims_map,
        zero_baseline_tensors=set(embedding_list),
        new_output_scope=new_output_scope,
        baseline_scope=baseline_scope,
        tensors_to_keep=tensors_to_keep)
    attributions = []
    for embedding_lookup, mean_grad, embedding in zip(
        embedding_lookup_list, attribution_hooks['mean_grads'], embedding_list):
      attributions.append(
          tf.reduce_sum(
              tf.squeeze(embedding_lookup, 0) * tf.expand_dims(mean_grad, 0) /
              (embedding + sys.float_info.epsilon), 1))
    attribution_hooks['bow_attributions'] = attributions
    attribution_hooks['num_evals'] = num_evals
  return attribution_hooks


def GetEmbeddingLookupList(signals_list,
                           embedding_vars,
                           sparse_ids,
                           sparse_weights=None,
                           combiners='sqrtn',
                           partition_strategies='mod'):
  """Get a list of embedding lookup tensors.

  Args:
    signals_list: A list of strings, representing names of features.
    embedding_vars: Dict mapping feature names to full embedding variables.
    sparse_ids: Dict mapping feature names to SparseTensors of their ids.
    sparse_weights: Either None, or a dict mapping feature names to
      SparseTensors of their weights (which can also be None).
    combiners: Either a common combiner type for all features ('mean', sqrtn' or
      'sum') or a dict mapping each feature name to a combiner type.
    partition_strategies: Either a common partition_strategy for all features
      ('mod' or 'div') or a dict mapping feature_names to partition_stratgies.

  Returns:
    embedding_lookup_list: A list of embedding lookup tensors used for bag of
      words attribution, aligned with signals_list.
  """
  assert isinstance(embedding_vars, dict) and isinstance(sparse_ids, dict)
  assert sparse_weights is None or isinstance(sparse_weights, dict)
  assert combiners in ('mean', 'sqrtn', 'sum') or isinstance(combiners, dict)
  assert (partition_strategies in ('mod', 'div') or
          isinstance(partition_strategies, dict))
  embedding_lookup_list = []
  for signal in signals_list:
    combiner = combiners[signal] if isinstance(combiners, dict) else combiners
    partition_strategy = (partition_strategies[signal] if isinstance(
        partition_strategies, dict) else partition_strategies)

    # Batch dimension should be 1 for attribution.
    with tf.control_dependencies(
        [tf.assert_equal(tf.shape(sparse_ids[signal])[0], 1)]):
      embedding_lookup = tf.nn.embedding_lookup(
          params=embedding_vars[signal],
          ids=tf.sparse_tensor_to_dense(sparse_ids[signal]),
          partition_strategy=partition_strategy)
    if sparse_weights is None or sparse_weights[signal] is None:
      num_vals = tf.size(sparse_ids[signal].values)
      if combiner == 'mean':
        embedding_weights = tf.fill([1, num_vals],
                                    1.0 / tf.to_float(num_vals))
      elif combiner == 'sqrtn':
        embedding_weights = tf.fill([1, num_vals],
                                    1.0 / tf.sqrt(tf.to_float(num_vals)))
      else:
        embedding_weights = tf.ones([1, num_vals], dtype=tf.float32)
    else:
      # Batch dimension should be 1 for attribution.
      with tf.control_dependencies(
          [tf.assert_equal(tf.shape(sparse_weights[signal])[0], 1)]):
        dense_weights = tf.sparse_tensor_to_dense(sparse_weights[signal])
      if combiner == 'mean':
        embedding_weights = dense_weights / tf.reduce_sum(dense_weights)
      elif combiner == 'sqrtn':
        embedding_weights = (
            dense_weights / tf.sqrt(tf.reduce_sum(tf.pow(dense_weights, 2))))
      else:
        embedding_weights = dense_weights
    embedding_lookup *= tf.expand_dims(embedding_weights, -1)
    embedding_lookup_list.append(embedding_lookup)
  return embedding_lookup_list


