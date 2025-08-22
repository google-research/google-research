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

"""Model definitions for automatic feature engineering."""

import collections
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
import feature_selection
import tensorflow as tf
import tensorflow_probability as tfp


_IDENTITY_FN = lambda x: x
_PLACEHOLDER_TENSOR = tf.zeros([])
_N_ARY_NUM_FEATURES = 5
_N_ARY_TO_UNARY_RATIO = 2
_NUM_QUANTILES = 3
_NUMERICAL_FEATURE_TYPE = 'numerical'
_CATEGORICAL_FEATURE_TYPE = 'categorical'
_SUM_FUNCTION_NAME = 'Sum'
_PROD_FUNCTION_NAME = 'Product'


def log_transform(inputs):
  """Creates the entrywise log transform function."""
  inputs_min = tf.math.reduce_min(inputs, axis=-1, keepdims=True)
  inputs = inputs - inputs_min
  return tf.math.log1p(inputs)


def idf(inputs):
  """Creates the identity transform function."""
  return inputs


class PolyTransform(tf.keras.layers.Layer):
  """Layer to learn polynomial transformations."""

  def __init__(self):
    super().__init__()
    # TODO(yihed): make exponent learnable. The current bottleneck is that 1)the
    # exponent needs to be integral, but tf.math.round is not differentiable,
    # and 2) exponentiation of negative floats by non-integral exponents leads
    # to nans.
    self.exp = 2.0

  def call(self, inputs):
    return tf.math.pow(inputs, self.exp)


class ZScaling(tf.keras.layers.Layer):
  """Layer to learn and apply z-scaling."""

  def __init__(self):
    super().__init__()
    self.gamma = tf.Variable(
        tf.ones(
            1,
        )
    )
    self.beta = tf.Variable(
        tf.zeros(
            1,
        )
    )

  def call(self, inputs):
    # TODO(yihed): save train-time statistics to apply at test time.
    mean = tf.math.reduce_mean(inputs, axis=0, keepdims=True)
    std = tf.math.reduce_std(inputs, axis=0, keepdims=True)
    return self.gamma * tf.math.divide_no_nan(inputs - mean, std) + self.beta


class Aggregate(tf.keras.layers.Layer):
  """Layer to aggregate by categorical features."""

  def __init__(
      self,
      num_cat_features,
      num_attribute_features,
      n_ary_num_features,
      feature_dim,
      activation_fn = 'gelu',
      aggregate_average = False,
  ):
    super().__init__()
    self.num_cat_features = num_cat_features
    self.num_attribute_features = num_attribute_features
    self.n_ary_num_features = n_ary_num_features
    self.cat_mask = tf.Variable(
        tf.random.uniform((self.num_cat_features,), maxval=1.0),
        dtype=tf.float32,
    )
    self.num_numerics = self.num_attribute_features - self.num_cat_features
    self.numeric_mask = tf.Variable(
        tf.random.uniform((self.num_numerics,), maxval=1.0), dtype=tf.float32
    )
    self.aggregate_average = aggregate_average

  def call(self, inputs, idx_inputs):
    """Forward pass for learnable aggregation layer.

    Args:
      inputs: attribute input features.
      idx_inputs: inputs containing index embeddings for categorical features.

    Returns:
      Aggregated features after learning which features to GroupBy by and which
      to aggregate.
    """
    cat_mask = tf.nn.softmax(self.cat_mask)
    top_cat_mask_val, top_idx = tf.math.top_k(cat_mask, k=1)
    # Categorical indices. This assumes categorical features precede numerical
    # features, as done during data processing.
    cat_idx = tf.expand_dims(idx_inputs[:, int(tf.squeeze(top_idx))], -1)
    cat_max = tf.squeeze(tf.math.reduce_max(cat_idx))
    cat_aggregate = tf.zeros((cat_max + 1, self.num_numerics))
    updates = inputs[:, self.num_cat_features :]
    cat_idx = tf.cast(cat_idx, tf.int32)
    # 2D tensor that collects the numeric aggregations grouped by a categorical
    # column.
    cat_aggregate = tf.tensor_scatter_nd_add(cat_aggregate, cat_idx, updates)
    if self.aggregate_average:
      count_ones = tf.ones_like(inputs[:, 0])
      count_aggregate = tf.zeros((cat_max + 1,))
      cat_counts = tf.tensor_scatter_nd_add(
          count_aggregate, cat_idx, count_ones
      )
      cat_aggregate = tf.math.divide_no_nan(
          cat_aggregate, tf.expand_dims(cat_counts, -1)
      )

    cat_aggregate = tf.gather(cat_aggregate, tf.squeeze(cat_idx), axis=0)

    # Select the aggregate numerical features.
    numeric_mask = tf.nn.softmax(self.numeric_mask)
    top_numeric_mask_val, top_idx = tf.math.top_k(
        numeric_mask, k=min(self.n_ary_num_features, self.num_numerics)
    )
    mask_confidence = (top_numeric_mask_val + top_cat_mask_val) / 2.0
    cat_aggregate = tf.gather(cat_aggregate, top_idx, axis=-1) * tf.expand_dims(
        mask_confidence, 0
    )
    return cat_aggregate


class Quantilize(tf.keras.layers.Layer):
  """Layer to apply quantilization into three quantiles."""

  def __init__(self):
    super().__init__()
    self.num_quantiles = _NUM_QUANTILES
    self.quantile_intervals = []
    interval_min = -1.0
    interval_max = 1.0
    self.quantile_len = (interval_max - interval_min) / self.num_quantiles
    for i in range(self.num_quantiles):
      self.quantile_intervals.append(interval_min + self.quantile_len * i)

  def call(self, inputs):
    # TODO(yihed): save training thresholds for inference.
    quantiles = tfp.stats.quantiles(inputs, self.num_quantiles, axis=0)
    # TODO(yihed): make more adaptable to different numbers of quantiles.
    input_thirds = []
    input_thirds.append(tf.where(inputs < quantiles[1, :], inputs, 0.0))

    input_thirds.append(
        tf.where(
            (inputs > quantiles[1, :]) & (inputs < quantiles[2, :]), inputs, 0.0
        )
    )
    input_thirds.append(tf.where(inputs > quantiles[2, :], inputs, 0.0))

    # Map each quantile evenly between -1 and 1.
    for i, third in enumerate(input_thirds):
      interval = quantiles[i + 1, :] - quantiles[i, :]
      input_thirds[i] = third * (
          tf.math.divide_no_nan(self.quantile_len, interval)
      )

    input_thirds[0] = tf.where(
        inputs < quantiles[1, :],
        self.quantile_intervals[0] + input_thirds[0],
        0.0,
    )
    input_thirds[1] = tf.where(
        (inputs > quantiles[1, :]) & (inputs < quantiles[2, :]),
        self.quantile_intervals[1] + input_thirds[1],
        0.0,
    )
    input_thirds[2] = tf.where(
        inputs > quantiles[2, :],
        self.quantile_intervals[2] + input_thirds[2],
        0.0,
    )

    return input_thirds[0] + input_thirds[1] + input_thirds[2]


class FeatureDiscoveryLayer(tf.keras.layers.Layer):
  """Feature selection and discovery layer."""

  def __init__(
      self,
      num_features,
      num_selected_features,
      transform_function = _IDENTITY_FN,
      keep_input_dims = False,
      use_softmax_mask = True,
      final_feature_dim = None,
      feature_type = _NUMERICAL_FEATURE_TYPE,
  ):
    """Initializes feature discovery layer.

    Applies a given transform function to the inputs.

    Args:
      num_features: number of total features.
      num_selected_features: target number of selected features.
      transform_function: transform function to apply to inputs.
      keep_input_dims: whether to keep input dimensions after executing
        transformation function.
      use_softmax_mask: whether to use softmax to learn mask.
      final_feature_dim: the feature dimension of the final output.
      feature_type: the type of feature this transform acts on.
    """
    super().__init__()
    self.num_features = num_features
    self.num_selected_features = min(num_features, num_selected_features)
    self.feature_selector = feature_selection.FeatureSelectionSparseMasks(
        num_features,
        self.num_selected_features,
        output_mask=True,
        use_softmax_mask=use_softmax_mask,
    )
    self.transform_function = transform_function
    self.keep_input_dims = keep_input_dims
    self.final_feature_dim = min(self.num_selected_features, final_feature_dim)
    self.feature_type = feature_type

  def set_sparse_mask(self, mask):
    """Sets the learned sparse mask for this layer."""
    self.sparse_mask = mask

  def call(
      self, inputs, training = True
  ):
    """Feature discovery layer forward pass.

    This call uses the input to learn the feature and transform masks, and zeros
    out non-selected features. Further details:go/automatic-feature-engineering.

    Args:
      inputs: the input features.
      training: whether currently training.

    Returns:
      Sparsified and transformed features, learned sparse mask, and the selected
      features to be used for the given transform.
    """
    if self.keep_input_dims:
      # Apply transform before masking for elementwise operations.
      inputs = self.transform_function(inputs)

    inputs_sparse, inputs_mask = self.feature_selector(inputs)

    inputs_mask = tf.reshape(inputs_mask, [-1])
    # TODO(yihed): consider using raw inputs rather than masked-out inputs.
    # non-selected features are zeroed-out
    _, top_idx = tf.math.top_k(inputs_mask, k=self.num_selected_features)

    # Unary operators outputs have same dimensionality as inputs, n-ary
    # operators output tensors with fewer dimensions than inputs.
    top_features = tf.gather(inputs_sparse, top_idx, axis=-1)
    if self.keep_input_dims:
      inputs_transformed = inputs_sparse
    else:
      inputs_transformed = self.transform_function(top_features)
      top_features = inputs_transformed
    self.set_sparse_mask(inputs_mask)
    return inputs_transformed, inputs_mask, top_features

  def visualize_transform(
      self, str_presentation, num_cat_features
  ):
    """Create an interpretable string representation using current layer.

    Modifies input list in-place.

    Args:
      str_presentation: List of string representations per index from previous
        layers.
      num_cat_features: number of categorical features.
    """
    _, top_idx = tf.math.top_k(self.sparse_mask, k=self.num_selected_features)
    transform_str = transform_to_str(self.transform_function)

    if self.feature_type == _NUMERICAL_FEATURE_TYPE:
      top_idx += num_cat_features

    if self.keep_input_dims:
      # e.g. for unary transforms like log.
      for i in top_idx:
        # This accounts for skip connections.
        if str_presentation[i]:
          str_presentation[i] = f'{transform_str}({str_presentation[i]})'
        # Append here to match up with cross_mask selection coefficients.
        str_presentation.append(str_presentation[i])
    else:
      # e.g. for n-ary transforms like reduce_sum.
      features = []
      for i in top_idx:
        if str_presentation[i]:
          features.append(str_presentation[i])
        if self.final_feature_dim > 1:
          str_transformed = f'{transform_str}({str_presentation[i]})'
          str_presentation.append(str_transformed)

      if self.final_feature_dim == 1:
        features = ', '.join(features)
        transform_str = f'{transform_str}({features})'
        str_presentation.append(transform_str)


class TemporalFeatureDiscoveryLayer(tf.keras.layers.Layer):
  """Discovery module for temporal features."""

  def __init__(
      self,
      feature_dim,
      n_temporal_features,
      context_len = 10,
  ):
    """Initializes temporal feature discovery module."""
    super().__init__()
    self.context_len = context_len
    # TODO(yihed): consider case when cat_embed_dim > 1.
    self.temporal_mask = tf.Variable(
        tf.random.uniform((context_len,), maxval=1.0), dtype=tf.float32
    )
    self.lag_mask = tf.Variable(
        tf.random.uniform((context_len,), maxval=1.0), dtype=tf.float32
    )

    self.n_temporal_features = n_temporal_features
    # TODO(yihed): explore making this threshold learnable.
    self.feature_selection_prob_thresh = 1.0 / context_len

  @classmethod
  def output_dim_per_temporal_feature(cls):
    """Output feature dimension per temporal feature.

    Each temporal feature produces a cumulative sum feature, a cumulative
    difference feature, and a lag feature.
    Returns:
      Processed putput feature dimension per temporal feature.
    """
    # TODO(yihed): update this return value to avoid the constant 3.
    return 7

  def call(
      self,
      temporal_inputs,
      training = True,
      separate_prob_coeff = True,
  ):
    """Temporal feature discovery module forward pass.

    Args:
      temporal_inputs: input tensor containing temporal features.
      training: whether currently training.
      separate_prob_coeff: whether to separate feature selection probability
        from the aggregation coefficients.

    Returns:
      Transformed temporal features.
    """
    # This mask simultaneously learns the sign of aggregation and the selection
    # coefficients.
    all_features_prob = tf.nn.softmax(tf.math.abs(self.temporal_mask))

    selected_feature_coeff = tf.where(
        all_features_prob > self.feature_selection_prob_thresh,
        self.temporal_mask,
        0,
    )
    # This incentivizes the mask to select features that are temporally adjacent
    temporal_regularizer = tf.reduce_sum(
        tf.math.abs(selected_feature_coeff[:-1] - selected_feature_coeff[1:])
    )

    if separate_prob_coeff:
      # TODO(yihed): introduce other aggregation schemes such as differencing.
      # Aggregating selected features with coefficients 1.
      selection_coeff = tf.where(
          all_features_prob > self.feature_selection_prob_thresh, 1.0, 0.0
      )
    else:
      selection_coeff = selected_feature_coeff

    selection_coeff = tf.tile(
        selection_coeff,
        [
            self.n_temporal_features,
        ],
    )
    selected_features = temporal_inputs * tf.expand_dims(selection_coeff, 0)
    selected_features = tf.reshape(
        selected_features, [-1, self.n_temporal_features, self.context_len]
    )
    # TODO(yihed): consider using percentages rather than raw values.
    selected_features_mean_diff = (
        selected_features[:, :, :-1] - selected_features[:, :, 1:]
    )
    selected_features_mean_diff = tf.reduce_mean(
        tf.math.abs(selected_features_mean_diff), -1
    )
    selected_features_sum = tf.reduce_sum(selected_features, -1)
    # Learn temporal lag probabilistically.
    lag_prob = tf.nn.softmax(self.lag_mask)
    lag_prob_max = tf.math.reduce_max(lag_prob, axis=-1)
    lag_prob = tf.where(lag_prob < lag_prob_max, 0.0, lag_prob)

    lag_features = lag_prob * selected_features
    lag_features = tf.math.reduce_sum(lag_features, axis=-1)

    # Compute features based on relative rather than absolute changes.
    temporal_inputs = tf.reshape(
        temporal_inputs, [-1, self.n_temporal_features, self.context_len]
    )
    temporal_mean, temporal_std, relative_mean, relative_std = (
        self.temporal_relative_stats(temporal_inputs)
    )
    return (
        tf.concat(
            [
                selected_features_sum,
                selected_features_mean_diff,
                lag_features,
                temporal_mean,
                temporal_std,
                relative_mean,
                relative_std,
            ],
            -1,
        ),
        temporal_regularizer,
    )

  def temporal_relative_stats(
      self, temporal_inputs
  ):
    temporal_mean = tf.math.reduce_mean(temporal_inputs, axis=-1)
    temporal_std = tf.math.reduce_std(temporal_inputs, axis=-1)

    temporal_max = tf.math.reduce_max(temporal_inputs, axis=-1, keepdims=True)
    temporal_min = tf.math.reduce_max(temporal_inputs, axis=-1, keepdims=True)
    # percentage changes w.r.t. amount of total variation per feature.
    temporal_relative = tf.math.divide_no_nan(
        temporal_inputs - temporal_inputs[:, :, :1], temporal_max - temporal_min
    )
    relative_mean = tf.math.reduce_mean(temporal_relative, axis=-1)
    relative_std = tf.math.reduce_std(temporal_relative, axis=-1)
    return temporal_mean, temporal_std, relative_mean, relative_std


class FeatureDiscoveryModel(tf.keras.layers.Layer):
  """Feature selection and discovery module."""

  def __init__(
      self,
      num_attribute_features,
      num_selected_features,
      feature_dim,
      num_mlp_layers,
      n_temporal_features,
      context_len = 10,
      infer_features = False,
      activation_fn = 'gelu',
      prod_num_args = 2,
      use_softmax_mask = True,
      num_cat_features = 0,
      n_reduce_ops = 3,
      n_aggregation_layers = 1,
  ):
    """Initializes feature selection module.

    Args:
      num_attribute_features: total number of attribute features.
      num_selected_features: target number of selected features.
      feature_dim: feature dimension for latent vectors.
      num_mlp_layers: number of MLP layers.
      n_temporal_features: number of temporal features.
      context_len: length of context window for temporal features.
      infer_features: whether to produce and save learned features.
      activation_fn: activation function for MLP layers.
      prod_num_args: number of arguments going into reduce_prod.
      use_softmax_mask: whether to use softmax to learn mask.
      num_cat_features: number of categorical features.
      n_reduce_ops: number of reduction operations such as reduce_prod.
      n_aggregation_layers: number of aggregation layers.
    """
    super().__init__()
    self.num_attribute_features = num_attribute_features
    num_selected_features = min(num_selected_features, num_attribute_features)
    self.num_selected_features = num_selected_features
    z_scaling = ZScaling()
    self.unary_transforms = [
        # This is an initial unary transform candidate, more transforms
        # will be added.
        # z_scaling,
    ]

    poly_transform = PolyTransform()
    quantilization = Quantilize()

    NAryTransform = collections.namedtuple(
        'NAryTransform',
        [
            'transform_func',
            'num_selected_features',
            'final_feature_dim',
            'feature_type',
        ],
    )
    n_ary_num_features = max(
        _N_ARY_NUM_FEATURES,
        (self.num_selected_features // _N_ARY_TO_UNARY_RATIO),
    )
    n_ary_num_features = min(n_ary_num_features, self.num_selected_features)
    # Learned masks select which subset of features to use as arguments to these
    # operators.
    reduce_sum_transforms = [
        NAryTransform(
            functools.partial(
                tf.math.reduce_sum,
                axis=-1,
                keepdims=True,
            ),
            prod_num_args,
            1,
            _NUMERICAL_FEATURE_TYPE,
        )
    ] * n_reduce_ops
    reduce_prod_transforms = [
        NAryTransform(
            functools.partial(
                tf.math.reduce_prod,
                axis=-1,
                keepdims=True,
            ),
            prod_num_args,
            1,
            _NUMERICAL_FEATURE_TYPE,
        )
    ] * n_reduce_ops
    self.n_ary_transforms = [
        *reduce_sum_transforms,
        # Taking the product across many entries can create vanishing gradients.
        *reduce_prod_transforms,
        NAryTransform(
            log_transform,
            n_ary_num_features,
            n_ary_num_features,
            _NUMERICAL_FEATURE_TYPE,
        ),
        NAryTransform(
            z_scaling,
            n_ary_num_features,
            n_ary_num_features,
            _NUMERICAL_FEATURE_TYPE,
        ),
        NAryTransform(
            poly_transform,
            n_ary_num_features,
            n_ary_num_features,
            _NUMERICAL_FEATURE_TYPE,
        ),
        NAryTransform(
            quantilization,
            n_ary_num_features,
            n_ary_num_features,
            _NUMERICAL_FEATURE_TYPE,
        ),
    ]
    self.unary_layers = []
    self.n_ary_layers = []
    self.unary_layernorms = []
    for transform in self.unary_transforms:
      # keep_input_dims=True so addition across layers makes sense.
      feature_layer = FeatureDiscoveryLayer(
          num_attribute_features,
          self.num_selected_features,
          transform,
          keep_input_dims=True,
          use_softmax_mask=use_softmax_mask,
      )
      self.unary_layers.append(feature_layer)
      self.unary_layernorms.append(tf.keras.layers.LayerNormalization())

    for transform in self.n_ary_transforms:
      if transform.feature_type == _NUMERICAL_FEATURE_TYPE:
        layer_n_features = num_attribute_features - num_cat_features
      elif transform.feature_type == _CATEGORICAL_FEATURE_TYPE:
        layer_n_features = num_cat_features
      else:
        layer_n_features = num_attribute_features

      feature_layer = FeatureDiscoveryLayer(
          layer_n_features,
          transform.num_selected_features,
          transform.transform_func,
          keep_input_dims=False,
          use_softmax_mask=use_softmax_mask,
          final_feature_dim=transform.final_feature_dim,
          feature_type=transform.feature_type,
      )
      self.n_ary_layers.append(feature_layer)

    raw_features_mlp_sequence = [
        tf.keras.layers.Dense(feature_dim, activation=activation_fn)
        for _ in range(num_mlp_layers)
    ]
    self.raw_features_mlp = tf.keras.Sequential(raw_features_mlp_sequence)

    self.n_temporal_features = n_temporal_features
    # mask for comparing importance across learned transforms.
    n_ary_output_dim = sum(
        [layer.final_feature_dim for layer in self.n_ary_layers]
    )
    self.cross_mask_dim = (
        self.num_selected_features * len(self.unary_transforms)
        + n_ary_output_dim
    )

    cross_mlp_sequence = [
        tf.keras.layers.Dense(feature_dim, activation=activation_fn)
        for _ in range(num_mlp_layers)
    ]
    self.cross_mlp = tf.keras.Sequential(cross_mlp_sequence)
    if n_temporal_features > 0:
      self.temporal_discovery_layer = TemporalFeatureDiscoveryLayer(
          feature_dim,
          n_temporal_features,
          context_len,
      )
      self.cross_mask_dim += (
          n_temporal_features
          * TemporalFeatureDiscoveryLayer.output_dim_per_temporal_feature()
      )

    self.num_cat_features = num_cat_features
    self.aggregation_layers = []
    if num_cat_features > 0:
      for _ in range(n_aggregation_layers):
        for aggregate_average in [True, False]:
          self.aggregation_layers.append(
              Aggregate(
                  num_cat_features,
                  self.num_attribute_features,
                  n_ary_num_features,
                  feature_dim,
                  aggregate_average=aggregate_average,
              )
          )
          self.cross_mask_dim += min(
              n_ary_num_features, self.num_attribute_features - num_cat_features
          )

    self.cross_mask = tf.Variable(
        tf.random.uniform((self.cross_mask_dim,), maxval=1.0), dtype=tf.float32
    )
    self.infer_features = infer_features

  def call(
      self,
      attribute_inputs,
      temporal_inputs = _PLACEHOLDER_TENSOR,
      idx_inputs = None,
  ):
    """Feature discovery module forward pass.

    This call uses the input to learn the feature and transform masks, and zeros
    out non-selected features.

    Args:
      attribute_inputs: SLM-processed input attribute features tensor.
      temporal_inputs: input temporal features tensor.
      idx_inputs: inputs containing the indices for categorical features.

    Returns:
      Transformed features, including both unary and n-ary transforms.
    """
    # Since inputs are SLM selected features, feeding this into the downstream
    # MLP means the identity function is used as a unary operator. Hence this
    # model includes SLM as a special case.
    raw_features_sparse = attribute_inputs
    latents = attribute_inputs
    selected_unary_features = []

    # save for visualizing transforms.
    # TODO(yihed): consider passing in SLM mask rather than inferring
    # non-zero indices here.
    raw_features_idx = tf.squeeze(tf.where(attribute_inputs[0]))
    self.save_raw_features_idx(raw_features_idx)

    for i, layer in enumerate(self.unary_layers):
      layer_latents, _, top_features = layer(latents)
      latents = self.unary_layernorms[i](latents + layer_latents)
      selected_unary_features.append(top_features)

    # Note unary_output is expected to keep the dimension of attribute_inputs.
    unary_output = latents
    # parallelize the n_ary layers, to allow each n_ary operation on
    # the unary outputs directly.
    selected_n_ary_features = []

    numerical_latents = unary_output[:, self.num_cat_features :]
    categorical_latents = unary_output[:, : self.num_cat_features]

    for layer in self.n_ary_layers:
      if layer.feature_type == _NUMERICAL_FEATURE_TYPE:
        layer_input = numerical_latents
      elif layer.feature_type == _CATEGORICAL_FEATURE_TYPE:
        layer_input = categorical_latents
      else:
        layer_input = unary_output
      latents, _, _ = layer(layer_input)
      # n_ary transforms output 1-D top output
      selected_n_ary_features.append(latents)

    raw_feature_latents = self.raw_features_mlp(raw_features_sparse)

    selected_unary_features.extend(selected_n_ary_features)
    if self.n_temporal_features > 0:
      temporal_features, temporal_regularizer = self.temporal_discovery_layer(
          temporal_inputs
      )

      selected_unary_features.append(temporal_features)
    else:
      temporal_regularizer = 0

    for aggregation_layer in self.aggregation_layers:
      aggregation_features = aggregation_layer(raw_features_sparse, idx_inputs)
      selected_unary_features.append(aggregation_features)

    cross_latents = tf.concat(selected_unary_features, axis=-1)
    if self.infer_features:
      self.save_learned_features(cross_latents)

    top_val, top_idx = tf.math.top_k(
        self.cross_mask,
        k=self.num_selected_features + self.n_temporal_features,
    )
    top_idx = tf.expand_dims(top_idx, -1)
    top_val = tf.nn.softmax(top_val)
    transform_prob = tf.scatter_nd(
        top_idx, top_val, tf.constant([self.cross_mask_dim])
    )
    learned_latents = cross_latents * tf.expand_dims(transform_prob, 0)

    cross_latents = self.cross_mlp(learned_latents)
    all_latents = tf.concat([raw_feature_latents, cross_latents], axis=-1)

    return (
        all_latents,
        temporal_regularizer,
    )

  def save_learned_features(self, learned_features):
    """Record newly discovered features."""
    # TODO(yihed): save config file with feature dimensions.
    self.learned_features = learned_features

  def save_raw_features_idx(self, raw_features_idx):
    """Save selected indices for raw input features."""
    self.raw_features_idx = raw_features_idx


def transform_to_str(transform):
  """Converts a transform function to a readable string."""
  if isinstance(transform, functools.partial):
    if transform.func == tf.math.reduce_prod:
      return _PROD_FUNCTION_NAME
    elif transform.func == tf.math.reduce_sum:
      return _SUM_FUNCTION_NAME
    else:
      return str(transform.func.__name__)
  elif isinstance(transform, tf.keras.layers.Layer):
    return transform.name
  elif callable(transform):
    # e.g. transform of function type.
    return transform.__name__
  else:
    return str(transform)


def recover_transforms(
    model,
    cat_features = None,
    numerical_features = None,
):
  """Retrieves transformation functions from model.

  Args:
    model: the discovery model to extract transforms from.
    cat_features: list of categorical feature names.
    numerical_features: list of numerical feature names.

  Returns:
    Dictionary of feature names along their scores.
    Feature names of discovered features, ranked by importance.
    Ranked ordering of the discovered features, ranked by by importance.
  """

  raw_features = [''] * model.num_attribute_features
  all_feature_names = cat_features + numerical_features
  assert tf.math.reduce_max(model.raw_features_idx) < (
      len(cat_features) + len(numerical_features)
  ), 'Selected feature indices must be within total feature range.'
  for idx in model.raw_features_idx:
    raw_features[idx] = all_feature_names[int(idx)]

  # Note the iteration order here must be the same as in forward pass, due to
  # construction of the cross mask.
  features = raw_features
  for transform_layers in [model.unary_layers, model.n_ary_layers]:
    for layer in transform_layers:
      layer.visualize_transform(features, len(cat_features))

  if hasattr(model, 'temporal_discovery_layer'):
    temporal_layer = model.temporal_discovery_layer
    all_features_prob = tf.nn.softmax(tf.math.abs(temporal_layer.temporal_mask))

    selected_feature_index = tf.where(
        all_features_prob > temporal_layer.feature_selection_prob_thresh
    )
    temporal_features = []
    for idx in selected_feature_index:
      temporal_features.append('t' + str(int(idx)))

    temporal_features_str = ', '.join(temporal_features) + '; '
    temporal_sum_strs = [
        'TemporalSum(' + temporal_features_str + str(i) + ')'
        for i in range(model.n_temporal_features)
    ]
    features.extend(temporal_sum_strs)

    temporal_diff_strs = [
        'TemporalDiff(' + temporal_features_str + str(i) + ')'
        for i in range(model.n_temporal_features)
    ]
    features.extend(temporal_diff_strs)
    lag_idx = tf.math.argmax(temporal_layer.lag_mask, axis=-1)
    temporal_lag_strs = [
        'TemporalLag(t' + str(int(lag_idx)) + ', ' + str(i) + ')'
        for i in range(model.n_temporal_features)
    ]
    features.extend(temporal_lag_strs)
    transform_names = [
        'TemporalMean',
        'TemporalStd',
        'RelTemporalMean',
        'RelTemporalStd',
    ]
    for transform_name in transform_names:
      for idx in range(model.n_temporal_features):
        features.append(transform_name + '(series' + str(idx) + ')')

  for aggregation_layer in model.aggregation_layers:
    cat_idx = tf.math.argmax(aggregation_layer.cat_mask)

    _, numeric_idx = tf.math.top_k(
        aggregation_layer.numeric_mask,
        k=min(
            aggregation_layer.num_numerics, aggregation_layer.n_ary_num_features
        ),
    )
    aggregate_str = (
        'GroupByThenAverage('
        if aggregation_layer.aggregate_average
        else 'GroupByThenSum('
    )
    for idx in numeric_idx:
      features.append(
          aggregate_str
          + cat_features[int(cat_idx)]
          + '; '
          + numerical_features[int(idx)]
          + ')'
      )

  n_top_features = model.cross_mask_dim
  cross_mask_scores, cross_mask_idx = tf.math.top_k(
      model.cross_mask, n_top_features
  )

  if model.num_attribute_features + model.cross_mask_dim != len(features):
    logging.error(
        'Attributes and cross mask dimensions do not add up to total #features.'
    )

  # The cross features start at index len(features) - model.cross_mask_dim.
  ranked_features = tf.gather(features[-n_top_features:], cross_mask_idx)
  return (
      dict(zip(ranked_features.numpy(), cross_mask_scores.numpy())),
      ranked_features.numpy().tolist(),
      cross_mask_idx,
  )
