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

"""Feature selection layers to be integrated into the architecture."""

import sys
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
# This import format necessary: b/164280272.
import tensorflow_addons.activations as tfa_activations


class FeatureSelectionSparseMasks(tf.keras.layers.Layer):
  """Feature selection module using sparse learnable masks."""

  def __init__(
      self,
      num_features: int,
      num_selected_features: Optional[int] = None,
      mask_temperature: float = 0.9,
      mask_perturbation_amplitude: float = 0.0,
      scale_before_sparsemax: bool = True,
      num_feature_scaler: Any = None,
      do_mixed_precision: bool = False,
      gt_salient_feature_indices: Optional[np.ndarray] = None,
      reduce_feature_dim: bool = False,
      output_mask: bool = False,
      use_softmax_mask: bool = False,
      epsilon: float = 0.00001,
      seed: Optional[int] = None,
  ):
    """Initializes feature selection module.

    Args:
      num_features: number of total features.
      num_selected_features: number of selected features.
      mask_temperature: amount to scale the mask to encourage higher or lower
        sparsity; encourages higher sparsity if > 1, and lower if < 1.
      mask_perturbation_amplitude: amplitude of noise to add to the mask to
        introduce stochasticity.
      scale_before_sparsemax: whether to scale mask before sparsemax, so
        sparsemax achieves target sparsity level.
      num_feature_scaler: object used for number of features tempering.
      do_mixed_precision: whether to use mixed precision.
      gt_salient_feature_indices: ground truth indices for salient features,
        applies only to synthetic data.
      reduce_feature_dim: whether to reduce feature dimension to speed up
        inference.
      output_mask: whether to return the learned sparse mask.
      use_softmax_mask: whether to use softmax mask for better latency.
      epsilon: small floating point value to bound away from 0.
      seed: seed for random number generation.
    """
    super().__init__()
    self.mask = tf.Variable(
        tf.random.uniform((1, num_features), maxval=1.0, seed=seed),
        dtype=tf.float32,
    )

    if reduce_feature_dim:
      initializer = tf.keras.initializers.GlorotNormal()
      self.feature_weight = tf.Variable(
          initializer(shape=(num_features, num_selected_features)),
          dtype=tf.float32,
      )
      self.feature_bias = tf.Variable(
          np.zeros(num_selected_features), dtype=tf.float32
      )

    self.num_features = num_features
    self.num_selected_features = min(num_selected_features, num_features)
    self.mask_temperature = mask_temperature
    # TODO(yihed): remove mask_perturbation_amplitude parameter and function.
    self.mask_perturbation_amplitude = mask_perturbation_amplitude
    self.scale_before_sparsemax = scale_before_sparsemax
    self.num_feature_scaler = num_feature_scaler
    self.do_mixed_precision = do_mixed_precision
    self.gt_salient_feature_indices = gt_salient_feature_indices
    self.reduce_feature_dim = reduce_feature_dim
    self._output_mask = output_mask
    # Mask cached after training to accelerate inference time.
    self.trained_mask = None
    self.trained_feature_weight = None
    self.trained_top_idx = None
    self.use_softmax_mask = use_softmax_mask
    self.epsilon = epsilon

  def softmax_mask(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    """Use softmax to learn the sparse selection mask.

    This achieves up to 40% latency improvements with insignificant degradation
      in accuracy
    Args:
      inputs: data to learn the selection mask on.

    Returns:
      Tensor where non-selected portions of the inputs are zeroed-out, as well
        as the learned mask if specified.
    """
    top_val, top_idx = tf.math.top_k(self.mask, k=self.num_selected_features)
    top_val = tf.reshape(top_val, [-1])
    top_val = tf.nn.softmax(top_val)
    top_idx = tf.reshape(top_idx, [-1, 1])

    transform_prob = tf.scatter_nd(
        top_idx, top_val, tf.constant([self.num_features])
    )
    if self._output_mask:
      return inputs * tf.expand_dims(transform_prob, 0), transform_prob
    else:
      return inputs * tf.expand_dims(transform_prob, 0)

  def fast_infer(self, inputs: tf.Tensor) -> tf.Tensor:
    masked_inputs = tf.gather(inputs, self.trained_top_idx, axis=-1)
    # Only apply weights and bias to the selected features.
    masked_inputs = (
        tf.expand_dims(masked_inputs, -1) * self.trained_feature_weight
    )
    return tf.math.reduce_sum(masked_inputs, axis=-2) + self.feature_bias

  def call(
      self, inputs: tf.Tensor, training: bool = True
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Feature selection module forward pass.

    This call uses the input to learn the feature mask, and zeros out
    non-selected features.

    Args:
      inputs: Tensor of dataset samples.
      training: whether is in training mode.

    Returns:
      Processed input tensor with non-selected features zeroed-out. If mutual_
        information, HSIC, or compute_entropy is enabled, also returns the
        sparse mask.
    """
    if self.use_softmax_mask:
      return self.softmax_mask(inputs)

    # TF requires all conditions to have same type, hence explicit "not None".
    if (
        not training
        and self.reduce_feature_dim
        and self.trained_mask is not None
    ):
      return self.fast_infer(inputs), self.trained_mask

    # Subtract the min for numerical stability (not often needed)
    # norm_mask = self.mask - tf.math.reduce_min(self.mask, keepdims=True)
    if self.num_feature_scaler is not None:
      num_selected_features = self.num_feature_scaler.get_num_features()
    else:
      num_selected_features = self.num_selected_features

    inputs_dtype = inputs.dtype
    scalar = tf.constant(1, dtype=inputs_dtype)
    if num_selected_features and self.scale_before_sparsemax:
      scalar, top_idx = self.compute_sparsifier(
          self.mask, num_selected_features, inputs_dtype
      )
    else:
      top_idx = None

    # Nonlinear mapping with sparsemax
    if self.do_mixed_precision:
      scalar = tf.cast(scalar, self.mask.dtype)

    # Noise to encourage feature selection exploration
    if training:
      mask_perturbation = tf.random.normal(
          shape=tf.shape(self.mask),
          mean=0.0,
          stddev=tf.math.reduce_mean(self.mask)
          * self.mask_perturbation_amplitude,
          dtype=tf.float32,
      )
    else:
      mask_perturbation = 0.0
    mask_argument = self.mask + mask_perturbation
    sparse_mask = (
        mask_argument
        if num_selected_features == self.num_features
        else tfa_activations.sparsemax(mask_argument * scalar)
    )

    # Note that sparsemax throws error if input is float16.
    if self.do_mixed_precision:
      sparse_mask = tf.cast(sparse_mask, inputs.dtype)

    # Subtract the minimum value for numerical stability. Note that sparsemax is
    # invariant to a constant bias.
    self.mask.assign(self.mask - tf.math.reduce_min(self.mask))

    if num_selected_features:
      non_zero_count = tf.cast(
          tf.math.count_nonzero(sparse_mask), dtype=inputs_dtype
      )

      # If the number of selected features is lower than the target, apply
      # temperature scaling to the mask to reduce the sparsity in selection.

      if (
          non_zero_count < tf.constant(num_selected_features, inputs_dtype)
          and training
      ):
        tempered_mask = self.mask * self.mask_temperature
        if self.do_mixed_precision:
          tempered_mask = tf.cast(tempered_mask, self.mask.dtype)
        self.mask.assign(tempered_mask)

      # Trim the number of selected features to the target, by masking out the
      # elements with the lowest coefficient values.
      if not self.scale_before_sparsemax and non_zero_count > tf.constant(
          num_selected_features, inputs_dtype
      ):
        top_k_mask = tf.cast(
            sparse_mask[0, :]
            >= tf.math.top_k(sparse_mask[0, :], num_selected_features)[0][-1],
            inputs_dtype,
        )
        # Filter out the top num_selected_features elements.
        sparse_mask = sparse_mask * tf.expand_dims(top_k_mask, 0)
        # Rescale the mask to add up to 1.
        sparse_mask = sparse_mask / tf.math.reduce_sum(sparse_mask)

    if not training:
      self.sparse_mask = sparse_mask
    masked_inputs = inputs * sparse_mask
    if self.reduce_feature_dim:
      if top_idx is None:
        top_idx = tf.math.top_k(sparse_mask, num_selected_features)[1]
      # Only gather the selected features.
      cur_weight = tf.gather(self.feature_weight, top_idx[0], axis=0)
      masked_inputs = tf.gather(masked_inputs, top_idx[0], axis=-1)
      # Only apply weights and bias to the selected features.
      masked_inputs = tf.expand_dims(masked_inputs, -1) * cur_weight
      masked_inputs = (
          tf.math.reduce_sum(masked_inputs, axis=-2) + self.feature_bias
      )

    if self._output_mask:
      return (masked_inputs, sparse_mask)
    else:
      return masked_inputs

  def save_mask(
      self,
  ):
    """Cache mask after training to accelerate inference."""
    scalar = 1.0
    if self.num_selected_features and self.scale_before_sparsemax:
      scalar, top_idx = self.compute_sparsifier(
          self.mask, self.num_selected_features, tf.float32
      )

    sparse_mask = tfa_activations.sparsemax(self.mask * scalar)
    self.trained_mask = sparse_mask

    if top_idx is None:
      top_idx = tf.math.top_k(sparse_mask, self.num_selected_features)[1]
    # Only gather the selected features.
    self.trained_feature_weight = tf.gather(
        self.feature_weight, top_idx[0], axis=0
    )
    self.trained_top_idx = top_idx[0]

  def print_mask(self, sparse_mask: tf.Tensor):
    """Logs the learned sparse mask."""
    print("Mask pattern:")
    tf.print(sparse_mask, summarize=self.num_features, output_stream=sys.stdout)
    sys.stdout.flush()
    print("Number of non-zero elements:")
    tf.print(tf.math.count_nonzero(sparse_mask), output_stream=sys.stdout)
    sys.stdout.flush()

    # Print correct feature selection rate if ground truth salient features
    # are known
    if (
        self.gt_salient_feature_indices is not None
        and self.gt_salient_feature_indices
    ):
      selected_feature_indices = tf.sparse.from_dense(sparse_mask).indices[:, 1]
      correct_feature_indices = tf.sets.intersection(
          self.gt_salient_feature_indices[None, :],
          selected_feature_indices[None, :],
      )
      correct_feature_rate = tf.cast(
          tf.size(correct_feature_indices), dtype=tf.float32
      ) / tf.cast(tf.size(selected_feature_indices), dtype=tf.float32)

      print("Correct feature selection rate:")
      tf.print(correct_feature_rate, output_stream=sys.stdout)
      sys.stdout.flush()

  def compute_sparsifier(
      self,
      feature_mask: tf.Variable,
      num_selected_features: int,
      inputs_dtype: tf.dtypes.DType,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute scalar multiplier to obtain desired sparsity.

    level with sparsemax.
    Args:
      feature_mask: feature mask used for scaling.
      num_selected_features: number of features to select.
      inputs_dtype: dtype of the inputs.

    Returns:
      Scalar multiplier before sparsemax, and the indices of
      the selected features.
    """
    if num_selected_features == self.num_features:
      # All features are used
      return (
          tf.constant(1, dtype=inputs_dtype),
          tf.expand_dims(tf.range(num_selected_features), 0),
      )
    num_selected_features += 1

    if self.do_mixed_precision:
      feature_mask = tf.cast(feature_mask, inputs_dtype)

    top_elements, top_idx = tf.math.top_k(
        feature_mask, k=min(self.num_features, num_selected_features + 1)
    )
    top_sum = tf.reduce_sum(top_elements)
    top_k_sum = top_sum - top_elements[0][-1]
    top_k_idx = top_idx[:, :-1]
    scalar = tf.constant(1, dtype=inputs_dtype)

    if ((num_selected_features + 1) * top_elements[0][-1] + 1 - top_sum) > 0:
      # Need to increase sparsity
      scalar = tf.math.reciprocal_no_nan(
          top_sum - (num_selected_features + 1) * top_elements[0][-1]
      )

    elif top_k_sum - num_selected_features * top_elements[0][-2] - 1 > 0:
      scalar = tf.math.reciprocal_no_nan(
          top_k_sum - num_selected_features * top_elements[0][-2]
      )

    scalar += self.epsilon
    return scalar, top_k_idx


class NumFeatureScaler:
  """Class to for scaling the number of features."""

  def __init__(
      self,
      x_train: tf.data.Dataset,
      n_epochs: int,
      batch_size: int,
      n_total_features: int,
      target_num_features: int,
      n_feature_updates: int = 5,
      effective_n_step_ratio: float = 0.5,
  ):
    """Allows tempering of the number of features selected.

    Args:
      x_train: training dataset.
      n_epochs: number of epochs.
      batch_size: batch size.
      n_total_features: number of total features.
      target_num_features: final desired number of features.
      n_feature_updates: number of num_selected_features changes.
      effective_n_step_ratio: 0.5 means only temper during the first half of all
        steps.
    """
    total_steps = int(len(x_train) / batch_size * n_epochs)

    self.feature_decrement = np.ceil(
        (n_total_features - target_num_features) / n_feature_updates
    )
    self.n_steps_per_change = np.floor(
        total_steps * effective_n_step_ratio / n_feature_updates
    )
    self.step_counter = 0
    self.target_num_features = target_num_features
    self.n_total_features = n_total_features

  def add_step(self):
    self.step_counter += 1

  def get_num_features(self) -> int:
    cur_num_features = int(
        self.n_total_features
        - self.feature_decrement * self.step_counter // self.n_steps_per_change
    )
    cur_num_features = max(cur_num_features, self.target_num_features)
    return cur_num_features
