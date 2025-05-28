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

"""Constructing the objective functions for multiple pretrain/dowstreams."""

import collections
from typing import Sequence, Set, Type

import jax
from jax import lax
from jax import numpy as jnp
import optax

from imp.max.core import constants
from imp.max.core import utils
from imp.max.data import config as data_config
from imp.max.evaluation import metrics as eval_metrics
from imp.max.optimization import config as opt_config
from imp.max.utils import typing

EPSILON = 1e-6
LARGENUM = 1e10
AggregationMethod = constants.ObjectiveAggregation
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName


def _get_modality_pair_name(modality_1, modality_2):
  return "_".join(sorted([modality_1, modality_2]))


def maybe_expand(inputs,
                 expand_axes,
                 target_rank):
  """Expands a given array along one or more axes if necessary."""

  input_rank = len(inputs.shape)
  if input_rank == target_rank:
    return inputs
  else:
    if isinstance(expand_axes, int):
      expand_axes = [expand_axes]

    if input_rank + len(expand_axes) != target_rank:
      raise ValueError("Invalid expand config")

    return jnp.expand_dims(inputs, expand_axes)


def l2_normalize(inputs, axis = -1):
  """Applies L2 normalization to the inputs."""
  return inputs / (jnp.linalg.norm(inputs, axis=axis, keepdims=True) + EPSILON)


class BaseObjective:
  """Base class for objective functions.

  Attributes:
    name: the name for this objective.
    loss_weight: a global loss weight for this objective.
    dtype: casts inputs to the dtype, if not None.
    aux_metrics: tuple of auxiliary metric functions to calculate on the model
      outputs.
  """

  def __init__(self,
               name,
               loss_weight,
               dtype = None):
    """Initializes the objective."""
    self.name = name
    self.loss_weight = loss_weight
    self.dtype = dtype
    self.aux_metrics = ()

  def __call__(
      self,
      data,
  ):
    """Applies the objective.

    Args:
      data: the data collection that contains inputs/targets/outputs.

    Returns:
      A tuple representing the loss and metrics dict respectively.
    """
    raise NotImplementedError


class ObjectiveAggregator:
  """Objective that aggregates multiple objectives within a dataset.

  Attributes:
    name: the name for this objective.
    loss_weight: a global loss weight for this objective.
    objectives: a sequence of objectives to apply.
    aggregation_method: the type of aggregation to use
    dtype: the dtype of the final loss. If None, defaults to float32.
  """

  def __init__(self,
               name,
               loss_weight,
               objectives,
               aggregation_method,
               dtype = None):
    """Initializes the objective."""
    self.name = name
    self.loss_weight = loss_weight
    self.objectives = objectives
    self.aggregation_method = aggregation_method
    self.dtype = dtype or jnp.float32

    if not objectives:
      raise ValueError("Objectives must not be empty!")

  def __call__(
      self,
      data,
  ):
    """Applies all objectives in sequence.

    Args:
      data: the data collection that contains inputs/targets/outputs.

    Returns:
      A tuple representing the aggregated loss and metrics dict respectively.
    """
    loss = 0.
    metrics = {}

    for objective in self.objectives:
      objective_loss, objective_metrics = objective(data)
      loss += jnp.asarray(objective_loss, dtype=self.dtype)
      metrics.update(objective_metrics)

    if self.aggregation_method == AggregationMethod.MEAN:
      loss /= len(self.objectives)
    elif self.aggregation_method != AggregationMethod.SUM:
      raise ValueError("Unknown loss aggregation method "
                       f"{self.aggregation_method}")

    loss *= self.loss_weight

    return loss, metrics


class BaseSequenceObjective(BaseObjective):
  """Base objective function for sequences of predictions.

  Attributes:
    name: The name for this objective.
    route_key: The route feature name in the data collection or its
      corresponding dataflow.
    predictions_key: The output feature name of the model predictions in the
      data collection or its corresponding dataflow.
    targets_key: The output feature name of the prediction targets in the
      data collection or its corresponding dataflow.
    modality_token_weights: A sequence of flattened dictionary items that maps
      modalities to their token feature names and their corresponding weights
      with which each modality-specific token is included in the final loss
      value. Non-existing pairs or zero-weight items will be excluded from the
      final loss calculation.
      An example input would be (("vision", "token_raw"), 0.9).
    loss_weight: A global loss weight for this objective.
    left_shift_targets: Whether to shift the targets by one token. This is only
      used in the autoregressive decoder training. If set to True, targets are
      shifted one token to the left and predictions are truncated at end by one
      token accordingly (so that predictions predict the next target at each
      position).
    dtype: Casts inputs to the dtype, if not None.
  """

  def __init__(
      self,
      name,
      route_key,
      predictions_key,
      targets_key,
      modality_token_weights,
      loss_weight,
      left_shift_targets = False,
      dtype = None,
  ):
    """Initializes a generic base sequence objective."""
    super().__init__(name=name, loss_weight=loss_weight, dtype=dtype)

    self.route_key = route_key
    self.predictions_key = predictions_key
    self.targets_key = targets_key
    self.left_shift_targets = left_shift_targets

    if not modality_token_weights:
      raise ValueError(
          "`modality_token_weights` must not be empty! Please specify the "
          "modality-token pairs for which the objective should be calculated.")

    modality_token_weights = utils.unflatten_unitemize_dict(
        modality_token_weights)
    for modality in modality_token_weights:
      for token_key in modality_token_weights[modality]:
        if modality_token_weights[modality][token_key] == 0:
          del modality_token_weights[modality][token_key]

    self.modality_token_weights = modality_token_weights

  @property
  def supported_modalities(self):
    """Returns all of the modalities supported under this instance."""
    return set(self.modality_token_weights.keys())

  def supported_modality_token_keys(self, modality):
    """Returns all of the token keys supported under this instance."""
    return set(self.modality_token_weights[modality].keys())

  def _verify_predictions_and_targets(
      self,
      predictions,
      targets,
      targets_mask
      ):
    """Verifies predictions, targets, and targets mask's shape."""
    predictions_temp_axis, targets_temp_axis = self._get_temporal_axes(
        predictions, targets)

    if (predictions.shape[:predictions_temp_axis + 1]
        != targets.shape[:targets_temp_axis + 1]):
      raise ValueError(
          "Targets and predictions should have the same batch/instance/length."
          f" Instead, received {targets.shape[:targets_temp_axis+1]} and "
          f"{predictions.shape[:predictions_temp_axis+1]}, respectively.")

    if (targets_mask is not None
        and targets.shape[:targets_temp_axis + 1] != targets_mask.shape):
      raise ValueError(
          "Targets and targets_mask should have the same batch/instance/length."
          f" Instead, received {targets.shape[:targets_temp_axis+1]} and "
          f"{targets_mask.shape}, respectively.")

  def _get_predictions_and_targets(
      self,
      outputs_collection,
      targets_collection,
      route_key,
      modality,
      token_key,
      predictions_key,
      targets_key,
  ):
    """Extracts the relevant predictions, targets, and target mask.

    Args:
      outputs_collection: The data collection that contains model predictions.
      targets_collection: The data collection that contains prediction targets.
      route_key: The route feature name in the data collection or its
        corresponding dataflow.
      modality: The modality for which we fetch predictions/targets.
      token_key: The token feature name in the data collection or its
        corresponding dataflow.
      predictions_key: The output feature name of the model predictions in the
        data collection or its corresponding dataflow.
      targets_key: The output feature name of the prediction targets in the
        data collection or its corresponding dataflow.

    Returns:
      A tuple representing the predictions, targets, and mask respectively.
    """
    raise NotImplementedError(
        "This is an abstract method. The method should be specified by "
        "inherited classes.")

  def _calculate_loss(self,
                      predictions,
                      targets):
    """Calculates the loss.

    Args:
      predictions: An array of shape [..., num_classes] containing the model
        predictions.
      targets: An array of shape [..., num_classes] containing the ground-truth
        targets.
    Returns:
      A float number containing the loss value according to the objective
      function.
    """
    raise NotImplementedError(
        "This is an abstract method. The loss calculation method "
        "should be specified by inherited classes.")

  def _get_temporal_axes(self,
                         predictions,
                         targets):
    """Gets the temporal axes depending on the target/prediction ranks."""
    predictions_temp_axis = -2
    if predictions.ndim == targets.ndim:
      targets_temp_axis = -2
    elif predictions.ndim == targets.ndim + 1:
      # In this case, the targets do not contain the `dim` axis and the last
      # axis is the temporal axis
      targets_temp_axis = -1
    else:
      raise ValueError(
          "Targets should have a rank in [pred_rank - 1, pred_rank]. "
          f"Instead, received {predictions.ndim=} and {targets.ndim=}.")

    predictions_temp_axis += predictions.ndim
    targets_temp_axis += targets.ndim
    return predictions_temp_axis, targets_temp_axis

  def _left_shift_targets(
      self,
      predictions,
      targets,
      targets_mask
  ):
    """Shifts targets one token left and truncates predictions accordingly."""
    # Sequence features are always assumed to have a shape of (..., length, dim)
    predictions_temp_axis, targets_temp_axis = self._get_temporal_axes(
        predictions, targets)
    length = predictions.shape[predictions_temp_axis]
    predictions_take_indices = lax.iota(jnp.int32, length - 1)
    targets_take_indices = 1 + predictions_take_indices
    precision = "float32" if self.dtype == jnp.float32 else "bfloat16"
    predictions = utils.take_along_axis(inputs=predictions,
                                        indices=predictions_take_indices,
                                        axis=predictions_temp_axis,
                                        precision=precision)
    targets = utils.take_along_axis(inputs=targets,
                                    indices=targets_take_indices,
                                    axis=targets_temp_axis,
                                    precision=precision)
    if targets_mask is not None:
      # Target mask is always assumed to have a shape of (..., length)
      targets_mask = utils.take_along_axis(inputs=targets_mask,
                                           indices=targets_take_indices,
                                           axis=-1,
                                           precision=precision)
    return predictions, targets, targets_mask

  def _get_default_metadata(self):
    """Fetches the defaullt metadata if that is not passed in the data."""
    if any([self.route_key is None,
            self.predictions_key is None,
            self.targets_key is None]):
      raise ValueError(
          "A valid `metadata` was not found in the data collection and the "
          "objective was instantiated with insufficient data flow information: "
          f"{self.route_key=}, {self.predictions_key=}, {self.targets_key=}. "
          "Please make sure you either pass a valid `metadata` or specify a "
          "valid data flow information.")

    dataflow = ()
    for modality in sorted(self.supported_modalities):
      for token_key in sorted(self.supported_modality_token_keys(modality)):
        datapass = (
            {
                DataFeatureType.OUTPUTS: {
                    self.route_key: {
                        modality: {
                            token_key: self.predictions_key,
                        },
                    },
                },
                DataFeatureType.TARGETS: {
                    self.route_key: {
                        modality: self.targets_key,
                    },
                },
            },
        )
        dataflow += datapass
    default_metadata = data_config.Metadata(
        dataflow=dataflow,
        taskflow=(),
    )
    return default_metadata

  def __call__(
      self,
      data
  ):
    """Computes loss value and metrics based on the preds/targets."""

    metrics = {}
    loss = jnp.asarray(0., dtype=self.dtype)
    num_losses = 0

    outputs_collection = data[DataFeatureType.OUTPUTS]
    targets_collection = data[DataFeatureType.TARGETS]
    metadata = data.get(DataFeatureType.METADATA, self._get_default_metadata())
    for dataflow in metadata.dataflow:
      outputs_flow = dataflow[DataFeatureType.OUTPUTS]
      targets_flow = dataflow[DataFeatureType.TARGETS]
      utils.verify_flow_exists_in_data(
          flow=outputs_flow,
          data=outputs_collection,
          flow_leaf_is_feature_name=True,
      )
      utils.verify_flow_exists_in_data(
          flow=targets_flow,
          data=targets_collection,
          flow_leaf_is_feature_name=True,
      )
      # TODO(b/277977414): Develop a Data API to encapsulate these boilerplates
      for route_key in targets_flow:
        for modality in targets_flow[route_key]:
          modality_loss = jnp.asarray(0., dtype=self.dtype)
          for token_key in outputs_flow[route_key][modality]:
            predictions_key = outputs_flow[route_key][modality][token_key]
            targets_key = targets_flow[route_key][modality]
            (predictions,
             targets,
             targets_mask) = self._get_predictions_and_targets(
                 outputs_collection,
                 targets_collection,
                 route_key,
                 modality,
                 token_key,
                 predictions_key,
                 targets_key,
             )

            self._verify_predictions_and_targets(
                predictions, targets, targets_mask)
            if self.dtype is not None:
              predictions = jnp.asarray(predictions, dtype=self.dtype)
              targets = jnp.asarray(targets, dtype=self.dtype)
              if targets_mask is not None:
                targets_mask = jnp.asarray(targets_mask, dtype=self.dtype)

            if self.left_shift_targets:
              predictions, targets, targets_mask = self._left_shift_targets(
                  predictions, targets, targets_mask
              )

            # Loss is calculated independently for each instance and modality.
            modality_token_loss = self._calculate_loss(predictions, targets)
            modality_token_loss = (
                modality_token_loss
                * self.modality_token_weights[modality][token_key]
            )

            # If `targets_mask` is not provided, we assume that the loss should
            # be taken uniformly on every position in the sequence. If more
            # fine-grained loss is needed, such as for padded sequences, pass
            # in `targets_mask`.
            if targets_mask is not None:
              if targets_mask.shape != modality_token_loss.shape:
                raise ValueError(
                    "Shape mismatch between the per-sample loss and the target "
                    f"mask! loss: {modality_token_loss.shape}, mask: "
                    f"{targets_mask.shape}.")
              modality_token_loss = (
                  (modality_token_loss * targets_mask).sum()
                  / (targets_mask.sum() + EPSILON))
            else:
              modality_token_loss = modality_token_loss.mean()

            metrics[f"{self.name}/{modality}/{token_key}/loss"] = (
                modality_token_loss
            )

            for aux_metric in self.aux_metrics:
              aux_metric_name = (
                  f"{self.name}/{modality}/{token_key}/{aux_metric.name}"
              )
              metrics[aux_metric_name] = aux_metric(
                  predictions, targets, mask=targets_mask)

            modality_loss += modality_token_loss
            num_losses += 1

          metrics[f"{self.name}/{modality}/loss"] = modality_loss
          loss += modality_loss

    # normalize total loss wrt number of loss calls (if any)
    if num_losses != 0:
      loss = loss * self.loss_weight / num_losses
    metrics[f"{self.name}/total_loss"] = loss

    return loss, metrics


class BaseCrossEntropy(BaseSequenceObjective):
  """Base Cross Entropy objective function for token sequences.

  Attributes:
    name: The name for this objective.
    route_key: The route feature name in the data collection or its
      corresponding dataflow.
    predictions_key: The output feature name of the model predictions in the
      data collection or its corresponding dataflow.
    targets_key: The output feature name of the prediction targets in the
      data collection or its corresponding dataflow.
    left_shift_targets: Whether to shift the targets by one token. This is only
      used in the autoregressive decoder training. If set to True, targets are
      shifted one token to the left and predictions are truncated at end by one
      token accordingly (so that predictions predict the next target at each
      position).
    modality_token_weights: A sequence of flattened dictionary items that maps
      modalities to their token feature names and their corresponding weights
      with which each modality-specific token is included in the final loss
      value. Non-existing pairs or zero-weight items will be excluded from the
      final loss calculation.
      An example input would be (("vision", "token_raw"), 0.9).
    loss_weight: A global loss weight for this objective.
    dtype: Casts inputs to the dtype, if not None.
    aux_metrics: tuple of auxiliary metric functions to calculate on the model
      outputs.
  """

  def __init__(
      self,
      name,
      route_key,
      predictions_key,
      targets_key,
      left_shift_targets,
      modality_token_weights,
      loss_weight,
      dtype = None,
  ):
    """Initializes a generic base cross entropy objective."""
    super().__init__(
        name=name,
        route_key=route_key,
        predictions_key=predictions_key,
        targets_key=targets_key,
        left_shift_targets=left_shift_targets,
        modality_token_weights=modality_token_weights,
        loss_weight=loss_weight,
        dtype=dtype)

  def _get_predictions_and_targets(
      self,
      outputs_collection,
      targets_collection,
      route_key,
      modality,
      token_key,
      predictions_key,
      targets_key,
  ):
    predictions = outputs_collection[route_key][modality][token_key][
        predictions_key
    ]
    targets = targets_collection[route_key][modality][targets_key]
    targets_mask = targets_collection[route_key][modality].get(
        DataFeatureName.TOKEN_MASK, None
    )
    return predictions, targets, targets_mask


# TODO(b/236642817): add mAP, AUC, and d-prime metrics
class SigmoidBinaryCrossEntropy(BaseCrossEntropy):
  """Sigmoid Binary Cross Entropy objective function for token sequences."""

  def _calculate_loss(
      self, predictions, targets
  ):
    # Note that sigmoid_binary_cross_entropy by default does NOT average over
    # the logits, hence we take an average over the last dim here
    return optax.sigmoid_binary_cross_entropy(predictions, targets).mean(
        axis=-1
    )


class SoftmaxCrossEntropy(BaseCrossEntropy):
  """Softmax Cross Entropy objective function for token sequences."""

  def __init__(
      self,
      name,
      route_key,
      predictions_key,
      targets_key,
      modality_token_weights,
      loss_weight,
      one_hot_targets = True,
      left_shift_targets = False,
      dtype = None,
  ):
    """Initializes the softmax cross entropy objective."""
    super().__init__(
        name=name,
        route_key=route_key,
        predictions_key=predictions_key,
        targets_key=targets_key,
        left_shift_targets=left_shift_targets,
        modality_token_weights=modality_token_weights,
        loss_weight=loss_weight,
        dtype=dtype)
    self.one_hot_targets = one_hot_targets
    self.aux_metrics = (
        eval_metrics.Accuracy(
            top=1,
            average_logits=False,
            one_hot_targets=one_hot_targets
            ).enable_jax_mode(),
    )

  def _calculate_loss(self,
                      predictions,
                      targets):
    # Note that softmax_cross_entropy by default averages over the logits,
    # hence we do not take an average over the last dim here
    if self.one_hot_targets:
      return optax.softmax_cross_entropy(predictions, targets)  # pytype: disable=bad-return-type  # numpy-scalars
    else:
      targets = jnp.asarray(targets, dtype=jnp.int32)
      return optax.softmax_cross_entropy_with_integer_labels(predictions,  # pytype: disable=bad-return-type  # numpy-scalars
                                                             targets)


class MeanSquaredError(BaseSequenceObjective):
  """Mean squared error (MSE) loss objective.

  Attributes:
    name: The name for this objective.
    route_key: The route feature name in the data collection or its
      corresponding dataflow.
    predictions_key: The output feature name of the model predictions in the
      data collection or its corresponding dataflow.
    targets_key: The output feature name of the prediction targets in the
      data collection or its corresponding dataflow.
    modality_token_weights: A sequence of flattened dictionary items that maps
      modalities to their token feature names and their corresponding weights
      with which each modality-specific token is included in the final loss
      value. Non-existing pairs or zero-weight items will be excluded from the
      final loss calculation.
      An example input would be (("vision", "token_raw"), 0.9).
    loss_weight: A global loss weight for this objective.
    dtype: Casts inputs to the dtype, if not None.
    aux_metrics: tuple of auxiliary metric functions to calculate on the model
      outputs.
    z_score_predictions: If True, the predictions will be normalized using
      z-score (normalize to mean and std).
    z_score_targets: If True, the target (groundtruth) features will be
      normalized using z-score (normalize to mean and std).
  """

  def __init__(
      self,
      name,
      route_key,
      predictions_key,
      targets_key,
      modality_token_weights,
      loss_weight,
      z_score_predictions = False,
      z_score_targets = False,
      left_shift_targets = False,
      dtype = None,
  ):
    """Initializes a generic base cross entropy objective."""
    super().__init__(
        name=name,
        route_key=route_key,
        predictions_key=predictions_key,
        targets_key=targets_key,
        modality_token_weights=modality_token_weights,
        left_shift_targets=left_shift_targets,
        loss_weight=loss_weight,
        dtype=dtype)
    self.z_score_predictions = z_score_predictions
    self.z_score_targets = z_score_targets

  def _z_score(self, array):
    """Normalizes an array to its mean and std (aka z-scoring)."""
    mean = array.mean(axis=-1, keepdims=True)
    std = jnp.std(array, axis=-1, keepdims=True)
    array = (array - mean) / (std + EPSILON)
    return array

  def _get_predictions_and_targets(
      self,
      outputs_collection,
      targets_collection,
      route_key,
      modality,
      token_key,
      predictions_key,
      targets_key,
  ):
    predictions = outputs_collection[route_key][modality][token_key][
        predictions_key
    ]
    targets = targets_collection[route_key][modality][targets_key]
    targets_mask = targets_collection[route_key][modality].get(
        DataFeatureName.TOKEN_MASK, None)

    if self.z_score_predictions:
      predictions = self._z_score(predictions)

    if self.z_score_targets:
      targets = self._z_score(targets)

    return predictions, targets, targets_mask

  def _calculate_loss(self,
                      predictions,
                      targets):
    return jnp.square(predictions - targets).mean(axis=-1)


class CrossModalNCE(BaseObjective):
  """Cross-Modal NCE objective function.

  Attributes:
    name: The name for this objective.
    hparams_route_key: The feature route of the outputs that contains the
      model hyperparameters collection.
    modality_pair_weights: A dictionary that maps pairs of modalities to weights
      with which each pair is included in the final loss value. Non-existing
      pairs or zero-weight items will be excluded from the final loss
      calculation. An example input would be (("vision", "text"): 0.9).
    temperature: The temperature of the NCE objective function.
    margin: The margin (bias) in the NCE objective function.
    loss_weight: A global loss weight for this objective.
    dtype: Casts inputs to the dtype, if not None.
    cross_modal_weights: A mapping between the (sorted) modality pair names and
      their corresponding weight. An example would be {"text_vision": 0.9}.
  """

  def __init__(
      self,
      name,
      hparams_route_key,
      modality_pair_weights,
      temperature,
      margin,
      loss_weight,
      dtype = None,
  ):
    super().__init__(name=name, loss_weight=loss_weight, dtype=dtype)
    if not modality_pair_weights:
      raise ValueError(
          "`modality_pair_weights` must not be empty! Please specify the "
          "modality pairs for which the objective should be calculated.")

    cross_modal_weights = {}
    for (modality_1, modality_2), weight in modality_pair_weights:
      pair_name = _get_modality_pair_name(modality_1, modality_2)
      if weight > 0.:
        cross_modal_weights[pair_name] = weight

    self.cross_modal_weights = cross_modal_weights
    self.hparams_route_key = hparams_route_key
    self.temperature = temperature
    self.margin = margin

  def _diagonal_accuracy(self,
                         features,
                         axis = 0):
    """Calculates the top-1 accuracy of features along the diagonal.

    This function is optimized for efficiency on large batch sizes to avoid
    slowdowns during training.

    Args:
      features: an array with shape (batch_size, batch_size, num_examples) that
        represents the dot product between all N^2 features in a batch.
      axis: the axis to perform the accuracy calculation on. Can either be 0
        or 1, representing row or column accuracy.

    Returns:
      The accuracy metric, a scalar value representing the ratio of values that
      were highest along the features' diagonal.
    """
    if not 0 <= axis <= 1:
      raise ValueError(f"Axis should be in (0, 1), got {axis}")

    # features should be of shape (B, B, N).
    b, b2, n = features.shape
    assert b == b2

    preds = jnp.argmax(features, axis)
    labels = lax.broadcasted_iota(jnp.int32, (b, n), 0)

    return jnp.asarray(preds == labels, features.dtype).mean()

  def _calculate_nce(
      self,
      modality_1,
      modality_2,
      temperature,
  ):
    """Calculate modality_1 vs. modality_2 pair-wise similarities.

    Args:
      modality_1: a batch of vectors for modality 1.
      modality_2: a batch of vectors for modality 2.
      temperature: a scalar for the temperature value.

    Returns:
      The pair of similarities and accuracies of modality_1 vs. modality_2
      and modality_2 vs. modality_1.
    """

    if self.dtype is not None:
      modality_1 = jnp.asarray(modality_1, dtype=self.dtype)
      modality_2 = jnp.asarray(modality_2, dtype=self.dtype)
      temperature = jnp.asarray(temperature, dtype=self.dtype)

    # Ensure temperature is nonzero to avoid potential NaNs.
    temperature += EPSILON

    # normalize embeddings
    modality_1 = l2_normalize(modality_1, axis=-1)  # (B, N1, D)
    modality_2 = l2_normalize(modality_2, axis=-1)  # (B, N2, D)

    # calculate cross-modal similarities for all samples -> (B, B, N1*N2)
    m1_vs_m2 = jnp.einsum("bmd,cnd->bcmn", modality_1, modality_2)
    m1_vs_m2 = m1_vs_m2.reshape(m1_vs_m2.shape[:2] + (-1,))

    # TODO(hassanak): add mask support
    # only keep positive pairs -> (B, N1*N2)
    sim_pos = jnp.einsum("bbn->bn", m1_vs_m2)

    # calculate the log sum exp (numerator) of the NCE loss.
    logsumexp_pos = jax.scipy.special.logsumexp(
        (sim_pos / temperature) - self.margin,
        axis=1,
    )  # (B, )

    # calculate the log sum exp (denominator) of the NCE loss.
    logsumexp_all_m1_vs_m2 = jax.scipy.special.logsumexp(
        m1_vs_m2 / temperature,
        axis=[1, 2],
    )  # (B, )
    logsumexp_all_m2_vs_m1 = jax.scipy.special.logsumexp(
        m1_vs_m2 / temperature,
        axis=[0, 2],
    )  # (B, )

    # calculate the loss.
    loss_m1_vs_m2 = logsumexp_all_m1_vs_m2 - logsumexp_pos
    loss_m2_vs_m1 = logsumexp_all_m2_vs_m1 - logsumexp_pos

    # average across batch samples
    loss_m1_vs_m2 = loss_m1_vs_m2.mean()
    loss_m2_vs_m1 = loss_m2_vs_m1.mean()

    # Calculate aux accuracy value across the batch.
    accuracy_m1_vs_m2 = self._diagonal_accuracy(m1_vs_m2, axis=0)
    accuracy_m2_vs_m1 = self._diagonal_accuracy(m1_vs_m2, axis=1)

    return loss_m1_vs_m2, loss_m2_vs_m1, accuracy_m1_vs_m2, accuracy_m2_vs_m1

  def __call__(
      self,
      data
  ):

    # Fetch outputs and hyperparams collections
    outputs_collection = data[DataFeatureType.OUTPUTS]
    common_space_collection = outputs_collection[DataFeatureRoute.COMMON_SPACE]
    hyperparams_collection = data.get(DataFeatureType.HYPERPARAMS, {})
    hyperparams_collection = hyperparams_collection.get(
        self.hparams_route_key, {})

    # Define what pairs of modalities could be used for cross-modal NCE
    paired_common_space = collections.defaultdict(
        collections.defaultdict(list).copy
        )
    for modality in common_space_collection:
      source_token_keys = set(common_space_collection[modality].keys())
      if len(source_token_keys) != 1:
        raise NotImplementedError
      else:
        source_token_key = source_token_keys.pop()
      common_space = common_space_collection[modality][source_token_key]
      for target_modality in common_space:
        pair_name = _get_modality_pair_name(modality, target_modality)
        paired_common_space[pair_name]["embeddings"].append(
            maybe_expand(common_space[target_modality], 1, 3)
        )
        paired_common_space[pair_name]["modalities"].append(
            target_modality
        )

    # Fetch the present common space modalities
    common_space_modalities = set(paired_common_space.keys())

    # Fetch the present objective modalities
    objective_modalities = set(self.cross_modal_weights.keys())

    # Check if the objective modalities exist in the common space
    if not objective_modalities.issubset(common_space_modalities):
      raise ValueError(
          "Some or all of the requested modalities do not exist in the common"
          f" space. The objective is configured with {objective_modalities=} "
          f"while {common_space_modalities=}."
      )

    # Fetch the modalities-of-interest for calculating the objective value
    valid_modalities = objective_modalities.intersection(
        common_space_modalities)

    # Temperature value can be passed by the model/data
    temperatures = hyperparams_collection.get(DataFeatureName.TEMPERATURE, {})

    # then calculate cross-modal NCE for pairs of modalities
    loss = jnp.asarray(0., dtype=self.dtype)
    metrics = {}

    # Use sorting for a deterministic ordering across hosts.
    for pair_name in sorted(list(valid_modalities)):
      embeddings = paired_common_space[pair_name]["embeddings"]
      modalities = paired_common_space[pair_name]["modalities"]
      if len(embeddings) != 2:
        raise ValueError(
            f"One modality is missing in {pair_name} common space!"
        )

      # Get the modality-specific temperature, or use the default static value.
      temperature = temperatures.get(pair_name, self.temperature)
      metrics[f"temperature/{pair_name}"] = jnp.asarray(temperature,
                                                        dtype=self.dtype)

      (loss_m1_vs_m2, loss_m2_vs_m1, accuracy_m1_vs_m2,
       accuracy_m2_vs_m1) = self._calculate_nce(
           modality_1=embeddings[0],
           modality_2=embeddings[1],
           temperature=temperature)

      loss_m1_vs_m2 *= self.cross_modal_weights[pair_name]
      loss_m2_vs_m1 *= self.cross_modal_weights[pair_name]

      loss += (loss_m1_vs_m2 + loss_m2_vs_m1) / 2

      # store losses in metrics to log
      name_m1_vs_m2 = f"{modalities[0]}_vs_{modalities[1]}"
      name_m2_vs_m1 = f"{modalities[1]}_vs_{modalities[0]}"
      metrics[f"{self.name}/{name_m1_vs_m2}"] = loss_m1_vs_m2
      metrics[f"{self.name}/{name_m2_vs_m1}"] = loss_m2_vs_m1
      metrics[f"{self.name}/{name_m1_vs_m2}/top_1"] = accuracy_m1_vs_m2
      metrics[f"{self.name}/{name_m2_vs_m1}/top_1"] = accuracy_m2_vs_m1

    loss = (loss / len(valid_modalities)) * self.loss_weight
    metrics[f"{self.name}/total_loss"] = loss

    return loss, metrics


_OBJECTIVE_MAP: dict[str, Type[BaseObjective] | Type[ObjectiveAggregator]] = {
    constants.Objective.OBJECTIVE_AGGREGATOR: ObjectiveAggregator,
    constants.Objective.CROSS_MODAL_NCE: CrossModalNCE,
    constants.Objective.SOFTMAX_CROSS_ENTROPY: SoftmaxCrossEntropy,
    constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY: SigmoidBinaryCrossEntropy,
    constants.Objective.MEAN_SQUARED_ERROR: MeanSquaredError,
}


def fetch_objective_cls(objective_name):
  """Fetches the objective class from the objective name."""
  if objective_name not in _OBJECTIVE_MAP:
    raise NotImplementedError(f"Unknown objective {objective_name!r}")
  return _OBJECTIVE_MAP[objective_name]


def get_objective(
    objective_configs,
):
  """Returns a sequence of objective functions."""

  cached_instances = {}
  all_objectives = ()
  for objective_config in objective_configs:
    objective_name = objective_config.name
    config_dict = objective_config.as_dict()

    # We re-use config instances with the exact same contents to avoid
    # re-computing objectives when they are later instantiated in the graph.
    unique_config_id = hash(tuple(objective_config.as_flat_dict().items()))

    if isinstance(objective_config, opt_config.ObjectiveAggregator):
      for objective in objective_config.objectives:
        if isinstance(objective, opt_config.ObjectiveAggregator):
          raise ValueError("Nesting objectives is discouraged. Please "
                           "flatten them into a single sequence instead.")
      config_dict["objectives"] = tuple(
          fetch_objective_cls(objective.name)(**objective.as_dict())
          for objective in objective_config.objectives
      )

    if unique_config_id not in cached_instances:
      cached_instances[unique_config_id] = (
          fetch_objective_cls(objective_name)(**config_dict)
          )

    # Concatenate all objective functions together.
    all_objectives += (cached_instances[unique_config_id],)

  if not all_objectives:
    raise ValueError(f"No supported objectives provided in {objective_configs}")

  return all_objectives
