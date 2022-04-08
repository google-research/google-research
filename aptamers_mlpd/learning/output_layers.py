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

"""Output layers (including loss functions and predictions) for aptamer models.

Note that throughout this module, 'target' refers to the target output, i.e.
a count within a pool of sequenced DNA. In tensors holding this information, the
axis is labeled 'target'.

For clarity, the selection target molecule (e.g the protein used during SELEX),
although it is a called a target in the experiment_proto is called an
affinity molecule or just molecule in this module. In tensors holding this
information, the axis is labeled 'affinity'.

Optionally, the output layer can also predict additional output, such as
parititon function of the minimal-free-engery secondary structure. In tensors
holding this information, the axis is labeld 'additional_output'.

For each output layer, there is a function called predict_counts which predicts
a value for each item in the 'target' axis and there is function called
predict_affinity which predicts a value for each affinity molecule in the
selection experiment. There is also a function called predict_additional_output
which predicts a value for each additional output. The training data (called
'outputs' in this module ) is counts in sequenced DNA pools so the 'target'
axis, or a subset, is used in training.

For each output layer, there are three functions related with loss calculation.
The function affinity_loss_per_example_and_target calculates the loss for each
sample on each of the affinity predictions. The function
loss_per_example_per_target calculates the loss for each sample on each of the
counts and additional output predictions, and when array data exist, also
calculates the affinity loss using affinity_loss_per_example_and_target.
The function avg_loss_per_target first calculates per-sample loss using
loss_per_example_per_target and returns the average across samples. If array
data exist, when calculating the average it ignores the count loss for samples
with all-zero count, and the affinity loss for samples with all-zero array data.

The FullyObserved model logits are the target outputs and optionally additioanl
output and the affinities are calculated from these logits (e.g. the value in
the final round).

The LatentAffinity model and its derivatives have logits for the affinity to
each affinity molecule and optionally additional output, and the target
predictions are calculated from the affinity value.
"""

import collections
import math

import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.contrib import labeled_tensor as lt

from ..util import selection
from .learning import config
from ..learning import utils


# names of output layer components; use these in preference to string literals
NORM_STANDARDIZE = 'STANDARDIZE'
NORM_BINARIZE = 'BINARIZE'
NORM_TOTAL_COUNTS = 'TOTAL_COUNTS'
NORM_SKIP = 'SKIP'

LOSS_SQUARED_ERROR = 'SQUARED_ERROR'
LOSS_CROSS_ENTROPY = 'CROSS_ENTROPY'
LOSS_POISSON_LOSS = 'POISSON_LOSS'
LOSS_ZERO_TRUNCATED_POISSON_LOSS = 'ZERO_TRUNCATED_POISSON_LOSS'

OUTPUT_FULLY_OBSERVED = 'FULLY_OBSERVED'
OUTPUT_LATENT_AFFINITY = 'LATENT_AFFINITY'
OUTPUT_LATENT_WITH_DEPS = 'LATENT_WITH_DEPS'
OUTPUT_LATENT_WITH_PRED_DEPS = 'LATENT_WITH_PRED_DEPS'
OUTPUT_LATENT_WITH_CROSS_DEPS = 'LATENT_WITH_CROSS_DEPS'

TARGETS_ALL_OUTPUTS = 'ALL_OUTPUTS'


class Error(Exception):
  pass


def standardizer(means, stddevs, log_transform=True):
  """Create a function to standardize counts data.

  Args:
    means: LabeledTensor with a 'target' axis giving the mean for each target.
    stddevs: LabeledTensor with a 'target' axis giving the std. dev. for each
      target.
    log_transform: optional boolean indicating whether or not we want to log
      transform counts before standardizing them.

  Returns:
    Function that maps a LabeledTensor of counts into a LabeledTensor with mean
    0 and standard deviation 1.
  """

  def transform(counts):
    if log_transform:
      counts = lt.log(1.0 + counts)
    selection_dict = {'target': list(counts.axes['target'].labels)}
    aligned_means = lt.select(means, selection_dict)
    aligned_stddevs = lt.select(stddevs, selection_dict)
    return (counts - aligned_means) / aligned_stddevs

  return transform


def binarizer(threshold):
  """Create a function to binarize counts data.

  Args:
    threshold: threshold (>=) to use when binarizing counts as True/False.

  Returns:
    Function that maps a LabeledTensor of counts into a LabeledTensor of zeros
    and ones.
  """

  def transform(counts):
    return lt.cast(counts >= float(threshold), tf.float32)

  return transform


def total_counts_normalizer(total_counts):
  """Create a function to normalize counts by total counts.

  Args:
    total_counts: LabeledTensor with a 'target' axis giving total counts for
      each round.

  Returns:
    Function that maps a LabeledTensor of counts into a float LabeledTensor
    giving the fraction of the entire round given by each count.
  """

  def transform(counts):
    aligned_counts = lt.select(
        total_counts, {'target': list(counts.axes['target'].labels)})
    return counts / lt.cast(aligned_counts, tf.float32)

  return transform


def _get_measurement_statistics(experiment_proto, statistic):
  """Extract measurement statistics from a selection_pb2.Experiment.

  Args:
    experiment_proto: selection_pb2.Experiment proto with saved measurement
      statistics.
    statistic: string giving the name of the statistic to pull out of the proto.

  Returns:
    LabeledTensor giving the value of the desired statistic for each output.
  """
  mapping = selection.extract_measurement_statistic(experiment_proto, statistic)
  output_names, data = list(zip(*sorted(mapping.items())))
  return lt.constant(data, axes=[('target', list(output_names))])


def normalizer(method,
               experiment_proto,
               standardize_log_transform=True,
               binarize_threshold=None):
  """Create a normalizer function.

  Args:
    method: one of {NORM_STANDARDIZE, NORM_BINARIZE, NORM_TOTAL_COUNTS,
      NORM_SKIP}.
    experiment_proto: selection_pb2.Experiment describing the experiment.
    standardize_log_transform: optional boolean indicating whether or not we
      want to log transform counts before standardizing them. Only relevant if
      you use NORM_STANDARIZE.
    binarize_threshold: optional integer giving the count threshold at which to
      binarize. Only relevant if you use NORM_BINARIZE.

  Returns:
    Function that maps LabeledTensor with unnormalized output values into
    another LabeledTensor with normalized outputs.

  Raises:
    ValueError: if the normalize method is unrecognized.
  """
  if method == NORM_STANDARDIZE:
    if standardize_log_transform:
      mean_statistic = 'mean_log_plus_one'
      std_dev_statistic = 'std_dev_log_plus_one'
    else:
      mean_statistic = 'mean'
      std_dev_statistic = 'std_dev'
    means = _get_measurement_statistics(experiment_proto, mean_statistic)
    stddevs = _get_measurement_statistics(experiment_proto, std_dev_statistic)
    return standardizer(means, stddevs, standardize_log_transform)
  elif method == NORM_BINARIZE:
    return binarizer(binarize_threshold)
  elif method == NORM_TOTAL_COUNTS:
    total_counts = _get_measurement_statistics(experiment_proto, 'total_depth')
    return total_counts_normalizer(total_counts)
  elif method == NORM_SKIP:
    return lambda x: x
  else:
    raise ValueError('unknown method for normalizer: %r' % method)


class AbstractLoss:
  """Training loss for count data."""

  def __init__(self, normalize):
    """Initialize a loss function.

    Args:
      normalize: callable to use for normalizing counts.
    """
    self.normalize = normalize

  def per_example_and_target(self, preds, targets):
    """Calculate loss per example and target output.

    Args:
      preds: Tensor with dtype=float32 and shape (mbsz, net.output_dims).
      targets: Dictionary with Tensor keys with dtype=int and shape
        (mbsz,) listing all targets.

    Returns:
      Tensor with shape (mbsz, n_targets).
    """
    with tf.name_scope('normalize'):
      normalized_targets = self.normalize(targets)

    with tf.name_scope('loss'):
      loss = self._calculate(preds, normalized_targets)

    return loss

  def per_example_and_target_array(self, preds, targets):
    """Calculate affinity loss per example and target output.

    For now we don't normalize microarray data (targets) when calculating
    affinity loss.

    Args:
      preds: Tensor with dtype=float32 and shape (mbsz, net.output_dims).
      targets: Dictionary with Tensor keys with dtype=int and shape
        (mbsz,) listing all targets.

    Returns:
      Tensor with shape (mbsz, n_targets).
    """
    with tf.name_scope('loss'):
      loss = self._calculate(preds, targets)
    return loss

  def _calculate(self, preds, targets):
    """Calculating the loss per example/output from transformed counts.

    Args:
      preds: Tensor with dtype=float32 and shape (mbsz, net.output_dims).
      targets: Tensor with dtype=float32 and shape (mbsz, n_rounds).

    Returns:
      Tensor with shape (mbsz, n_output_rounds) and dtype=float32.
    """
    raise NotImplementedError


class SquaredError(AbstractLoss):
  """Loss function that uses squared error loss."""

  def _calculate(self, preds, targets):
    """See method on base class."""
    return lt.square(targets - preds)


class CrossEntropy(AbstractLoss):
  """Loss function that uses cross entropy loss."""

  def _calculate(self, preds, targets):
    """See method on base class."""
    return lt.nn.sigmoid_cross_entropy_with_logits(preds, targets)


class PoissonLoss(AbstractLoss):
  """Loss function based on the log-likelihood for a Poisson distribution.

  You should *not* normalize target counts before passing them to PoissonLoss.
  Use loss_norm=NORM_SKIP.

  TODO(shoyer): consider refactoring to avoid the need to define a normalizer
  for each loss even if it doesn't use it.
  """

  def _calculate(self, preds, targets):
    """See method on base class."""
    # TODO(shoyer): replace this with lt.nn.log_poisson_loss when that exists
    loss = tf.nn.log_poisson_loss(targets, preds)
    return lt.LabeledTensor(loss, targets.axes)


def zero_truncated_log_poisson_loss(
    targets, log_input, compute_full_loss=False, name=None):
  """Calculate log-loss for a zero-truncated Poisson distribution.

  See tf.nn.log_poisson_loss for details and sanity checks.

  Args:
    targets: A `Tensor` of the same type and shape as `log_input`.
    log_input: A `Tensor` of type `float32` or `float64`.
    compute_full_loss: whether to compute the full loss. If false, a constant
      term is dropped in favor of more efficient optimization.
    name: optional name for this op.

  Returns:
    A `Tensor` of the same shape as `log_input` with the componentwise
    logistic losses.
  """
  with tf.name_scope(
      name, 'ZeroTruncatedLogPoissonLoss', [targets, log_input]) as scope:
    targets = tf.convert_to_tensor(targets, name='targets')
    log_input = tf.convert_to_tensor(log_input, name='log_input')

    input_ = tf.exp(log_input)
    zeros = tf.zeros_like(targets)

    # We switch to a different formula to avoid numerical stability problems
    # when log_input is small (see the go link above for details).
    approximate = log_input < -5.0
    # Use the nested-tf.where trick (go/tf-where-nan) to avoid NaNs in the
    # gradient. The value when approximate is True is arbitrary:
    input2 = tf.where(approximate, 1.0 + zeros, input_)
    result = tf.where(approximate,
                      0.5 * input_ - (targets - 1.0) * log_input,
                      input_ - targets * log_input + tf.log1p(-tf.exp(-input2)))

    # The zero-truncated Poisson distribution isn't meaningfully defined for
    # targets less than one (i.e., zero):
    result = tf.where(targets < 1.0, np.nan + zeros, result)

    if compute_full_loss:
      # this is the same approximation used in tf.nn.log_poisson_loss
      stirling_approx = (targets * (tf.log(targets) - 1.0) +
                         0.5 * tf.log((2 * math.pi) * targets))
      result += tf.where(targets <= 1.0, zeros, stirling_approx)

    return tf.identity(result, name=scope)


class ZeroTruncatedPoissonLoss(AbstractLoss):
  """Loss function based on a zero-truncated Poisson distribution.

  As for PoissonLoss, use loss_norm=NORM_SKIP here.
  """

  def _calculate(self, preds, targets):
    """See method on base class."""
    loss = zero_truncated_log_poisson_loss(targets, preds)
    return lt.LabeledTensor(loss, targets.axes)


class AbstractOutputLayer:
  """Combined output layer and loss function.

  Output layers map LabeledTensors with the axes `[batch_axis, logit_axis]`
  into a training loss (for each target in `target_axis`) and predictions for
  some or all outputs.

  Attributes:
    loss: AbstractLoss subclass indicating the loss to use for training.
    logit_axis: labeled_tensor.Axis labeling inputs to the output layer.
    target_axis: labeled_tensor.Axis labeling the output layer
      predictions/targets used in computing the loss.
    params: List of tf.Variable objects to optimize.
  """

  def average_loss_per_target(self, logits, outputs, include_array=True):
    """Calculate averaged over examples.

    This is the loss to use for training. If affinity loss is calculated and
    "include_array" is set to True, the count loss for the novel sequences
    included in the microarray and the affinity loss for the sequences not
    included in the microarray are excluded from the average loss calculation.
    Otherwise, return the average count loss over all samples.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].
      outputs: LabeledTensor with dtype=float32 and axes [batch, output_axis].
      include_array: Optional boolean variable indicating whether to also
                     compute affinity loss against binding array data.

    Returns:
      LabeledTensor with type=float32 with axes [output_axis].
    """
    # should be independent of mini-batch size
    loss_matrix = self.loss_per_example_and_target(logits,
                                                   outputs,
                                                   include_array)

    if bool(set(self.binding_arrays_map.keys()) &
            set(outputs.axes['output'].labels)) and include_array:
      count_loss = lt.select(loss_matrix,
                             {'target': list(self.target_axis.labels)})
      # Only the count loss for the samples with at least one non-zero
      # count output will be kept.
      loss_matrix_keep_idx = lt.reduce_any(lt.not_equal(
          lt.select(outputs, {'output': list(self.target_axis.labels)})
          , 0.0), 'output')
      loss_matrix_keep = lt.boolean_mask(count_loss, loss_matrix_keep_idx)
      reduce_loss_matrix = utils.reduce_nanmean(loss_matrix_keep, 'batch')

      affinity_loss = lt.select(
          loss_matrix, {'target': list(self.binding_arrays_map.keys())})
      # Only the affinity loss for the samples with at least one non-zero
      # affinity output wil be kept.
      affinity_loss_keep_idx = lt.reduce_any(
          lt.not_equal(
              lt.select(outputs,
                        {'output': list(self.binding_arrays_map.keys())}), 0.0),
          'output')
      affity_loss_keep = lt.boolean_mask(affinity_loss, affinity_loss_keep_idx)
      reduce_affity_loss = utils.reduce_nanmean(affity_loss_keep, 'batch')
      # Count loss and affinity loss are concatenated
      avg_loss = lt.concat([reduce_loss_matrix, reduce_affity_loss], 'target')

      # Only the additional output loss for the samples with at least one
      # non-zero output value wil be kept.
      if self.additional_output_axis:
        ao_labels = list(self.additional_output_axis.labels)
        af_loss = lt.select(loss_matrix, {'target': ao_labels})
        af_loss_keep_idx = lt.reduce_any(
            lt.not_equal(lt.select(outputs, {'output': ao_labels}), 0.0),
            'output')
        af_loss_keep = lt.boolean_mask(af_loss, af_loss_keep_idx)
        reduce_af_loss = utils.reduce_nanmean(af_loss_keep, 'batch')
        avg_loss = lt.concat([avg_loss, reduce_af_loss], 'target')

    else:
      avg_loss = utils.reduce_nanmean(loss_matrix, 'batch')

    return avg_loss

  def loss_per_example_and_target(self, logits, outputs, include_array=True):
    """Calculate loss per example and output prediction.

    Our evaluation script uses the result of this function to calculate various
    informative summaries of the loss.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].
      outputs: LabeledTensor with dtype=float32 and axes [batch, output_axis].
        These outputs should include everything from the preprocessing, whether
        it is used in the loss or not.
      include_array: optional boolean for whether to calculate affinity loss

    Returns:
      LabeledTensor with dtype=float32 and axes [batch, target_axis] giving
      loss for each target.
    """
    raise NotImplementedError

  def affinity_loss_per_example_and_target(self, logits, outputs):
    """Calculate loss per example on predicting affinity.

    This calls "predict_affinity" which assumably has been implemented in the
    current output layer to predict affinity, and calculates the loss against
    the array output.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].
      outputs: LabeledTensor with dtype=float32 and axes [batch, output_axis].
        These outputs should include everything from the preprocessing, whether
        it is used in the loss or not.

    Returns:
      LabeledTensor with dtype=float32 and axes [batch, target_axis] giving
      loss for each target.
    """
    affinity_pred = _affinities_to_binding_arrays(self.binding_arrays_map,
                                                  self.predict_affinity(logits))

    affinity_pred = lt.rename_axis(affinity_pred, 'output', 'target')
    array_output = lt.rename_axis(
        lt.select(outputs, {'output': list(self.binding_arrays_map.keys())}),
        'output', 'target')

    return self.loss.per_example_and_target_array(affinity_pred, array_output)

  # TODO(shoyer): consider adding probabilistic versions of these prediction
  # methods (adding another dimension to the output tensors for n_samples) if we
  # get around to making models that explicitly model uncertainty.

  def predict_outputs(self, logits, outputs=None):
    """Predict a score that should correlate with each output.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].
      outputs: optional LabeledTensor with dtype=float32 and axes [batch,
        output_axis]. Note that different output layers may not be directly
        comparable if they make sure of `outputs` from prior rounds of selection
        in predictions.

    Returns:
      LabeledTensor with dtype=float32 and axes [batch, output_axis] giving
      predictions for each count and binding array.
    """
    predicted_counts = lt.rename_axis(
        self.predict_counts(logits, outputs), 'target', 'output')

    if self.binding_arrays_map:
      predicted_affinity = self.predict_affinity(logits)
      predicted_binding_arrays = lt.pack([
          lt.select(predicted_affinity, {'affinity': target})
          for target in self.binding_arrays_map.values()
      ], ('output', list(self.binding_arrays_map.keys())),
                                         axis_position=1)
      preds = lt.concat([predicted_counts, predicted_binding_arrays], 'output')
    else:
      preds = predicted_counts

    if self.additional_output_axis:
      predicted_additional_output = lt.rename_axis(
          self.predict_additional_output(logits), 'target', 'output')
      preds = lt.concat([preds, predicted_additional_output], 'output')
    return preds

  def predict_counts(self, logits, outputs=None):
    """Make count predictions from logits and counts.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].
      outputs: optional LabeledTensor with dtype=float32 and axes [batch,
        output_axis]. Not all `predict_count` methods require outputs. It is the
        responsibility of implementations that do use `outputs` to ensure that
        this method respects the casual structure of the experiment.

    Returns:
      preds: LabeledTensor with dtype=float32 and axes [batch, target_axis].
    """
    raise NotImplementedError

  def predict_affinity(self, logits):
    """Predict a score that should correlate with affinity.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].

    Returns:
      LabeledTensor with dtype=float32 and axes [batch, affinity_axis] giving
      predictions for each affinity.
    """
    raise NotImplementedError

  def predict_additional_output(self, logits):
    """Make predictions on auxiliary information such as secondary structure.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].

    Returns:
      LabeledTensor with dtype=float32 and axes [batch, target_axis] giving
      predictions for each additional output.
    """
    raise NotImplementedError


def _targets_from_outputs(outputs, target_axis):
  selected = lt.select(outputs, {'output': list(target_axis.labels)})
  targets = lt.reshape(selected, ['output'], [target_axis])
  return targets


def _binding_arrays_map(experiment_proto):
  result = collections.OrderedDict()
  for array in experiment_proto.binding_arrays:
    if len(array.target_concentrations) != 1:
      raise ValueError('can only process binding arrays with a single target: '
                       '%r' % array)
    target, = list(array.target_concentrations.keys())
    result[array.name] = target
  return result


def _affinities_to_binding_arrays(binding_arrays_map, affinities):
  return lt.pack([
      lt.select(affinities, {'affinity': target})
      for target in binding_arrays_map.values()
  ], ('output', list(binding_arrays_map.keys())),
                 axis_position=1)


def get_target_names(experiment_proto, target_names=None):
  """Get or validate target names for training.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    target_names: optional list of strings giving target names to train
      against. By default, use all output counts.

  Returns:
    List of strings giving target names.

  Raises:
    ValueError: if any target_names are not output counts.
  """
  output_count_names = selection.non_input_count_names(experiment_proto)
  if target_names is None or TARGETS_ALL_OUTPUTS in target_names:
    target_names = output_count_names
  else:
    invalid_targets = set(target_names) - set(output_count_names)
    if invalid_targets:
      raise ValueError('invalid target names: %r' % list(invalid_targets))
  return sorted(target_names)


def get_additional_output_names(experiment_proto, ao_names=None):
  """Get or validate additional output names for training.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment.
    ao_names: optional list of strings giving additional output names to train
      against. By default, use all additional output in the experiment proto.

  Returns:
    List of strings giving additional output names.

  Raises:
    ValueError: if any ao_names are not found in the experiment proto.
  """
  ao_names_in_proto = selection.all_additional_output_names(experiment_proto)
  if ao_names is None:
    ao_names = ao_names_in_proto
  else:
    invalid_ao = set(ao_names) - set(ao_names_in_proto)
    if invalid_ao:
      raise ValueError('invalid additional output '
                       'names: %r' % list(invalid_ao))
  return sorted(ao_names)


class FullyObserved(AbstractOutputLayer):
  """Output layer and loss for the fully observed model with SSE loss.

  In this model, we directly predict observed standardized and optionally
  log-transformed counts.

  The average of the network output for the last two rounds is used as the
  predicted affinity score.
  """

  def __init__(self,
               experiment_proto,
               loss,
               affinity_target_map=None,
               target_names=None,
               additional_output=None):
    """Initialize a FullyObserved output layer.

    Args:
      experiment_proto: selection_pb2.Experiment describing the experiment.
      loss: instance of an AbstractLoss subclass used for computing loss on this
        output layer.
      affinity_target_map: dictionary with one entry for each selection target
        molecule (e.g. protein) and the list of target output values to be used
        to calculate that target molecule's affinity. This dictionary is
        optional to create this OutputLayer but is required to calculate
        affinity. (In other words, during training it is unnecessary but for
        inference it is usually required.)
      target_names: optional list of strings giving target names to train
        against.
      additional_output: optional list of strings containing all the
        additional output to predict.

    Raises:
      Error: if the affinity_target_map is invalid.
    """
    self.loss = loss

    target_names = get_target_names(experiment_proto, target_names)
    additional_output = get_additional_output_names(experiment_proto,
                                                    additional_output)
    if additional_output:
      self.additional_output_axis = lt.Axis('additional_output',
                                            additional_output)
    else:
      self.additional_output_axis = None
    self.count_axis = self.target_axis = lt.Axis('target', target_names)
    self.logit_axis = lt.Axis('target', target_names+additional_output)

    self.binding_arrays_map = _binding_arrays_map(experiment_proto)

    self.params = []

    self.affinity_target_axis = self.affinity_target_lt = None
    if affinity_target_map:
      affinity_target_map = config.DEFAULT_AFFINITY_TARGET_MAPS[
          affinity_target_map]
      # make sure that every target in the affinity_target_map is in the logits
      # (otherwise the target is silently ignored, could be dangerous)
      target_names = self.count_axis.labels
      affinity_names = list(affinity_target_map.keys())
      for (affinity,
           desired_target_names) in affinity_target_map.items():
        for desired_name in desired_target_names:
          if desired_name not in target_names:
            raise Error('The desired target name %s for the affinity molecule'
                        '%s is not found in the logit target names.\n'
                        'logit target names: %s\n', desired_name,
                        affinity, target_names)

      array = np.zeros((len(affinity_names), len(target_names)), dtype=int)
      for i, affinity in enumerate(affinity_names):
        for j, target in enumerate(target_names):
          if target in affinity_target_map[affinity]:
            array[i, j] = 1
      self.affinity_axis = lt.Axis('affinity', affinity_names)
      self.affinity_target_lt = lt.LabeledTensor(
          tf.constant(
              array, dtype=tf.float32, name='affinity_targets'),
          [self.affinity_axis, self.count_axis])

  def loss_per_example_and_target(self, logits, outputs, include_array=True):
    """See method on base class."""
    targets = _targets_from_outputs(outputs, self.logit_axis)
    loss = self.loss.per_example_and_target(logits, targets)
    if bool(set(self.binding_arrays_map.keys()) &
            set(outputs.axes['output'].labels)) and include_array:
      affinity_loss = self.affinity_loss_per_example_and_target(logits, outputs)
      return lt.concat([loss, affinity_loss], 'target')
    else:
      return loss

  def predict_counts(self, logits, outputs=None):
    """See method on base class."""
    if self.additional_output_axis:
      return lt.select(logits, {'target': list(self.target_axis.labels)})
    else:
      return logits

  def predict_additional_output(self, logits):
    if not self.additional_output_axis:
      raise Error(
          'Tries to calculate additional output while no such output specified')
    return lt.select(logits,
                     {'target': list(self.additional_output_axis.labels)})

  def predict_affinity(self, logits):
    """See method on base class."""

    if not self.affinity_target_lt:
      raise Error(
          'No affinity_target_map has been designated. This FullyObserved '
          'layer cannot calculate the affinity. The FullyObserved layer '
          'must be initialized with an affinity_target_map to be capable '
          'of calculating affinity.')

    # then do matrix multiple to turn (target) X (target by protein)
    # to a vector of length protein. For proteins with multiple targets, the
    # multiplication takes the sum of the values.
    if self.additional_output_axis:
      count_logits = lt.select(logits,
                               {'target': list(self.target_axis.labels)})
    else:
      count_logits = logits
    output_per_affinity = lt.matmul(count_logits, self.affinity_target_lt)

    return output_per_affinity


def _get_selection_signs(affinity_names, output_names, experiment_proto):
  """Calculate selection signs from an experiment_proto.

  Selection signs indicate whether a larger affinity score should correspond
  to a larger output count (+1), a smaller count (-1) or should not make any
  difference at all (0). We use it to constrain output layer weights in our
  latent affinity models.

  Args:
    affinity_names: list of strings giving names for each latent affinity.
    output_names: list of strings giving names for each target.
    experiment_proto: selection_pb2.Experiment describing the experiment.

  Returns:
    numpy.ndarray with dtype=int consisting of values drawn from {+1, -1, 0}
    with shape (n_affinity_names, n_output_counts).

  Raises:
    ValueError: If any selections are both target and background.
  """
  n_aff_names = len(affinity_names)
  n_outputs = len(output_names)
  signs = np.zeros((n_aff_names, n_outputs), dtype=int)
  for i, aff_name in enumerate(affinity_names):
    for j, output in enumerate(output_names):
      round_proto = selection.round_from_count_name(output, experiment_proto)

      in_target = bool(round_proto.target_concentrations[aff_name] > 0)
      in_background = bool(round_proto.background_concentrations[aff_name] > 0)
      if in_target and in_background:
        raise ValueError('compounds cannot be both target and background in '
                         'the same round of selection')
      elif in_target:
        signs[i, j] = 1
      elif in_background:
        signs[i, j] = -1

      if output == round_proto.negative_reads.name:
        signs[i, j] *= -1

  return signs


class LatentAffinity(AbstractOutputLayer):
  """Output layer and loss for a basic latent affinity model with SSE loss.

  In latent affinity models, we use the output of the feedforward network
  ("logits") directly as scores for aptamer binding affinity. The loss function
  specifies how to map these affinities onto high throughput sequencing counts.
  """

  def __init__(self, experiment_proto, loss, target_names=None,
               additional_output=None):
    """Initialize a LatentAffinity output layer.

    Args:
      experiment_proto: selection_pb2.Experiment describing the experiment.
      loss: instance of an AbstractLoss subclass used for computing loss on this
        output layer.
      target_names: optional list of strings giving target names to train
        against.
      additional_output: optional list of strings containing all the
        additional output to predict.

    Raises:
      ValueError: if any target_names are not counts.
    """
    self.loss = loss

    affinity_names = selection.all_target_and_background_names(experiment_proto)
    additional_output = get_additional_output_names(experiment_proto,
                                                    additional_output)
    target_names = get_target_names(experiment_proto, target_names)
    self.target_axis = lt.Axis('target', target_names)

    if additional_output:
      self.additional_output_axis = lt.Axis('additional_output',
                                            additional_output)
    else:
      self.additional_output_axis = None
    self.logit_axis = lt.Axis('target', affinity_names+additional_output)
    self.affinity_axis = lt.Axis('affinity', affinity_names)
    self.all_target_axis = lt.Axis('target', target_names+additional_output)

    self.all_count_names = selection.all_count_names(experiment_proto)
    self.binding_arrays_map = _binding_arrays_map(experiment_proto)

    signs = _get_selection_signs(affinity_names, target_names, experiment_proto)
    self.selection_signs = lt.LabeledTensor(
        tf.constant(
            signs, dtype=tf.float32, name='selection_signs'),
        [self.affinity_axis, self.target_axis])

    # TODO(shoyer): consider if there's a sane way to make lt.Variable
    affinity_weights = tf.Variable(
        tf.ones_like(
            signs, dtype=tf.float32), name='affinity_weights')
    bias = tf.Variable(tf.zeros([self.target_axis.size]), name='bias')
    self.params = [affinity_weights, bias]

    self.affinity_weights = lt.LabeledTensor(
        tf.convert_to_tensor(affinity_weights),
        [self.affinity_axis, self.target_axis])
    self.bias = lt.LabeledTensor(tf.convert_to_tensor(bias), [self.target_axis])

  def predict_counts(self, logits, outputs=None):  # pylint: disable=unused-argument
    """Make count predictions from logits and counts.

    Args:
      logits: LabeledTensor with dtype=float32 and axes [batch, logit_axis].
      outputs: LabeledTensor with dtype=float32 and axes [batch, output_axis].
        Unused by the base class but in the signature for the benefit of
        subclasses that use counts from previous rounds to help predict future
        rounds. It is the responsibility of the implementation using `outputs`
        to ensure that this method respects the casual structure of the
        experiment.

    Returns:
      preds: LabeledTensor with dtype=float32 and axes [batch, target_axis].
    """
    # TODO(shoyer): consider using tf.nn.softplus instead of abs here
    weights = abs(self.affinity_weights) * self.selection_signs
    if self.additional_output_axis:
      affinity_logits = lt.rename_axis(
          lt.select(logits, {'target': list(self.affinity_axis.labels)}),
          'target', 'affinity')
    else:
      affinity_logits = lt.rename_axis(logits, 'target', 'affinity')
    preds = lt.matmul(affinity_logits, weights) + self.bias
    return preds

  def loss_per_example_and_target(self, logits, outputs, include_array=True):
    """See method on base class."""
    with tf.name_scope('predictions'):
      if self.additional_output_axis:
        affinity_logits = lt.select(logits,
                                    {'target': list(self.affinity_axis.labels)})
        ao_logits = lt.select(logits,
                              {'target':
                               list(self.additional_output_axis.labels)})
        count_preds = self.predict_counts(affinity_logits, outputs)
        preds = lt.concat([count_preds, ao_logits], 'target')
      else:
        preds = self.predict_counts(logits, outputs)
    targets = _targets_from_outputs(outputs, self.all_target_axis)
    loss = self.loss.per_example_and_target(preds, targets)
    if bool(set(self.binding_arrays_map.keys()) &
            set(outputs.axes['output'].labels)) and include_array:
      affinity_loss = self.affinity_loss_per_example_and_target(logits, outputs)
      return lt.concat([loss, affinity_loss], 'target')
    else:
      return loss

  def predict_affinity(self, logits):
    """See method on base class."""
    if self.additional_output_axis:
      return lt.rename_axis(
          lt.select(logits, {'target': list(self.affinity_axis.labels)}),
          'target', 'affinity')
    else:
      return lt.rename_axis(logits, 'target', 'affinity')

  def predict_additional_output(self, logits):
    if not self.additional_output_axis:
      raise Error(
          'Tries to calculate additional output while no such output specified')
    return lt.select(logits,
                     {'target': list(self.additional_output_axis.labels)})


class LatentAffinityWithDeps(LatentAffinity):
  """LatentAffinity model with round-to-round dependencies.

  This model works like LatentAffinity model, but we include a prior for each
  count, which is the count from the previous round. We can directly use
  previous round counts even though we won't have these at prediction time,
  because we only need these counts for training, not predictions.
  """

  def __init__(self, experiment_proto, loss, deps_normalize, target_names=None,
               additional_output=None):
    """Initialize a LatentAffinityWithDeps output layer.

    Args:
      experiment_proto: selection_pb2.Experiment describing the experiment.
      loss: instance of an AbstractLoss subclass used for computing loss on this
        output layer.
      deps_normalize: Normalizer instance used for normalizing dependency
        counts.
      target_names: optional list of strings giving target names to train
        against.
      additional_output: optional list of strings containing all the
        additional output to predict.
    """
    super(LatentAffinityWithDeps, self).__init__(
        experiment_proto, loss,
        target_names=target_names, additional_output=additional_output)
    self.deps_normalize = deps_normalize

    self.parent_count_names = selection.parent_counts(experiment_proto)

    init_value = 0.1 * tf.ones((self.target_axis.size,), dtype=tf.float32)
    prev_round_scale = tf.Variable(init_value, name='prev_round_scale')
    self.params.append(prev_round_scale)

    self.prev_round_scale = lt.LabeledTensor(
        tf.convert_to_tensor(prev_round_scale), [self.target_axis])

  def _normed_prev_round_counts(self, input_counts,
                                input_counts_label='output'):
    """Create a Tensor with normalized counts from the previous round.

    Args:
      input_counts: LabeledTensor with dtype=float32 and axes
        [batch, input_counts_label].
      input_counts_label: Name of the axis in input_counts that contains the
        count data to use. For LatentAffinityWithDeps that uses the
        actual count values, input_counts will be the outputs tensor and the
        label will be 'output'. For LatentAffinityWithPredDeps, input_counts
        will be the predictions for these counts and the axis will be 'target'.

    Returns:
      preds: LabeledTensor with dtype=float32 and axes [batch, target_axis].
    """
    parent_lookup = {}
    for k, parent in self.parent_count_names.items():
      if parent in input_counts.axes:
        parent_lookup[k] = lt.select(input_counts,
                                     {input_counts_label: parent})
    default_tensor = lt.LabeledTensor(
        tf.zeros_like(input_counts[:, 0]), [input_counts.axes['batch']])
    parent_tensors = [
        parent_lookup.get(k, default_tensor) for k in self.target_axis.labels
    ]
    parent_counts = lt.pack(parent_tensors, self.target_axis, axis_position=1)
    normed_counts = self.deps_normalize(parent_counts)
    return normed_counts

  def predict_counts(self, logits, outputs):
    """See method on base class."""
    preds = super(LatentAffinityWithDeps, self).predict_counts(logits, outputs)
    preds += self.prev_round_scale * self._normed_prev_round_counts(outputs)
    return preds


class LatentAffinityWithPredictedDeps(LatentAffinityWithDeps):
  """Experimental version that uses *predicted* instead of *observed* counts."""

  def predict_counts(self, logits, outputs):
    """See method on base class."""
    preds = LatentAffinity.predict_counts(self, logits, outputs)
    preds += self.prev_round_scale * self._normed_prev_round_counts(preds,
                                                                    'target')

    return preds


class LatentAffinityWithCrossDeps(LatentAffinityWithDeps):
  """Experimental version of LatentAffinityWithDeps with another predictor.

  This version may do a better job of capturing the physical selection process,
  because it predicts high counts only if there is both appearence in the
  previous round *and* high affinity.
  """

  def __init__(self, experiment_proto, loss, deps_normalize, target_names=None,
               additional_output=None):
    """See method on base class."""
    super(LatentAffinityWithCrossDeps, self).__init__(
        experiment_proto, loss, deps_normalize, target_names=target_names,
        additional_output=additional_output)

    init_value = tf.zeros(
        (self.affinity_axis.size, self.target_axis.size), dtype=tf.float32)
    logit_by_prev_count = tf.Variable(init_value, name='logit_by_prev_count')
    self.params.append(logit_by_prev_count)

    self.logit_by_prev_count = lt.LabeledTensor(
        tf.convert_to_tensor(logit_by_prev_count),
        [self.affinity_axis, self.target_axis])

  def predict_counts(self, logits, outputs):
    """See method on base class."""
    preds = super(LatentAffinityWithCrossDeps, self).predict_counts(logits,
                                                                    outputs)
    interact_weights = abs(self.logit_by_prev_count) * self.selection_signs
    # We're calling _normed_prev_round_counts a second time here with the same
    # arguments, but that's actually OK because TensorFlow automatically
    # consolidates these calls.
    if self.additional_output_axis:
      affinity_logits = lt.rename_axis(
          lt.select(logits, {'target': list(self.affinity_axis.labels)}),
          'target', 'affinity')
    else:
      affinity_logits = lt.rename_axis(logits, 'target', 'affinity')
    preds += (lt.matmul(affinity_logits, interact_weights) *
              self._normed_prev_round_counts(outputs))
    return preds


def create_output_layer(experiment_proto, hps=None):
  """Create an output layer of the provided type.

  Args:
    experiment_proto: selection_pb2.Experiment describing the experiment. Must
      have pre-computed statistics necessary for normalizing counts.
    hps: optional tf.HParams with at least these fields:
      - output_layer: string indicating an OutputLayer class.
      - loss: string indicating a Loss class.
      - dependency_norm: string indicating the name of an normalization
        method to use for count dependencies in the OutputLayer.
      - loss_norm: string indicating the name of an normalization method to
        use in the loss.
      - standardize_log_transform: boolean indicating whether or not we want to
        log transform counts before standardizing them. Only relevant for if
        you use standardization for normalization.
      - binarize_threshold: integer giving the count threshold at which to
        binarize. Only relevant for losses that binarize.
      - target_names: list of strings giving target names to train against.
      - affinity_target_map: Only required for FULLY_OBSERVED models when the
        output layer will be used to calculate affinity, this dictionary maps
        each selection affinity molecule (e.g. protein) to a set of target
        outputs (i.e. sequencing count pools) to be used when calculating
        affinity.

  Returns:
    AbstractOutputLayer instance of the type indicated by `name`.

  Raises:
    ValueError: If any of `output_layer`, `loss`, `dependency_norm`
      or `loss_norm` are not recognized.
  """
  if hps is None:
    hps = tf.HParams(
        output_layer=OUTPUT_FULLY_OBSERVED,
        loss=LOSS_SQUARED_ERROR,
        dependency_norm=NORM_STANDARDIZE,
        loss_norm=NORM_STANDARDIZE,
        standardize_log_transform=True,
        binarize_threshold=1,
        target_names=[TARGETS_ALL_OUTPUTS])

  normalize = normalizer(hps.loss_norm, experiment_proto,
                         hps.standardize_log_transform, hps.binarize_threshold)
  deps_normalize = normalizer(hps.dependency_norm, experiment_proto,
                              hps.standardize_log_transform,
                              hps.binarize_threshold)

  if hps.loss_name == LOSS_SQUARED_ERROR:
    loss = SquaredError(normalize)
  elif hps.loss_name == LOSS_CROSS_ENTROPY:
    loss = CrossEntropy(normalize)
  elif hps.loss_name == LOSS_POISSON_LOSS:
    if hps.loss_norm != NORM_SKIP:
      raise ValueError('invalid normalization for poisson loss')
    loss = PoissonLoss(normalize)
  elif hps.loss_name == LOSS_ZERO_TRUNCATED_POISSON_LOSS:
    if hps.loss_norm != NORM_SKIP:
      raise ValueError('invalid normalization for zero-truncated poisson loss')
    loss = ZeroTruncatedPoissonLoss(normalize)
  else:
    raise ValueError('unrecognized loss: %r' % hps.loss_name)

  if hps.additional_output:
    additional_output = hps.additional_output.split(',')
  else:
    additional_output = []
  if hps.output_layer == OUTPUT_FULLY_OBSERVED:
    output_layer = FullyObserved(experiment_proto, loss,
                                 hps.affinity_target_map, hps.target_names,
                                 additional_output)
  elif hps.output_layer == OUTPUT_LATENT_AFFINITY:
    output_layer = LatentAffinity(experiment_proto, loss, hps.target_names,
                                  additional_output)
  elif hps.output_layer == OUTPUT_LATENT_WITH_DEPS:
    output_layer = LatentAffinityWithDeps(experiment_proto, loss,
                                          deps_normalize, hps.target_names,
                                          additional_output)
  elif hps.output_layer == OUTPUT_LATENT_WITH_PRED_DEPS:
    output_layer = LatentAffinityWithPredictedDeps(experiment_proto, loss,
                                                   deps_normalize,
                                                   hps.target_names,
                                                   additional_output)
  elif hps.output_layer == OUTPUT_LATENT_WITH_CROSS_DEPS:
    output_layer = LatentAffinityWithCrossDeps(experiment_proto, loss,
                                               deps_normalize,
                                               hps.target_names,
                                               additional_output)
  else:
    raise ValueError('unrecognized output_layer: %r' % hps.output_layer)

  return output_layer
