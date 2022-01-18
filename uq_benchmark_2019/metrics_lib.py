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

# Lint as: python2, python3
"""Library of calibration metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.special
import six
from six.moves import range
import tensorflow.compat.v2 as tf



def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
  """A helper function which histograms a vector of probabilities into bins.

  Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1}
    bins: Number of equal width bins to bin predictions into in [0, 1], or an
      array representing bin edges.

  Returns:
    bin_edges: Numpy vector of floats containing the edges of the bins
      (including leftmost and rightmost).
    accuracies: Numpy vector of floats for the average accuracy of the
      predictions in each bin.
    counts: Numpy vector of ints containing the number of examples per bin.
  """
  _validate_probabilities(probabilities)
  _check_rank_nonempty(rank=1,
                       probabilities=probabilities,
                       ground_truth=ground_truth)

  if len(probabilities) != len(ground_truth):
    raise ValueError(
        'Probabilies and ground truth must have the same number of elements.')

  if [v for v in ground_truth if v not in [0., 1., True, False]]:
    raise ValueError(
        'Ground truth must contain binary labels {0,1} or {False, True}.')

  if isinstance(bins, int):
    num_bins = bins
  else:
    num_bins = bins.size - 1

  # Ensure probabilities are never 0, since the bins in np.digitize are open on
  # one side.
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
  indices = np.digitize(probabilities, bin_edges, right=True)
  accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
  return bin_edges, accuracies, counts


def bin_centers_of_mass(probabilities, bin_edges):
  probabilities = np.where(probabilities == 0, 1e-8, probabilities)
  indices = np.digitize(probabilities, bin_edges, right=True)
  return np.array([np.mean(probabilities[indices == i])
                   for i in range(1, len(bin_edges))])


def expected_calibration_error(probabilities, ground_truth, bins=15):
  """Compute the expected calibration error of a set of preditions in [0, 1].

  Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
  Returns:
    Float: the expected calibration error.
  """

  probabilities = probabilities.flatten()
  ground_truth = ground_truth.flatten()
  bin_edges, accuracies, counts = bin_predictions_and_accuracies(
      probabilities, ground_truth, bins)
  bin_centers = bin_centers_of_mass(probabilities, bin_edges)
  num_examples = np.sum(counts)

  ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
      np.abs(bin_centers[i] - accuracies[i]))
                for i in range(bin_centers.size) if counts[i] > 0])
  return ece


def accuracy_top_k(probabilities, labels, top_k):
  """Computes the top-k accuracy of predictions.

  A prediction is considered correct if the ground-truth class is among the k
  classes with the highest predicted probabilities.

  Args:
    probabilities: Array of probabilities of shape [num_samples, num_classes].
    labels: Integer array labels of shape [num_samples].
    top_k: Integer. Number of highest-probability classes to consider.
  Returns:
    float: Top-k accuracy of predictions.
  """
  _, ground_truth = _filter_top_k(probabilities, labels, top_k)
  return ground_truth.any(axis=-1).mean()


def _filter_top_k(probabilities, labels, top_k):
  """Extract top k predicted probabilities and corresponding ground truths."""

  labels_one_hot = np.zeros(probabilities.shape)
  labels_one_hot[np.arange(probabilities.shape[0]), labels] = 1

  if top_k is None:
    return probabilities, labels_one_hot

  # Negate probabilities for easier use with argpartition (which sorts from
  # lowest)
  negative_prob = -1. * probabilities

  ind = np.argpartition(negative_prob, top_k-1, axis=-1)
  top_k_ind = ind[:, :top_k]
  rows = np.expand_dims(np.arange(probabilities.shape[0]), axis=1)
  lowest_k_negative_probs = negative_prob[rows, top_k_ind]
  output_probs = -1. * lowest_k_negative_probs

  labels_one_hot_k = labels_one_hot[rows, top_k_ind]
  return output_probs, labels_one_hot_k


def get_multiclass_predictions_and_correctness(probabilities, labels, top_k=1):
  """Returns predicted class, correctness boolean vector."""
  _validate_probabilities(probabilities, multiclass=True)
  _check_rank_nonempty(rank=1, labels=labels)
  _check_rank_nonempty(rank=2, probabilities=probabilities)

  if top_k == 1:
    class_predictions = np.argmax(probabilities, -1)
    top_k_probs = probabilities[np.arange(len(labels)), class_predictions]
    is_correct = np.equal(class_predictions, labels)
  else:
    top_k_probs, is_correct = _filter_top_k(probabilities, labels, top_k)

  return top_k_probs, is_correct


def expected_calibration_error_multiclass(probabilities, labels, bins=15,
                                          top_k=1):
  """Computes expected calibration error from Guo et al. 2017.

  For details, see https://arxiv.org/abs/1706.04599.
  Note: If top_k is None, this only measures calibration of the argmax
    prediction.

  Args:
    probabilities: Array of probabilities of shape [num_samples, num_classes].
    labels: Integer array labels of shape [num_samples].
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
    top_k: Integer or None. If integer, use the top k predicted
      probabilities in ECE calculation (can be informative for problems with
      many classes and lower top-1 accuracy). If None, use all classes.
  Returns:
    float: Expected calibration error.
  """
  top_k_probs, is_correct = get_multiclass_predictions_and_correctness(
      probabilities, labels, top_k)
  top_k_probs = top_k_probs.flatten()
  is_correct = is_correct.flatten()
  return expected_calibration_error(top_k_probs, is_correct, bins)


# TODO(yovadia): Write unit-tests.
def compute_accuracies_at_confidences(labels, probs, thresholds):
  """Compute accuracy of samples above each confidence threshold.

  Args:
    labels: Array of integer categorical labels.
    probs: Array of categorical probabilities.
    thresholds: Array of floating point probability thresholds in [0, 1).
  Returns:
    accuracies: Array of accuracies over examples with confidence > T for each T
        in thresholds.
    counts: Count of examples with confidence > T for each T in thresholds.
  """
  assert probs.shape[:-1] == labels.shape

  predict_class = probs.argmax(-1)
  predict_confidence = probs.max(-1)

  shape = (len(thresholds),) + probs.shape[:-2]
  accuracies = np.zeros(shape)
  counts = np.zeros(shape)

  eq = np.equal(predict_class, labels)
  for i, thresh in enumerate(thresholds):
    mask = predict_confidence >= thresh
    counts[i] = mask.sum(-1)
    accuracies[i] = np.ma.masked_array(eq, mask=~mask).mean(-1)
  return accuracies, counts


def brier_scores(labels, probs=None, logits=None):
  """Compute elementwise Brier score.

  Args:
    labels: Tensor of integer labels shape [N1, N2, ...]
    probs: Tensor of categorical probabilities of shape [N1, N2, ..., M].
    logits: If `probs` is None, class probabilities are computed as a softmax
      over these logits, otherwise, this argument is ignored.
  Returns:
    Tensor of shape [N1, N2, ...] consisting of Brier score contribution from
    each element. The full-dataset Brier score is an average of these values.
  """
  assert (probs is None) != (logits is None)
  if probs is None:
    probs = scipy.special.softmax(logits, axis=-1)
  nlabels = probs.shape[-1]
  flat_probs = probs.reshape([-1, nlabels])
  flat_labels = labels.reshape([len(flat_probs)])

  plabel = flat_probs[np.arange(len(flat_labels)), flat_labels]
  out = np.square(flat_probs).sum(axis=-1) - 2 * plabel
  return out.reshape(labels.shape)


def brier_decompositions(labels, probs):
  """Compute Brier decompositions for batches of datasets.

  Args:
    labels: Tensor of integer labels shape [S1, S2, ..., N]
    probs: Tensor of categorical probabilities of shape [S1, S2, ..., N, M].
  Returns:
    Tensor of shape [S1, S2, ..., 3] consisting of 3-component Brier
    decompositions for each series of probabilities and labels. The components
    are ordered as <uncertainty, resolution, reliability>.
  """
  labels = tf.cast(labels, tf.int32)
  probs = tf.cast(probs, tf.float32)
  batch_shape = labels.shape[:-1]
  flatten, unflatten = _make_flatten_unflatten_fns(batch_shape)
  labels = flatten(labels)
  probs = flatten(probs)
  out = []
  for labels_i, probs_i in zip(labels, probs):
    out_i = brier_decomposition(labels_i, probabilities=probs_i)
    out.append(tf.stack(out_i, axis=-1))
  out = tf.stack(out)
  return unflatten(out)


def brier_decomposition(labels=None, logits=None, probabilities=None):
  r"""Decompose the Brier score into uncertainty, resolution, and reliability.

  [Proper scoring rules][1] measure the quality of probabilistic predictions;
  any proper scoring rule admits a [unique decomposition][2] as
  `Score = Uncertainty - Resolution + Reliability`, where:

  * `Uncertainty`, is a generalized entropy of the average predictive
    distribution; it can both be positive or negative.
  * `Resolution`, is a generalized variance of individual predictive
    distributions; it is always non-negative.  Difference in predictions reveal
    information, that is why a larger resolution improves the predictive score.
  * `Reliability`, a measure of calibration of predictions against the true
    frequency of events.  It is always non-negative and a lower value here
    indicates better calibration.

  This method estimates the above decomposition for the case of the Brier
  scoring rule for discrete outcomes.  For this, we need to discretize the space
  of probability distributions; we choose a simple partition of the space into
  `nlabels` events: given a distribution `p` over `nlabels` outcomes, the index
  `k` for which `p_k > p_i` for all `i != k` determines the discretization
  outcome; that is, `p in M_k`, where `M_k` is the set of all distributions for
  which `p_k` is the largest value among all probabilities.

  The estimation error of each component is O(k/n), where n is the number
  of instances and k is the number of labels.  There may be an error of this
  order when compared to `brier_score`.

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  [2]: Jochen Broecker.  Reliability, sufficiency, and the decomposition of
       proper scores.
       Quarterly Journal of the Royal Meteorological Society, Vol. 135, 2009.
       https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.456

  Args:
    labels: Tensor, (n,), with tf.int32 or tf.int64 elements containing ground
      truth class labels in the range [0,nlabels].
    logits: Tensor, (n, nlabels), with logits for n instances and nlabels.
    probabilities: Tensor, (n, nlabels), with predictive probability
      distribution (alternative to logits argument).

  Returns:
    uncertainty: Tensor, scalar, the uncertainty component of the
      decomposition.
    resolution: Tensor, scalar, the resolution component of the decomposition.
    reliability: Tensor, scalar, the reliability component of the
      decomposition.
  """
  if (logits is None) == (probabilities is None):
    raise ValueError(
        'brier_decomposition expects exactly one of logits or probabilities.')
  if probabilities is None:
    probabilities = scipy.special.softmax(logits, axis=1)
  _, nlabels = probabilities.shape  # Implicit rank check.

  # Compute pbar, the average distribution
  pred_class = tf.argmax(probabilities, axis=1, output_type=tf.int32)
  confusion_matrix = tf.math.confusion_matrix(pred_class, labels, nlabels,
                                              dtype=tf.float32)
  dist_weights = tf.reduce_sum(confusion_matrix, axis=1)
  dist_weights /= tf.reduce_sum(dist_weights)
  pbar = tf.reduce_sum(confusion_matrix, axis=0)
  pbar /= tf.reduce_sum(pbar)

  # dist_mean[k,:] contains the empirical distribution for the set M_k
  # Some outcomes may not realize, corresponding to dist_weights[k] = 0
  dist_mean = confusion_matrix / tf.expand_dims(
      tf.reduce_sum(confusion_matrix, axis=1) + 1.0e-7, 1)

  # Uncertainty: quadratic entropy of the average label distribution
  uncertainty = -tf.reduce_sum(tf.square(pbar))

  # Resolution: expected quadratic divergence of predictive to mean
  resolution = tf.square(tf.expand_dims(pbar, 1) - dist_mean)
  resolution = tf.reduce_sum(dist_weights * tf.reduce_sum(resolution, axis=1))

  # Reliability: expected quadratic divergence of predictive to true
  prob_true = tf.gather(dist_mean, pred_class, axis=0)
  reliability = tf.reduce_sum(tf.square(prob_true - probabilities), axis=1)
  reliability = tf.reduce_mean(reliability)

  return uncertainty, resolution, reliability


def soften_probabilities(probs, epsilon=1e-8):
  """Returns heavily weighted average of categorical distribution and uniform.

  Args:
    probs: Categorical probabilities of shape [num_samples, num_classes].
    epsilon: Small positive value for weighted average.
  Returns:
    epsilon * uniform + (1-epsilon) * probs
  """
  uniform = np.ones_like(probs) / probs.shape[-1]
  return epsilon * uniform + (1-epsilon) * probs


def get_quantile_bins(num_bins, probs, top_k=1):
  """Find quantile bin edges.

  Args:
    num_bins: int, number of bins desired.
    probs: Categorical probabilities of shape [num_samples, num_classes].
    top_k: int, number of highest-predicted classes to consider in binning.
  Returns:
    Numpy vector, quantile bin edges.
  """
  edge_percentiles = np.linspace(0, 100, num_bins+1)

  if len(probs.shape) == 1:
    probs = np.stack([probs, 1-probs]).T

  if top_k == 1:
    max_probs = probs.max(-1)
  else:
    unused_labels = np.zeros(probs.shape[0]).astype(np.int32)
    max_probs, _ = _filter_top_k(probs, unused_labels, top_k)

  bins = np.percentile(max_probs, edge_percentiles)
  bins[0], bins[-1] = 0., 1.
  return bins


def _validate_probabilities(probabilities, multiclass=False):
  if np.max(probabilities) > 1. or np.min(probabilities) < 0.:
    raise ValueError('All probabilities must be in [0,1].')
  if multiclass and not np.allclose(1, np.sum(probabilities, axis=-1),
                                    atol=1e-5):
    raise ValueError(
        'Multiclass probabilities must sum to 1 along the last dimension.')


def _check_rank_nonempty(rank, **kwargs):
  for key, array in six.iteritems(kwargs):
    if len(array) <= 1 or array.ndim != rank:
      raise ValueError(
          '%s must be a rank-1 array of length > 1; actual shape is %s.' %
          (key, array.shape))


def _make_flatten_unflatten_fns(batch_shape):
  """Builds functions for flattening and unflattening batch dimensions."""
  batch_shape = tuple(batch_shape)
  batch_rank = len(batch_shape)
  ndims = np.prod(batch_shape)

  def flatten_fn(x):
    x_shape = tuple(x.shape)
    if x_shape[:batch_rank] != batch_shape:
      raise ValueError('Expected batch-shape=%s; received array of shape=%s' %
                       (batch_shape, x_shape))
    flat_shape = (ndims,) + x_shape[batch_rank:]
    return tf.reshape(x, flat_shape)

  def unflatten_fn(x):
    x_shape = tuple(x.shape)
    if x_shape[0] != ndims:
      raise ValueError('Expected batch-size=%d; received shape=%s' %
                       (ndims, x_shape))
    return tf.reshape(x, batch_shape + x_shape[1:])
  return flatten_fn, unflatten_fn
