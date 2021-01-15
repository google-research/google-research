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

"""Calibration metrics."""
import numpy as np

import caltrain.bin_methods as bm


class CalibrationMetric():
  """Class to compute the calibration error.

  Let N = num examples, K = num classes, and B = num bins.
  """

  def __init__(self,
               ce_type="quant",
               num_bins=15,
               bin_method="equal_examples",
               norm=2,
               multiclass_setting="top_label"):
    """Initialize calibration metric class.

    Args:
      ce_type: str describing the type of calibration error to compute. Must be
        "quant", "bn" or "neighborhood".
      num_bins: int for number of bins.
      bin_method: string for binning technique to use. Must be either
        "equal_width", "equal_examples" or "".
      norm: integer for norm to use to compute the calibration error. Norm
        should be > 0.
      multiclass_setting: string specifying the type of multiclass calibration
        error to compute. Must be "top_label" or "marginal". If "top_label",
        computes the calibration error of the top class. If "marginal", computes
        the marginal calibration error.
    """
    if bin_method not in ["equal_width", "equal_examples", ""]:
      raise NotImplementedError("Bin method not supported.")
    if multiclass_setting not in ["top_label", "marginal"]:
      raise NotImplementedError(
          f"Multiclass setting {multiclass_setting} not supported.")
    if bin_method == "equal_width" or ce_type.startswith("ew"):
      self.bin_method = bm.BinEqualWidth(num_bins)
    elif bin_method == "equal_examples" or ce_type.startswith("em"):
      self.bin_method = bm.BinEqualExamples(num_bins)
    elif bin_method == "None":
      self.bin_method = None
    else:
      raise NotImplementedError(f"Bin method {bin_method} not supported.")

    self.ce_type = ce_type
    self.norm = norm
    self.num_bins = num_bins
    self.multiclass_setting = multiclass_setting
    self.configuration_str = "{}_bins:{}_{}_norm:{}_{}".format(
        ce_type, num_bins, bin_method, norm, multiclass_setting)

  def get_configuration_str(self):
    return self.configuration_str

  def predict_top_label(self, fx, y):
    """Compute confidence scores and correctness of predicted labels.

    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.

    Returns:
      fx_top: np.ndarray of shape [N, 1] for confidence score of top label.
      hits: np.ndarray of shape [N, 1] denoting whether or not top label
        is a correct prediction or not.
    """
    picked_classes = np.argmax(fx, axis=1)
    labels = np.argmax(y, axis=1)
    hits = 1 * np.array(picked_classes == labels, ndmin=2).transpose()
    fx_top = np.max(fx, axis=1, keepdims=True)
    return fx_top, hits

  def compute_error(self, fx, y):
    """Compute the calibration error given softmax fx and one hot labels.

    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.

    Returns:
      calibration error
    """
    if len(fx.shape) == 1:
      print("WARNING: reshaping fx, assuming K=1")
      fx = fx.reshape(len(fx), 1)
    if len(y.shape) == 1:
      print("WARNING: reshaping one hot labels, assuming K=1")
      y = y.reshape(len(y), 1)

    if np.max(fx) > 1.:
      raise ValueError("Maximum score of {} is greater than 1.".format(
          np.max(fx)))
    if np.min(fx) < 0.:
      raise ValueError("Minimum score of {} is less than 0.".format(np.min(fx)))

    if np.max(y) > 1.:
      raise ValueError("Maximum label of {} is greater than 1.".format(
          np.max(y)))
    if np.min(y) < 0.:
      raise ValueError("Minimum label of {} is less than 0.".format(np.min(y)))

    num_classes = fx.shape[1]
    if self.multiclass_setting == "top_label" and num_classes > 1:
      fx, y = self.predict_top_label(fx, y)

    if self.num_bins > 0 and self.bin_method:
      binned_fx, binned_y, bin_sizes, bin_indices = self._bin_data(fx, y)

    if self.ce_type in ["ew_ece_bin", "em_ece_bin"]:
      calibration_error = self._compute_error_all_binned(
          binned_fx, binned_y, bin_sizes)
    elif self.ce_type in ["label_binned"]:
      calibration_error = self._compute_error_label_binned(
          fx, binned_y, bin_indices)
    elif self.ce_type.endswith(("sweep")):
      calibration_error = self._compute_error_monotonic_sweep(fx, y)
    else:
      raise NotImplementedError("Calibration error {} not supported.".format(
          self.ce_type))

    return calibration_error

  def _compute_error_no_bins(self, fx, y):
    """Compute error without binning."""
    num_examples = fx.shape[0]
    ce = pow(np.abs(fx - y), self.norm)
    return pow(ce.sum() / num_examples, 1. / self.norm)

  def _compute_error_all_binned(self, binned_fx, binned_y, bin_sizes):
    """Compute calibration error given binned data."""
    num_examples = np.sum(bin_sizes[:, 0])
    ce = pow(np.abs(binned_fx - binned_y), self.norm) * bin_sizes
    ce_sum = pow(ce.sum() / num_examples, 1. / self.norm)
    return ce_sum

  def _compute_error_label_binned(self, fx, binned_y, bin_indices):
    """Compute label binned calibration error."""
    num_examples = fx.shape[0]
    num_classes = fx.shape[1]
    ce_sum = 0.0
    for k in range(num_classes):
      for i in range(num_examples):
        ce_sum += pow(
            np.abs(fx[i, k] - binned_y[bin_indices[i, k], k]), self.norm)
    ce_sum = pow(ce_sum / num_examples, 1. / self.norm)
    return ce_sum

  def _bin_data(self, fx, y):
    """Bin fx and y.

    Args:
      fx: np.ndarray of shape [N, K] for predicted confidence fx.
      y: np.ndarray of shape [N, K] for one-hot-encoded labels.

    Returns:
      A tuple containing:
        - binned_fx: np.ndarray of shape [B, K] containing mean
            predicted score for each bin and class
        - binned_y: np.ndarray of shape [B, K]
            containing mean empirical accuracy for each bin and class
        - bin_sizes: np.ndarray of shape [B, K] containing number
            of examples in each bin and class
    """
    bin_indices = self.bin_method.compute_bin_indices(fx)
    num_classes = fx.shape[1]

    binned_fx = np.zeros((self.num_bins, num_classes))
    binned_y = np.zeros((self.num_bins, num_classes))
    bin_sizes = np.zeros((self.num_bins, num_classes))

    for k in range(num_classes):
      for bin_idx in range(self.num_bins):
        indices = np.where(bin_indices[:, k] == bin_idx)[0]
        # Disable for Numpy containers.
        # pylint: disable=g-explicit-length-test
        if len(indices) > 0:
          # pylint: enable=g-explicit-length-test
          mean_score = np.mean(fx[:, k][indices])
          mean_accuracy = np.mean(y[:, k][indices])
          bin_size = len(indices)
        else:
          mean_score = 0.0
          mean_accuracy = 0.0
          bin_size = 0
        binned_fx[bin_idx][k] = mean_score
        binned_y[bin_idx][k] = mean_accuracy
        bin_sizes[bin_idx][k] = bin_size

    return binned_fx, binned_y, bin_sizes, bin_indices

  def _compute_error_monotonic_sweep(self, fx, y):
    """Compute ECE using monotonic sweep method."""
    fx = np.squeeze(fx)
    y = np.squeeze(y)
    non_nan_inds = np.logical_not(np.isnan(fx))
    fx = fx[non_nan_inds]
    y = y[non_nan_inds]

    if self.ce_type == "em_ece_sweep":
      bins = self.em_monotonic_sweep(fx, y)
    elif self.ce_type == "ew_ece_sweep":
      bins = self.ew_monotonic_sweep(fx, y)
    n_bins = np.max(bins) + 1
    ece, _ = self._calc_ece_postbin(n_bins, bins, fx, y)
    return ece

  def _calc_ece_postbin(self, n_bins, bins, fx, y):
    """Calculate ece_bin after bins are computed and determine monotonicity."""
    ece = 0.
    monotonic = True
    last_ym = -1000
    for i in range(n_bins):
      cur = bins == i
      if any(cur):
        fxm = np.mean(fx[cur])
        ym = np.mean(y[cur])
        if ym < last_ym:  # determine if predictions are monotonic
          monotonic = False
        last_ym = ym
        n = np.sum(cur)
        ece += n * pow(np.abs(ym - fxm), self.norm)
    return (pow(ece / fx.shape[0], 1. / self.norm)), monotonic

  def em_monotonic_sweep(self, fx, y):
    """Monotonic bin sweep using equal mass binning scheme."""
    sort_ix = np.argsort(fx)
    n_examples = fx.shape[0]
    bins = np.zeros((n_examples), dtype=int)

    prev_bins = np.zeros((n_examples), dtype=int)
    for n_bins in range(2, n_examples):
      bins[sort_ix] = np.minimum(
          n_bins - 1, np.floor(
              (np.arange(n_examples) / n_examples) * n_bins)).astype(int)
      _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
      if not monotonic:
        return prev_bins
      prev_bins = np.copy(bins)
    return bins

  def ew_monotonic_sweep(self, fx, y):
    """Monotonic bin sweep using equal width binning scheme."""
    n_examples = fx.shape[0]
    bins = np.zeros((n_examples), dtype=int)
    prev_bins = np.zeros((n_examples), dtype=int)
    for n_bins in range(2, n_examples):
      bins = np.minimum(n_bins - 1, np.floor(fx * n_bins)).astype(int)
      _, monotonic = self._calc_ece_postbin(n_bins, bins, fx, y)
      if not monotonic:
        return prev_bins
      prev_bins = np.copy(bins)
    return bins
