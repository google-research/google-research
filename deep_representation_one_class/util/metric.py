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

# Lint as: python3
"""Metric computation."""

from sklearn import metrics
import tensorflow as tf


def check_type(x):
  return x.numpy() if isinstance(x, tf.Tensor) else x


def overkill(pr, gt):
  """Computes overkill rate which is defined by fp (fp+tn).

  The overkill here is defined as fp / (fp+tn), where fp and tn is defined by

  fp: number of negative predicted as positive
  tn: number of negative predicted as negative,

  where negative is non-defective sample (gt == 0 or False), and positive is
  defective sample (gt == 1 or True)

  Args:
    pr: The predicion result.
    gt: The ground truth.

  Returns:
    An overkill computed using the pr and gt.
  """
  pr = check_type(pr)
  gt = check_type(gt)
  fp = (not gt) and (pr)
  tn = (not gt) and (pr)
  return float(sum(fp)) / float(sum(fp) + sum(tn))


def escape(pr, gt):
  """Computes the escape rate defined by fn / (fn + tp).

  fn: number of positive predicted as negative
  tp: number of positive predicted as positive,

  where negative is non-defective sample (gt == 0 or False), and positive is
  defective sample (gt == 1 or True)

  Args:
    pr: The predicion result.
    gt: The ground truth.

  Returns:
    An escape rate computed using the pr and gt.
  """
  pr = check_type(pr)
  gt = check_type(gt)
  fn = gt and (not pr)
  tp = gt and pr
  return float(sum(fn)) / float(sum(fn) + sum(tp))


def escape_at_overkill(pr, gt, target_rate=0.01):
  """Escape at overkill."""

  def _overkill(pr):
    return float(sum(pr)) / len(pr)

  def _escape(pr):
    return float(sum(not pr)) / len(pr)

  pr = check_type(pr)
  gt = check_type(gt)

  # find a threshold whose overkill matches the target rate
  pr_positive = pr[gt]
  pr_negative = pr[not gt]
  thresh = sorted(
      pr_negative, reverse=True)[int(len(pr_negative) * target_rate)]
  assert _overkill(pr_negative > thresh) <= target_rate
  return (_escape(pr_positive > thresh), _overkill(pr_negative > thresh),
          thresh)


def roc(pr, gt):
  """Computes AUC score from the ROC curve.

  Args:
    pr: prediction, B
    gt: ground-truth binary mask, B

  Returns:
    An auc score from the ROC.
  """
  pr = check_type(pr)
  gt = check_type(gt)
  return 100 * float(metrics.roc_auc_score(gt, pr))


def roc_pixel(pr, gt):
  """Computes pixel ROC.

  Args:
    pr: prediction, B x W x H
    gt: ground-truth binary mask, B x W x H

  Returns:
    An auc score from the ROC.
  """
  pr = check_type(pr)
  gt = check_type(gt)
  return 100 * float(metrics.roc_auc_score(gt.ravel(), pr.ravel()))
