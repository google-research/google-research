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

"""Custom metrics for TensorFlow."""

from typing import Sequence, Optional
import tensorflow as tf


class AUCWithMaskedClass(tf.keras.metrics.AUC):
  """Computes AUC while ignoring class with id equal to `-1`.

  Assumes binary `{0, 1}` classes with a masked `{-1}` class.
  """

  def __init__(self, with_logits = False, **kwargs):
    super(AUCWithMaskedClass, self).__init__(**kwargs)
    self.with_logits = with_logits

  @tf.autograph.experimental.do_not_convert
  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.
      sample_weight: Input value is ignored. Parameter present to match
        signature with parent class where mask `{-1}` is the sample weight.
    Returns: `None`
    """
    if self.with_logits:
      y_pred = tf.math.sigmoid(y_pred)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    super(AUCWithMaskedClass, self).update_state(
        y_true, y_pred, sample_weight=mask)


class PrecisionWithMaskedClass(tf.keras.metrics.Precision):
  """Computes precision while ignoring class with id equal to `-1`.

  Assumes binary `{0, 1}` classes with a masked `{-1}` class.
  """

  def __init__(self, with_logits = False, **kwargs):
    super(PrecisionWithMaskedClass, self).__init__(**kwargs)
    self.with_logits = with_logits

  @tf.autograph.experimental.do_not_convert
  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.
      sample_weight: Input value is ignored. Parameter present to match
        signature with parent class where mask `{-1}` is the sample weight.
    Returns: `None`
    """
    if self.with_logits:
      y_pred = tf.math.sigmoid(y_pred)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    super(PrecisionWithMaskedClass, self).update_state(
        y_true, y_pred, sample_weight=mask)


class RecallWithMaskedClass(tf.keras.metrics.Recall):
  """Computes recall while ignoring class with id equal to `-1`.

  Assumes binary `{0, 1}` classes with a masked `{-1}` class.
  """

  def __init__(self, with_logits = False, **kwargs):
    super(RecallWithMaskedClass, self).__init__(**kwargs)
    self.with_logits = with_logits

  @tf.autograph.experimental.do_not_convert
  def update_state(self,
                   y_true,
                   y_pred,
                   sample_weight = None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values.
      y_pred: Predicted values.
      sample_weight: Input value is ignored. Parameter present to match
        signature with parent class where mask `{-1}` is the sample weight.
    Returns: `None`
    """
    if self.with_logits:
      y_pred = tf.math.sigmoid(y_pred)
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    super(RecallWithMaskedClass, self).update_state(
        y_true, y_pred, sample_weight=mask)
