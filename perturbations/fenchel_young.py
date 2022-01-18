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

# Lint as: python3
"""Implementation of a Fenchel-Young loss using perturbation techniques."""

from typing import Callable, Optional

import gin
import tensorflow.compat.v2 as tf

from perturbations import perturbations


@gin.configurable
class FenchelYoungLoss(tf.keras.losses.Loss):
  """Implementation of a Fenchel Young loss."""

  def __init__(
      self,
      func = None,
      num_samples = 1000,
      sigma = 0.01,
      noise = perturbations._NORMAL,
      batched = True,
      maximize = True,
      reduction = tf.keras.losses.Reduction.SUM):
    """Initializes the Fenchel-Young loss.

    Args:
     func: the function whose argmax is to be differentiated by perturbation.
     num_samples: (int) the number of perturbed inputs.
     sigma: (float) the amount of noise to be considered
     noise: (str) the noise distribution to be used to sample perturbations.
     batched: whether inputs to the func will have a leading batch dimension
      (True) or consist of a single example (False). Defaults to True.
     maximize: (bool) whether to maximize or to minimize the input function.
     reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `SUM`. When used in custom training loops under the scope
      of `tf.distribute.Strategy`, must be set to `NONE` or `SUM`.
    """
    super().__init__(reduction=reduction, name='fenchel_young')
    self._batched = batched
    self._maximize = maximize
    self.func = func
    self.perturbed = perturbations.perturbed(func=func,
                                             num_samples=num_samples,
                                             sigma=sigma,
                                             noise=noise,
                                             batched=batched)

  def call(self, y_true, theta):
    """The Fenchel-Young loss mainly defines a gradient.

    We provide a meaningful forward pass for convenience only.

    Args:
     y_true: tf.Tensor containing the supervised label.
     theta: tf.Tensor which is the output of the network, before applying the
      passed function `func`.

    Returns:
     tf.Tensor which is the value of the loss. What actually matters is only its
     gradient.
    """

    @tf.custom_gradient
    def forward(theta):
      diff = self.perturbed(theta) - tf.cast(y_true, dtype=theta.dtype)
      if not self._maximize:
        diff = -diff

      def grad(dy):
        if self._batched:  # dy has shape (batch_size,) in this case.
          dy = tf.reshape(dy, [tf.shape(dy)[0]] + (diff.shape.rank - 1) * [1])
        return dy * diff

      # Computes per-example loss for batched inputs. If the total loss for the
      # batch is the desired output, use `SUM` or `SUM_OVER_BATCH` as reduction.
      if self._batched:
        loss = tf.reduce_sum(
            tf.reshape(diff, [tf.shape(diff)[0], -1]) ** 2, axis=-1)
      else:  # Computes loss for unbatched inputs.
        loss = tf.reduce_sum(diff ** 2)

      return loss, grad

    return forward(theta)
