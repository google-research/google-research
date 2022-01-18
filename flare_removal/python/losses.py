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

"""Loss functions for training a lens flare reduction model."""
from typing import Callable, Dict, Mapping, Optional, Union

import tensorflow as tf

from flare_removal.python import vgg


def get_loss(name):
  """Returns the loss function object for the given name.

  Supported configs:
  - "l1":
    Pixel-wise MAE.
  - "l2":
    Pixel-wise MSE.
  - "perceptual" (or "percep"):
    Perceptual loss implemented using a pre-trained VGG19 network, plus L1 loss.
    The two losses have equal weights.

  Args:
    name: One of the three configs above. Not case-sensitive.

  Returns:
    A Keras `Loss` object.
  """
  name = name.lower()
  if name == 'l2':
    return tf.keras.losses.MeanSquaredError()
  elif name == 'l1':
    return tf.keras.losses.MeanAbsoluteError()
  elif name in ['percep', 'perceptual']:
    loss_fn = CompositeLoss()
    loss_fn.add_loss(PerceptualLoss(), weight=1.0)
    loss_fn.add_loss('L1', weight=1.0)
    return loss_fn
  else:
    raise ValueError(f'Unrecognized loss function name: {name}')


class PerceptualLoss(tf.keras.losses.Loss):
  """A perceptual loss function based on the VGG-19 model.

  The loss function is defined as a weighted sum of the L1 loss at various
  tap-out layers of the network.
  """
  DEFAULT_COEFFS = {
      'block1_conv2': 1 / 2.6,
      'block2_conv2': 1 / 4.8,
      'block3_conv2': 1 / 3.7,
      'block4_conv2': 1 / 5.6,
      'block5_conv2': 10 / 1.5,
  }

  def __init__(self,
               coeffs = None,
               name = 'perceptual'):
    """Initializes a perceptual loss instance.

    Args:
      coeffs: Key-value pairs where the keys are the tap-out layer names, and
        the values are their coefficients in the weighted sum. Defaults to the
        `self.DEFAULT_COEFFS`.
      name: Name of this Tensorflow object.
    """
    super(PerceptualLoss, self).__init__(name=name)
    coeffs = coeffs or self.DEFAULT_COEFFS
    layers, self._coeffs = zip(*coeffs.items())
    self._model = vgg.Vgg19(tap_out_layers=layers)

  def call(self, y_true, y_pred):
    """Invokes the loss function.

    See base class for details.

    Do not call this method directly. Use the __call__() method instead.

    Args:
      y_true: ground-truth image batch, with shape [B, H, W, C].
      y_pred: predicted image batch, with the same shape.

    Returns:
      A [B, 1, 1] tensor containing the perceptual loss values. Note that
      according to the base class's specs, if the inputs have D dimensions, the
      output must have D-1 dimensions. Hence the [B, 1, 1] shape.
    """
    true_features = self._model(y_true)
    pred_features = self._model(y_pred)
    total_loss = tf.constant(0.0)
    for ft, fp, coeff in zip(true_features, pred_features, self._coeffs):
      # MAE only reduces the last dimension, leading to a [B, H, W]-tensor.
      loss = tf.keras.losses.MAE(ft, fp)
      # Further reduce on the H and W dimensions.
      loss = tf.reduce_mean(loss, axis=[1, 2], keepdims=True)
      total_loss += loss * coeff
    return total_loss


class CompositeLoss(tf.keras.losses.Loss):
  """A weighted sum of individual loss functions for images.

  Attributes:
    losses: Mapping from Keras loss objects to weights.
  """

  def __init__(self, name = 'composite'):
    """Initializes an instance with given weights.

    Args:
      name: Optional name for this Tensorflow object.
    """
    super(CompositeLoss, self).__init__(name=name)
    self.losses: Dict[tf.keras.losses.Loss, float] = {}

  def add_loss(self, loss, weight):
    """Adds a component loss to the composite with specific weight.

    Args:
      loss: A Keras loss object or identifier. All standard Keras loss
        identifiers are supported (e.g., string like "mse", loss functions, and
        `tf.keras.losses.Loss` objects). In addition, strings "l1" and "l2" are
        also supported. Cannot be a loss that is already added to this
        `CompositeLoss`.
      weight: Weight associated with this loss. Must be > 0.

    Raises:
      ValueError: If the given `loss` already exists, or if `weight` is empty or
      <= 0.
    """
    if weight <= 0.0:
      raise ValueError(f'Weight must be > 0, but is {weight}.')
    if isinstance(loss, str):
      loss = loss.lower()
      loss = {'l1': 'mae', 'l2': 'mse'}.get(loss, loss)
      loss_fn = tf.keras.losses.get(loss)
    else:
      loss_fn = loss
    if loss_fn in self.losses:
      raise ValueError('The same loss already exists.')
    self.losses[loss_fn] = weight  # pytype: disable=container-type-mismatch  # typed-keras

  def call(self, y_true, y_pred):
    """See base class."""
    assert self.losses, 'At least one component loss must be added.'
    loss_sum = tf.constant(0.0)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    for loss, weight in self.losses.items():
      loss_sum = loss(y_true, y_pred) * weight + loss_sum
    return loss_sum
