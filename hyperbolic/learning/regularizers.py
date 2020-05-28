# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regularizers for CF embeddings."""

# pytype: skip-file
import abc
import tensorflow.compat.v2 as tf


class CFRegularizer(tf.keras.regularizers.Regularizer, abc.ABC):
  """CF embedding regularizers."""

  def __init__(self, reg_weight):
    """Initializes CF embedding regularizer.

    Args:
      reg_weight: regularization weight
    """
    super(CFRegularizer, self).__init__()
    self.reg_weight = tf.keras.backend.cast_to_floatx(reg_weight)

  def __call__(self, x):
    """Compute regularization for input embeddings.

    Args:
      x: Tensor of size batch_size x embedding_dimension to regularize.

    Returns:
      Regularization term.
    """
    if not self.reg_weight:
      return tf.keras.backend.constant(0.)
    else:
      return self.reg_weight * self.compute_norm(x)

  @abc.abstractmethod
  def compute_norm(self, x):
    """Computes embeddings' norms for regularization."""
    pass

  def get_config(self):
    return {'reg_weight': float(self.reg_weight)}


class NoReg(CFRegularizer):
  """No regularization."""

  def __call__(self, x):
    return tf.keras.backend.constant(0.)

  def get_config(self):
    return {'reg_weight': 0}


class L1(CFRegularizer):
  """L1 regularization."""

  def compute_norm(self, x):
    return tf.reduce_sum(tf.abs(x))


class L2(CFRegularizer):
  """L2 regularization."""

  def compute_norm(self, x):
    return tf.reduce_sum(tf.square(x))


class L3(CFRegularizer):
  """L3 regularization."""

  def compute_norm(self, x):
    return tf.reduce_sum(tf.abs(x)**3)


class N2(CFRegularizer):
  """Nuclear 2-norm regularization."""

  def compute_norm(self, x):
    return tf.reduce_sum(tf.norm(x, ord=2, axis=1)**3)
