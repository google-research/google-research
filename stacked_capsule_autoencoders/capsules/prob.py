# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Probability Distributions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
from tensorflow import nest
import tensorflow_probability as tfp

from stacked_capsule_autoencoders.capsules.math_ops import safe_log
from stacked_capsule_autoencoders.capsules.tensor_ops import make_brodcastable

try:  # pylint:disable=g-statement-before-imports
  import functools32 as functools  # pylint:disable=g-import-not-at-top
except ImportError:
  import functools  # pylint:disable=g-import-not-at-top

tfd = tfp.distributions


class MixtureDistribution(object):
  """Mixture."""

  def __init__(self, mixing_logits, component_stats, component_class,
               presence=None):
    """Builds the module.

    Args:
      mixing_logits: tensor [B, k, ...] with k the number of components.
      component_stats: list of tensors of shape [B, k, ...] or broadcastable
        to these shapes; they are argument to the chosen distribution class.
      component_class: callable; returns a distribution object.
      presence: [B, k] tensor of floats in [0, 1] or None.
    """
    super(MixtureDistribution, self).__init__()
    if presence is not None:
      mixing_logits += make_brodcastable(safe_log(presence), mixing_logits)

    self._mixing_logits = mixing_logits

    component_stats = nest.flatten(component_stats)
    self._distributions = component_class(*component_stats)
    self._presence = presence

  def _maybe_mask(self, tensor):
    if self._presence is None:
      return tensor

    pres = make_brodcastable(self._presence, tensor)
    return tensor * pres

  @property
  def mixing_log_prob(self):
    return self._mixing_logits - tf.reduce_logsumexp(self._mixing_logits, 1,
                                                     keepdims=True)

  @property
  def mixing_prob(self):
    # we don't need to take presence into account any more,
    # because its log was added to mixing logits
    return tf.nn.softmax(self._mixing_logits, 1)

  def _component_log_prob(self, x):
    lp = self._distributions.log_prob(x)
    return lp

  def log_prob(self, x):
    x = tf.expand_dims(x, 1)
    lp = self._component_log_prob(x)
    return tf.reduce_logsumexp(lp + self.mixing_log_prob, 1)

  def sample(self):
    raise NotImplementedError

  def mean(self):
    return tf.reduce_sum(self.mixing_prob * self._distributions.mean(), 1)

  def mode(self, straight_through_gradient=False, maximum=False):
    """Mode of the distribution.

    Args:
      straight_through_gradient: Boolean; if True, it uses the straight-through
        gradient estimator for the mode. Otherwise there is no gradient
        with respect to the mixing coefficients due to the `argmax` op.
      maximum: if True, attempt to return the highest-density mode.

    Returns:
      Mode.
    """
    mode_value = self._distributions.mode()
    mixing_log_prob = self.mixing_log_prob

    if maximum:
      mixing_log_prob += self._maybe_mask(self._component_log_prob(mode_value))

    mask = tf.one_hot(tf.argmax(mixing_log_prob, axis=1),
                      mixing_log_prob.shape[1], axis=1)

    if straight_through_gradient:
      soft_mask = tf.nn.softmax(mixing_log_prob, axis=1)
      mask = tf.stop_gradient(mask - soft_mask) + soft_mask

    return tf.reduce_sum(mask * mode_value, 1)


