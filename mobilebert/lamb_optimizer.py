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

"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow.compat.v1 as tf

from mobilebert import distill_util


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               exclude_from_layer_adaptation=None,
               name="LAMBOptimizer",
               use_layer_wise_warmup=False,
               total_warmup_phases=0,
               num_train_steps=0):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)
    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
    # arg is None.
    if exclude_from_layer_adaptation:
      self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
    else:
      self.exclude_from_layer_adaptation = exclude_from_weight_decay
    self.use_layer_wise_warmup = use_layer_wise_warmup
    if total_warmup_phases == 0:
      self.steps_per_phase = 1
    else:
      self.steps_per_phase = num_train_steps // total_warmup_phases

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    background_lr = distill_util.get_background_lr(
        global_step=global_step, steps_per_phase=self.steps_per_phase)
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue
      param_name = self._get_variable_name(param.name)
      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      if self.use_layer_wise_warmup:
        # Use model-specific name spaces to get layer id.
        if param_name.startswith("bert/encoder/layer_"):
          layer_id = int(param_name[len("bert/encoder/layer_"):].split("/",
                                                                       1)[0])
          layer_wise_lr = distill_util.layer_wise_learning_rate(
              layer_id=layer_id,
              steps_per_phase=self.steps_per_phase,
              background_lr=background_lr)
          layer_wise_gate = tf.where(
              tf.math.greater(layer_wise_lr, 0.0), 1.0, 0.0)
        else:
          layer_wise_lr = 0.0
          layer_wise_gate = 0.0
      else:
        layer_wise_lr = 1.0
        layer_wise_gate = 1.0
      # Standard Adam update.
      next_m = layer_wise_gate * (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = layer_wise_gate * (
          tf.multiply(self.beta_2, v) +
          tf.multiply(1.0 - self.beta_2, tf.square(grad)))
      update = next_m / (tf.sqrt(next_v) + self.epsilon)
      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += layer_wise_gate * self.weight_decay_rate * param
      ratio = 1.0
      if self._do_layer_adaptation(param_name):
        w_norm = tf.linalg.norm(param, ord=2)
        g_norm = tf.linalg.norm(update, ord=2)
        ratio = tf.where(tf.math.greater(w_norm, 0), tf.where(
            tf.math.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)
      update_with_lr = layer_wise_lr * ratio * self.learning_rate * update
      next_param = param - update_with_lr
      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _do_layer_adaptation(self, param_name):
    """Whether to do layer-wise learning rate adaptation for `param_name`."""
    if self.exclude_from_layer_adaptation:
      for r in self.exclude_from_layer_adaptation:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
