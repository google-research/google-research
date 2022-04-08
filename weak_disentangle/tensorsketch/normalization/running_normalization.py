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

"""Running Normalization utilities."""

# pylint: disable=g-bad-import-order, g-direct-tensorflow-import
import tensorflow.compat.v1 as tf

from weak_disentangle.tensorsketch.normalization.base import Norm
from weak_disentangle.tensorsketch.modules.base import build_with_name_scope
from weak_disentangle.tensorsketch.utils import assign_moving_average


class RunningNorm(Norm):
  """Running Normalization class.
  """

  NAME = "running_norm"

  def __init__(self, affine=True, momentum=0.9, epsilon=1e-5, fuse=None,
               use_out_hook=None, name=None):
    super().__init__(name=name)
    self.affine = affine
    self.scale = None
    self.bias = None
    self.running_mean = None
    self.running_variance = None
    self.momentum = momentum
    self.epsilon = epsilon
    self.fuse = fuse
    self.use_out_hook = use_out_hook
    raise NotImplementedError("This module is not properly implemented yet.")

  @build_with_name_scope
  def build_parameters(self, x):
    num_dims = self.num_dims = x.shape[-1]
    if self.fuse is None:
      self.fuse = len(x.shape) == 4

    if self.affine:
      self.scale = tf.Variable(tf.ones([num_dims]), trainable=True)
      self.bias = tf.Variable(tf.zeros([num_dims]), trainable=True)

    if self.fuse and not self.affine:
      self.scale = tf.ones([num_dims])
      self.bias = tf.zeros([num_dims])

    self.running_mean = tf.Variable(tf.zeros([num_dims]), trainable=False)
    self.running_variance = tf.Variable(tf.ones([num_dims]), trainable=False)

  def reset_parameters(self):
    if self.affine:
      self.scale.assign(tf.ones(self.scale.shape))
      self.bias.assign(tf.zeros(self.bias.shape))

    self.running_mean.assign(tf.zeros(self.running_mean.shape))
    self.running_variance.assign(tf.ones(self.running_variance.shape))

  def forward(self, x):
    return self.normalize(x, self.scale, self.bias,
                          self.momentum, self.epsilon, self.training,
                          self.running_mean, self.running_variance, self.fuse)

  @staticmethod
  def normalize(x, scale, bias, momentum, epsilon, training=True,
                running_mean=None, running_variance=None, fuse=False):
    if fuse:
      x, mean, variance = tf.nn.fused_batch_norm(x, scale, bias,
                                                 mean=running_mean,
                                                 variance=running_variance,
                                                 epsilon=epsilon,
                                                 is_training=False)
    else:
      x = tf.nn.batch_normalization(x, running_mean, running_variance,
                                    bias, scale, epsilon)
      mean, variance = tf.nn.moments(x, list(range(len(x.shape) - 1)))

    if training:
      assign_moving_average(running_mean, mean, momentum)
      assign_moving_average(running_variance, variance, momentum)

    return x

  @staticmethod
  def add(module, affine=True, momentum=0.9, epsilon=1e-5, fuse=None,
          use_out_hook=True):
    Norm.add(module,
             RunningNorm(affine, momentum, epsilon, fuse, use_out_hook),
             use_out_hook=use_out_hook)

  @staticmethod
  def remove(module, use_out_hook=False):
    Norm.remove(module, RunningNorm.NAME, use_out_hook)

  def extra_repr(self):
    main = "({}, {}, {}".format(self.affine,
                                self.momentum,
                                self.epsilon)

    if self.use_out_hook is not None:
      main += ", out" if self.use_out_hook else ", in"
    main += ")"
    return main
