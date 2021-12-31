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
"""Custom components of vatt, e.g. BN_ReLU, etc.."""

from absl import logging

import tensorflow as tf


class BNReLU(tf.keras.layers.Layer):
  """Does BN + ReLU with cross replica option."""

  def __init__(self,
               bn_config,
               use_xreplica_bn=True,
               use_relu=True,
               use_bn=True,
               name="bn_relu"):
    super(BNReLU, self).__init__(name=name)
    self.use_relu = use_relu
    self.use_bn = use_bn
    assert use_bn or use_relu, "Either relu or bn should be specified"
    if use_bn:
      if use_xreplica_bn:
        logging.info("Using Cross Replica BatchNorm.")
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            **bn_config)
      else:
        self.bn = tf.keras.layers.BatchNormalization(**bn_config)

  def call(self,
           inputs,
           is_training=False):
    if self.use_bn:
      inputs = self.bn(inputs, training=is_training)
    if self.use_relu:
      inputs = tf.nn.relu(inputs)

    return inputs


class NonLinearProj(tf.keras.layers.Layer):
  """Non-linear projection head."""

  def __init__(self,
               d_inner,
               d_embd,
               bn_config,
               use_xreplica_bn=True,
               use_inner_bn=True,
               use_bn_out=False,
               name="non_linear_proj"):
    super(NonLinearProj, self).__init__(name=name)
    self._bn_config = bn_config
    self._use_xreplica_bn = use_xreplica_bn
    self._use_inner_bn = use_inner_bn
    self._use_bn_out = use_bn_out

    self.dense_inner = None
    if d_inner is not None:
      self.dense_inner = tf.keras.layers.Dense(
          d_inner, name="final_projection_inner")
    self.bn_relu = BNReLU(
        bn_config=self._bn_config,
        use_xreplica_bn=self._use_xreplica_bn,
        use_relu=True,
        use_bn=self._use_inner_bn,
        name="final_projection_inner_bn_relu")
    self.dense_final = tf.keras.layers.Dense(
        d_embd, use_bias=not self._use_bn_out, name="final_projection")
    if self._use_bn_out:
      self.bn_out = BNReLU(
          bn_config=self._bn_config,
          use_xreplica_bn=self._use_xreplica_bn,
          use_relu=False,
          use_bn=True,
          name="final_projection_bn")

  def call(self,
           inputs,
           is_training):
    if self.dense_inner is None:
      d_inner = inputs.shape[-1]
      self.dense_inner = tf.keras.layers.Dense(d_inner,
                                               name="final_projection_inner")

    inputs = self.dense_inner(inputs)
    inputs = self.bn_relu(inputs, is_training)
    inputs = self.dense_final(inputs)
    if self._use_bn_out:
      inputs = self.bn_out(inputs, is_training)

    return inputs


class ReluDenseBN(tf.keras.layers.Layer):
  """Relu + Dense + BN module."""

  def __init__(self,
               d_model,
               pre_bn=False,
               bn_config=None,
               use_xreplica_bn=True,
               name="relue_dense_relu"):
    super(ReluDenseBN, self).__init__(name=name)
    self.pre_bn = pre_bn
    if use_xreplica_bn:
      logging.info("Using Cross Replica BatchNorm in Relu-Dense-BN.")
      bn_module = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_module = tf.keras.layers.BatchNormalization

    if bn_config is None:
      bn_config = {"scale": True}
    if use_xreplica_bn:
      bn_config.update({"momentum": 0.9})

    if self.pre_bn:
      self.pre_bn = bn_module(**bn_config)

    self.dense = tf.keras.layers.Dense(d_model,
                                       use_bias=False,
                                       name="linear_projection")
    self.bn = bn_module(**bn_config)

  def call(self, inputs, training):
    if self.pre_bn:
      inputs = self.pre_bn(inputs, training)

    inputs = tf.nn.relu(inputs)
    inputs = self.dense(inputs)
    inputs = self.bn(inputs, training)

    return inputs
