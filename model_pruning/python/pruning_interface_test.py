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

# Lint as: python3
"""Tests for pruning_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from model_pruning.python import pruning
from model_pruning.python import pruning_interface


class MockWeightInit(object):
  """Mock class providing weight initialization config params."""

  @staticmethod
  def Constant(scale):
    """Constant initialization."""
    return {"scale": scale}


class MockLSTMVars(object):
  """Mock LSTM vars."""

  def __init__(self):
    self.wm = None
    self.mask = None
    self.threshold = None


class MockLSTMCell(object):
  """Mock LSTM cell."""

  def __init__(self):
    self._private_vars = {}
    self._private_theta = {}
    self.vars = MockLSTMVars()

  def CreateVariable(self, name, var_params, theta_fn=None, trainable=False):
    dtype = var_params["dtype"]
    shape = var_params["shape"]
    scale = var_params["init"]["scale"]

    v_init = tf.constant_initializer(value=scale, dtype=dtype)
    with tf.variable_scope("MockLSTMCell"):
      var = tf.get_variable(name, shape, dtype, v_init, trainable=trainable)
    value = var
    if theta_fn is not None:
      value = theta_fn(value)
    self._private_vars[name] = var
    self._private_theta[name] = value

    if name == "wm":
      self.vars.wm = var
    elif name == "mask":
      self.vars.mask = var
    elif name == "threshold":
      self.vars.threshold = var
    else:
      raise ValueError("name {} is not supported".format(name))


class PruningSpeechUtilsTest(tf.test.TestCase):
  PARAM_LIST = [
      "name=test", "threshold_decay=0.9", "pruning_frequency=10",
      "sparsity_function_end_step=100", "target_sparsity=0.9",
      "weight_sparsity_map=[conv1:0.8,conv2/kernel:0.8]",
      "block_dims_map=[dense1:4x4,dense2:1x4]"
  ]
  TEST_HPARAMS = ",".join(PARAM_LIST)

  def setUp(self):
    super(PruningSpeechUtilsTest, self).setUp()
    # Add global step variable to the graph
    self.global_step = tf.train.get_or_create_global_step()
    # Add sparsity
    self.sparsity = tf.Variable(0.5, name="sparsity")
    # Parse hparams
    self.pruning_hparams = pruning.get_pruning_hparams().parse(
        self.TEST_HPARAMS)

    self.pruning_obj = pruning.Pruning(
        self.pruning_hparams, global_step=self.global_step)

    def MockWeightParamsFn(shape, init=None, dtype=None):
      if init is None:
        init = MockWeightInit.Constant(0.0)
      if dtype is None:
        dtype = tf.float32
      return {"dtype": dtype, "shape": shape, "init": init}

    self.mock_weight_params_fn = MockWeightParamsFn
    self.mock_lstmobj = MockLSTMCell()
    self.wm_pc = np.zeros((2, 2))

  def testApplyPruning(self):
    pruning_obj = pruning_interface.apply_pruning(self.pruning_obj,
                                                  self.pruning_hparams,
                                                  self.mock_weight_params_fn,
                                                  MockWeightInit,
                                                  self.mock_lstmobj,
                                                  self.wm_pc, tf.float32)

    self.assertEqual(pruning_obj, self.pruning_obj)

  def testGetPruningUpdate(self):
    mask_update_op = pruning_interface.get_pruning_update(
        self.pruning_obj, self.pruning_hparams)
    self.assertNotEqual(mask_update_op, tf.no_op())


if __name__ == "__main__":
  tf.test.main()
