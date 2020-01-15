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

# python3
"""Utilities for logging, neural network activations, and initializations."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
import tensorflow.compat.v1 as tf
import numpy as np
import gin
from functools import partial
import functools

from weak_disentangle import tensorsketch as ts


@gin.configurable
def log(*args, debug=False):
  if debug:
    print(*args)
  else:
    tf.logging.info(" ".join(map(str, args)))


def reset_parameters(m):
  m.reset_parameters()


# pylint: disable=invalid-name
def add_act(m, Act):
  m.act = Act()
  m.out_hooks.update(dict(act=lambda self, x: self.act(x)))


def remove_act(m):
  del m.act
  del m.out_hooks["act"]


# pylint: disable=unused-argument
@gin.configurable
def initializer(kernel, bias, method, layer):
  if method == "pytorch":
    pytorch_init(kernel, bias)
  elif method == "keras":
    keras_init(kernel, bias)


def pytorch_init(kernel, bias):
  fan_in, _ = ts.utils.compute_fan(kernel)
  limit = np.sqrt(1 / fan_in)
  kernel.assign(tf.random.uniform(kernel.shape, -limit, limit))

  if bias is not None:
    bias.assign(tf.random.uniform(bias.shape, -limit, limit))


def keras_init(kernel, bias):
  fan_in, fan_out = ts.utils.compute_fan(kernel)
  limit = np.sqrt(6 / (fan_in + fan_out))
  kernel.assign(tf.random.uniform(kernel.shape, -limit, limit))

  if bias is not None:
    bias.assign(tf.zeros(bias.shape))
