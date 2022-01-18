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

"""Custom getters for use with the tensorflow get_variable "custom_getter"."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class DynamicCustomGetter(object):
  """A dynamic custom getter to modify variables.
  """

  def __init__(self):
    self._stack = []

  def __call__(self, getter, name, *args, **kwargs):
    """The custom getter.

    Do not call directly, instead pass to a variable_scope.

    Args:
      getter: callable the default getter
      name: str name of variable to get
      *args: Additional args
      **kwargs: Additional kwargs

    Returns:
      tf.Tensor or tf.Variable
    """
    if self._stack:
      return getter(name, *args, **kwargs)
    else:
      return self._stack[-1](getter, name, *args, **kwargs)

  @contextlib.contextmanager
  def use_getter(self, getter):
    """Context for using local variables."""
    self._stack.append(getter)
    yield
    self._stack.pop(-1)


@gin.configurable
class ESCustomGetter(object):
  """A custom getter that perturbs weights.

  For use when computing evolutionary strategies(ES) gradient estimators.
  """

  def __init__(self, std=0.01):
    """ESCustomGetter initializer.

    Args:
      std: float
        standard deviation for ES
    """
    self._do_antithetic = False
    self._always_resample = False
    self._no_sample = False
    self._common_noise = {}
    self.std = std
    self.verbose = False

  def __call__(self, getter, name, *args, **kwargs):
    """The custom getter.

    Do not call directly, instead pass to a variable_scope.

    Args:
      getter: callable the default getter
      name: str name of variable to get
      *args: additional args
      **kwargs: additional kwargs

    Returns:
      tf.Tensor or tf.Variable
    """
    var = getter(name, *args, **kwargs)
    if self._no_sample:
      return var

    if kwargs["trainable"]:
      if self.verbose:
        tf.logging.info("Doing getter on %s" % name)
      return var + self.get_perturbation(name, var)
    else:
      return var

  def gradient_scale(self):
    return 1. / (2 * self.std**2)

  def get_perturbation(self, name, var):
    """Get the perturbation for a variable."""
    if self._always_resample:
      return tf.random_normal(shape=var.shape, dtype=tf.float32) * self.std

    if name not in self._common_noise:
      with tf.control_dependencies(None):
        self._common_noise[name] = tf.random_normal(
            shape=var.shape, dtype=tf.float32)
    perturb = self._common_noise[name] * self.std

    if self._do_antithetic:
      return -perturb
    else:
      return perturb

  def get_perturbations(self, var_list):
    return [self.get_perturbation(v.op.name, v) for v in var_list]

  @contextlib.contextmanager
  def antithetic_sample(self, verbose=False):
    assert not self._do_antithetic, "Do not nest antithetic_sample"
    self._do_antithetic = True
    was_verbose = self.verbose
    self.verbose = verbose
    yield
    self._do_antithetic = False
    self.verbose = was_verbose

  @contextlib.contextmanager
  def always_resample(self):
    self._always_resample = True
    yield
    self._always_resample = False

  @contextlib.contextmanager
  def no_sample(self):
    self._no_sample = True
    yield
    self._no_sample = False
