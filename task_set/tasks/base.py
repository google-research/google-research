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

"""Base classes for tasks.

All tasks inherit from `BaseTask`.
"""

import abc
import collections
from typing import Union, Dict, Text, Callable, Tuple, Any, Optional, List
import sonnet as snt

from task_set import datasets
from task_set import variable_replace
import tensorflow.compat.v1 as tf

nest = tf.nest


class _BaseModel(snt.AbstractModule):
  """Sonnet module that exposes variables.

  By default, sonnet modules are a function of only data, with variables hidden.
  This class exposes a `call_with_values` function which instead explicitally
  is a function of both variables and inputs. This is useful for meta-learning
  learning applications.
  """

  def __init__(self,
               module_fn,
               name = "BaseModel"):
    """Initialize a _BaseModel that wraps the module_fn.

    Args:
      module_fn: Function that returns a sonnet module that will be wrapped.
      name: Name of this sonnet module.
    """
    self.context = variable_replace.VariableReplaceGetter(verbose=False)
    super(_BaseModel, self).__init__(name=name, custom_getter=self.context)

    with self._enter_variable_scope():
      self.mod = module_fn()

  def _build(self, inp):
    """The forward pass.

    Do not call this with __call__, instead use
    call_with_values or call_with_variables.
    Args:
      inp: input batch of data passed to module.

    Returns:
      output of module
    """
    return self.mod(inp)

  def call_with_values(self, values, inp):
    """Call with values instead of the hidden tf.Variable.

    Args:
      values: dict {str:tf.Tensor} map of names to tensors. This is used for
        variable substitution.
      inp: input passed to mlp

    Returns:
      output of module
    """
    with self.context.use_value_dict(values):
      return self(inp)

  def call_with_variables(self, inp):
    """Call this module but use the tf.Variables created from sonnet module.

    Args:
      inp: input tensor data passed to module

    Returns:
      output of module
    """
    with self.context.use_variables():
      return self(inp)

  def get_initialized_value_dict(self):
    """Get initial values for current weights.

    Returns:
      dict of string names to tf.Tensor.
    """
    return self.context.get_initialized_value_dict()

  def get_variable_dict(self):
    """Get a dictionary of variables created by this module.

    Returns:
      dict of string names to tf.Variable.
    """
    return self.context.get_variable_dict()


class BaseTask(snt.AbstractModule):  # pytype: disable=ignored-metaclass
  """Baseclass for all tasks.

  A task represents a possibly stochastic parametric function mapping params to
  a scalar loss value.

  Attributes:
    name: str containing the name of the task.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, **kwargs):
    self.name = None
    super(BaseTask, self).__init__(**kwargs)

  def _build(self):
    raise ValueError("Do not call this! Sonnet module used only for variable"
                     "scoping")

  def gradients(self, loss,
                params):
    """"Returns an ordered dict mapping parameter name to gradients."""
    grads = tf.gradients(loss, list(params.values()))
    name_grads = [(name, g) for name, g in zip(params.keys(), grads)]
    return collections.OrderedDict(name_grads)

  @abc.abstractmethod
  def call_split(
      self,
      params,
      split,
      batch = None,
      with_metrics = False
  ):
    """Perform a forward pass of the task."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_batch(self, split):
    """Get a batch of train data if required, otherwise None."""
    raise NotImplementedError()

  @abc.abstractmethod
  def initial_params(self):
    """Initial values of parameters."""
    raise NotImplementedError()

  @abc.abstractmethod
  def current_params(self):
    """tf.Variables for the current parameters."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_variables(self):
    raise NotImplementedError()


LossAndAux = collections.namedtuple("LossAndAux", ["loss", "aux"])


class DatasetModelTask(BaseTask):
  """A task consisting of a dataset and a sonnet module."""

  def __init__(self, loss_fn,
               datasets_obj, **kwargs):
    """Initializer.

    Args:
      loss_fn: A function that returns a sonnet module. The sonnet module should
        be a function that takes in the datatype produced by `dataset_fn` and
        should return either a scalar loss value, or a `LossAndAux` object.
      datasets_obj: A Datasets object containing the `tf.data.Dataset`s to use
        for this task.
      **kwargs:
    """
    super(DatasetModelTask, self).__init__(**kwargs)

    self.datasets = datasets_obj
    self.iterators = nest.map_structure(lambda x: x.make_one_shot_iterator(),
                                        self.datasets)

    self.base_model = _BaseModel(loss_fn)

    # This is a hack to create the tf variables of the base_model before usage.
    with tf.name_scope("dummy_graph"):
      b = self.iterators.train.get_next()
      loss_tmp = self.base_model.call_with_variables(b)
      if isinstance(loss_tmp, LossAndAux):
        loss_tmp = loss_tmp.loss
      if not loss_tmp.shape:
        raise ValueError("Wrapped loss must return a scalar!")

  def get_batch(self, split):
    """Get a batch of data from the given split.

    Args:
      split: split to take data from.

    Returns:
      A batch of data.
    """
    if split == datasets.Split.TRAIN:
      return self.iterators.train.get_next()

    elif split == datasets.Split.VALID_INNER:
      return self.iterators.valid_inner.get_next()

    elif split == datasets.Split.VALID_OUTER:
      return self.iterators.valid_outer.get_next()

    elif split == datasets.Split.TEST:
      return self.iterators.test.get_next()

    else:
      raise ValueError("Split not supported.")

  def call_split(
      self,
      params,
      split,
      batch = None,
      with_metrics = False
  ):
    """Perform a forward pass with current params.

    Args:
      params: params to use for forward pass.
      split: split of data to use.
      batch: optional batch of data to compute over.
      with_metrics: flag to turn off and off extra metrics.

    Returns:
      Scalar loss computed over a batch of data.
    """
    if batch is None:
      batch = self.get_batch(split)

    loss_or_aux = self.base_model.call_with_values(params, batch)

    if isinstance(loss_or_aux, LossAndAux):
      loss = loss_or_aux.loss
      aux = loss_or_aux.aux
    else:
      loss = loss_or_aux
      aux = {}

    if with_metrics:
      return loss, aux
    else:
      return loss

  def initial_params(self):
    """"Returns an ordered dict mapping parameter name to initial value."""
    return self.base_model.get_initialized_value_dict()

  def current_params(self):
    """"Returns an ordered dict mapping parameter name to current value."""
    return self.base_model.get_variable_dict()

  def get_variables(self):
    return list(self.base_model.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))
