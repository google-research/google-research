# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Allows logging of tensors in tf.Modules on TPU and in tf.functions.

The module allows tf.Modules with the LoggingModule mixin to log tensors at any
point in time without returning these tensors:

>>> class MyModule(tf.Module, LoggingModule):
>>>   def my_function(self, input_tensor):
>>>     ...
>>>     avg_value = tf.reduce_mean(input_tensor)
>>>     self.log('avg_value', avg_value)
>>>     ...

By default, the `self.log(...)` call is a no-op. However, using LoggingTape the
logging tensors can be "reacquired" in the same tf.function and then returned
by the tf.function.

>>> my_tf_module = MyModule()
>>>
>>> @tf.function
>>> def my_function(input_tensor):
>>>   with LoggingTape(my_tf_module) as logged_tensors:
>>>     result = my_tf_module.my_function(input_tensor)
>>>   return result, logged_tensors
"""

import collections
import tensorflow as tf


class _LoggingDict(object):
  """Custom dictionary wrapper used to avoid dependency checks."""

  def __init__(self, dict_):
    self._dict = dict_

  def __setitem__(self, key, item):
    self._dict[key] = item

  def __getitem__(self, key):
    return self._dict[key]

  def __repr__(self):
    return repr(self._dict)

  def __len__(self):
    return len(self._dict)

  def __delitem__(self, key):
    del self._dict[key]

  def __getattr__(self, name):
    return getattr(self._dict, name)

  def __contains__(self, item):
    return item in self._dict

  def __iter__(self):
    return iter(self._dict)


class LoggingModule(object):
  """Mixin that allows tf.Modules to add tensors for logging."""
  _logging_dict = None

  def log(self, key, tensor):
    """Registers `tensor` as `key` for logging."""
    if self._logging_dict is not None:
      if key in self._logging_dict:  # pylint: disable=unsupported-membership-test
        raise ValueError('Logging key already exists in current contex.')
      if not tf.is_tensor(tensor):
        raise ValueError('`tensor` needs to be a Tensor.')
      self._logging_dict[key] = tensor  # pylint: disable=unsupported-assignment-operation

  def set_logging_dict(self, logging_dict):
    """Sets the logging dict."""
    if not isinstance(logging_dict, dict):
      raise ValueError('`logging_dict` is not a dict.')
    if self._logging_dict is not None:
      raise ModuleAlreadyTapedError('Submodule `%s` is already taped.' %
                                    str(self))
    self._logging_dict = _LoggingDict(logging_dict)

  def unset_logging_dict(self):
    """Unsets the current logging dict."""
    self._logging_dict = None


class LoggingTape(object):
  """Context manager that allows to collect logging tensors from LoggingModules.

  Sample usage:

  >>> with LoggingTape(my_tf_module) as logged_tensors:
  >>>   result = my_tf_module.arbitrary_function(input_tensor)
  >>> return result, logged_tensors
  """

  def __init__(self, tracked_modules):
    """Creates a LoggingTape.

    Args:
      tracked_modules: Specifies a list of tf.Modules from which tensors should
        be collected for logging (if they are LoggingModules) Can either be a
        list of tf.Modules or a single tf.Module, in which case the tracked set
        of modules consists of the tf.Module and its set of submodules at
        construction time of the LoggingTape.
    """
    if isinstance(tracked_modules, tf.Module):
      tracked_modules = [tracked_modules] + list(tracked_modules.submodules)
    self._tracked_modules = list(tracked_modules)

  def __enter__(self):
    """Enters the context manager."""
    logged_tensors = collections.OrderedDict()
    for submodule in self._tracked_modules:
      if not isinstance(submodule, LoggingModule):
        continue
      submodule.set_logging_dict(logged_tensors)
    return logged_tensors

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the context manager."""
    del exc_type, exc_value, traceback
    for submodule in self._tracked_modules:
      if not isinstance(submodule, LoggingModule):
        continue
      submodule.unset_logging_dict()


class ModuleAlreadyTapedError(BaseException):
  """Error that signifies a submodule is already taped using LoggingTape."""
