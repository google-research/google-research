# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utils to do math in gin."""
import collections

import enum
import functools
import math
from typing import Any, Sequence, Optional

import gin


@gin.register
def mult(a, b):
  return a * b


@gin.register
def div(a, b):
  return a / b


@gin.register
def add(a, b):
  return a + b


@gin.register
def subtract(a, b):
  return a - b


@gin.register
def as_int(a):
  return int(a)


@gin.register
def ceil(a):
  return int(math.ceil(a))


@gin.register
def logical_not(a):
  return not a


@gin.register
def cond(condition, a, b):
  if condition:
    return a
  else:
    return b


@gin.register
def formatstr(format_string, values):
  """Utility function to format a string in gin config.

  Example: formatstr('Hello {}!', ['John']) -> 'Hello John!'.
  Args:
    format_string: A string of pattern to format.
    values: A sequence of string-convertible values to fill in the pattern. The
      number of values, i.e. len(values), need to match the number of '{}' in
      the format string. The content of the sequence needs to be convertible to
      strings.

  Returns:
    The formatted string.
  """
  return format_string.format(*values)


@gin.register
def get_enum_value(x):
  """Return the value of an input enum."""
  return x.value


_remap = collections.defaultdict(dict)


@gin.configurable
def set_remap(**kwargs):
  """Setter function to specify remapping of arguments via gin.

  This function updates a global state to remap input arguments of different
  functions. This global state is used in `remap_to` function decorator.

  Args:
    **kwargs: A dictionary of named mappings. A mapping is used for a single
      function and is a mapping from argument names accepted in the wrapper
      function to the argument names in the wrapped function. If argument names
      are the same, do not explicitly specify it here.
  """
  for key, value in kwargs.items():
    _remap[key].update(value)


def allow_remapping(fn = None, name = ''):
  """Decorator to allow remapping arguments of a function via gin.

  Please see `set_remap` for background. Use this function to annotate functions
  that might need remapping. This is an example usage:

  >>> @allow_remapping
  >>> def f(x, y):
  >>>   ...

  >>> # In gin config
  >>> set_remap:
  >>>   f = {'a': 'x', 'b': 'y'}

  This results in decorated f to behave exactly as f unless it gets invoked with
  `_do_remap=True`, in which case it accepts 'a' and 'b' arguments and route
  them to 'x' and 'y' respectively.

  You can also specify another name for remap to use:

  >>> @allow_remapping(name='new_f')
  >>> def f(x, y):
  >>>   ...

  >>> # In gin config
  >>> set_remap:
  >>>   new_f = {'a'.: 'x', 'b': 'y'}

  Args:
    fn: the base function to allow remapping on.
    name: the name of the remapping dict that is specified in gin to remap kw
      argument of the `fn`. This remapping should be specified via the
      `set_remap` function. If empty, we use `fn.__qualname__` but replace '.'
      with '_' because gin treats '.' as a separator and it cannot be used a
      part of a qualifier name.

  Returns:
    a wrapped function that invokes `fn` with no change unless it is called with
    `_do_remap=True` argument in which it remaps the keyword arguments according
    to `set_remap` specifications.
  """

  def wrapper(base):

    @functools.wraps(base)
    def wrapped(*args, _do_remap=False, **kwargs):
      # trigger gin populating the remap dictionary by calling set_remap()
      set_remap()
      remap = _remap[name or base.__qualname__.replace('.', '_')]
      mapped_kwargs = {
          remap[k] if k in remap and _do_remap else k: v
          for k, v in kwargs.items()
      }
      return base(*args, **mapped_kwargs)

    return wrapped

  if fn is not None:
    return wrapper(fn)
  else:
    return wrapper
