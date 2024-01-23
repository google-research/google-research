# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""jax-effects experimental APIs."""

import functools
from typing import Any, Callable, ParamSpec, TypeVar

from jax_effects._src import core
from jax_effects._src import handler

Handler = core.Handler

################################################################################
# Handler decorator
################################################################################

# Unique object, used as the default value for `arg` in `handle`.
_DEFAULT_HANDLER_ARGUMENT = handler._DEFAULT_HANDLER_ARGUMENT  # pylint: disable=protected-access


T = TypeVar('T')
P = ParamSpec('P')
Result = TypeVar('Result')
ReturnFn = Callable[[Result], Any] | Callable[[T, Result], Any]


def handle(
    arg = _DEFAULT_HANDLER_ARGUMENT,
    parameterized = None,
    return_fun = None,
    **handler_impls,
):
  """Decorator function for registering effect handlers."""

  def decorator(f):
    @functools.wraps(f)
    def body(*args, **kwargs):
      with Handler(
          arg=arg,
          parameterized=parameterized,
          return_fun=return_fun,
          **handler_impls,
      ) as h:
        h.result = f(*args, **kwargs)
      return h.result

    return body

  return decorator
