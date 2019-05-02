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

"""Automatically generated random variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect

from tensorflow_probability import distributions
from tensorflow_probability.python.util import docstring as docstring_util
from simple_probabilistic_programming.random_variable import RandomVariable
from simple_probabilistic_programming.trace import traceable


def make_random_variable(distribution_cls):
  """Factory function to make random variable given distribution class."""
  @traceable
  @functools.wraps(distribution_cls, assigned=("__module__", "__name__"))
  @docstring_util.expand_docstring(
      cls=distribution_cls.__name__,
      doc=inspect.cleandoc(distribution_cls.__init__.__doc__))
  def func(*args, **kwargs):
    # pylint: disable=g-doc-args
    """Create a random variable for ${cls}.

    See ${cls} for more details.

    Returns:
      RandomVariable.

    #### Original Docstring for Distribution

    ${doc}
    """
    # pylint: enable=g-doc-args
    sample_shape = kwargs.pop("sample_shape", ())
    value = kwargs.pop("value", None)
    return RandomVariable(distribution=distribution_cls(*args, **kwargs),
                          sample_shape=sample_shape,
                          value=value)
  return func


__all__ = ["make_random_variable"]
_globals = globals()
for candidate_name in sorted(dir(distributions)):
  candidate = getattr(distributions, candidate_name)
  if (inspect.isclass(candidate) and
      candidate != distributions.Distribution and
      issubclass(candidate, distributions.Distribution)):

    _globals[candidate_name] = make_random_variable(candidate)
    __all__.append(candidate_name)
