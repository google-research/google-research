# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Timeout function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading


class FunctionTimeoutError(Exception):
  """Timeout Error."""
  pass


class RunWithTimeout(object):
  """Runs a python function with a timeout and gets a returned value.

  NOTE(leeley): This class is forked from answer in
  https://stackoverflow.com/questions/46858493/python-run-a-function-with-a-timeout-and-get-a-returned-value
  I added a FunctionTimeoutError when time limit is reached.
  """

  def __init__(self, function, args, name=None):
    """Initializer.

    Args:
      function: Callable, function to run.
      args: Tuple of function arguments.
      name: String, the name of the function. Default None, function.__name__
          will be used.
    """
    self.function = function
    self.args = args
    self.answer = None
    if name is None:
      self.name = function.__name__
    else:
      self.name = name

  def _worker(self):
    self.answer = self.function(*self.args)

  def run(self, time_limit_seconds):
    """Runs function.

    Args:
      time_limit_seconds: Float, timeout limit in seconds.

    Returns:
      output of function.

    Raises:
      FunctionTimeoutError: If output of the answer is None.
    """
    thread = threading.Thread(target=self._worker)
    thread.start()
    thread.join(time_limit_seconds)
    if self.answer is None:
      raise FunctionTimeoutError(
          '%s timeout after %f.' % (self.name, time_limit_seconds))
    return self.answer
