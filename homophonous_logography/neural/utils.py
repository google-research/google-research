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

"""Collection of simple utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


class DualLogger(object):
  """Log to file and terminal: https://stackoverflow.com/questions/14906764."""

  def __init__(self, filename, mode="wt"):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
      os.makedirs(directory)
    self.terminal = sys.stdout
    self.log = open(filename, mode, encoding="utf-8")

  def _flush(self):
    self.terminal.flush()
    self.log.flush()

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)
    self._flush()

  def flush(self):
    # This flush method is needed for python 3 compatibility.
    self._flush()
    pass
