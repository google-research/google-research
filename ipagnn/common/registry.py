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

# Lint as: python3
"""A Registry is a singleton for registering interchangeable objects of a kind.
"""


class Registry(object):
  """A singleton for registering objects of a particular kind."""

  def __init__(self):
    self.items = {}

  def __call__(self, fn):
    key = fn.__name__
    self.items[key] = fn
    return fn

  def __getitem__(self, key):
    return self.items[key]
