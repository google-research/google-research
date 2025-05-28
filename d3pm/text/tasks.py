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

"""Single import to access / gin register tasks."""

from d3pm.text import diffusion  # pylint: disable=unused-import
from d3pm.text import types


class TaskRegistry:
  """A registry containing all tasks supported by D3PM."""

  def __init__(self):
    self.tasks = {}

  def clear(self):
    self.tasks = {}

  def register(self, name, task):
    self.tasks[name] = task

  def list_tasks(self):
    """Returns a list of the available tasks."""

    msg = "Available Tasks:\n\n"
    for name in self.tasks:
      msg += "* " + name + "\n"

    return msg

  def load(self, name):
    """Load a task registered with the TaskRegistry."""

    if name not in self.tasks:
      info_string = self.list_tasks()
      raise ValueError(
          f"Unable to find a tasks with the name {name}.\n\n{info_string}.")

    return self.tasks[name]


_REGISTRY = TaskRegistry()


def load(name):
  """Load a tasks registered with the D3PM task registry.

  Args:
    name: the name of the task to load.

  Returns:
    a D3PM task.
  """
  return _REGISTRY.load(name)


def register(name, task):
  """Register a task with the registry.

  Args:
    name: the name of the task to register.
    task: a task to register.

  Returns:
    a training and validation task.
  """
  _REGISTRY.register(name, task)

  return task
