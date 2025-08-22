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

"""Task utility function to define the modeling Task API.
"""
import abc
import enum
from typing import Callable, Optional, Sequence, Any

import flax
from flax import struct
import gin

from utils.types import LossFn
from utils.types import Metric
from utils.types import NestedTextDictArray


BaseTaskHead = Any


@gin.configurable
class TaskSet(enum.Enum):
  """Task type for multi-task learning."""
  DETECTION = 'detection'
  BASE = 'base'


@flax.struct.dataclass
class BaseTask(abc.ABC):
  """Base Task abstract class for other tasks to inherit from."""
  head: Optional[Callable[Ellipsis, BaseTaskHead]] = None
  _loss: Optional[LossFn] = None
  metric: Optional[Metric] = None
  name: str = TaskSet.BASE.value
  loss_weight: float = 1.0

  def loss(self, outputs, labels, **kwargs):
    """Compute task-specific loss from multitask outputs and labels.

    First filter the task-specific outputs and labels and then call the
    task-specific loss.
    Args:
      outputs: A dictionary with the following structure:
        key name: {task_name}/{output_name}.
        value content: corresponding task specific outputs.
      labels: A dictionary with the following structure:
        key name: {task_name}/{label_name}.
        value content: corresponding task specific labels.
      **kwargs: Additional arguments.

    Returns:
      task_model_loss: Total model loss for this task.
      task_loss: A dictionary of task-specific losses.

    Raises:
      KeyError: Missing task-specific outputs or labels.
    """
    del kwargs
    task_outputs, task_labels = (self.filter_by_task(outputs),
                                 self.filter_by_task(labels))
    if not task_outputs:
      raise KeyError(f'No task outputs for task: {self.name}!')
    if task_labels is None:
      raise KeyError(f'No task labels for task: {self.name}!')

    task_loss = self._loss(task_outputs, task_labels)

    if isinstance(task_loss, dict):
      model_loss = task_loss['model_loss']
    else:
      model_loss = task_loss
    return model_loss, self.prepend_by_task(task_loss)

  @classmethod
  def filter_by_task(cls, inputs):
    """Filter the input dictionary by task name."""
    return inputs[cls.name]

  @classmethod
  def unfilter_by_task(cls, inputs):
    """Un-filter the input dictionary by task name."""
    return {cls.name: inputs}

  @classmethod
  def prepend_by_task(cls, inputs):
    """Prepend the input dictionary by task name.

    Args:
      inputs: A dictionary of arrays.

    Returns:
      outputs: A dictionary where the scope name is prepended. For example,
        'box_outputs' -> 'detection/box_outputs'.
    """
    outputs = {}
    pattern = f'{cls.name}/'
    if isinstance(inputs, dict):
      for k, v in inputs.items():
        outputs[pattern + k] = v
    else:
      outputs[pattern] = inputs
    return outputs

# Task for multitask learning.
Task = BaseTask
Tasks = Sequence[BaseTask]


@gin.register
@flax.struct.dataclass
class DetectionTask(BaseTask):
  """Detection task template."""
  name: str = struct.field(
      pytree_node=False, init=True, default=TaskSet.DETECTION.value)
