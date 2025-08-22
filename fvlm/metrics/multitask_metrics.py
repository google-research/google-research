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

"""Evaluation metrics for Multi-Task setup."""
from clu import metrics
import flax
import gin

from utils import gin_utils
from utils.task_utils import Tasks
from utils.types import DictArray
from utils.types import NestedDictArray


@gin.configurable
def get_multitask_metrics(metric_tasks = ()):
  """A wrapper to receive tasks from gin config."""

  @flax.struct.dataclass
  class MultiTaskMetric(metrics.Metric):
    """MultiTaskMetric.

    This metric aggregates sub-metrics in the metric_dict and return the metrics
    of all of them by calling them separately.

    Attributes:
      tasks: A sequence of tasks to compute metrics over.
    """
    tasks: Tasks = metric_tasks

    @classmethod
    @gin_utils.allow_remapping(name='get_multitask_metrics')
    def from_model_output(cls, outputs,
                          labels):
      """Accumulates model outputs for evaluation.

      Args:
        outputs: A dictionary with the following structure:
          key name: Task name.
          value content: A dictionary to corresponding task specific outputs.
        labels: A dictionary with the following structure:
          key name: Task name.
          value content: A dictionary corresponding task specific labels.

      Returns:
        A metric object initialized from the outputs and labels.

      Raises:
        KeyError: Missing task-specific outputs or labels.
      """
      new_tasks = []
      for task in cls.tasks:
        task_outputs, task_labels = (
            task.filter_by_task(outputs), task.filter_by_task(labels))
        if not task_outputs:
          raise KeyError(f'No task outputs for task: {task.name}!')
        if task_labels is None:
          raise KeyError(f'No task labels for task: {task.name}!')

        metric = task.metric.from_model_output(task_outputs, task_labels)
        new_tasks.append(type(task)(metric=metric))

      return cls(tasks=new_tasks)

    def merge(self, other):
      new_tasks = []
      assert len(self.tasks) == len(other.tasks)
      for task, other_task in zip(self.tasks, other.tasks):
        metric = task.metric.merge(other_task.metric)
        new_tasks.append(type(task)(metric=metric))

      return type(self)(tasks=new_tasks)

    def reduce(self):
      new_tasks = []
      for task in self.tasks:
        metric = task.metric.reduce()
        new_tasks.append(type(task)(metric=metric))

      return type(self)(tasks=new_tasks)

    def compute(self):
      output_metric = {}
      for task in self.tasks:
        task_metric = task.metric.compute()
        output_metric.update(task.prepend_by_task(task_metric))

      return output_metric

  return MultiTaskMetric
