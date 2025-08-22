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

"""Multi-task loss wrapper.
"""
import gin

from utils import gin_utils
from utils.task_utils import Tasks
from utils.types import DictArray
from utils.types import NestedDictArray


@gin.configurable
def get_multitask_loss_fn(tasks = ()):
  """Return multitask loss function.

  Args:
    tasks: A sequence of tasks to aggregate losses as defined here:
      learning/brain/voyager/val/utils/task_utils.py.

  Returns:
    A multitask loss function.
  """

  @gin_utils.allow_remapping(name='multitask_loss_fn')
  def multitask_loss_fn(outputs, labels,
                        **kwargs):
    """Compute Multi-task loss from model outputs and groundtruth labels.

    Args:
      outputs: A dictionary with the following structure:
        key name: {task_name}/{output_name}.
        value content: corresponding task specific outputs.
      labels: A dictionary with the following structure:
        key name: {task_name}/{label_name}.
        value content: corresponding task specific labels.
      **kwargs: Additional arguments.

    Returns:
      model_loss_dict: A dictionary of combined losses for the multi-task model.
        key name: {task_name}/{loss_name}.
        value: corresponding task specific losses.
        The total combined loss is stored in 'model_loss'.
    """
    del kwargs
    model_loss_dict = {}
    model_loss = 0.
    for task in tasks:
      task_model_loss, task_loss_dict = task.loss(outputs, labels)
      model_loss += task_model_loss * task.loss_weight  # Update total loss.
      model_loss_dict.update(task_loss_dict)  # Store task-specific loss.

    model_loss_dict['model_loss'] = model_loss
    return model_loss_dict

  return multitask_loss_fn
