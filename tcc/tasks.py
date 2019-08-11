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

"""List all available tasks."""

from tcc.evaluation.algo_loss import AlgoLoss
from tcc.evaluation.classification import Classification
from tcc.evaluation.event_completion import EventCompletion
from tcc.evaluation.few_shot_classification import FewShotClassification
from tcc.evaluation.kendalls_tau import KendallsTau

TASK_NAME_TO_TASK_CLASS = {
    'algo_loss': AlgoLoss,
    'kendalls_tau': KendallsTau,
    'classification': Classification,
    'event_completion': EventCompletion,
    'few_shot_classification': FewShotClassification,
}


def get_tasks(task_names):
  """Returns evaluation tasks."""
  iterator_tasks = {}
  embedding_tasks = {}

  for task_name in list(set(task_names)):
    if task_name not in TASK_NAME_TO_TASK_CLASS.keys():
      raise ValueError('%s not supported yet.' % task_name)

    task = TASK_NAME_TO_TASK_CLASS[task_name]()
    if task.downstream_task:
      embedding_tasks[task_name] = task
    else:
      iterator_tasks[task_name] = task

  return iterator_tasks, embedding_tasks
