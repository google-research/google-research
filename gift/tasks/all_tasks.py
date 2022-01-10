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

"""Mapping of all defined tasks in the project.

 task name --> task class.
"""
from gift.tasks import task

ALL_SINGLE_ENV_TASKS = {
    'cls': task.ClassificationTask,
}

# TODO(samiraabnar): Refactor the desgin of the task and dataset classes.
ALL_MULTI_ENV_TASKS = {
    'multi_env_cls': task.MultiEnvClassificationTask,
    'multi_env_irm_cls': task.MultiEnvIRMClassificationTask,
    'multi_env_vrex_cls': task.MultiEnvVRexClassificationTask,
}

ALL_MULTI_ENV_WITH_REPS = {
    'multi_env_dm_cls':
        task.MultiEnvLinearDomainMappingClassification,
    'multi_env_nl_dm_cls':
        task.MultiEnvNonLinearDomainMappingClassification,
    'multi_env_hungarian_dm_cls':
        task.MultiEnvHungarianDomainMappingClassification,
    'multi_env_identity_dm_cls':
        task.MultiEnvIdentityDomainMappingClassification,
    'multi_env_sinkhorn_dm_cls':
        task.MultiEnvSinkhornDomainMappingClassification,
}

ALL_MULTI_ENV_DOMAIN_ADVERSARIALS = {
    'multi_env_dann_cls': task.MultiEnvDannClassification
}

ALL_TASKS = {}
ALL_TASKS.update(ALL_SINGLE_ENV_TASKS)
ALL_TASKS.update(ALL_MULTI_ENV_TASKS)
ALL_TASKS.update(ALL_MULTI_ENV_WITH_REPS)
ALL_TASKS.update(ALL_MULTI_ENV_DOMAIN_ADVERSARIALS)


def get_task_class(task_name):
  """Maps dataset name to a dataset_builder.

  Args:
    task_name: string; Name of the task.

  Returns:
    A dataset builder.
  """
  if task_name not in ALL_TASKS.keys():
    raise ValueError('Unrecognized task: {}'.format(task_name))
  return ALL_TASKS[task_name]
