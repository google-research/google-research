# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Various envs to load."""

from typing import Any
from acme import wrappers
import dm_env



def _create_environment(
    task_class,
    task_name,
    **kwargs):
  if task_class == 'd4rl':
    from jrl.envs import d4rl
    return d4rl.create_d4rl_env(task_name, **kwargs)
  elif task_class == 'dm_control':
    from jrl.envs import dm_control
    return dm_control.create_dm_control_env(task_name)
  else:
    raise NotImplementedError('task class not handled!')


def create_environment(
    task_class,
    task_name,
    single_precision = False,
    **kwargs):
  env = _create_environment(task_class, task_name, **kwargs)
  if single_precision:
    env = wrappers.SinglePrecisionWrapper(env)

  return env
