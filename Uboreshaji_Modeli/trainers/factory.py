# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Factory for instantiating trainer strategies."""


from Uboreshaji_Modeli.common import config
from Uboreshaji_Modeli.trainers import base
from Uboreshaji_Modeli.trainers import detection


def get_trainer(task_type):
  """Returns the trainer strategy for the given task type."""
  if task_type == config.TaskType.DETECTION:
    return detection.DetectionTrainer()


  raise ValueError(f"Unsupported task type: {task_type}")
