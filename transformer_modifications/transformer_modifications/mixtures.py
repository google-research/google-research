# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Transformer Modifications Mixtures."""

import seqio
from t5.data.glue_utils import get_super_glue_weight_mapping

from transformer_modifications.transformer_modifications import tasks  # pylint: disable=unused-import

MixtureRegistry = seqio.MixtureRegistry
TaskRegistry = seqio.TaskRegistry

_super_glue_tasks_envocab = {}
for task, value in get_super_glue_weight_mapping().items():
  _super_glue_tasks_envocab[task + "_envocab"] = value

MixtureRegistry.add(
    "super_glue_v102_proportional_envocab",
    list(_super_glue_tasks_envocab.items()))
