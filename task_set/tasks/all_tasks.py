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

# python3
# pylint: disable=unused-import
"""Load all tasks. Each import registers some tasks."""
from task_set.tasks import conv_fc
from task_set.tasks import conv_pooling
from task_set.tasks import language_model
from task_set.tasks import losg_tasks
from task_set.tasks import maf
from task_set.tasks import mlp
from task_set.tasks import mlp_ae
from task_set.tasks import mlp_vae
from task_set.tasks import nvp
from task_set.tasks import quadratic
from task_set.tasks import rnn_text_classification
from task_set.tasks import synthetic_sequence
from task_set.tasks.fixed import all_tasks
