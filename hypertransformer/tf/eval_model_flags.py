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

"""Hyper-Transformer flags."""
from absl import flags

flags.DEFINE_integer('eval_batch_size', 200, 'Evaluation batch size.')
flags.DEFINE_integer('num_eval_batches', 16, 'Number of batches to evaluate '
                     'for the same task.')
flags.DEFINE_integer('num_task_evals', 512, 'Number of different "tasks" to '
                     'evaluate.')
flags.DEFINE_string('eval_datasets', '', 'List of datasets to evaluate.')
