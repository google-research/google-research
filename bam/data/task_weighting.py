# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Code for weighting examples from different tasks based on dataset sizes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _multiples_and_weights(config):
  """Helper for weighting GLUE datasets.

  Concatenating all the train sets together and then shuffling the examples
  causes large datasets to dominate the training, resulting in poor performance
  on small datasets. This has some hacky logic to produce (1) "multiples" for
  each dataset so the multi-task train set contains small datasets multiple
  times, so those examples are seen more often and (2) weights for each dataset,
  which also allows for smaller datasets to have influence on training.
  Overall the effect is that tasks are weighted according to
  dataset_size^config.task_weight_exponent.

  Args:
    config: a configure.Config object
  Returns:
    How many copies and weights for each dataset.
  """

  dataset_sizes = {
      "cola": 8551,
      "mnli": 392702,
      "mrpc": 7336,
      "qnli": 108436,
      "qqp": 363869,
      "sst": 67349,
      "sts": 11498,
      "rte": 2490
  }

  def map_values(f, d):
    return {k: f(v) for k, v in d.items()}

  def map_kv(f, d):
    return {k: f(k, v) for k, v in d.items()}

  def normalize(d):
    total = float(sum(d.values()))
    return map_values(lambda v: v / total, d)

  dataset_weights = map_values(lambda s: s ** config.task_weight_exponent,
                               dataset_sizes)
  dataset_weights = normalize(dataset_weights)
  correction = dataset_sizes["mnli"] / dataset_weights["mnli"]
  dataset_tgts = map_values(lambda v: v * correction, dataset_weights)
  dataset_multiples = map_kv(
      lambda task, tgt: round((tgt + 0.01) / dataset_sizes[task]), dataset_tgts)
  new_dataset_sizes = map_kv(
      lambda task, multiple: dataset_sizes[task] * multiple, dataset_multiples)
  weights_after_multiples = map_values(
      lambda v: v * len(dataset_sizes),
      normalize({task: dataset_weights[task] / new_dataset_sizes[task]
                 for task in new_dataset_sizes}))

  return dataset_multiples, weights_after_multiples


def get_task_multiple(task, split):
  if split != "train":
    return 1
  if task.config.dataset_multiples:
    multiples, _ = _multiples_and_weights(task.config)
    return int(multiples[task.name] + 1e-5)
  return 1


def get_task_weights(config, sizes):
  """Get task weights according to dataset sizes."""

  if config.dataset_multiples:
    _, weights = _multiples_and_weights(config)
    return weights
  else:
    if config.task_weight_exponent < 0:
      return {task_name: 1.0 for task_name in sizes}
    n_examples = sum(sizes.values())
    weights = {task_name: 1.0 / (size**(1 - config.task_weight_exponent))
               for task_name, size in sizes.items()}
    expected_weight = sum([weights[task_name] * sizes[task_name] / n_examples
                           for task_name in weights])
    weights = {task_name: w / expected_weight
               for task_name, w in weights.items()}
    return weights
