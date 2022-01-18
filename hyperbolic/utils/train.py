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

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training utility functions."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np

FLAGS = flags.FLAGS


def get_config_dict(external_config):
  """Maps FLAGS to dictionnary in order to save it in json format."""
  config = {}
  for _, arg_dict in external_config.items():
    for arg, _ in arg_dict.items():
      config[arg] = getattr(FLAGS, arg)
  return config


def count_params(model):
  """Counts the total number of trainable parameters in a KG embedding model.

  Args:
    model: A tf.keras.Model KG embedding model.

  Returns:
    Integer representing the number of trainable variables.
  """
  total = 0
  for x in model.trainable_variables:
    total += np.prod(x.shape)
  return total


def ranks_to_metrics_dict(ranks):
  """Calculates metrics, returns metrics as a dict."""
  mean_rank = np.mean(ranks)
  mean_reciprocal_rank = np.mean(1. / ranks)
  hits_at = {}
  for k in (1, 3, 10):
    hits_at[k] = np.mean(ranks <= k)*100
  return {
      'MR': mean_rank,
      'MRR': mean_reciprocal_rank,
      'hits@[1,3,10]': hits_at
  }


def metric_dict_full_and_random(ranks, random_ranks):
  results = ranks_to_metrics_dict(ranks)
  rand_results = ranks_to_metrics_dict(random_ranks)
  results.update({z + '_r': rand_results[z] for z in rand_results})
  return results


def format_metrics(metrics, split):
  """Formats metrics for logging.

  Args:
    metrics: Dictionary with metrics.
    split: String indicating the KG dataset split.

  Returns:
    String with formatted metrics.
  """
  result = format_partial_metrics(metrics, split)
  result += '\n'
  result += format_partial_metrics(metrics, split, extra='_r')
  return result


def format_partial_metrics(metrics, split, extra=''):
  result = '\t {} MR{}: {:.2f} | '.format(split, extra, metrics['MR'+extra])
  result += 'MRR{}: {:.3f} | '.format(extra, metrics['MRR'+extra])
  result += 'H@1{}: {:.3f} | '.format(extra, metrics['hits@[1,3,10]'+extra][1])
  result += 'H@3{}: {:.3f} | '.format(extra, metrics['hits@[1,3,10]'+extra][3])
  result += 'H@10{}: {:.3f}'.format(extra, metrics['hits@[1,3,10]'+extra][10])
  return result


def format_tree_stats(nodes_per_level, used_nodes, mean_deg, std_deg):
  result = '\t Tree stats:'
  temp = nodes_per_level + [1]
  for l in used_nodes:
    result += '\n\t l:{} | Used: {} / {} | Deg {} +/- {}'.format(
        l, used_nodes[l], temp[l], mean_deg[l], std_deg[l])
  return result
