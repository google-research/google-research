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
"""Utility functions printing tabulated outputs."""

import tabulate
import tensorflow.compat.v2 as tf


def make_table(tabulate_dict, iteration, header_freq):
  table = tabulate.tabulate([tabulate_dict.values()],
                            tabulate_dict.keys(),
                            tablefmt='simple',
                            floatfmt='8.4f')
  table_split = table.split('\n')
  if iteration % header_freq == 0:
    table = '\n'.join([table_split[1]] + table_split)
  else:
    table = table_split[2]
  return table


def make_header(tabulate_dict):
  table = tabulate.tabulate([tabulate_dict.values()],
                            tabulate_dict.keys(),
                            tablefmt='simple',
                            floatfmt='8.4f')
  table_split = table.split('\n')
  table = '\n'.join([table_split[1]] + table_split[:2])
  return table


def make_logging_dict(train_stats, test_stats, ensemble_stats):
  logging_dict = {}
  for prefix, stat_dict in zip(['train/', 'test/', 'test/ens_'],
                               [train_stats, test_stats, ensemble_stats]):
    for stat_name, stat_val in stat_dict.items():
      logging_dict[prefix + stat_name] = stat_val
  return logging_dict
