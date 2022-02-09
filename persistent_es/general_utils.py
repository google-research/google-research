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

"""General utility functions.
"""
from jax.tree_util import tree_flatten


def recursive_keys(dictionary, upper_key=''):
  all_keys = []
  for key, value in dictionary.items():
    try:
      value.keys()  # Try to see if value is a dictionary with keys
      all_keys += recursive_keys(value, key)
    except:
      all_keys += ['{}/{}'.format(upper_key, key)]
  return all_keys


def count_params(params):
  value_flat, value_tree = tree_flatten(params)
  return sum([v.size for v in value_flat])


def save_command(argv, fname):
  with open(fname, 'w') as f:
    f.write('\n'.join(['python {} \\'.format(argv[0])] +
                      ['    {} \\'.format(line) for line in argv[1:-1]] +
                      ['    {}'.format(argv[-1])]))
