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

"""Makita Config for ElasticNet + greedy selection active learning."""
import ml_collections


def get_config():
  return ml_collections.ConfigDict({
      'model_config':
          ml_collections.ConfigDict({
              'model_type':
                  'gp',
              'hyperparameters':
                  ml_collections.ConfigDict(),
              'tuning_hyperparameters': [{}],
              'features':
                  ml_collections.ConfigDict({
                      'feature_type': 'fingerprint',
                      'params': {
                          'feature_column': 'Smiles',
                          'fingerprint_size': 128,
                          'fingerprint_radius': 2
                      }
                  }),
              'targets':
                  ml_collections.ConfigDict({
                      'feature_type': 'number',
                      'params': {
                          'feature_column': 'dG',
                      }
                  })
          }),
      'selection_config':
          ml_collections.ConfigDict({
              'selection_type': 'greedy',
              'hyperparameters': ml_collections.ConfigDict({}),
              'num_elements': 2,
              'selection_columns': ['Smiles', 'dG', 'DockingScore']
          }),
      'metadata':
          'Small test for active learning.',
      'cycle_dir':
          '',
      'training_pool':
          '',
      'virtual_library':
          '',
  })
