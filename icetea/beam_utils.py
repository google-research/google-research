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

"""Beam Utils!

Functions to manage the interface between the Beam pipeline and the experiments.
"""
from icetea import utils


def data(pipeline_seed, pipeline_data):
  if pipeline_data['data_name'] != 'ukb':
    return utils.DataSimulation(seed=pipeline_seed, param_data=pipeline_data)
  else:
    return utils.LoadImages(seed=pipeline_seed, param_data=pipeline_data)


def organize_param_methods(pipeline_data, param_methods):
  for i, param in enumerate(param_methods):
    yield [i, param, pipeline_data[0], pipeline_data[1]]


def methods(pipeline):
  simulation, _ = utils.experiments(
      data=data(pipeline[2], pipeline[3]),
      seed=pipeline[0],
      param_method=pipeline[1])
  return simulation


def convert_dict_to_csv_record(input_dict):
  # Turns dictionary values into a comma-separated value formatted string.
  return ','.join(map(str, input_dict.values()))


def print_tee(pipeline):
  # Only used for debugging.
  print(pipeline)
  return pipeline
