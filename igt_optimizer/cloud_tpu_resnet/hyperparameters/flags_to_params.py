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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Functions to override model parameters from command-line flags."""

from igt_optimizer.cloud_tpu_resnet.hyperparameters import params_dict

ESSENTIAL_FLAGS = ['tpu', 'data_dir', 'model_dir']


def override_params_from_input_flags(params, input_flags):
  """Update params dictionary with input flags.

  Args:
    params: ParamsDict object containing dictionary of model parameters.
    input_flags: All the flags with non-null value of overridden model
      parameters.

  Returns:
    ParamsDict object containing dictionary of model parameters.
  """
  if params is None:
    raise ValueError(
        'Input dictionary is empty. It is expected to be loaded with default '
        'values')

  if not isinstance(params, params_dict.ParamsDict):
    raise ValueError(
        'The base parameter set must be a ParamsDict, was: {}'.format(
            type(params)))

  essential_flag_dict = {}
  for key in ESSENTIAL_FLAGS:
    flag_value = input_flags.get_flag_value(key, None)

    if flag_value is None:
      raise ValueError('Flag {} could not be None.'.format(key))
    else:
      essential_flag_dict[key] = flag_value

  params_dict.override_params_dict(params, essential_flag_dict, is_strict=False)

  normal_flag_dict = get_dictionary_from_flags(params.as_dict(), input_flags)

  params_dict.override_params_dict(params, normal_flag_dict, is_strict=False)

  return params


def get_dictionary_from_flags(params, input_flags):
  """Generate dictionary from non-null flags.

  Args:
    params: Python dictionary of model parameters.
    input_flags: All the flags with non-null value of overridden model
      parameters.

  Returns:
    Python dict of overriding model parameters.
  """
  flag_dict = {}
  for k, v in params.items():
    if isinstance(v, dict):
      d = get_dictionary_from_flags(v, input_flags)
      flag_dict[k] = d
    else:
      flag_value = input_flags.get_flag_value(k, None)

      if flag_value is not None:
        flag_dict[k] = flag_value

  return flag_dict
