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

"""Utils."""
# Lint as: python3

import pprint
from absl import logging

from official.core import train_utils
from official.modeling import hyperparams


def config_override(params, flags_obj):
  """Override ExperimentConfig according to flags."""
  # Change runtime.tpu to the real tpu.
  params.override({
      'runtime': {
          'tpu': flags_obj.tpu,
      }
  })

  # Get the first level of override from `--config_file`.
  #   `--config_file` is typically used as a template that specifies the common
  #   override for a particular experiment.
  for config_file in flags_obj.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)

  # Get the second level of override from `--params_override`.
  #   `--params_override` is typically used as a further override over the
  #   template. For example, one may define a particular template for training
  #   ResNet50 on ImageNet in a config fid pass it via `--config_file`,
  #   then define different learning rates and pass it via `--params_override`.
  if flags_obj.params_override:
    params = hyperparams.override_params_dict(
        params, flags_obj.params_override, is_strict=True)

  params.validate()
  params.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s', pp.pformat(params.as_dict()))

  model_dir = flags_obj.model_dir
  if 'train' in flags_obj.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  return params
