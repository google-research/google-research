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

"""Main function to train and eval vatt models."""

import pprint

from absl import app
from absl import flags
from absl import logging

from vatt.configs import factory as config_factory
from vatt.experiments import finetune
from vatt.experiments import pretrain

flags.DEFINE_string('task', 'PRETRAIN', 'PRETRAIN or FINETUNE.')
flags.DEFINE_string('mode', 'train', 'train or eval.')
flags.DEFINE_string('model_dir', None, 'Default path for the experiment.')
flags.DEFINE_string('model_arch', 'Tx_FAC', 'Arch of the model.')
flags.DEFINE_string('override_checkpoint', None,
                    ('Path to a checkpoint for initialization. '
                     'If this is passed, the model is initialized from this '
                     'checkpoint, even if there is a valid latest checkpoint '
                     'inside the model_dir.'))
flags.DEFINE_string('config_file', None,
                    ('Path to a YAML config file containing the dictionary of '
                     'parameters to override the original params defined '
                     'under configs/'))
flags.DEFINE_string('params_override', None,
                    'A safe_dumped str of a dictionary of parameters')
flags.DEFINE_string('strategy_type', 'tpu', 'Type of the distribution strategy')
flags.DEFINE_string('tpu', None, 'Address of the TPU device')


FLAGS = flags.FLAGS


def get_params():
  """Constructs the configuration of the experiment."""

  task = FLAGS.task
  model_arch = FLAGS.model_arch
  params = config_factory.build_experiment_configs(
      task=task,
      model_arch=model_arch,
      )

  if FLAGS.config_file:
    params.override_from_file(FLAGS.config_file)

  if FLAGS.params_override:
    params.override_from_str(FLAGS.params_override)

  params.override({
      'mode': FLAGS.mode,
      'model_dir': FLAGS.model_dir,
      'checkpoint_path': FLAGS.override_checkpoint,
      'strategy_config': {'tpu': FLAGS.tpu,
                          'distribution_strategy': FLAGS.strategy_type},
  })

  return params


def main(argv):
  del argv  # Unused.

  params = get_params()
  logging.info('Model Parameters: %s', pprint.pformat(params.as_dict()))

  if params.task.lower() == 'pretrain':
    executor = pretrain.get_executor(params=params)

  elif params.task.lower() == 'finetune':
    executor = finetune.get_executor(params=params)

  else:
    raise ValueError('Task not found: %s.' % params.task)

  return executor.run(mode=params.mode)


if __name__ == '__main__':
  app.run(main)
