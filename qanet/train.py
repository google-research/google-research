# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

r"""Experiment launcher training and eval."""
from absl import flags
import tensorflow as tf

from qanet import experiment
from qanet.util import configurable


flags.DEFINE_string('config', '', 'Flattened config')
flags.DEFINE_string('config_file', '', 'Path to json config file')
flags.DEFINE_string('model_dir', '', 'Output directory for model')
flags.DEFINE_string('mode', 'train_and_eval', 'train, eval, or train_and_eval')

FLAGS = flags.FLAGS


def main(_):
  """Parse config flag and config files then run experiment."""

  # Load config
  parsed_config = configurable.load_config(FLAGS.config_file, FLAGS.config)

  if 'fn' not in parsed_config:
    # Fill in the default template
    parsed_config = configurable.merge(
        parsed_config, fn='ConfigurableExperiment')

  configurable.save_config(FLAGS.model_dir, parsed_config)
  # TODO(thangluong): consider use FLAGS if we do need to use this often
  parsed_config['log_device_placement'] = False
  parsed_config['allow_soft_placement'] = True
  if 'data_format' not in parsed_config:
    parsed_config['data_format'] = 'squad'
  assert FLAGS.model_dir
  experiment_fn = experiment.create_experiment_fn(
      default_config=parsed_config, return_distributed_spec=False)
  exp = experiment_fn(hparams=None, model_dir=FLAGS.model_dir)
  if FLAGS.mode == 'train':
    exp.train()
  elif FLAGS.mode == 'eval':
    exp.evaluate()
  elif FLAGS.mode == 'train_and_eval':
    exp.train_and_evaluate()
  else:
    raise ValueError('Unknown mode %s' % FLAGS.mode)


if __name__ == '__main__':
  tf.app.run()
