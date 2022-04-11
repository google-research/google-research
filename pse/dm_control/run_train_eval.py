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

r"""Wraps drq_sac_agent and expands the root_dir for nightly baselines.

"""
import os

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import tensorflow.compat.v2 as tf

from pse.dm_control import train_eval_flags  # pylint:disable=unused-import
from pse.dm_control.agents import pse_drq_train_eval

FLAGS = flags.FLAGS
flags.DEFINE_bool('debugging', False,
                  'If True, we set additional logging and run in eager mode.')


def set_random_seed(seed):
  """Set random seed for reproducibility."""
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)


@gin.configurable(module='evaluator')
def evaluate(max_train_step=int(1e+8)):  # pylint: disable=unused-argument
  pass


def main(argv):
  del argv
  logging.set_verbosity(logging.INFO)
  if FLAGS.seed is not None:
    set_random_seed(FLAGS.seed)
    logging.info('Random seed %d', FLAGS.seed)
    trial_suffix = f'{FLAGS.trial_id}/seed_{FLAGS.seed}'
  else:
    trial_suffix = str(FLAGS.trial_id)

  expanded_root_dir = os.path.join(
      FLAGS.root_dir, FLAGS.env_name, trial_suffix)
  if FLAGS.load_pretrained and (FLAGS.pretrained_model_dir is not None):
    pretrained_model_dir = os.path.join(
        FLAGS.pretrained_model_dir, FLAGS.env_name, trial_suffix)
  else:
    pretrained_model_dir = None
  if FLAGS.debugging:
    tf.debugging.set_log_device_placement(True)
    tf.config.experimental_run_functions_eagerly(True)

  gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

  pse_drq_train_eval.train_eval(
      expanded_root_dir,
      FLAGS.env_name,
      num_train_steps=FLAGS.num_train_steps,
      policy_save_interval=FLAGS.policy_save_interval,
      checkpoint_interval=FLAGS.checkpoint_interval,
      load_pretrained=FLAGS.load_pretrained,
      pretrained_model_dir=pretrained_model_dir,
      contrastive_loss_weight=FLAGS.contrastive_loss_weight,
      contrastive_loss_temperature=FLAGS.contrastive_loss_temperature,
      image_encoder_representation=FLAGS.image_encoder_representation,
      reverb_port=FLAGS.reverb_port,
      eval_interval=FLAGS.eval_interval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
