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

# Lint as: python3
"""Utility to perform policy evaluations for measuring generalization."""

import math
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow.compat.v2 as tf
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import triggers

from pse.dm_control import train_eval_flags  # pylint:disable=unused-import
from pse.dm_control.utils import env_utils

EVALUATED_STEPS_FILE = 'evaluated_steps.txt'
FLAGS = flags.FLAGS


@gin.configurable(module='drq_agent')
def train_eval(data_dir=None):  # pylint: disable=unused-argument
  pass


@gin.configurable(module='evaluator')
def evaluate(env_name,
             saved_model_dir,
             env_load_fn=env_utils.load_dm_env_for_eval,
             num_episodes=1,
             eval_log_dir=None,
             continuous=False,
             max_train_step=math.inf,
             seconds_between_checkpoint_polls=5,
             num_retries=100,
             log_measurements=lambda metrics, current_step: None):
  """Evaluates a checkpoint directory.

  Checkpoints for the saved model to evaluate are assumed to be at the same
  directory level as the saved_model dir. ie:

  * saved_model_dir: root_dir/policies/greedy_policy
  * checkpoints_dir: root_dir/checkpoints

  Args:
    env_name: Name of the environment to evaluate in.
    saved_model_dir: String path to the saved model directory.
    env_load_fn: Function to load the environment specified by env_name.
    num_episodes: Number or episodes to evaluate per checkpoint.
    eval_log_dir: Optional path to output summaries of the evaluations. If None
      a default directory relative to the saved_model_dir will be used.
    continuous: If True all the evaluation will keep polling for new
      checkpoints.
    max_train_step: Maximum train_step to evaluate. Once a train_step greater or
      equal to this is evaluated the evaluations will terminate. Should set to
      <= train_eval.num_iterations to ensure that eval terminates.
    seconds_between_checkpoint_polls: The amount of time in seconds to wait
      between polls to see if new checkpoints appear in the continuous setting.
    num_retries: Number of retries for reading checkpoints.
    log_measurements: Function to log measurements.

  Raises:
    IOError: on repeated failures to read checkpoints after all the retries.
  """
  split = os.path.split(saved_model_dir)
  # Remove trailing slash if we have one.
  if not split[-1]:
    saved_model_dir = split[0]

  env = env_load_fn(env_name)

  # Load saved model.
  saved_model_path = os.path.join(saved_model_dir, 'saved_model.pb')
  while continuous and not tf.io.gfile.exists(saved_model_path):
    logging.info('Waiting on the first checkpoint to become available at: %s',
                 saved_model_path)
    time.sleep(seconds_between_checkpoint_polls)

  for _ in range(num_retries):
    try:
      policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
          saved_model_dir, load_specs_from_pbtxt=True)
      break
    except (tf.errors.OpError, tf.errors.DataLossError, IndexError,
            FileNotFoundError):
      logging.warning(
          'Encountered an error while loading a policy. This can '
          'happen when reading a checkpoint before it is fully written. '
          'Retrying...')
      time.sleep(seconds_between_checkpoint_polls)
  else:
    logging.error('Failed to load a checkpoint after retrying: %s',
                  saved_model_dir)

  if max_train_step and policy.get_train_step() > max_train_step:
    logging.info(
        'Policy train_step (%d) > max_train_step (%d). No evaluations performed.',
        policy.get_train_step(), max_train_step)
    return

  # Assume saved_model dir is of the form: root_dir/policies/greedy_policy. This
  # requires going up two levels to get the root_dir.
  root_dir = os.path.dirname(os.path.dirname(saved_model_dir))
  log_dir = eval_log_dir or os.path.join(root_dir, 'eval')

  # evaluated_file = os.path.join(log_dir, EVALUATED_STEPS_FILE)
  evaluated_checkpoints = set()

  train_step = tf.Variable(policy.get_train_step(), dtype=tf.int64)
  metrics = actor.eval_metrics(buffer_size=num_episodes)
  eval_actor = actor.Actor(
      env,
      policy,
      train_step,
      metrics=metrics,
      episodes_per_run=num_episodes,
      summary_dir=log_dir)

  checkpoint_list = _get_checkpoints_to_evaluate(evaluated_checkpoints,
                                                 saved_model_dir)

  latest_eval_step = policy.get_train_step()
  while (checkpoint_list or continuous) and latest_eval_step < max_train_step:
    while not checkpoint_list and continuous:
      logging.info('Waiting on new checkpoints to become available.')
      time.sleep(seconds_between_checkpoint_polls)
      checkpoint_list = _get_checkpoints_to_evaluate(evaluated_checkpoints,
                                                     saved_model_dir)
    checkpoint = checkpoint_list.pop()
    for _ in range(num_retries):
      try:
        policy.update_from_checkpoint(checkpoint)
        break
      except (tf.errors.OpError, IndexError):
        logging.warning(
            'Encountered an error while evaluating a checkpoint. This can '
            'happen when reading a checkpoint before it is fully written. '
            'Retrying...')
        time.sleep(seconds_between_checkpoint_polls)
    else:
      # This seems to happen rarely. Just skip this checkpoint.
      logging.error('Failed to evaluate checkpoint after retrying: %s',
                    checkpoint)
      continue

    logging.info('Evaluating:\n\tStep:%d\tcheckpoint: %s',
                 policy.get_train_step(), checkpoint)
    eval_actor.train_step.assign(policy.get_train_step())

    train_step = policy.get_train_step()
    if triggers.ENV_STEP_METADATA_KEY in policy.get_metadata():
      env_step = policy.get_metadata()[triggers.ENV_STEP_METADATA_KEY].numpy()
      eval_actor.training_env_step = env_step

    if latest_eval_step <= train_step:
      eval_actor.run_and_log()
      latest_eval_step = policy.get_train_step()
    else:
      logging.info(
          'Skipping over train_step %d to avoid logging backwards in time.',
          train_step)
    evaluated_checkpoints.add(checkpoint)


def _get_checkpoints_to_evaluate(evaluated_checkpoints, saved_model_dir):
  """Get an ordered list of checkpoint directories that have not been evaluated.

  Note that the checkpoints are in reversed order here, because we are popping
  the checkpoints later.

  Args:
    evaluated_checkpoints: a set of checkpoint directories that have already
      been evaluated.
    saved_model_dir: directory where checkpoints are saved. Often
      root_dir/policies/greedy_policy.

  Returns:
    A sorted list of checkpoint directories to be evaluated.
  """
  checkpoints_dir = os.path.join(
      os.path.dirname(saved_model_dir), 'checkpoints', '*')
  checkpoints = tf.io.gfile.glob(checkpoints_dir)
  # Sort checkpoints, such that .pop() will return the most recent one.
  return sorted(list(set(checkpoints) - evaluated_checkpoints))


def main(_):
  logging.set_verbosity(logging.INFO)
  logging.info('root_dir is %s', FLAGS.root_dir)

  gin.parse_config_files_and_bindings(FLAGS.gin_files, FLAGS.gin_bindings)

  if FLAGS.trial_id is not None:
    expanded_root_dir = os.path.join(FLAGS.root_dir, FLAGS.env_name,
                                     str(FLAGS.trial_id))
  else:
    expanded_root_dir = os.path.join(FLAGS.root_dir, FLAGS.env_name)

  if FLAGS.seed is not None:
    expanded_root_dir = os.path.join(expanded_root_dir, f'seed_{FLAGS.seed}')

  saved_model_dir = os.path.join(expanded_root_dir, 'policies', 'greedy_policy')

  log_measurements = lambda metrics, current_step: None

  evaluate(
      env_name=FLAGS.env_name,
      saved_model_dir=saved_model_dir,
      eval_log_dir=FLAGS.eval_log_dir,
      continuous=FLAGS.continuous,
      log_measurements=log_measurements)


if __name__ == '__main__':
  app.run(main)
