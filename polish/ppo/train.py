# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Launch training."""
import os
from absl import app
from absl import flags
from absl import logging
import gin
import gym
import numpy as np
import tensorflow as tf
from polish.env import parallel_env
from polish.ppo import ppo_input_fn
from polish.ppo import ppo_model_fn
from polish.ppo import ppo_trainer
from polish.utils import directory_handling
from polish.utils import tf_utils

flags.DEFINE_integer(
    'task_id', 0,
    'Index of the current borg task.  e.g., if the corresponding borg job has '
    '4 replicas, the 4 tasks will have task ids 0, 1, 2, respecively 3.')
flags.DEFINE_string('master', 'local', 'BNS name of the TF runtime to use.')
flags.DEFINE_string(
    'tpu_job_name', None, 'The name of TPU job. Typically '
    'TPU job name is auto-inferred within TPUEstimator. '
    'No need to set this FLAG as it is automatically set by the launcher.')
flags.DEFINE_multi_string(
    'gin_config', [], 'List of paths to the gin config files. No need to '
    'set this FLAG, as it is set by the launcher.')
flags.DEFINE_string(
    'gin_config_file',
    './polish/ppo/train_cpu.gin',
    'Gin config file. Necessary for local runs.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings. '
    'No need to set this FLAG.')
flags.DEFINE_string(
    'workdir', '/tmp/', 'Base directory of the studies. '
    'No need to set this FLAG, as it is set by the launcher. '
    ' If you decide to launch TensorBoard manually, '
    '`work_dir` path is directory you should use as --logdir')

FLAGS = flags.FLAGS


@gin.configurable
def ppo_train(env_name=gin.REQUIRED, env_seed=gin.REQUIRED):
  """Launch PPO and MCTS training.

  Args:
    env_name: name of gym environment to make. the supported environments are:
      /robotics/reinforcement_learning/environments/gym_mujoco/__init__.py
    env_seed: the seed to set the environment to.
  """
  # Make a gym environment.
  gym_env = gym.make(env_name)
  gym_env.seed(env_seed)
  np.random.seed(0)
  tf.random.set_random_seed(0)

  # Bound action_space and state_space sizes.
  with gin.unlock_config():
    action_space = gym_env.action_space.shape[0]
    state_space = gym_env.observation_space.shape[0]

    # The size of `action_space` & `state_space` can only be determined after
    # creating the gym environment (gym_env). The following gin bindings
    # update the `env_action_space` and `env_state_space` values in different
    # classes/functions.
    gin.bind_parameter('PpoModelFn.env_action_space', action_space)
    gin.bind_parameter('PpoInputFn.env_action_space', action_space)
    gin.bind_parameter('MCTSPlayer.env_action_space', action_space)
    gin.bind_parameter('MCTSNode.env_action_space', action_space)

    gin.bind_parameter('PpoInputFn.env_state_space', state_space)
    gin.bind_parameter('serving_input_fn.env_state_space', state_space)

  # Query the checkpoint and summary directory.
  checkpoint_dir_str = gin.query_parameter('checkpoint_dir/macro.value')
  summary_dir_str = gin.query_parameter('summary_dir/macro.value')
  train_state_file_str = gin.query_parameter('train_state_file/macro.value')

  # Create the `model_fn` for TF estimator.
  model_fn = ppo_model_fn.PpoModelFn()
  # Create an estimator for prediction used for data sampling
  # from policy network.
  estimator = tf_utils.create_estimator(
      working_dir=checkpoint_dir_str, model_fn=model_fn)

  # Create a ParallelEnv instance for collecting policy and MCTS rollout data.
  env_wrapper = parallel_env.ParallelEnv(
      env=gym_env,
      estimator=estimator,
      checkpoint_dir=checkpoint_dir_str,
      serving_input_fn=tf_utils.serving_input_fn)

  # Create `input_fn` for TF estimator. `input_fn` is where you need to
  # perform data preparation for the TF estimator.
  input_fn = ppo_input_fn.PpoInputFn(
      env_wrapper=env_wrapper,
      model_fn=model_fn,
      checkpoint_dir=checkpoint_dir_str,
      summary_dir=summary_dir_str,
      train_state_file=train_state_file_str)

  # Instantiate from PpoTrainer class.
  trainer = ppo_trainer.PpoTrainer(
      input_fn=input_fn, model_fn=model_fn, checkpoint_dir=checkpoint_dir_str)

  logging.info('Before Training!')
  # Run training for a pre-defined number of steps.
  trainer.train()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments for train.')

  if not FLAGS.gin_config:
    # Run the experiments locally.
    gin.parse_config_file(FLAGS.gin_config_file)
  else:
    # Run the experiments on a server.
    gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)

  # create the `checkpoing` and `summary` directory used during training
  # to save/restore the model and write TF summaries.
  checkpoint_dir_str = gin.query_parameter('checkpoint_dir/macro.value')
  summary_dir_str = gin.query_parameter('summary_dir/macro.value')
  mcts_checkpoint_dir_str = os.path.join(checkpoint_dir_str, 'mcts_data')
  app_directories = [
      checkpoint_dir_str, summary_dir_str, mcts_checkpoint_dir_str
  ]

  for d in app_directories:
    directory_handling.ensure_dir_exists(d)

  ppo_train()


if __name__ == '__main__':
  app.run(main)
