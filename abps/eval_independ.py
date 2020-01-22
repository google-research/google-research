# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

r"""Eval independent train DQN on Atari environments.

Additional flags are available such as `--replay_buffer_capacity` and
`--n_step_update`.

"""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
from tf_agents.environments import suite_atari

from abps import baseline_runners

FLAGS = flags.FLAGS

# AtariPreprocessing runs 4 frames at a time, max-pooling over the last 2
# frames. We need to account for this when computing things like update
# intervals.
ATARI_FRAME_SKIP = 4


def get_run_args():
  """Builds a dict of run arguments from flags."""
  run_args = {}
  run_args['is_eval'] = FLAGS.is_eval
  if FLAGS.n_step_update:
    run_args['n_step_update'] = FLAGS.n_step_update
  if FLAGS.enable_functions:
    run_args['enable_functions'] = FLAGS.enable_functions
  if FLAGS.dqn_type:
    run_args['dqn_type'] = FLAGS.dqn_type
  if FLAGS.learning_rate:
    run_args['learning_rate'] = FLAGS.learning_rate
  if FLAGS.hparam_path:
    run_args['hparam_path'] = FLAGS.hparam_path
  if FLAGS.eval_parallel_size:
    run_args['eval_parallel_size'] = FLAGS.eval_parallel_size
  if FLAGS.num_iterations:
    run_args['num_iterations'] = FLAGS.num_iterations
  # evaler specific args
  if FLAGS.eval_episode_per_iteration:
    run_args['eval_episode_per_iteration'] = FLAGS.eval_episode_per_iteration
  if FLAGS.eval_interval_secs:
    run_args['eval_interval_secs'] = FLAGS.eval_interval_secs
  if FLAGS.eval_epsilon_greedy:
    run_args['eval_epsilon_greedy'] = FLAGS.eval_epsilon_greedy
  if FLAGS.ucb_coeff:
    run_args['ucb_coeff'] = FLAGS.ucb_coeff
  if FLAGS.num_worker:
    run_args['num_worker'] = FLAGS.num_worker
  if FLAGS.eval_agents:
    run_args['eval_agents'] = FLAGS.eval_agents.split(',')
    logging.info('found eval agents:%s', run_args['eval_agents'])
  # if FLAGS.select_policy_way:
  #   run_args['select_policy_way'] = FLAGS.select_policy_way

  return run_args


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.enable_resource_variables()
  if FLAGS.select_policy_way == 'independent':
    runner = baseline_runners.EvalRunner(
        root_dir=FLAGS.root_dir,
        env_name=suite_atari.game(name=FLAGS.game_name),
        **get_run_args())
  runner.run()


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
