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

r"""Train and Eval DQN on Atari environments.

Training and evaluation proceeds alternately in iterations, where each
iteration consists of a 1M frame training phase followed by a 500K frame
evaluation phase. In the literature, some papers report averages of the train
phases, while others report averages of the eval phases.

This example is configured to use dopamine.atari.preprocessing, which, among
other things, repeats every action it receives for 4 frames, and then returns
the max-pool over the last 2 frames in the group. In this example, when we
refer to "ALE frames" we refer to the frames before the max-pooling step (i.e.
the raw data available for processing). Because of this, many of the
configuration parameters (like initial_collect_steps) are divided by 4 in the
body of the trainer (e.g. if you want to evaluate with 400 frames in the
initial collection, you actually only need to .step the environment 100 times).

For a good survey of training on Atari, see Machado, et al. 2017:
https://arxiv.org/pdf/1709.06009.pdf.

To run:

```bash
tf_agents/agents/dqn/examples/v1/train_eval_atari \
  --root_dir=$HOME/atari/pong \
  --atari_roms_path=/tmp
  --alsologtostderr
```

Additional flags are available such as `--replay_buffer_capacity` and
`--n_step_update`.

"""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf
from tf_agents.environments import suite_atari

from abps import abps_runners

FLAGS = flags.FLAGS

# AtariPreprocessing runs 4 frames at a time, max-pooling over the last 2
# frames. We need to account for this when computing things like update
# intervals.
ATARI_FRAME_SKIP = 4


def get_run_args():
  """Builds a dict of run arguments from flags."""
  run_args = {}
  run_args['enable_functions'] = FLAGS.enable_functions
  run_args['pbt'] = FLAGS.pbt
  run_args['online_eval_use_train'] = FLAGS.online_eval_use_train
  run_args['create_hparam'] = FLAGS.create_hparam

  if FLAGS.n_step_update:
    run_args['n_step_update'] = FLAGS.n_step_update
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
  # trainer specific args
  if FLAGS.initial_collect_steps:
    run_args['initial_collect_steps'] = FLAGS.initial_collect_steps
  if FLAGS.replay_buffer_capacity:
    run_args['replay_buffer_capacity'] = FLAGS.replay_buffer_capacity
  if FLAGS.train_steps_per_iteration:
    run_args['train_steps_per_iteration'] = FLAGS.train_steps_per_iteration
  if FLAGS.update_policy_iteration:
    run_args['update_policy_iteration'] = FLAGS.update_policy_iteration
  if FLAGS.ucb_coeff:
    run_args['ucb_coeff'] = FLAGS.ucb_coeff
  if FLAGS.select_policy_way:
    run_args['select_policy_way'] = FLAGS.select_policy_way
  if FLAGS.epsilon_decay_selection:
    run_args['epsilon_decay_selection'] = FLAGS.epsilon_decay_selection
  if FLAGS.bandit_ucb_coeff:
    run_args['bandit_ucb_coeff'] = FLAGS.bandit_ucb_coeff
  if FLAGS.bandit_buffer_size:
    run_args['bandit_buffer_size'] = FLAGS.bandit_buffer_size
  if FLAGS.pbt_period:
    run_args['pbt_period'] = FLAGS.pbt_period
  if FLAGS.pbt_low:
    run_args['pbt_low'] = FLAGS.pbt_low
  if FLAGS.pbt_high:
    run_args['pbt_high'] = FLAGS.pbt_high
  if FLAGS.pbt_percent_low:
    run_args['pbt_percent_low'] = FLAGS.pbt_percent_low
  if FLAGS.pbt_percent_top:
    run_args['pbt_percent_top'] = FLAGS.pbt_percent_top
  if FLAGS.num_worker:
    run_args['num_worker'] = FLAGS.num_worker
  if FLAGS.architect_prob:
    run_args['architect_prob'] = [
        float(x) for x in FLAGS.architect_prob.split(',')
    ]
    logging.info('using architect_prob:%s', run_args['architect_prob'])
  if FLAGS.train_agents:
    run_args['train_agents'] = FLAGS.train_agents.split(',')
    logging.info('training agents:%s', run_args['train_agents'])

  return run_args


def main(_):
  tf.disable_eager_execution()
  logging.set_verbosity(logging.INFO)
  tf.enable_resource_variables()
  runner = abps_runners.TrainRunner(
      root_dir=FLAGS.root_dir,
      env_name=suite_atari.game(name=FLAGS.game_name),
      **get_run_args())
  runner.run()


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
