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

r"""Train and Eval multi-agent PPO for multi-agent gridworld.

Each agent learns an independent policy.

Note: this code always assumes the network has an RNN to track other agents'
state.

To run:

```bash tensorboard.sh --port=2222 --logdir /tmp/multigrid/ppo/

python -m multiagent_train_eval.py --root_dir=/tmp/multigrid/ppo/
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import app
from absl import flags
from absl import logging
import gin

from tf_agents.system import system_multiprocessing
# Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl import gym_multigrid
from social_rl.multiagent_tfagents import football_gym_env
from social_rl.multiagent_tfagents import multiagent_gym_suite
from social_rl.multiagent_tfagents import multiagent_metrics
from social_rl.multiagent_tfagents import multiagent_ppo
from social_rl.multiagent_tfagents import multiagent_train_eval
from social_rl.multiagent_tfagents import utils
from social_rl.multiagent_tfagents.joint_attention import attention_ppo_agent

FLAGS = flags.FLAGS

flags.DEFINE_string('attention_bonus_type', 'kld',
                    'Method for computing attention bonuses.')
flags.DEFINE_float('bonus_ratio', 0.00, 'Final multiplier for bonus rewards.')
flags.DEFINE_integer('bonus_timescale', int(1e6),
                     'Attention bonuses scale linearly until this point.')


def main(_):
  logging.set_verbosity(logging.INFO)

  agent_class = functools.partial(
      attention_ppo_agent.MultiagentAttentionPPO,
      attention_bonus_type=FLAGS.attention_bonus_type,
      bonus_ratio=FLAGS.bonus_ratio,
      bonus_timescale=FLAGS.bonus_timescale
      )

  if 'academy' in FLAGS.env_name:
    env_load_fn = football_gym_env.load
    gin.bind_parameter('construct_attention_networks.use_stacks', True)
    gin.bind_parameter('AttentionMultiagentPPOPolicy.use_stacks', True)
  else:
    env_load_fn = multiagent_gym_suite.load

  multiagent_train_eval.train_eval(
      FLAGS.root_dir,
      env_load_fn=env_load_fn,
      agent_class=agent_class,
      env_name=FLAGS.env_name,
      num_environment_steps=FLAGS.num_environment_steps,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_parallel_environments=FLAGS.num_parallel_environments,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      num_epochs=FLAGS.num_epochs,
      num_eval_episodes=FLAGS.num_eval_episodes,
      train_checkpoint_interval=FLAGS.train_checkpoint_interval,
      policy_checkpoint_interval=FLAGS.policy_checkpoint_interval,
      log_interval=FLAGS.log_interval,
      summary_interval=FLAGS.summary_interval,
      actor_fc_layers=(FLAGS.actor_fc_layers_size, FLAGS.actor_fc_layers_size),
      value_fc_layers=(FLAGS.value_fc_layers_size, FLAGS.value_fc_layers_size),
      lstm_size=(FLAGS.lstm_size,),
      conv_filters=FLAGS.conv_filters,
      conv_kernel=FLAGS.conv_kernel,
      direction_fc=FLAGS.direction_fc,
      debug=FLAGS.debug,
      inactive_agent_ids=tuple(),
      random_seed=FLAGS.random_seed,
      reinit_checkpoint_dir=FLAGS.reinit_checkpoint_dir,
      use_attention_networks=True)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  system_multiprocessing.handle_main(lambda _: app.run(main))
