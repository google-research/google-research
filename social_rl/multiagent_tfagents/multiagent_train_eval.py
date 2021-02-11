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

import math
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.drivers import tf_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing
from tf_agents.utils import common

# Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl import gym_multigrid
from social_rl.multiagent_tfagents import multiagent_gym_suite
from social_rl.multiagent_tfagents import multiagent_metrics
from social_rl.multiagent_tfagents import multiagent_ppo

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'MultiGrid-DoorKey-16x16-v0',
                    'Name of an environment')
flags.DEFINE_integer('replay_buffer_capacity', 3001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer(
    'num_parallel_environments', 4,
    'Number of environments to run in parallel. Originally 30')
flags.DEFINE_integer('num_environment_steps', 25000000,
                     'Number of environment steps to run before finishing.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'collect_episodes_per_iteration', 8,
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('num_eval_episodes', 2,
                     'The number of episodes to run eval on. Originally 30.')
flags.DEFINE_integer('train_checkpoint_interval', 500, '')
flags.DEFINE_integer('policy_checkpoint_interval', 500, '')
flags.DEFINE_integer('log_interval', 15, '')
flags.DEFINE_integer('summary_interval', 50, '')

flags.DEFINE_integer('conv_filters', 16, '')
flags.DEFINE_integer('conv_kernel', 3, '')
flags.DEFINE_integer(
    'direction_fc', 5,
    'Number of fully-connected neurons connecting the one-hot'
    'direction to the main LSTM.')
flags.DEFINE_integer('actor_fc_layers_size', 32, '')
flags.DEFINE_integer('value_fc_layers_size', 32, '')
flags.DEFINE_integer('lstm_size', 256, '')

flags.DEFINE_boolean('debug', False, 'Turn on debugging and tf functions off.')
flags.DEFINE_list('inactive_agent_ids', None,
                  'List of agent IDs to fix and not train')
flags.DEFINE_integer('random_seed', 0, 'Random seed for policy and env.')
flags.DEFINE_string('reinit_checkpoint_dir', None, 'Checkpoint for reinit.')
FLAGS = flags.FLAGS

# Loss value that is considered too high and training will be terminated.
MAX_LOSS = 1e9
# How many steps does the loss have to be diverged for (too high, inf, nan)
# after the training terminates. This should prevent termination on short loss
# spikes.
TERMINATE_AFTER_DIVERGED_LOSS_STEPS = 100


@gin.configurable
def train_eval(
    root_dir,
    env_name='MultiGrid-Empty-5x5-v0',
    env_load_fn=multiagent_gym_suite.load,
    random_seed=None,
    # Architecture params
    actor_fc_layers=(64, 64),
    value_fc_layers=(64, 64),
    lstm_size=(64,),
    conv_filters=64,
    conv_kernel=3,
    direction_fc=5,
    entropy_regularization=0.,
    # Specialized agents
    inactive_agent_ids=tuple(),
    # Params for collect
    num_environment_steps=25000000,
    collect_episodes_per_iteration=30,
    num_parallel_environments=5,
    replay_buffer_capacity=1001,  # Per-environment
    # Params for train
    num_epochs=2,
    learning_rate=1e-4,
    # Params for eval
    num_eval_episodes=2,
    eval_interval=5,
    # Params for summaries and logging
    train_checkpoint_interval=100,
    policy_checkpoint_interval=100,
    log_interval=10,
    summary_interval=10,
    summaries_flush_secs=1,
    use_tf_functions=True,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    eval_metrics_callback=None,
    reinit_checkpoint_dir=None,
    debug=True):
  """A simple train and eval for PPO."""
  tf.compat.v1.enable_v2_behavior()

  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  if debug:
    logging.info('In debug mode, turning tf_functions off')
    use_tf_functions = False

  for a in inactive_agent_ids:
    logging.info('Fixing and not training agent %d', a)

  # Load multiagent gym environment and determine number of agents
  gym_env = env_load_fn(env_name)
  n_agents = gym_env.n_agents

  # Set up logging
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      multiagent_metrics.AverageReturnMetric(
          n_agents, buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    if random_seed is not None:
      tf.compat.v1.set_random_seed(random_seed)

    logging.info('Creating %d environments...', num_parallel_environments)
    eval_tf_env = tf_py_environment.TFPyEnvironment(gym_env)
    tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn(env_name)] * num_parallel_environments))


    logging.info('Preparing to train...')
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        multiagent_metrics.AverageReturnMetric(
            n_agents, batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=num_parallel_environments)
    ]

    logging.info('Creating agent...')
    tf_agent = multiagent_ppo.MultiagentPPO(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        n_agents=n_agents,
        learning_rate=learning_rate,
        actor_fc_layers=actor_fc_layers,
        value_fc_layers=value_fc_layers,
        lstm_size=lstm_size,
        conv_filters=conv_filters,
        conv_kernel=conv_kernel,
        direction_fc=direction_fc,
        entropy_regularization=entropy_regularization,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        inactive_agent_ids=inactive_agent_ids)
    tf_agent.initialize()
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    logging.info('Allocating replay buffer ...')
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)
    logging.info('RB capacity: %i', replay_buffer.capacity)

    # If reinit_checkpoint_dir is provided, the last agent in the checkpoint is
    # reinitialized. The other agents are novices.
    # Otherwise, all agents are reinitialized from train_dir.
    if reinit_checkpoint_dir:
      reinit_checkpointer = common.Checkpointer(
          ckpt_dir=reinit_checkpoint_dir,
          agent=tf_agent,
      )
      reinit_checkpointer.initialize_or_restore()
      temp_dir = os.path.join(train_dir, 'tmp')
      agent_checkpointer = common.Checkpointer(
          ckpt_dir=temp_dir,
          agent=tf_agent.agents[:-1],
      )
      agent_checkpointer.save(global_step=0)
      tf_agent = multiagent_ppo.MultiagentPPO(
          tf_env.time_step_spec(),
          tf_env.action_spec(),
          n_agents=n_agents,
          learning_rate=learning_rate,
          actor_fc_layers=actor_fc_layers,
          value_fc_layers=value_fc_layers,
          lstm_size=lstm_size,
          conv_filters=conv_filters,
          conv_kernel=conv_kernel,
          direction_fc=direction_fc,
          entropy_regularization=entropy_regularization,
          num_epochs=num_epochs,
          debug_summaries=debug_summaries,
          summarize_grads_and_vars=summarize_grads_and_vars,
          train_step_counter=global_step,
          inactive_agent_ids=inactive_agent_ids,
          non_learning_agents=list(range(n_agents - 1)))
      agent_checkpointer = common.Checkpointer(
          ckpt_dir=temp_dir, agent=tf_agent.agents[:-1])
      agent_checkpointer.initialize_or_restore()
      tf.io.gfile.rmtree(temp_dir)
      eval_policy = tf_agent.policy
      collect_policy = tf_agent.collect_policy

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=multiagent_metrics.MultiagentMetricsGroup(
            train_metrics, 'train_metrics'))
    if not reinit_checkpoint_dir:
      train_checkpointer.initialize_or_restore()
    logging.info('Successfully initialized train checkpointer')

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
    logging.info('Successfully initialized policy saver.')

    print('Using TFDriver')
    collect_driver = tf_driver.TFDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        max_episodes=collect_episodes_per_iteration,
        disable_tf_function=not use_tf_functions)

    def train_step():
      trajectories = replay_buffer.gather_all()
      return tf_agent.train(experience=trajectories)

    if use_tf_functions:
      tf_agent.train = common.function(tf_agent.train, autograph=False)
      train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    # How many consecutive steps was loss diverged for.
    loss_divergence_counter = 0

    # Save operative config as late as possible to include used configurables.
    if global_step.numpy() == 0:
      config_filename = os.path.join(
          train_dir, 'operative_config-{}.gin'.format(global_step.numpy()))
      with tf.io.gfile.GFile(config_filename, 'wb') as f:
        f.write(gin.operative_config_str())

    total_episodes = 0
    logging.info('Commencing train loop!')
    while environment_steps_metric.result() < num_environment_steps:
      global_step_val = global_step.numpy()

      # Evaluation
      if global_step_val % eval_interval == 0:
        if debug:
          logging.info('Performing evaluation at step %d', global_step_val)
        results = multiagent_metrics.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
            use_function=use_tf_functions,
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        multiagent_metrics.log_metrics(eval_metrics)

      # Collect data
      if debug:
        logging.info('Collecting at step %d', global_step_val)
      start_time = time.time()
      time_step = tf_env.reset()
      policy_state = collect_policy.get_initial_state(tf_env.batch_size)
      collect_driver.run(time_step, policy_state)
      collect_time += time.time() - start_time

      total_episodes += collect_episodes_per_iteration
      if debug:
        logging.info('Have collected a total of %d episodes', total_episodes)

      # Train
      if debug:
        logging.info('Training at step %d', global_step_val)
      start_time = time.time()
      total_loss, extra_loss = train_step()
      replay_buffer.clear()
      train_time += time.time() - start_time

      # Check for exploding losses.
      if (math.isnan(total_loss) or math.isinf(total_loss) or
          total_loss > MAX_LOSS):
        loss_divergence_counter += 1
        if loss_divergence_counter > TERMINATE_AFTER_DIVERGED_LOSS_STEPS:
          logging.info('Loss diverged for too many timesteps, breaking...')
          break
      else:
        loss_divergence_counter = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=step_metrics)

      if global_step_val % log_interval == 0:
        logging.info('step = %d, total loss = %f', global_step_val, total_loss)
        for a in range(n_agents):
          if not inactive_agent_ids or a not in inactive_agent_ids:
            logging.info('Loss for agent %d = %f', a, extra_loss[a].loss)
        steps_per_sec = ((global_step_val - timed_at_step) /
                         (collect_time + train_time))
        logging.info('%.3f steps/sec', steps_per_sec)
        logging.info('collect_time = %.3f, train_time = %.3f', collect_time,
                     train_time)
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        if global_step_val % train_checkpoint_interval == 0:
          train_checkpointer.save(global_step=global_step_val)

        if global_step_val % policy_checkpoint_interval == 0:
          policy_checkpointer.save(global_step=global_step_val)
          saved_model_path = os.path.join(
              saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
          saved_model.save(saved_model_path)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

    # One final eval before exiting.
    results = multiagent_metrics.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
        use_function=use_tf_functions,
    )
    if eval_metrics_callback is not None:
      eval_metrics_callback(results, global_step.numpy())
    multiagent_metrics.log_metrics(eval_metrics)


def main(_):
  logging.set_verbosity(logging.INFO)
  inactive_agent_ids = tuple()
  if FLAGS.inactive_agent_ids:
    inactive_agent_ids = [int(fid) for fid in FLAGS.inactive_agent_ids]
  train_eval(
      FLAGS.root_dir,
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
      inactive_agent_ids=inactive_agent_ids,
      random_seed=FLAGS.random_seed,
      reinit_checkpoint_dir=FLAGS.reinit_checkpoint_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  system_multiprocessing.handle_main(lambda _: app.run(main))
