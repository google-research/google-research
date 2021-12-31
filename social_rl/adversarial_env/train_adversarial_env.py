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

r"""Train and Eval launch script for PAIRED / adversarial environment training.

To run:

```bash tensorboard.sh --port=2222 --logdir /tmp/adversarial_env/

python -m train_adversarial_env --root_dir=/tmp/adversarial_env/
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.metrics import tf_metrics
from tf_agents.system import system_multiprocessing

from social_rl import gym_multigrid  # Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl.adversarial_env import adversarial_driver
from social_rl.adversarial_env import adversarial_env
from social_rl.adversarial_env import adversarial_env_parallel
from social_rl.adversarial_env import adversarial_eval
from social_rl.adversarial_env import agent_train_package


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_name', 'MultiGrid-Adversarial-v0',
                    'Name of an environment')
flags.DEFINE_integer('replay_buffer_capacity', 3001,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('num_parallel_envs', 4,
                     'Number of environments to run in parallel. Originally 30')
flags.DEFINE_integer('num_train_steps', 500000,
                     'Number of train steps to run before finishing. Number of '
                     'train steps per environment episodes differs based on how'
                     'many agents are being trained.')
flags.DEFINE_integer('num_epochs', 25,
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer('collect_episodes_per_iteration', 4,
                     'The number of episodes to take in the environment before '
                     'each update. This is the total across all parallel '
                     'environments.')
flags.DEFINE_integer('num_eval_episodes', 2,
                     'The number of episodes to run eval on. Originally 30.')
flags.DEFINE_boolean('debug', False, 'Turn on debugging and tf functions off.')
flags.DEFINE_boolean('agent_regret_off', False,
                     'If True, agents will train with normal reward not regret')
flags.DEFINE_boolean('unconstrained_adversary', False,
                     'If True, does not use a second agent to constrain the '
                     'adversary environment.')
flags.DEFINE_boolean('domain_randomization', False,
                     'If True, do not use adversarial training, just randomize'
                     'block positions.')
flags.DEFINE_float('percent_random_episodes', 0.,
                   'The % of episodes trained with domain randomization.')
flags.DEFINE_boolean('no_adversary_rnn', False,
                     'If True, will not use an RNN to parameterize the '
                     'adversary environment')
flags.DEFINE_boolean('flexible_protagonist', False,
                     'Which agent assumes the role of protagonist depends on '
                     'the scores of both agents')
flags.DEFINE_integer('protagonist_episode_length', None,
                     'Number of steps for protagonist episodes.')
flags.DEFINE_integer('adversary_population_size', 1,
                     'Number of adversaries in the population')
flags.DEFINE_integer('protagonist_population_size', 1,
                     'Number of training agents in the population')
flags.DEFINE_integer('antagonist_population_size', 1,
                     'Number of agents aligned with the adversary')
flags.DEFINE_boolean('combined_population', False,
                     'If True, will have a single population of agents. Which'
                     'agent is the antagonist depends on which gets the highest'
                     'score for a given round.')
flags.DEFINE_float('block_budget_weight', 0.,
                   'Coefficient used to impose block budget on the adversary.')
FLAGS = flags.FLAGS

# Loss value that is considered too high and training will be terminated.
MAX_LOSS = 1e9
# How many steps does the loss have to be diverged for (too high, inf, nan)
# after the training terminates. This should prevent termination on short loss
# spikes.
TERMINATE_AFTER_DIVERGED_STEPS = 100


@gin.configurable
def train_eval(
    root_dir,
    env_name='MultiGrid-Adversarial-v0',
    random_seed=None,
    # PAIRED parameters
    agents_learn_with_regret=True,
    non_negative_regret=True,
    unconstrained_adversary=False,
    domain_randomization=False,
    percent_random_episodes=0.,
    protagonist_episode_length=None,
    flexible_protagonist=False,
    adversary_population_size=1,
    protagonist_population_size=1,
    antagonist_population_size=1,
    combined_population=False,
    block_budget_weight=0,
    # Agent architecture params
    actor_fc_layers=(32, 32),
    value_fc_layers=(32, 32),
    lstm_size=(128,),
    conv_filters=8,
    conv_kernel=3,
    direction_fc=5,
    entropy_regularization=0.,
    # Adversary architecture params
    adversary_env_rnn=True,
    adv_actor_fc_layers=(32, 32),
    adv_value_fc_layers=(32, 32),
    adv_lstm_size=(128,),
    adv_conv_filters=16,
    adv_conv_kernel=3,
    adv_timestep_fc=10,
    adv_entropy_regularization=0.,
    # Params for collect
    num_train_steps=500000,
    collect_episodes_per_iteration=30,
    num_parallel_envs=5,
    replay_buffer_capacity=1001,  # Per-environment
    # Params for train
    num_epochs=25,
    learning_rate=1e-4,
    # Params for eval
    num_eval_episodes=5,
    eval_interval=10,
    # Params for summaries and logging
    train_checkpoint_interval=100,
    policy_checkpoint_interval=100,
    log_interval=5,
    summary_interval=5,
    summaries_flush_secs=1,
    use_tf_functions=True,
    debug_summaries=True,
    summarize_grads_and_vars=True,
    eval_metrics_callback=None,
    debug=True):
  """Adversarial environment train and eval."""
  tf.compat.v1.enable_v2_behavior()

  if debug:
    logging.info('In debug mode. Disabling tf functions.')
    use_tf_functions = False

  if combined_population:
    # The number of train steps per environment episodes differs based on the
    # number of agents trained per episode. Adjust value when training a
    # population of agents per episode.
    # The number of agents must change from 3 (for protagonist, antagonist,
    # adversary) to protagonist population size + adversary
    num_train_steps = num_train_steps / 3 * (protagonist_population_size + 1)

  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  gym_env = adversarial_env.load(env_name)

  # Set up logging
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)

  # Initialize global step and random seed
  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    if random_seed is not None:
      tf.compat.v1.set_random_seed(random_seed)

    # Create environments
    logging.info('Creating %d environments...', num_parallel_envs)
    eval_tf_env = adversarial_env.AdversarialTFPyEnvironment(
        adversarial_env_parallel.AdversarialParallelPyEnvironment(
            [lambda: adversarial_env.load(env_name)] * num_eval_episodes))
    tf_env = adversarial_env.AdversarialTFPyEnvironment(
        adversarial_env_parallel.AdversarialParallelPyEnvironment(
            [lambda: adversarial_env.load(env_name)] * num_parallel_envs))


    logging.info('Preparing to train...')
    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    # Logging for special environment metrics
    env_metrics_names = [
        'DistanceToGoal',
        'NumBlocks',
        'DeliberatePlacement',
        'NumEnvEpisodes',
        'GoalX',
        'GoalY',
        'IsPassable',
        'ShortestPathLength',
        'ShortestPassablePathLength',
        'SolvedPathLength',
        'TrainEpisodesCollected',
    ]
    env_train_metrics = []
    env_eval_metrics = []
    for mname in env_metrics_names:
      env_train_metrics.append(adversarial_eval.AdversarialEnvironmentScalar(
          batch_size=num_parallel_envs, name=mname))
      env_eval_metrics.append(adversarial_eval.AdversarialEnvironmentScalar(
          batch_size=num_eval_episodes, name=mname))

    # Create (populations of) both agents that learn to navigate the environment
    agents = {}
    for agent_name in ['agent', 'adversary_agent']:
      if (agent_name == 'adversary_agent' and
          (domain_randomization or unconstrained_adversary or
           combined_population)):
        # Antagonist agent not needed for baselines
        continue

      max_steps = gym_env.max_steps
      if protagonist_episode_length is not None and agent_name == 'agent':
        max_steps = protagonist_episode_length

      if agent_name == 'agent':
        population_size = protagonist_population_size
      else:
        population_size = antagonist_population_size

      agents[agent_name] = []
      for i in range(population_size):
        logging.info('Creating agent... %s %d', agent_name, i)
        agents[agent_name].append(agent_train_package.AgentTrainPackage(
            tf_env,
            global_step,
            root_dir,
            step_metrics,
            name=agent_name,
            use_tf_functions=use_tf_functions,
            max_steps=max_steps,
            replace_reward=(not unconstrained_adversary
                            and agents_learn_with_regret),
            id_num=i,

            # Architecture hparams
            learning_rate=learning_rate,
            actor_fc_layers=actor_fc_layers,
            value_fc_layers=value_fc_layers,
            lstm_size=lstm_size,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            scalar_fc=direction_fc,
            entropy_regularization=entropy_regularization,

            # Training & logging settings
            num_epochs=num_epochs,
            num_eval_episodes=num_eval_episodes,
            num_parallel_envs=num_parallel_envs,
            replay_buffer_capacity=replay_buffer_capacity,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars))

    if not domain_randomization:
      xy_dim = None
      if 'Reparam' in env_name:
        xy_dim = gym_env.width

      # Create (population of) environment-generating adversaries
      agents['adversary_env'] = []
      for i in range(adversary_population_size):
        logging.info('Creating adversary environment %d', i)
        agents['adversary_env'].append(agent_train_package.AgentTrainPackage(
            tf_env,
            global_step,
            root_dir,
            step_metrics,
            name='adversary_env',
            is_environment=True,
            use_rnn=adversary_env_rnn,
            use_tf_functions=use_tf_functions,
            max_steps=gym_env.adversary_max_steps,
            replace_reward=True,
            non_negative_regret=non_negative_regret,
            xy_dim=xy_dim,
            id_num=i,
            block_budget_weight=block_budget_weight,

            # Architecture hparams
            learning_rate=learning_rate,
            actor_fc_layers=adv_actor_fc_layers,
            value_fc_layers=adv_value_fc_layers,
            lstm_size=adv_lstm_size,
            conv_filters=adv_conv_filters,
            conv_kernel=adv_conv_kernel,
            scalar_fc=adv_timestep_fc,
            entropy_regularization=adv_entropy_regularization,

            # Training & logging settings
            num_epochs=num_epochs,
            num_eval_episodes=num_eval_episodes,
            num_parallel_envs=num_parallel_envs,
            replay_buffer_capacity=replay_buffer_capacity,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars))

    logging.info('Creating adversarial drivers')
    if unconstrained_adversary or domain_randomization or combined_population:
      adversary_agent = None
    else:
      adversary_agent = agents['adversary_agent']

    if domain_randomization:
      adversary_env = None
    else:
      adversary_env = agents['adversary_env']

    collect_driver = adversarial_driver.AdversarialDriver(
        tf_env,
        agents['agent'],
        adversary_agent,
        adversary_env,
        env_metrics=env_train_metrics,
        collect=True,
        disable_tf_function=True,  # TODO(natashajaques): enable tf functions
        debug=debug,
        combined_population=combined_population,
        flexible_protagonist=flexible_protagonist)
    eval_driver = adversarial_driver.AdversarialDriver(
        eval_tf_env,
        agents['agent'],
        adversary_agent,
        adversary_env,
        env_metrics=env_eval_metrics,
        collect=False,
        disable_tf_function=True,  # TODO(natashajaques): enable tf functions
        debug=False,
        combined_population=combined_population,
        flexible_protagonist=flexible_protagonist)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    # Save operative config as late as possible to include used configurables.
    if global_step.numpy() == 0:
      config_filename = os.path.join(
          train_dir, 'operative_config-{}.gin'.format(global_step.numpy()))
      with tf.io.gfile.GFile(config_filename, 'wb') as f:
        f.write(gin.operative_config_str())

    total_episodes = 0
    logging.info('Commencing train loop!')
    # Note that if there are N agents, the global step will increase at N times
    # the rate for the same number of train episodes (because it increases for
    # each agent trained. Therefore it is important to divide the train steps by
    # N when plotting).
    while global_step.numpy() <= num_train_steps:
      global_step_val = global_step.numpy()

      # Evaluation
      if global_step_val % eval_interval == 0:
        if debug:
          logging.info('Performing evaluation at step %d', global_step_val)
        results = adversarial_eval.eager_compute(
            eval_driver,
            agents,
            env_metrics=env_eval_metrics,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics'
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        adversarial_eval.log_metrics(agents, env_eval_metrics)

      # Used to interleave randomized episodes with adversarial training
      random_episodes = False
      if percent_random_episodes > 0:
        chance_random = random.random()
        if chance_random < percent_random_episodes:
          random_episodes = True
          if debug: logging.info('RANDOM EPISODE')

      # Collect data
      if debug: logging.info('Collecting at step %d', global_step_val)
      start_time = time.time()
      train_idxs = collect_driver.run(random_episodes=random_episodes)
      collect_time += time.time() - start_time
      if debug:
        logging.info('Trained agents: %s', ', '.join(train_idxs))

      # Log total episodes collected
      total_episodes += collect_episodes_per_iteration
      eps_metric = [tf.convert_to_tensor(total_episodes, dtype=tf.float32)]
      env_train_metrics[-1](eps_metric)
      env_eval_metrics[-1](eps_metric)
      if debug:
        logging.info('Have collected a total of %d episodes', total_episodes)

      # Train
      if debug: logging.info('Training at step %d', global_step_val)
      start_time = time.time()
      for name, agent_list in agents.items():
        if random_episodes and name == 'adversary_env':
          # Don't train the adversary on randomly generated episodes
          continue

        # Train the agents selected by the driver this training run
        for agent_idx in train_idxs[name]:
          agent = agent_list[agent_idx]
          if debug: logging.info('\tTraining %s %d', name, agent_idx)
          agent.total_loss, agent.extra_loss = agent.train_step()
          agent.replay_buffer.clear()

          # Check for exploding losses.
          if (math.isnan(agent.total_loss) or math.isinf(agent.total_loss) or
              agent.total_loss > MAX_LOSS):
            agent.loss_divergence_counter += 1
            if agent.loss_divergence_counter > TERMINATE_AFTER_DIVERGED_STEPS:
              logging.info('Loss diverged for too many timesteps, breaking...')
              break
          else:
            agent.loss_divergence_counter = 0

        # Log train metrics to tensorboard
        for train_metric in agent.train_metrics:
          train_metric.tf_summaries(
              train_step=global_step, step_metrics=step_metrics)
        if agent.is_environment:
          agent.env_train_metric.tf_summaries(
              train_step=global_step, step_metrics=step_metrics)

      # Global environment stats logging
      for metric in env_train_metrics:
        metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)
      if debug:
        logging.info('Train metrics for step %d', global_step_val)
        adversarial_eval.log_metrics(agents, env_train_metrics)

      train_time += time.time() - start_time

      # Print output logging statements
      if global_step_val % log_interval == 0:
        for name, agent_list in agents.items():
          for i, agent in enumerate(agent_list):
            print('Loss for', name, i, '=', agent.total_loss)
        steps_per_sec = (
            (global_step_val - timed_at_step) / (collect_time + train_time))
        logging.info('%.3f steps/sec', steps_per_sec)
        logging.info('collect_time = %.3f, train_time = %.3f', collect_time,
                     train_time)
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar(
              name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        # Save checkpoints for all agent types and population members
        if global_step_val % train_checkpoint_interval == 0:
          for name, agent_list in agents.items():
            for i, agent in enumerate(agent_list):
              if debug:
                logging.info('Saving checkpoint for agent %s %d', name, i)
              agent.train_checkpointer.save(global_step=global_step_val)
        if global_step_val % policy_checkpoint_interval == 0:
          for name, agent_list in agents.items():
            for i, agent in enumerate(agent_list):
              agent.policy_checkpointer.save(global_step=global_step_val)
              saved_model_path = os.path.join(
                  agent.saved_model_dir,
                  'policy_' + ('%d' % global_step_val).zfill(9))
              agent.saved_model.save(saved_model_path)

        timed_at_step = global_step_val
        collect_time = 0
        train_time = 0

    if total_episodes > 0:
      # Save one final checkpoint for all agent types and population members
      for name, agent_list in agents.items():
        for i, agent in enumerate(agent_list):
          if debug:
            logging.info('Saving checkpoint for agent %s %d', name, i)
          agent.train_checkpointer.save(global_step=global_step_val)
      for name, agent_list in agents.items():
        for i, agent in enumerate(agent_list):
          agent.policy_checkpointer.save(global_step=global_step_val)
          saved_model_path = os.path.join(
              agent.saved_model_dir,
              'policy_' + ('%d' % global_step_val).zfill(9))
          agent.saved_model.save(saved_model_path)

      # One final eval before exiting.
      results = adversarial_eval.eager_compute(
          eval_driver,
          agents,
          env_metrics=env_eval_metrics,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix='Metrics'
      )
      if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
      adversarial_eval.log_metrics(agents, env_eval_metrics)


def main(_):
  logging.set_verbosity(logging.INFO)
  train_eval(
      FLAGS.root_dir,
      env_name=FLAGS.env_name,
      agents_learn_with_regret=not FLAGS.agent_regret_off,
      unconstrained_adversary=FLAGS.unconstrained_adversary,
      domain_randomization=FLAGS.domain_randomization,
      percent_random_episodes=FLAGS.percent_random_episodes,
      adversary_env_rnn=not FLAGS.no_adversary_rnn,
      protagonist_episode_length=FLAGS.protagonist_episode_length,
      flexible_protagonist=FLAGS.flexible_protagonist,
      adversary_population_size=FLAGS.adversary_population_size,
      protagonist_population_size=FLAGS.protagonist_population_size,
      antagonist_population_size=FLAGS.antagonist_population_size,
      combined_population=FLAGS.combined_population,
      block_budget_weight=FLAGS.block_budget_weight,
      num_train_steps=FLAGS.num_train_steps,
      collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
      num_parallel_envs=FLAGS.num_parallel_envs,
      replay_buffer_capacity=FLAGS.replay_buffer_capacity,
      num_epochs=FLAGS.num_epochs,
      num_eval_episodes=FLAGS.num_eval_episodes,
      debug=FLAGS.debug)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  system_multiprocessing.handle_main(lambda _: app.run(main))
