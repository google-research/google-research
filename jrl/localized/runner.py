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

"""Generic offline RL main file."""


import logging
import os

from absl import app
from absl import flags

import gin

import acme
from acme import specs
from acme.utils import counting

import jax
import chex
from jax.config import config as jax_config
import tensorflow as tf
from tensorflow.compat.v1 import gfile

from jrl import agents
from jrl import data
from jrl import envs
from jrl import evaluation
from jrl.localized import runner_flags  # pylint:disable=unused-import
from jrl.utils import logger_utils
from jrl.utils import saved_model_lib


# from jax.experimental.jax2tf.examples import saved_model_lib
# saved_model_lib = jax2tf.examples.saved_model_lib

FLAGS = flags.FLAGS

def build_create_learner_logger(root_dir):
  summary_dir = os.path.join(root_dir, 'summary')
  def logger_fn():
    lg = logger_utils.create_default_logger(
        label='learner',
        tf_summary_logdir=summary_dir,
        save_data=True,
        # step_filter_delta=1_000,
        step_filter_delta=100,
        # step_filter_delta=1,
        # time_delta=1.0,
        asynchronous=True,
        # print_fn=None,
        # serialize_fn=,
        # steps_key='steps',
        steps_key='learner_steps',
        # extra_primary_keys=,
    )
    return lg
  return logger_fn


def build_create_env_loop_logger(
    root_dir,
    label='eval_loop',
    steps_key='learner_steps'):
  summary_dir = os.path.join(root_dir, 'summary')
  def logger_fn():
    lg = logger_utils.create_default_logger(
        label=label,
        tf_summary_logdir=summary_dir,
        save_data=True,
        # step_filter_delta=1_000,
        step_filter_delta=1,
        # time_delta=1.0,
        asynchronous=True,
        # print_fn=None,
        # serialize_fn=,
        # steps_key='steps',
        steps_key=steps_key,
        # extra_primary_keys=,
    )
    return lg
  return logger_fn


def main(_):
  if FLAGS.debug_nans:
    jax_config.update("jax_debug_nans", True)
  if FLAGS.spoof_multi_device:
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

  gin.parse_config_files_and_bindings(FLAGS.gin_configs, FLAGS.gin_bindings)

  # Optionally disable jit and pmap.
  if FLAGS.disable_jit:
    chex.fake_jit().__enter__()
    chex.fake_pmap().__enter__()


  # We cleanup a root_dir so that on preemptions we start from scratch.
  if gfile.Exists(FLAGS.root_dir):
    logging.info('Logdir (%s) already exists, removing it...', FLAGS.root_dir)
    gfile.DeleteRecursively(FLAGS.root_dir)

  create_env_fn = lambda: envs.create_environment(
    FLAGS.task_class, FLAGS.task_name, FLAGS.single_precision_env)
  create_data_iter_fn = (
    lambda: data.create_data_iter(
        FLAGS.task_class, FLAGS.task_name, FLAGS.batch_size))

  # for testing the data loader
  # batch = next(create_data_iter_fn())
  # import pdb; pdb.set_trace()
  # import time
  # data_iter = create_data_iter_fn()
  # for _ in range(50):
  #   begin_time = time.time()
  #   for _ in range(20):
  #     next(data_iter)
  #   print(f'\n\n{(time.time() - begin_time)/20.}\n\n')

  environment = create_env_fn()
  spec = specs.make_environment_spec(environment)
  # import time
  # import numpy as np
  # for _ in range(10):
  #   print('EPISODE')
  #   ep_begin = time.time()
  #   environment.reset()
  #   for _ in range(10):
  #     environment.step(np.zeros(shape=(12,)))
  #   ep_time = time.time() - ep_begin
  #   print('\n\n', ep_time)

  train_logger_factory = build_create_learner_logger(FLAGS.root_dir)
  rl_components = agents.create_agent(
      FLAGS.algorithm,
      spec,
      create_data_iter_fn,
      train_logger_factory)
  builder = rl_components.make_builder()

  counter = counting.Counter(time_delta=0.)
  is_offline_agent = not builder.make_replay_tables(spec)
  networks = rl_components.make_networks()
  seed = FLAGS.seed
  random_key = jax.random.PRNGKey(seed)
  learner_counter = counting.Counter(counter, 'learner', time_delta=0.)
  if is_offline_agent:
    random_key, sub_key = jax.random.split(random_key)
    learner = builder.make_learner(
        sub_key,
        networks,
        dataset=iter(()),  # dummy iterator
        counter=learner_counter)
    variable_source = learner
    train_loop = learner
  else:
    actor_counter_name = 'actor'
    steps_label = f'{actor_counter_name}_steps'

    agent = rl_components.make_agent( # pytype: disable=attribute-error
        networks,
        learner_counter,
        FLAGS.seed,)
    variable_source = agent._learner

    actor_counter = counting.Counter(counter, actor_counter_name, time_delta=0.)
    train_loop_cls = acme.EnvironmentLoop
    train_loop = train_loop_cls(
        environment,
        agent,
        counter=actor_counter,
        logger=build_create_env_loop_logger(
            FLAGS.root_dir,
            label='train_env_loop',
            steps_key=steps_label)()
    )

  random_key, sub_key = jax.random.split(random_key)
  all_eval_loops = []

  eval_actor = builder.make_actor(
      random_key=sub_key,
      policy_network=rl_components.make_eval_behavior_policy(networks),
      variable_source=variable_source)
  eval_env = create_env_fn()
  eval_counter = counting.Counter(
      counter,
      'eval_loop',
      time_delta=0.)
  # eval_loop = acme.EnvironmentLoop(
  #     eval_env,
  #     eval_actor,
  #     counter=eval_counter,
  #     label='eval_loop',
  #     logger=create_eval_loop_logger())
  eval_loop = evaluation.EvaluatorStandardWithFinalRewardLogging(
      eval_actor=eval_actor,
      environment=eval_env,
      num_episodes=FLAGS.episodes_per_eval,
      counter=eval_counter,
      logger=build_create_env_loop_logger(FLAGS.root_dir, label='eval_loop')(),
      eval_sync=None,
      progress_counter_name='eval_actor_steps',
      min_steps_between_evals=None,
      self_cleanup=False)
  all_eval_loops.append(eval_loop)

  if FLAGS.eval_with_q_filter:
    random_key, sub_key = jax.random.split(random_key)

    # pytype: disable=attribute-error
    old_value = builder._config.eval_with_q_filter
    builder._config.eval_with_q_filter = True
    q_filter_eval_actor = builder.make_actor(
        random_key=sub_key,
        policy_network=rl_components.make_eval_behavior_policy(
            networks, force_eval_with_q_filter=True, q_filter_with_unif=True),
        variable_source=variable_source,)
    builder._config.eval_with_q_filter = old_value
    # pytype: enable=attribute-error
    q_filter_eval_env = create_env_fn()
    q_filter_eval_counter = counting.Counter(
        counter,
        'q_filter_eval_loop',
        time_delta=0.)
    q_filter_eval_loop = evaluation.EvaluatorStandardWithFinalRewardLogging(
        eval_actor=q_filter_eval_actor,
        environment=q_filter_eval_env,
        num_episodes=FLAGS.episodes_per_eval,
        counter=q_filter_eval_counter,
        logger=build_create_env_loop_logger(
            FLAGS.root_dir, label='q_filter_eval_loop')(),
        eval_sync=None,
        progress_counter_name='q_filter_eval_actor_steps',
        min_steps_between_evals=None,
        self_cleanup=False)
    all_eval_loops.append(q_filter_eval_loop)

    # pytype: disable=attribute-error
    old_value = builder._config.eval_with_q_filter
    builder._config.eval_with_q_filter = True
    q_filter_eval_actor = builder.make_actor(
        random_key=sub_key,
        policy_network=rl_components.make_eval_behavior_policy(
            networks, force_eval_with_q_filter=True, q_filter_with_unif=False),
        variable_source=variable_source,)
    builder._config.eval_with_q_filter = old_value
    # pytype: enable=attribute-error
    q_filter_eval_env = create_env_fn()
    q_filter_eval_counter = counting.Counter(
        counter,
        'q_filter_eval_loop_no_unif',
        time_delta=0.)
    q_filter_eval_loop = evaluation.EvaluatorStandardWithFinalRewardLogging(
        eval_actor=q_filter_eval_actor,
        environment=q_filter_eval_env,
        num_episodes=FLAGS.episodes_per_eval,
        counter=q_filter_eval_counter,
        logger=build_create_env_loop_logger(
            FLAGS.root_dir, label='q_filter_eval_loop_no_unif')(),
        eval_sync=None,
        progress_counter_name='q_filter_no_unif_eval_actor_steps',
        min_steps_between_evals=None,
        self_cleanup=False)
    all_eval_loops.append(q_filter_eval_loop)


  # Run the training loop interleaved with some evaluations.
  assert FLAGS.num_steps % FLAGS.eval_every_steps == 0
  num_iterations = FLAGS.num_steps // FLAGS.eval_every_steps

  for _ in range(num_iterations):
    train_loop.run(num_steps=FLAGS.eval_every_steps)
    for el in all_eval_loops:
      el.run_once()

  # Final eval at the end of training
  for el in all_eval_loops:
    el.run_once()


  # saved model policy
  if FLAGS.create_saved_model_actor:
    _select_action = rl_components.make_eval_behavior_policy(networks)
    def select_action(params, x):
      obs, rng_seed = x[0], x[1]
      rng = jax.random.PRNGKey(rng_seed)
      return _select_action(params, rng, obs)
    input_spec = (
        tf.TensorSpec(eval_env.reset().observation.shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    saved_model_lib.convert_and_save_model(
        select_action,
        params=eval_actor._variable_client.params,  # pytype: disable=attribute-error
        model_dir=os.path.join(FLAGS.root_dir, 'saved_model'),
        input_signatures=[input_spec],)


  # Make sure to properly tear down the evaluators.
  # For e.g. to flush the files recording the episodes.
  for el in all_eval_loops:
    if hasattr(el, 'tear_down'):
      el.tear_down()


if __name__ == '__main__':
  app.run(main)
