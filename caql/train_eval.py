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

"""Train_eval for CAQL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf
from caql import agent_policy
from caql import caql_agent
from caql import epsilon_greedy_policy
from caql import gaussian_noise_policy
from caql import replay_memory as replay_memory_lib
from caql import utils

tf.disable_v2_behavior()
tf.enable_resource_variables()

flags.DEFINE_integer('seed', 0, 'The random seed instance.')

flags.DEFINE_string('env_name', 'Pendulum', 'Environment to evaluate on.')
flags.DEFINE_float('discount_factor', 0.99, 'Discount factor.')
flags.DEFINE_integer('time_out', 200, 'Environment time-out.')
flags.DEFINE_list(
    'action_bounds', None,
    'Comma separated list of min and max values for action '
    'variables. All action variables will have the same bounds. '
    'e.g., -.5,.5')

flags.DEFINE_integer('max_iterations', 10000, 'Maximum number of iterations.')
flags.DEFINE_integer('num_episodes_per_iteration', 1, '')
flags.DEFINE_integer('collect_experience_parallelism', 1,
                     'Number of threads for parallel experience collection.')

flags.DEFINE_list(
    'hidden_layers', '32,16',
    'Comma separated list of number of hidden units in each '
    'hidden layer. e.g., 32,16')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')
flags.DEFINE_integer('replay_memory_capacity', 100000,
                     'Capacity of replay buffer.')
flags.DEFINE_integer('train_steps_per_iteration', 20, '')
flags.DEFINE_integer('target_update_steps', 1, '')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for q_function.')
flags.DEFINE_float('learning_rate_action', 0.005, 'Learning rate for actions.')
flags.DEFINE_float('learning_rate_ga', 0.01,
                   'Learning rate for gradient ascent optimizer. Ignored if '
                   'gradient ascent optimizer is not used.')
flags.DEFINE_integer('action_maximization_iterations', 20,
                     'Iterations for inner gradient ascent.')

flags.DEFINE_float('tau_copy', 0.001, 'Portion to copy.')
flags.DEFINE_bool('clipped_target', True, 'Enable clipped double DQN.')
flags.DEFINE_integer(
    'hard_update_steps', 5000,
    'Number of gradient steps for hard-updating a target '
    'network. This is used only when `clipped_target` flag is '
    'enabled.')

flags.DEFINE_string('checkpoint_dir', None, 'Model checkpoint directory.')
flags.DEFINE_string('result_dir', None, 'Model result file dir.')

flags.DEFINE_bool('l2_loss_flag', True,
                  'True/False Flag to use l2_loss (as a baseline comparison).')
flags.DEFINE_bool('simple_lambda_flag', False,
                  'True/False Flag to use simple lambda.')
flags.DEFINE_bool('dual_filter_clustering_flag', False,
                  'Flags to use dual filter and clustering.')
flags.DEFINE_float('tolerance_init', None,
                   'Initial value for tolerance of max-Q solver.')
flags.DEFINE_float('tolerance_min', 1e-4,
                   'Minimum value for tolerance of max-Q solver.')
flags.DEFINE_float('tolerance_max', 100.0,
                   'Maximum value for tolerance of max-Q solver.')
flags.DEFINE_float('tolerance_decay', None,
                   'Decay rate for tolerance of max-Q solver.')
flags.DEFINE_bool('warmstart', True,
                  'Flags of warmstarting action maximization')
flags.DEFINE_bool('dual_q_label', True,
                  'Use dual max-Q label for action function training if True. '
                  'Otherwise, use primal max-Q label.')

flags.DEFINE_enum('solver', 'gradient_ascent',
                  ['dual', 'gradient_ascent', 'cross_entropy', 'ails', 'mip'],
                  'Solver to use for maxq.')
flags.DEFINE_float('initial_lambda', 1.0, 'Initial lambda for hinge loss.')
flags.DEFINE_enum(
    'exploration_policy', 'gaussian', ['egreedy', 'gaussian', 'none'],
    'Exploration policy to use. Choose "egreedy" for '
    'epsilon-greedy or "gaussian" for gaussian noise or "none" for no-exp.')

flags.DEFINE_float('epsilon', 1.0, 'Epsilon for epsilon-greedy exploration.')
flags.DEFINE_float('epsilon_decay', 0.999,
                   'Decay rate for epsilon-greedy exploration.')
flags.DEFINE_float('epsilon_min', 0.025,
                   'Epsilon minimum for epsilon-greedy exploration.')
flags.DEFINE_float('sigma', 1.0, 'Sigma for gaussian-noise exploration.')
flags.DEFINE_float('sigma_decay', 0.999,
                   'Decay rate for gaussian-noise exploration.')
flags.DEFINE_float('sigma_min', 0.025,
                   'Sigma minimum for gaussian-noise exploration.')

FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)

  assert FLAGS.replay_memory_capacity > FLAGS.batch_size * FLAGS.train_steps_per_iteration
  replay_memory = replay_memory_lib.ReplayMemory(
      name='ReplayBuffer', capacity=FLAGS.replay_memory_capacity)
  replay_memory.restore(FLAGS.checkpoint_dir)

  env = utils.create_env(FLAGS.env_name)
  state_spec, action_spec = utils.get_state_and_action_specs(
      env, action_bounds=FLAGS.action_bounds)

  hidden_layers = [int(h) for h in FLAGS.hidden_layers]

  summary_writer = None
  if FLAGS.result_dir is not None:
    hparam_dict = {
        'env_name': FLAGS.env_name,
        'discount_factor': FLAGS.discount_factor,
        'time_out': FLAGS.time_out,
        'action_bounds': FLAGS.action_bounds,
        'max_iterations': FLAGS.max_iterations,
        'num_episodes_per_iteration': FLAGS.num_episodes_per_iteration,
        'collect_experience_parallelism': FLAGS.collect_experience_parallelism,
        'hidden_layers': FLAGS.hidden_layers,
        'batch_size': FLAGS.batch_size,
        'train_steps_per_iteration': FLAGS.train_steps_per_iteration,
        'target_update_steps': FLAGS.target_update_steps,
        'learning_rate': FLAGS.learning_rate,
        'learning_rate_action': FLAGS.learning_rate_action,
        'learning_rate_ga': FLAGS.learning_rate_ga,
        'action_maximization_iterations': FLAGS.action_maximization_iterations,
        'tau_copy': FLAGS.tau_copy,
        'clipped_target': FLAGS.clipped_target,
        'hard_update_steps': FLAGS.hard_update_steps,
        'l2_loss_flag': FLAGS.l2_loss_flag,
        'simple_lambda_flag': FLAGS.simple_lambda_flag,
        'dual_filter_clustering_flag': FLAGS.dual_filter_clustering_flag,
        'solver': FLAGS.solver,
        'initial_lambda': FLAGS.initial_lambda,
        'tolerance_init': FLAGS.tolerance_init,
        'tolerance_min': FLAGS.tolerance_min,
        'tolerance_max': FLAGS.tolerance_max,
        'tolerance_decay': FLAGS.tolerance_decay,
        'warmstart': FLAGS.warmstart,
        'dual_q_label': FLAGS.dual_q_label,
        'seed': FLAGS.seed,
    }
    if FLAGS.exploration_policy == 'egreedy':
      hparam_dict.update({
          'epsilon': FLAGS.epsilon,
          'epsilon_decay': FLAGS.epsilon_decay,
          'epsilon_min': FLAGS.epsilon_min,
      })
    elif FLAGS.exploration_policy == 'gaussian':
      hparam_dict.update({
          'sigma': FLAGS.sigma,
          'sigma_decay': FLAGS.sigma_decay,
          'sigma_min': FLAGS.sigma_min,
      })

    utils.save_hparam_config(hparam_dict, FLAGS.result_dir)
    summary_writer = tf.summary.FileWriter(FLAGS.result_dir)

  with tf.Session() as sess:
    agent = caql_agent.CaqlAgent(
        session=sess,
        state_spec=state_spec,
        action_spec=action_spec,
        discount_factor=FLAGS.discount_factor,
        hidden_layers=hidden_layers,
        learning_rate=FLAGS.learning_rate,
        learning_rate_action=FLAGS.learning_rate_action,
        learning_rate_ga=FLAGS.learning_rate_ga,
        action_maximization_iterations=FLAGS.action_maximization_iterations,
        tau_copy=FLAGS.tau_copy,
        clipped_target_flag=FLAGS.clipped_target,
        hard_update_steps=FLAGS.hard_update_steps,
        batch_size=FLAGS.batch_size,
        l2_loss_flag=FLAGS.l2_loss_flag,
        simple_lambda_flag=FLAGS.simple_lambda_flag,
        dual_filter_clustering_flag=FLAGS.dual_filter_clustering_flag,
        solver=FLAGS.solver,
        dual_q_label=FLAGS.dual_q_label,
        initial_lambda=FLAGS.initial_lambda,
        tolerance_min_max=[FLAGS.tolerance_min, FLAGS.tolerance_max])

    saver = tf.train.Saver(max_to_keep=None)
    step = agent.initialize(saver, FLAGS.checkpoint_dir)

    iteration = int(step / FLAGS.train_steps_per_iteration)
    if iteration >= FLAGS.max_iterations:
      return

    greedy_policy = agent_policy.AgentPolicy(action_spec, agent)
    if FLAGS.exploration_policy == 'egreedy':
      epsilon_init = max(FLAGS.epsilon * (FLAGS.epsilon_decay**iteration),
                         FLAGS.epsilon_min)
      behavior_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
          greedy_policy, epsilon_init, FLAGS.epsilon_decay,
          FLAGS.epsilon_min)
    elif FLAGS.exploration_policy == 'gaussian':
      sigma_init = max(FLAGS.sigma * (FLAGS.sigma_decay**iteration),
                       FLAGS.sigma_min)
      behavior_policy = gaussian_noise_policy.GaussianNoisePolicy(
          greedy_policy, sigma_init, FLAGS.sigma_decay, FLAGS.sigma_min)
    elif FLAGS.exploration_policy == 'none':
      behavior_policy = greedy_policy

    logging.info('Start with iteration %d, step %d, %s', iteration, step,
                 behavior_policy.params_debug_str())

    while iteration < FLAGS.max_iterations:
      utils.collect_experience_parallel(
          num_episodes=FLAGS.num_episodes_per_iteration,
          session=sess,
          behavior_policy=behavior_policy,
          time_out=FLAGS.time_out,
          discount_factor=FLAGS.discount_factor,
          replay_memory=replay_memory)

      if (replay_memory.size <
          FLAGS.batch_size * FLAGS.train_steps_per_iteration):
        continue

      tf_summary = None
      if summary_writer:
        tf_summary = tf.Summary()

      q_function_losses = []
      q_vals = []
      lambda_function_losses = []
      action_function_losses = []
      portion_active_data = []
      portion_active_data_and_clusters = []
      ts_begin = time.time()

      # 'step' can be started from any number if the program is restored from
      # a checkpoint after crash or pre-emption.
      local_step = step % FLAGS.train_steps_per_iteration
      while local_step < FLAGS.train_steps_per_iteration:
        minibatch = replay_memory.sample(FLAGS.batch_size)
        if FLAGS.tolerance_decay is not None:
          tolerance_decay = FLAGS.tolerance_decay**iteration
        else:
          tolerance_decay = None

        # Leave summary only for the last one.
        agent_tf_summary_vals = None
        if local_step == FLAGS.train_steps_per_iteration - 1:
          agent_tf_summary_vals = []

        # train q_function and lambda_function networks
        (q_function_loss, target_q_vals, lambda_function_loss,
         best_train_label_batch, portion_active_constraint,
         portion_active_constraint_and_cluster) = (
             agent.train_q_function_network(
                 minibatch,
                 FLAGS.tolerance_init,
                 tolerance_decay,
                 FLAGS.warmstart,
                 agent_tf_summary_vals))

        action_function_loss = agent.train_action_function_network(
            best_train_label_batch)

        q_function_losses.append(q_function_loss)
        q_vals.append(target_q_vals)
        lambda_function_losses.append(lambda_function_loss)
        action_function_losses.append(action_function_loss)
        portion_active_data.append(portion_active_constraint)
        portion_active_data_and_clusters.append(
            portion_active_constraint_and_cluster)

        local_step += 1
        step += 1
        if step % FLAGS.target_update_steps == 0:
          agent.update_target_network()
        if FLAGS.clipped_target and step % FLAGS.hard_update_steps == 0:
          agent.update_target_network2()

      elapsed_secs = time.time() - ts_begin
      steps_per_sec = FLAGS.train_steps_per_iteration / elapsed_secs

      iteration += 1
      logging.info(
          'Iteration: %d, steps per sec: %.2f, replay memory size: %d, %s, '
          'avg q_function loss: %.3f, '
          'avg lambda_function loss: %.3f, '
          'avg action_function loss: %.3f '
          'avg portion active data: %.3f '
          'avg portion active data and cluster: %.3f ',
          iteration, steps_per_sec, replay_memory.size,
          behavior_policy.params_debug_str(),
          np.mean(q_function_losses), np.mean(lambda_function_losses),
          np.mean(action_function_losses), np.mean(portion_active_data),
          np.mean(portion_active_data_and_clusters))

      if tf_summary:
        if agent_tf_summary_vals:
          tf_summary.value.extend(agent_tf_summary_vals)
        tf_summary.value.extend([
            tf.Summary.Value(tag='steps_per_sec', simple_value=steps_per_sec),
            tf.Summary.Value(
                tag='avg_q_loss', simple_value=np.mean(q_function_loss)),
            tf.Summary.Value(tag='avg_q_val', simple_value=np.mean(q_vals)),
            tf.Summary.Value(
                tag='avg_portion_active_data',
                simple_value=np.mean(portion_active_data)),
            tf.Summary.Value(
                tag='avg_portion_active_data_and_cluster',
                simple_value=np.mean(portion_active_data_and_clusters))
        ])

      behavior_policy.update_params()
      utils.periodic_updates(
          iteration=iteration,
          train_step=step,
          replay_memories=(replay_memory,),
          greedy_policy=greedy_policy,
          use_action_function=True,
          saver=saver,
          sess=sess,
          time_out=FLAGS.time_out,
          tf_summary=tf_summary)

      if summary_writer and tf_summary:
        summary_writer.add_summary(tf_summary, step)

  logging.info('Training is done.')
  env.close()


if __name__ == '__main__':
  app.run(main)
