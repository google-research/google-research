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

"""Script for training and evaluating RPC agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import rpc_agent
import rpc_utils
from six.moves import range
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import data_converter
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS

DEFAULT_KL_CONSTRAINT = 1.0
DEFAULT_DUAL_LR = 1.0


@gin.configurable
def train_eval(
    root_dir,
    env_name='HalfCheetah-v2',
    num_iterations=3000000,
    actor_fc_layers=(),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    initial_collect_steps=10000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    dual_learning_rate=3e-4,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=0.1,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=50000,
    policy_checkpoint_interval=50000,
    rb_checkpoint_interval=50000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None,
    latent_dim=10,
    log_prob_reward_scale=0.0,
    predictor_updates_encoder=False,
    predict_prior=True,
    use_recurrent_actor=False,
    rnn_sequence_length=20,
    clip_max_stddev=10.0,
    clip_min_stddev=0.1,
    clip_mean=30.0,
    predictor_num_layers=2,
    use_identity_encoder=False,
    identity_encoder_single_stddev=False,
    kl_constraint=1.0,
    eval_dropout=(),
    use_residual_predictor=True,
    gym_kwargs=None,
    predict_prior_std=True,
    random_seed=0,):
  """A simple train and eval for SAC."""
  np.random.seed(random_seed)
  tf.random.set_seed(random_seed)
  if use_recurrent_actor:
    batch_size = batch_size // rnn_sequence_length
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):

    _build_env = functools.partial(suite_gym.load, environment_name=env_name,  # pylint: disable=invalid-name
                                   gym_env_wrappers=(), gym_kwargs=gym_kwargs)

    tf_env = tf_py_environment.TFPyEnvironment(_build_env())
    eval_vec = []  # (name, env, metrics)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]
    eval_tf_env = tf_py_environment.TFPyEnvironment(_build_env())
    name = ''
    eval_vec.append((name, eval_tf_env, eval_metrics))

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()
    if latent_dim == 'obs':
      latent_dim = observation_spec.shape[0]

    def _activation(t):
      t1, t2 = tf.split(t, 2, axis=1)
      low = -np.inf if clip_mean is None else -clip_mean
      high = np.inf if clip_mean is None else clip_mean
      t1 = rpc_utils.squash_to_range(t1, low, high)

      if clip_min_stddev is None:
        low = -np.inf
      else:
        low = tf.math.log(tf.exp(clip_min_stddev) - 1.0)
      if clip_max_stddev is None:
        high = np.inf
      else:
        high = tf.math.log(tf.exp(clip_max_stddev) - 1.0)
      t2 = rpc_utils.squash_to_range(t2, low, high)
      return tf.concat([t1, t2], axis=1)

    if use_identity_encoder:
      assert latent_dim == observation_spec.shape[0]
      obs_input = tf.keras.layers.Input(observation_spec.shape)
      zeros = 0.0 * obs_input[:, :1]
      stddev_dim = 1 if identity_encoder_single_stddev else latent_dim
      pre_stddev = tf.keras.layers.Dense(stddev_dim, activation=None)(zeros)
      ones = zeros + tf.ones((1, latent_dim))
      pre_stddev = pre_stddev * ones  # Multiply to broadcast to latent_dim.
      pre_mean_stddev = tf.concat([obs_input, pre_stddev], axis=1)
      output = tfp.layers.IndependentNormal(latent_dim)(pre_mean_stddev)
      encoder_net = tf.keras.Model(inputs=obs_input,
                                   outputs=output)
    else:
      encoder_net = tf.keras.Sequential([
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(256, activation='relu'),
          tf.keras.layers.Dense(
              tfp.layers.IndependentNormal.params_size(latent_dim),
              activation=_activation,
              kernel_initializer='glorot_uniform'),
          tfp.layers.IndependentNormal(latent_dim),
      ])

    # Build the predictor net
    obs_input = tf.keras.layers.Input(observation_spec.shape)
    action_input = tf.keras.layers.Input(action_spec.shape)

    class ConstantIndependentNormal(tfp.layers.IndependentNormal):
      """A keras layer that always returns N(0, 1) distribution."""

      def call(self, inputs):
        loc_scale = tf.concat([
            tf.zeros((latent_dim,)),
            tf.fill((latent_dim,), tf.math.log(tf.exp(1.0) - 1))
        ],
                              axis=0)
        # Multiple by [B x 1] tensor to broadcast batch dimension.
        loc_scale = loc_scale * tf.ones_like(inputs[:, :1])
        return super(ConstantIndependentNormal, self).call(loc_scale)

    if predict_prior:
      z = encoder_net(obs_input)
      if not predictor_updates_encoder:
        z = tf.stop_gradient(z)
      za = tf.concat([z, action_input], axis=1)
      if use_residual_predictor:
        za_input = tf.keras.layers.Input(za.shape[1])
        loc_scale = tf.keras.Sequential(
            predictor_num_layers * [tf.keras.layers.Dense(256, activation='relu')] + [  # pylint: disable=line-too-long
                tf.keras.layers.Dense(
                    tfp.layers.IndependentNormal.params_size(latent_dim),
                    activation=_activation,
                    kernel_initializer='zeros'),
            ])(za_input)
        if predict_prior_std:
          combined_loc_scale = tf.concat([
              loc_scale[:, :latent_dim] + za_input[:, :latent_dim],
              loc_scale[:, latent_dim:]
          ],
                                         axis=1)
        else:
          # Note that softplus(log(e - 1)) = 1.
          combined_loc_scale = tf.concat([
              loc_scale[:, :latent_dim] + za_input[:, :latent_dim],
              tf.math.log(np.e - 1) * tf.ones_like(loc_scale[:, latent_dim:])
          ],
                                         axis=1)
        dist = tfp.layers.IndependentNormal(latent_dim)(combined_loc_scale)
        output = tf.keras.Model(inputs=za_input, outputs=dist)(za)
      else:
        assert predict_prior_std
        output = tf.keras.Sequential(
            predictor_num_layers * [tf.keras.layers.Dense(256, activation='relu')] +  # pylint: disable=line-too-long
            [tf.keras.layers.Dense(
                tfp.layers.IndependentNormal.params_size(latent_dim),
                activation=_activation,
                kernel_initializer='zeros'),
             tfp.layers.IndependentNormal(latent_dim),
             ])(za)
    else:
      # scale is chosen by inverting the softplus function to equal 1.
      if len(obs_input.shape) > 2:
        input_reshaped = tf.reshape(
            obs_input, [-1, tf.math.reduce_prod(obs_input.shape[1:])])
        #  Multiply by [B x 1] tensor to broadcast batch dimension.
        za = tf.zeros(latent_dim + action_spec.shape[0],) * tf.ones_like(input_reshaped[:, :1])  # pylint: disable=line-too-long
      else:
        #  Multiple by [B x 1] tensor to broadcast batch dimension.
        za = tf.zeros(latent_dim + action_spec.shape[0],) * tf.ones_like(obs_input[:, :1])  # pylint: disable=line-too-long
      output = tf.keras.Sequential([
          ConstantIndependentNormal(latent_dim),
      ])(
          za)
    predictor_net = tf.keras.Model(inputs=(obs_input, action_input),
                                   outputs=output)
    if use_recurrent_actor:
      ActorClass = rpc_utils.RecurrentActorNet  # pylint: disable=invalid-name
    else:
      ActorClass = rpc_utils.ActorNet  # pylint: disable=invalid-name
    actor_net = ActorClass(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        encoder=encoder_net,
        predictor=predictor_net,
        fc_layers=actor_fc_layers)

    critic_net = rpc_utils.CriticNet(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
    critic_net_2 = None
    target_critic_net_1 = None
    target_critic_net_2 = None

    tf_agent = rpc_agent.RpAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        critic_network_2=critic_net_2,
        target_critic_network=target_critic_net_1,
        target_critic_network_2=target_critic_net_2,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    dual_optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=dual_learning_rate)
    tf_agent.initialize()

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)
    replay_observer = [replay_buffer.add_batch]

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
    ]
    kl_metric = rpc_utils.AverageKLMetric(
        encoder=encoder_net,
        predictor=predictor_net,
        batch_size=tf_env.batch_size)
    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy

    checkpoint_items = {
        'ckpt_dir': train_dir,
        'agent': tf_agent,
        'global_step': global_step,
        'metrics': metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        'dual_optimizer': dual_optimizer,
    }
    train_checkpointer = common.Checkpointer(**checkpoint_items)

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps,
        transition_observers=[kl_metric])

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration,
        transition_observers=[kl_metric])

    if use_tf_functions:
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    if replay_buffer.num_frames() == 0:
      # Collect initial replay data.
      logging.info(
          'Initializing replay buffer by collecting experience for %d steps '
          'with a random policy.', initial_collect_steps)
      initial_collect_driver.run()

    for name, eval_tf_env, eval_metrics in eval_vec:
      results = metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix='Metrics-%s' % name,
      )
      if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
      metric_utils.log_metrics(eval_metrics, prefix=name)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0
    train_time_acc = 0
    env_time_acc = 0

    if use_recurrent_actor:  # default from sac/train_eval_rnn.py
      num_steps = rnn_sequence_length + 1
      def _filter_invalid_transition(trajectories, unused_arg1):
        return tf.reduce_all(~trajectories.is_boundary()[:-1])

      tf_agent._as_transition = data_converter.AsTransition(  # pylint: disable=protected-access
          tf_agent.data_context, squeeze_time_dim=False)
    else:
      num_steps = 2

      def _filter_invalid_transition(trajectories, unused_arg1):
        return ~trajectories.is_boundary()[0]
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=num_steps).unbatch().filter(_filter_invalid_transition)

    dataset = dataset.batch(batch_size).prefetch(5)
    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)

    @tf.function
    def train_step():
      experience, _ = next(iterator)

      prior = predictor_net((experience.observation[:, 0],
                             experience.action[:, 0]), training=False)
      z_next = encoder_net(experience.observation[:, 1], training=False)
      # predictor_kl is a vector of size batch_size.
      predictor_kl = tfp.distributions.kl_divergence(z_next, prior)

      with tf.GradientTape() as tape:
        tape.watch(actor_net._log_kl_coefficient)  # pylint: disable=protected-access
        dual_loss = -1.0 * actor_net._log_kl_coefficient * (  # pylint: disable=protected-access
            tf.stop_gradient(tf.reduce_mean(predictor_kl)) - kl_constraint)
      dual_grads = tape.gradient(dual_loss, [actor_net._log_kl_coefficient])  # pylint: disable=protected-access
      grads_and_vars = list(zip(dual_grads, [actor_net._log_kl_coefficient]))  # pylint: disable=protected-access
      dual_optimizer.apply_gradients(grads_and_vars)

      # Clip the dual variable so exp(log_kl_coef) <= 1e6.
      log_kl_coef = tf.clip_by_value(actor_net._log_kl_coefficient,  # pylint: disable=protected-access
                                     -1.0 * np.log(1e6), np.log(1e6))
      actor_net._log_kl_coefficient.assign(log_kl_coef)  # pylint: disable=protected-access

      with tf.name_scope('dual_loss'):
        tf.compat.v2.summary.scalar(name='dual_loss',
                                    data=tf.reduce_mean(dual_loss),
                                    step=global_step)
        tf.compat.v2.summary.scalar(name='log_kl_coefficient',
                                    data=actor_net._log_kl_coefficient,  # pylint: disable=protected-access
                                    step=global_step)

      z_entropy = z_next.entropy()
      log_prob = prior.log_prob(z_next.sample())
      with tf.name_scope('rp-metrics'):
        common.generate_tensor_summaries('predictor_kl', predictor_kl,
                                         global_step)
        common.generate_tensor_summaries('z_entropy', z_entropy, global_step)
        common.generate_tensor_summaries('log_prob', log_prob, global_step)
        common.generate_tensor_summaries('z_mean', z_next.mean(), global_step)
        common.generate_tensor_summaries('z_stddev', z_next.stddev(),
                                         global_step)
        common.generate_tensor_summaries('prior_mean', prior.mean(),
                                         global_step)
        common.generate_tensor_summaries('prior_stddev', prior.stddev(),
                                         global_step)

      if log_prob_reward_scale == 'auto':
        coef = tf.stop_gradient(tf.exp(actor_net._log_kl_coefficient))  # pylint: disable=protected-access
      else:
        coef = log_prob_reward_scale
      tf.debugging.check_numerics(
          tf.reduce_mean(predictor_kl), 'predictor_kl is inf or nan.')
      tf.debugging.check_numerics(coef, 'coef is inf or nan.')
      new_reward = experience.reward - coef * predictor_kl[:, None]

      experience = experience._replace(reward=new_reward)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    # Save the hyperparameters
    operative_filename = os.path.join(root_dir, 'operative.gin')
    with tf.compat.v1.gfile.Open(operative_filename, 'w') as f:
      f.write(gin.operative_config_str())
      print(gin.operative_config_str())

    global_step_val = global_step.numpy()
    while global_step_val < num_iterations:
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      env_time_acc += time.time() - start_time
      train_start_time = time.time()
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
      train_time_acc += time.time() - train_start_time
      time_acc += time.time() - start_time

      global_step_val = global_step.numpy()

      if global_step_val % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step_val,
                     train_loss.loss)
        steps_per_sec = (global_step_val - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        train_steps_per_sec = (global_step_val - timed_at_step) / train_time_acc
        logging.info('Train: %.3f steps/sec', train_steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='train_steps_per_sec',
            data=train_steps_per_sec,
            step=global_step)
        env_steps_per_sec = (global_step_val - timed_at_step) / env_time_acc
        logging.info('Env: %.3f steps/sec', env_steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='env_steps_per_sec', data=env_steps_per_sec, step=global_step)
        timed_at_step = global_step_val
        time_acc = 0
        train_time_acc = 0
        env_time_acc = 0

      for train_metric in train_metrics + [kl_metric]:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step_val % eval_interval == 0:
        start_time = time.time()
        for name, eval_tf_env, eval_metrics in eval_vec:
          results = metric_utils.eager_compute(
              eval_metrics,
              eval_tf_env,
              eval_policy,
              num_episodes=num_eval_episodes,
              train_step=global_step,
              summary_writer=eval_summary_writer,
              summary_prefix='Metrics-%s' % name,
          )
          if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step_val)
          metric_utils.log_metrics(eval_metrics, prefix=name)
        logging.info('Evaluation: %d min', (time.time() - start_time) / 60)
        for prob_dropout in eval_dropout:
          rpc_utils.eval_dropout_fn(
              eval_tf_env, actor_net, global_step, prob_dropout=prob_dropout)

      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

      if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)


def main(_):
  os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  if 'xm_parameters' in FLAGS and FLAGS.xm_parameters:
    hparams = json.loads(FLAGS.xm_parameters)
    with gin.unlock_config():
      for (key, value) in hparams.items():
        print('Setting: %s = %s' % (key, value))
        gin.bind_parameter(key, value)

  root_dir = FLAGS.root_dir
  train_eval(root_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
