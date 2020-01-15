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

r"""Train and Eval Safety-Constrained SAC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import os.path as osp
import time

from absl import flags
from absl import logging
from .algorithm import agents
import gin
import tensorflow.compat.v1 as tf
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from .utils import misc


FLAGS = flags.FLAGS

# Loss value that is considered too high and training will be terminated.
MAX_LOSS = 1e9

# How many steps does the loss have to be diverged for (too high, inf, nan)
# after the training terminates. This should prevent termination on short loss
# spikes.
TERMINATE_AFTER_DIVERGED_LOSS_STEPS = 100


@gin.configurable
def train(
    root_dir,
    load_root_dir=None,
    env_load_fn=None,
    env_name=None,
    num_parallel_environments=1,  # pylint: disable=unused-argument
    agent_class=None,
    initial_collect_random=True,  # pylint: disable=unused-argument
    initial_collect_driver_class=None,
    collect_driver_class=None,
    num_global_steps=1000000,
    train_steps_per_iteration=1,
    train_metrics=None,
    # Safety Critic training args
    train_sc_steps=10,
    train_sc_interval=300,
    online_critic=False,
    # Params for eval
    run_eval=False,
    num_eval_episodes=30,
    eval_interval=1000,
    eval_metrics_callback=None,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    keep_rb_checkpoint=False,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    early_termination_fn=None,
    env_metric_factories=None):  # pylint: disable=unused-argument
  """A simple train and eval for SC-SAC."""

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  train_metrics = train_metrics or []

  if run_eval:
    eval_dir = os.path.join(root_dir, 'eval')
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ] + [tf_py_metric.TFPyMetric(m) for m in train_metrics]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf_env = env_load_fn(env_name)
    if not isinstance(tf_env, tf_py_environment.TFPyEnvironment):
      tf_env = tf_py_environment.TFPyEnvironment(tf_env)

    if run_eval:
      eval_py_env = env_load_fn(env_name)
      eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    print('obs spec:', observation_spec)
    print('action spec:', action_spec)

    if online_critic:
      resample_metric = tf_py_metric.TfPyMetric(
          py_metrics.CounterMetric('unsafe_ac_samples'))
      tf_agent = agent_class(
          time_step_spec,
          action_spec,
          train_step_counter=global_step,
          resample_metric=resample_metric)
    else:
      tf_agent = agent_class(
          time_step_spec, action_spec, train_step_counter=global_step)

    tf_agent.initialize()

    # Make the replay buffer.
    collect_data_spec = tf_agent.collect_data_spec

    logging.info('Allocating replay buffer ...')
    # Add to replay buffer and other agent specific observers.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collect_data_spec, max_length=1000000)
    logging.info('RB capacity: %i', replay_buffer.capacity)
    logging.info('ReplayBuffer Collect data spec: %s', collect_data_spec)

    agent_observers = [replay_buffer.add_batch]
    if online_critic:
      online_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          collect_data_spec, max_length=10000)

      online_rb_ckpt_dir = os.path.join(train_dir, 'online_replay_buffer')
      online_rb_checkpointer = common.Checkpointer(
          ckpt_dir=online_rb_ckpt_dir,
          max_to_keep=1,
          replay_buffer=online_replay_buffer)

      clear_rb = common.function(online_replay_buffer.clear)
      agent_observers.append(online_replay_buffer.add_batch)

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
    ] + [tf_py_metric.TFPyMetric(m) for m in train_metrics]

    if not online_critic:
      eval_policy = tf_agent.policy
    else:
      eval_policy = tf_agent._safe_policy  # pylint: disable=protected-access

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec, action_spec)
    if not online_critic:
      collect_policy = tf_agent.collect_policy
    else:
      collect_policy = tf_agent._safe_policy  # pylint: disable=protected-access

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    safety_critic_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'safety_critic'),
        safety_critic=tf_agent._safety_critic_network,  # pylint: disable=protected-access
        global_step=global_step)
    rb_ckpt_dir = os.path.join(train_dir, 'replay_buffer')
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=rb_ckpt_dir, max_to_keep=1, replay_buffer=replay_buffer)

    if load_root_dir:
      load_root_dir = os.path.expanduser(load_root_dir)
      load_train_dir = os.path.join(load_root_dir, 'train')
      misc.load_pi_ckpt(load_train_dir, tf_agent)  # loads tf_agent

    if load_root_dir is None:
      train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()
    safety_critic_checkpointer.initialize_or_restore()

    collect_driver = collect_driver_class(
        tf_env, collect_policy, observers=agent_observers + train_metrics)

    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    if not rb_checkpointer.checkpoint_exists:
      logging.info('Performing initial collection ...')
      common.function(
          initial_collect_driver_class(
              tf_env,
              initial_collect_policy,
              observers=agent_observers + train_metrics).run)()
      last_id = replay_buffer._get_last_id()  # pylint: disable=protected-access
      logging.info('Data saved after initial collection: %d steps', last_id)
      tf.print(
          replay_buffer._get_rows_for_id(last_id),  # pylint: disable=protected-access
          output_stream=logging.info)

    if run_eval:
      results = metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix='Metrics',
      )
      if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
      metric_utils.log_metrics(eval_metrics)
      if FLAGS.viz_pm:
        eval_fig_dir = osp.join(eval_dir, 'figs')
        if not tf.io.gfile.isdir(eval_fig_dir):
          tf.io.gfile.makedirs(eval_fig_dir)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, num_steps=2).prefetch(3)
    iterator = iter(dataset)
    if online_critic:
      online_dataset = online_replay_buffer.as_dataset(
          num_parallel_calls=3, num_steps=2).prefetch(3)
      online_iterator = iter(online_dataset)

      @common.function
      def critic_train_step():
        """Builds critic training step."""
        experience, buf_info = next(online_iterator)
        if env_name in [
            'IndianWell', 'IndianWell2', 'IndianWell3', 'DrunkSpider',
            'DrunkSpiderShort'
        ]:
          safe_rew = experience.observation['task_agn_rew']
        else:
          safe_rew = agents.process_replay_buffer(
              online_replay_buffer, as_tensor=True)
          safe_rew = tf.gather(safe_rew, tf.squeeze(buf_info.ids), axis=1)
        ret = tf_agent.train_sc(experience, safe_rew)
        clear_rb()
        return ret

    @common.function
    def train_step():
      experience, _ = next(iterator)
      ret = tf_agent.train(experience)
      return ret

    if not early_termination_fn:
      early_termination_fn = lambda: False

    loss_diverged = False
    # How many consecutive steps was loss diverged for.
    loss_divergence_counter = 0
    mean_train_loss = tf.keras.metrics.Mean(name='mean_train_loss')
    if online_critic:
      mean_resample_ac = tf.keras.metrics.Mean(name='mean_unsafe_ac_samples')
      resample_metric.reset()

    while (global_step.numpy() <= num_global_steps and
           not early_termination_fn()):
      # Collect and train.
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      if online_critic:
        mean_resample_ac(resample_metric.result())
        resample_metric.reset()
        if time_step.is_last():
          resample_ac_freq = mean_resample_ac.result()
          mean_resample_ac.reset_states()
          tf.compat.v2.summary.scalar(
              name='unsafe_ac_samples', data=resample_ac_freq, step=global_step)

      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
        mean_train_loss(train_loss.loss)

      if online_critic:
        if global_step.numpy() % train_sc_interval == 0:
          for _ in range(train_sc_steps):
            sc_loss, lambda_loss = critic_train_step()  # pylint: disable=unused-variable

      total_loss = mean_train_loss.result()
      mean_train_loss.reset_states()
      # Check for exploding losses.
      if (math.isnan(total_loss) or math.isinf(total_loss) or
          total_loss > MAX_LOSS):
        loss_divergence_counter += 1
        if loss_divergence_counter > TERMINATE_AFTER_DIVERGED_LOSS_STEPS:
          loss_diverged = True
          break
      else:
        loss_divergence_counter = 0

      time_acc += time.time() - start_time

      if global_step.numpy() % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step.numpy(), total_loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      global_step_val = global_step.numpy()
      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)
        safety_critic_checkpointer.save(global_step=global_step_val)

      if global_step_val % rb_checkpoint_interval == 0:
        if online_critic:
          online_rb_checkpointer.save(global_step=global_step_val)
        rb_checkpointer.save(global_step=global_step_val)

      if run_eval and global_step.numpy() % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)
        if FLAGS.viz_pm:
          savepath = 'step{}.png'.format(global_step_val)
          savepath = osp.join(eval_fig_dir, savepath)
          misc.record_episode_vis_summary(eval_tf_env, eval_policy, savepath)

  if not keep_rb_checkpoint:
    misc.cleanup_checkpoints(rb_ckpt_dir)

  if loss_diverged:
    # Raise an error at the very end after the cleanup.
    raise ValueError('Loss diverged to {} at step {}, terminating.'.format(
        total_loss, global_step.numpy()))

  return total_loss
