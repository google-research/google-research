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

r"""Script for training the RCE agent.

Example usage:
  python train_eval.py --root_dir=~/c_learning/sawyer_drawer_open \
    --gin_bindings='train_eval.env_name="sawyer_drawer_open"'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import rce_agent
import rce_envs
from six.moves import range
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


@gin.configurable
def bce_loss(y_true, y_pred, label_smoothing=0):
  loss_fn = tf.keras.losses.BinaryCrossentropy(
      label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.NONE)
  return loss_fn(y_true[:, None], y_pred[:, None])


@gin.configurable
class ClassifierCriticNetwork(critic_network.CriticNetwork):
  """Creates a critic network."""

  def __init__(self,
               input_tensor_spec,
               observation_fc_layer_params=None,
               action_fc_layer_params=None,
               joint_fc_layer_params=None,
               kernel_initializer=None,
               last_kernel_initializer=None,
               name='ClassifierCriticNetwork'):
    super(ClassifierCriticNetwork, self).__init__(
        input_tensor_spec,
        observation_fc_layer_params=observation_fc_layer_params,
        action_fc_layer_params=action_fc_layer_params,
        joint_fc_layer_params=joint_fc_layer_params,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=last_kernel_initializer,
        name=name,
    )

    last_layers = [
        tf.keras.layers.Dense(
            1,
            activation=tf.math.sigmoid,
            kernel_initializer=last_kernel_initializer,
            name='value')
    ]
    self._joint_layers = self._joint_layers[:-1] + last_layers


@gin.configurable
def train_eval(
    root_dir,
    env_name='HalfCheetah-v2',
    # The SAC paper reported:
    # Hopper and Cartpole results up to 1000000 iters,
    # Humanoid results up to 10000000 iters,
    # Other mujoco tasks up to 3000000 iters.
    num_iterations=3000000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Params for collect
    # Follow https://github.com/haarnoja/sac/blob/master/examples/variants.py
    # HalfCheetah and Ant take 10000 initial collection steps.
    # Other mujoco tasks take 1000.
    # Different choices roughly keep the initial episodes about the same.
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
    gamma=0.99,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=200000,
    # policy_checkpoint_interval=50000,
    rb_checkpoint_interval=50000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    random_seed=0,
    actor_min_std=1e-3,  # Added for numerical stability.
    n_step=10):
  """A simple train and eval for SAC."""
  np.random.seed(random_seed)
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf_env = rce_envs.load_env(env_name)
    eval_tf_env = rce_envs.load_env(env_name)
    if env_name == 'sawyer_lift':
      eval_tf_env.MODE = 'eval'

    expert_obs = rce_envs.get_data(tf_env.envs[0], env_name=env_name)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    proj_net = functools.partial(
        tanh_normal_projection_network.TanhNormalProjectionNetwork,
        std_transform=lambda t: actor_min_std + tf.nn.softplus(t))
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=proj_net)
    critic_net = ClassifierCriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    tf_agent = rce_agent.RceAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=bce_loss,
        gamma=gamma,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        n_step=n_step)
    tf_agent.initialize()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes,
                                       batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes,
                                              batch_size=tf_env.batch_size)
    ]
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
    ]

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        max_to_keep=None)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    replay_observer = [replay_buffer.add_batch]

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + train_metrics,
        num_steps=collect_steps_per_iteration)

    if use_tf_functions:
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    # Save the hyperparameters
    operative_filename = os.path.join(root_dir, 'operative.gin')
    with tf.compat.v1.gfile.Open(operative_filename, 'w') as f:
      f.write(gin.operative_config_str())
      print(gin.operative_config_str())

    if replay_buffer.num_frames() == 0:
      # Collect initial replay data.
      logging.info(
          'Initializing replay buffer by collecting experience for %d steps '
          'with a random policy.', initial_collect_steps)
      initial_collect_driver.run()

    results = metric_utils.eager_compute(
        eval_metrics,
        eval_tf_env,
        eval_policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
    )
    del results
    metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0
    env_time_acc = 0

    def _filter_invalid_transition(trajectories, unused_arg1):
      return ~trajectories.is_boundary()[0]

    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=2 if n_step is None else n_step)
    dataset = dataset.unbatch()
    dataset = dataset.filter(_filter_invalid_transition)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(5)
    iterator = iter(dataset)

    ### Expert dataset
    expert_dataset = tf.data.Dataset.from_tensors(expert_obs)
    expert_dataset = expert_dataset.unbatch()
    expert_dataset = expert_dataset.repeat().shuffle(int(1e6))

    expert_dataset = expert_dataset.batch(batch_size, drop_remainder=True)
    expert_iterator = iter(expert_dataset)

    def train_step():
      experience, _ = next(iterator)
      expert_experience = next(expert_iterator)
      return tf_agent.train(experience=(experience, expert_experience))

    if use_tf_functions:
      train_step = common.function(train_step)

    global_step_val = global_step.numpy()
    while global_step_val < num_iterations:
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      env_time_acc += time.time() - start_time
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
      time_acc += time.time() - start_time

      global_step_val = global_step.numpy()

      if global_step_val % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step_val,
                     train_loss.loss)
        steps_per_sec = (global_step_val - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        env_steps_per_sec = (global_step_val - timed_at_step) / env_time_acc
        logging.info('Env: %.3f steps/sec', env_steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='env_steps_per_sec', data=env_steps_per_sec, step=global_step)

        timed_at_step = global_step_val
        time_acc = 0
        env_time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step_val % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        metric_utils.log_metrics(eval_metrics)

      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      # if global_step_val % policy_checkpoint_interval == 0:
      #   policy_checkpointer.save(global_step=global_step_val)
#
      if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)
    return train_loss


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  root_dir = FLAGS.root_dir
  train_eval(root_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
