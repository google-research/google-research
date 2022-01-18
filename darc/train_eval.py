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

"""Main training script. See README.md for usage instructions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import classifiers
import darc_agent
import darc_envs
import gin
import numpy as np
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

flags.DEFINE_string(
    "root_dir",
    None,
    "Root directory for writing logs/summaries/checkpoints.",
)
flags.DEFINE_multi_string("gin_file", None, "Path to the trainer config files.")
flags.DEFINE_multi_string("gin_bindings", None, "Gin binding to pass through.")

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    environment_name="broken_reacher",
    num_iterations=1000000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    initial_collect_steps=10000,
    real_initial_collect_steps=10000,
    collect_steps_per_iteration=1,
    real_collect_interval=10,
    replay_buffer_capacity=1000000,
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=256,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    classifier_learning_rate=3e-4,
    alpha_learning_rate=3e-4,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=0.99,
    reward_scale_factor=0.1,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=10000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=50000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=True,
    summarize_grads_and_vars=False,
    train_on_real=False,
    delta_r_warmup=0,
    random_seed=0,
    checkpoint_dir=None,
):
  """A simple train and eval for SAC."""
  np.random.seed(random_seed)
  tf.random.set_seed(random_seed)
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, "train")
  eval_dir = os.path.join(root_dir, "eval")

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)

  if environment_name == "broken_reacher":
    get_env_fn = darc_envs.get_broken_reacher_env
  elif environment_name == "half_cheetah_obstacle":
    get_env_fn = darc_envs.get_half_cheetah_direction_env
  elif environment_name.startswith("broken_joint"):
    base_name = environment_name.split("broken_joint_")[1]
    get_env_fn = functools.partial(
        darc_envs.get_broken_joint_env, env_name=base_name)
  elif environment_name.startswith("falling"):
    base_name = environment_name.split("falling_")[1]
    get_env_fn = functools.partial(
        darc_envs.get_falling_env, env_name=base_name)
  else:
    raise NotImplementedError("Unknown environment: %s" % environment_name)

  eval_name_list = ["sim", "real"]
  eval_env_list = [get_env_fn(mode) for mode in eval_name_list]

  eval_metrics_list = []
  for name in eval_name_list:
    eval_metrics_list.append([
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, name="AverageReturn_%s" % name),
    ])

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf_env_real = get_env_fn("real")
    if train_on_real:
      tf_env = get_env_fn("real")
    else:
      tf_env = get_env_fn("sim")

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork),
    )
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer="glorot_uniform",
        last_kernel_initializer="glorot_uniform",
    )

    classifier = classifiers.build_classifier(observation_spec, action_spec)

    tf_agent = darc_agent.DarcAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        classifier=classifier,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        classifier_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=classifier_learning_rate),
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
        train_step_counter=global_step,
    )
    tf_agent.initialize()

    # Make the replay buffer.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity,
    )
    replay_observer = [replay_buffer.add_batch]

    real_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity,
    )
    real_replay_observer = [real_replay_buffer.add_batch]

    sim_train_metrics = [
        tf_metrics.NumberOfEpisodes(name="NumberOfEpisodesSim"),
        tf_metrics.EnvironmentSteps(name="EnvironmentStepsSim"),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size,
            name="AverageReturnSim",
        ),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size,
            name="AverageEpisodeLengthSim",
        ),
    ]
    real_train_metrics = [
        tf_metrics.NumberOfEpisodes(name="NumberOfEpisodesReal"),
        tf_metrics.EnvironmentSteps(name="EnvironmentStepsReal"),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size,
            name="AverageReturnReal",
        ),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size,
            name="AverageEpisodeLengthReal",
        ),
    ]

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())
    collect_policy = tf_agent.collect_policy

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(
            sim_train_metrics + real_train_metrics, "train_metrics"),
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "policy"),
        policy=eval_policy,
        global_step=global_step,
    )
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "replay_buffer"),
        max_to_keep=1,
        replay_buffer=(replay_buffer, real_replay_buffer),
    )

    if checkpoint_dir is not None:
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
      assert checkpoint_path is not None
      train_checkpointer._load_status = train_checkpointer._checkpoint.restore(   # pylint: disable=protected-access
          checkpoint_path)
      train_checkpointer._load_status.initialize_or_restore()  # pylint: disable=protected-access
    else:
      train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    if replay_buffer.num_frames() == 0:
      initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
          tf_env,
          initial_collect_policy,
          observers=replay_observer + sim_train_metrics,
          num_steps=initial_collect_steps,
      )
      real_initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
          tf_env_real,
          initial_collect_policy,
          observers=real_replay_observer + real_train_metrics,
          num_steps=real_initial_collect_steps,
      )

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=replay_observer + sim_train_metrics,
        num_steps=collect_steps_per_iteration,
    )

    real_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env_real,
        collect_policy,
        observers=real_replay_observer + real_train_metrics,
        num_steps=collect_steps_per_iteration,
    )

    config_str = gin.operative_config_str()
    logging.info(config_str)
    with tf.compat.v1.gfile.Open(os.path.join(root_dir, "operative.gin"),
                                 "w") as f:
      f.write(config_str)

    if use_tf_functions:
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      real_initial_collect_driver.run = common.function(
          real_initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      real_collect_driver.run = common.function(real_collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data.
    if replay_buffer.num_frames() == 0:
      logging.info(
          "Initializing replay buffer by collecting experience for %d steps with "
          "a random policy.",
          initial_collect_steps,
      )
      initial_collect_driver.run()
      real_initial_collect_driver.run()

    for eval_name, eval_env, eval_metrics in zip(eval_name_list, eval_env_list,
                                                 eval_metrics_list):
      metric_utils.eager_compute(
          eval_metrics,
          eval_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix="Metrics-%s" % eval_name,
      )
      metric_utils.log_metrics(eval_metrics)

    time_step = None
    real_time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories, unused_arg1):
      return ~trajectories.is_boundary()[0]

    dataset = (
        replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=2).unbatch().filter(
                _filter_invalid_transition).batch(batch_size).prefetch(5))
    real_dataset = (
        real_replay_buffer.as_dataset(
            sample_batch_size=batch_size, num_steps=2).unbatch().filter(
                _filter_invalid_transition).batch(batch_size).prefetch(5))

    # Dataset generates trajectories with shape [Bx2x...]
    iterator = iter(dataset)
    real_iterator = iter(real_dataset)

    def train_step():
      experience, _ = next(iterator)
      real_experience, _ = next(real_iterator)
      return tf_agent.train(experience, real_experience=real_experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    for _ in range(num_iterations):
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )
      assert not policy_state  # We expect policy_state == ().
      if (global_step.numpy() % real_collect_interval == 0 and
          global_step.numpy() >= delta_r_warmup):
        real_time_step, policy_state = real_collect_driver.run(
            time_step=real_time_step,
            policy_state=policy_state,
        )

      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
      time_acc += time.time() - start_time

      global_step_val = global_step.numpy()

      if global_step_val % log_interval == 0:
        logging.info("step = %d, loss = %f", global_step_val, train_loss.loss)
        steps_per_sec = (global_step_val - timed_at_step) / time_acc
        logging.info("%.3f steps/sec", steps_per_sec)
        tf.compat.v2.summary.scalar(
            name="global_steps_per_sec", data=steps_per_sec, step=global_step)
        timed_at_step = global_step_val
        time_acc = 0

      for train_metric in sim_train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=sim_train_metrics[:2])
      for train_metric in real_train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=real_train_metrics[:2])

      if global_step_val % eval_interval == 0:
        for eval_name, eval_env, eval_metrics in zip(eval_name_list,
                                                     eval_env_list,
                                                     eval_metrics_list):
          metric_utils.eager_compute(
              eval_metrics,
              eval_env,
              eval_policy,
              num_episodes=num_eval_episodes,
              train_step=global_step,
              summary_writer=eval_summary_writer,
              summary_prefix="Metrics-%s" % eval_name,
          )
          metric_utils.log_metrics(eval_metrics)

      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

      if global_step_val % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step_val)
    return train_loss


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

  train_eval(FLAGS.root_dir)


if __name__ == "__main__":
  flags.mark_flag_as_required("root_dir")
  app.run(main)
