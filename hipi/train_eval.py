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

"""Script for training the agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import relabelling_replay_buffer
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
import utils

flags.DEFINE_string("root_dir", None,
                    "Root directory for writing logs/summaries/checkpoints.")
flags.DEFINE_multi_string("gin_file", None, "Path to the trainer config files.")
flags.DEFINE_multi_string("gin_bindings", None, "Gin binding to pass through.")

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    env_name="HalfCheetah-v2",
    num_iterations=1000000,
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Params for collect
    initial_collect_steps=10000,
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
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=100000,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=500000000,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    relabel_type=None,
    num_future_states=4,
    max_episode_steps=100,
    random_seed=0,
    eval_task_list=None,
    constant_task=None,  # Whether to train on a single task
    clip_critic=None,
):
  """A simple train and eval for SAC."""
  np.random.seed(random_seed)
  if relabel_type == "none":
    relabel_type = None
  assert relabel_type in [None, "future", "last", "soft", "random"]
  if constant_task:
    assert relabel_type is None
  if eval_task_list is None:
    eval_task_list = []
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, "train")
  eval_dir = os.path.join(root_dir, "eval")

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      utils.AverageSuccessMetric(
          max_episode_steps=max_episode_steps, buffer_size=num_eval_episodes),
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    tf_env, task_distribution = utils.get_env(
        env_name, constant_task=constant_task)
    eval_tf_env, _ = utils.get_env(
        env_name, max_episode_steps, constant_task=constant_task)

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=utils.normal_projection_net,
    )
    if isinstance(clip_critic, float):
      output_activation_fn = lambda x: clip_critic * tf.sigmoid(x)
    elif isinstance(clip_critic, tuple):
      assert len(clip_critic) == 2
      min_val, max_val = clip_critic
      output_activation_fn = (lambda x:  # pylint: disable=g-long-lambda
                              (max_val - min_val) * tf.sigmoid(x) + min_val)
    else:
      output_activation_fn = None
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        output_activation_fn=output_activation_fn,
    )

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
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
        train_step_counter=global_step,
    )
    tf_agent.initialize()

    # Make the replay buffer.
    replay_buffer = relabelling_replay_buffer.GoalRelabellingReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        batch_size=1,
        max_length=replay_buffer_capacity,
        task_distribution=task_distribution,
        actor=actor_net,
        critic=critic_net,
        gamma=gamma,
        relabel_type=relabel_type,
        sample_batch_size=batch_size,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        num_future_states=num_future_states,
    )

    env_steps = tf_metrics.EnvironmentSteps(prefix="Train")
    train_metrics = [
        tf_metrics.NumberOfEpisodes(prefix="Train"),
        env_steps,
        utils.AverageSuccessMetric(
            prefix="Train",
            max_episode_steps=max_episode_steps,
            buffer_size=num_eval_episodes,
        ),
        tf_metrics.AverageReturnMetric(
            prefix="Train",
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size,
        ),
        tf_metrics.AverageEpisodeLengthMetric(
            prefix="Train",
            buffer_size=num_eval_episodes,
            batch_size=tf_env.batch_size,
        ),
    ]

    eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        tf_env.time_step_spec(), tf_env.action_spec())

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, "train_metrics"),
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, "policy"),
        policy=eval_policy,
        global_step=global_step,
    )

    train_checkpointer.initialize_or_restore()

    data_collector = utils.DataCollector(
        tf_env,
        tf_agent.collect_policy,
        replay_buffer,
        max_episode_steps=max_episode_steps,
        observers=train_metrics,
    )

    if use_tf_functions:
      tf_agent.train = common.function(tf_agent.train)
    else:
      tf.config.experimental_run_functions_eagerly(True)

    # Save the config string as late as possible to catch
    # as many object instantiations as possible.
    config_str = gin.operative_config_str()
    logging.info(config_str)
    with tf.compat.v1.gfile.Open(os.path.join(root_dir, "operative.gin"),
                                 "w") as f:
      f.write(config_str)

    # Collect initial replay data.
    logging.info(
        "Initializing replay buffer by collecting experience for %d steps with "
        "a random policy.",
        initial_collect_steps,
    )
    for _ in range(initial_collect_steps):
      data_collector.step(initial_collect_policy)
    data_collector.reset()
    logging.info("Replay buffer initial size: %d", replay_buffer.num_frames())

    logging.info("Computing initial eval metrics")
    for task in [None] + eval_task_list:
      with utils.FixedTask(eval_tf_env, task):
        prefix = "Metrics" if task is None else "Metrics-%s" % str(task)
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix=prefix,
        )
        metric_utils.log_metrics(eval_metrics)

    time_acc = 0
    env_time_acc = 0
    train_time_acc = 0
    env_steps_before = env_steps.result().numpy()

    if use_tf_functions:
      tf_agent.train = common.function(tf_agent.train)

    logging.info("Starting training")
    for _ in range(num_iterations):
      start_time = time.time()
      data_collector.step()
      env_time_acc += time.time() - start_time
      train_time_start = time.time()
      for _ in range(train_steps_per_iteration):
        experience = replay_buffer.get_batch()
        train_loss = tf_agent.train(experience)
        total_loss = train_loss.loss
      train_time_acc += time.time() - train_time_start
      time_acc += time.time() - start_time

      if global_step.numpy() % log_interval == 0:
        logging.info("step = %d, loss = %f", global_step.numpy(), total_loss)

        combined_steps_per_sec = (env_steps.result().numpy() -
                                  env_steps_before) / time_acc
        train_steps_per_sec = (env_steps.result().numpy() -
                               env_steps_before) / train_time_acc
        env_steps_per_sec = (env_steps.result().numpy() -
                             env_steps_before) / env_time_acc
        logging.info(
            "%.3f combined steps / sec: %.3f env steps/sec, %.3f train steps/sec",
            combined_steps_per_sec,
            env_steps_per_sec,
            train_steps_per_sec,
        )
        tf.compat.v2.summary.scalar(
            name="combined_steps_per_sec",
            data=combined_steps_per_sec,
            step=env_steps.result(),
        )
        tf.compat.v2.summary.scalar(
            name="env_steps_per_sec",
            data=env_steps_per_sec,
            step=env_steps.result(),
        )
        tf.compat.v2.summary.scalar(
            name="train_steps_per_sec",
            data=train_steps_per_sec,
            step=env_steps.result(),
        )
        time_acc = 0
        env_time_acc = 0
        train_time_acc = 0
        env_steps_before = env_steps.result().numpy()

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

      if global_step.numpy() % eval_interval == 0:

        for task in [None] + eval_task_list:
          with utils.FixedTask(eval_tf_env, task):
            prefix = "Metrics" if task is None else "Metrics-%s" % str(task)
            logging.info(prefix)
            metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix=prefix,
            )
            metric_utils.log_metrics(eval_metrics)

      global_step_val = global_step.numpy()
      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)

    return train_loss


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)
  root_dir = FLAGS.root_dir
  train_eval(root_dir)


if __name__ == "__main__":
  flags.mark_flag_as_required("root_dir")
  app.run(main)
