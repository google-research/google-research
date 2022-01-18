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

# Lint as: python3
r"""Train and eval DrQ (SAC) and PSEs + DrQ (SAC)."""
import os

from absl import logging
import gin
import tensorflow.compat.v2 as tf
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import replay_buffer_utils
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from pse.dm_control.agents import pse_drq_agent
from pse.dm_control.utils import dataset_utils
from pse.dm_control.utils import env_utils
from pse.dm_control.utils import model_utils
from pse.dm_control.utils import networks


def img_summary(experience, summary_writer, train_step):
  """Generates image summaries for the augmented images."""
  obs = experience['experience'].observation['pixels']
  if experience['augmented_obs']:
    aug_obs = experience['augmented_obs'][0]['pixels']
    aug_next_obs = experience['augmented_next_obs'][0]['pixels']
    images = tf.stack([
        obs[0, :, :, 0:3],
        aug_obs[:, :, 0:3],
        aug_next_obs[:, :, 0:3],
    ], axis=0)
  else:
    images = tf.expand_dims(obs[0, Ellipsis, 0:3], axis=0)
  with summary_writer.as_default(), \
       common.soft_device_placement(), \
    tf.compat.v2.summary.record_if(True):
    tf.summary.image('Sample crops', images, max_outputs=10, step=train_step)


def contrastive_img_summary(episode_tuple, agent, summary_writer, train_step):
  """Generates image summaries for the augmented images."""
  _, sim_matrix = agent.contrastive_metric_loss(
      episode_tuple, return_representation=True)
  sim_matrix = tf.expand_dims(tf.expand_dims(sim_matrix, axis=0), axis=-1)
  with summary_writer.as_default(), \
       common.soft_device_placement(), \
    tf.compat.v2.summary.record_if(True):
    tf.summary.image('Sim matrix', sim_matrix, step=train_step)


@gin.configurable(module='drq_agent')
def train_eval(
    root_dir,
    # Dataset params
    env_name,
    data_dir=None,
    load_pretrained=False,
    pretrained_model_dir=None,
    img_pad=4,
    frame_shape=(84, 84, 3),
    frame_stack=3,
    num_augmentations=2,  # K and M in DrQ
    # Training params
    contrastive_loss_weight=1.0,
    contrastive_loss_temperature=0.5,
    image_encoder_representation=True,
    initial_collect_steps=1000,
    num_train_steps=3000000,
    actor_fc_layers=(1024, 1024),
    critic_joint_fc_layers=(1024, 1024),
    # Agent params
    batch_size=256,
    actor_learning_rate=1e-3,
    critic_learning_rate=1e-3,
    alpha_learning_rate=1e-3,
    encoder_learning_rate=1e-3,
    actor_update_freq=2,
    gamma=0.99,
    target_update_tau=0.01,
    target_update_period=2,
    reward_scale_factor=1.0,
    # Replay params
    reverb_port=None,
    replay_capacity=100000,
    # Others
    checkpoint_interval=10000,
    policy_save_interval=5000,
    eval_interval=10000,
    summary_interval=250,
    debug_summaries=False,
    eval_episodes_per_run=10,
    summarize_grads_and_vars=False):
  """Trains and evaluates SAC."""
  collect_env = env_utils.load_dm_env_for_training(
      env_name,
      frame_shape,
      frame_stack=frame_stack)
  eval_env = env_utils.load_dm_env_for_eval(
      env_name,
      frame_shape,
      frame_stack=frame_stack)

  logging.info('Data directory: %s', data_dir)
  logging.info('Num train steps: %d', num_train_steps)
  logging.info('Contrastive loss coeff: %.2f', contrastive_loss_weight)
  logging.info(
      'Contrastive loss temperature: %.4f', contrastive_loss_temperature)
  logging.info('load_pretrained: %s', 'yes' if load_pretrained else 'no')
  logging.info('encoder representation: %s',
               'yes' if image_encoder_representation else 'no')

  load_episode_data = (contrastive_loss_weight > 0)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))

  train_step = train_utils.create_train_step()
  image_encoder = networks.ImageEncoder(observation_tensor_spec)

  actor_net = model_utils.Actor(
      observation_tensor_spec,
      action_tensor_spec,
      image_encoder=image_encoder,
      fc_layers=actor_fc_layers,
      image_encoder_representation=image_encoder_representation)

  critic_net = networks.Critic((observation_tensor_spec, action_tensor_spec),
                               image_encoder=image_encoder,
                               joint_fc_layers=critic_joint_fc_layers)
  critic_net_2 = networks.Critic((observation_tensor_spec, action_tensor_spec),
                                 image_encoder=image_encoder,
                                 joint_fc_layers=critic_joint_fc_layers)

  target_image_encoder = networks.ImageEncoder(observation_tensor_spec)
  target_critic_net_1 = networks.Critic(
      (observation_tensor_spec, action_tensor_spec),
      image_encoder=target_image_encoder)
  target_critic_net_2 = networks.Critic(
      (observation_tensor_spec, action_tensor_spec),
      image_encoder=target_image_encoder)

  agent = pse_drq_agent.DrQSacModifiedAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      critic_network_2=critic_net_2,
      target_critic_network=target_critic_net_1,
      target_critic_network_2=target_critic_net_2,
      actor_update_frequency=actor_update_freq,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_learning_rate),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_learning_rate),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=alpha_learning_rate),
      contrastive_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=encoder_learning_rate),
      contrastive_loss_weight=contrastive_loss_weight,
      contrastive_loss_temperature=contrastive_loss_temperature,
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      use_log_alpha_in_alpha_loss=False,
      gradient_clipping=None,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step,
      num_augmentations=num_augmentations)
  agent.initialize()

  # Setup the replay buffer.
  reverb_replay, rb_observer = (
      replay_buffer_utils.get_reverb_buffer_and_observer(
          agent.collect_data_spec,
          sequence_length=2,
          replay_capacity=replay_capacity,
          port=reverb_port))

  # pylint: disable=g-long-lambda
  if num_augmentations == 0:
    image_aug = lambda traj, meta: (dict(
        experience=traj, augmented_obs=[], augmented_next_obs=[]), meta)
  else:
    image_aug = lambda traj, meta: pse_drq_agent.image_aug(
        traj, meta, img_pad, num_augmentations)
  augmented_dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).unbatch().map(
          image_aug, num_parallel_calls=3)
  augmented_iterator = iter(augmented_dataset)

  trajs = augmented_dataset.batch(batch_size).prefetch(50)
  if load_episode_data:
    # Load full episodes and zip them
    episodes = dataset_utils.load_episodes(
        os.path.join(data_dir, 'episodes2'), img_pad)
    episode_iterator = iter(episodes)
    dataset = tf.data.Dataset.zip((trajs, episodes)).prefetch(10)
  else:
    dataset = trajs
  experience_dataset_fn = lambda: dataset

  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  learning_triggers = [
      triggers.PolicySavedModelTrigger(
          saved_model_dir, agent, train_step, interval=policy_save_interval),
      triggers.StepPerSecondLogTrigger(train_step, interval=summary_interval),
  ]

  agent_learner = model_utils.Learner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=experience_dataset_fn,
      triggers=learning_triggers,
      checkpoint_interval=checkpoint_interval,
      summary_interval=summary_interval,
      load_episode_data=load_episode_data,
      use_kwargs_in_agent_train=True,
      # Turn off the initialization of the optimizer variables since, the agent
      # expects different batching for the `training_data_spec` and
      # `train_argspec` which can't be handled in general by the initialization
      # logic in the learner.
      run_optimizer_variable_init=False)

  # If we haven't trained yet make sure we collect some random samples first to
  # fill up the Replay Buffer with some experience.
  train_dir = os.path.join(root_dir, learner.TRAIN_DIR)

  # Code for loading pretrained policy.
  if load_pretrained:
    # Note that num_train_steps is same as the max_train_step we want to
    # load the pretrained policy for our experiments
    pretrained_policy = model_utils.load_pretrained_policy(
        pretrained_model_dir, num_train_steps)
    initial_collect_policy = pretrained_policy

    agent.policy.update_partial(pretrained_policy)
    agent.collect_policy.update_partial(pretrained_policy)
    logging.info('Restored pretrained policy.')
  else:
    initial_collect_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec())
  initial_collect_actor = actor.Actor(
      collect_env,
      initial_collect_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer])
  logging.info('Doing initial collect.')
  initial_collect_actor.run()

  tf_collect_policy = agent.collect_policy
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_collect_policy, use_tf_function=True)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      observers=[rb_observer],
      metrics=actor.collect_metrics(buffer_size=10),
      summary_dir=train_dir,
      summary_interval=summary_interval,
      name='CollectActor')

  # If restarting with train_step > 0, the replay buffer will be empty
  # except for random experience. Populate the buffer with some on-policy
  # experience.
  if load_pretrained or (agent_learner.train_step_numpy > 0):
    for _ in range(batch_size * 50):
      collect_actor.run()

  tf_greedy_policy = agent.policy
  greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
      tf_greedy_policy, use_tf_function=True)

  eval_actor = actor.Actor(
      eval_env,
      greedy_policy,
      train_step,
      episodes_per_run=eval_episodes_per_run,
      metrics=actor.eval_metrics(buffer_size=10),
      summary_dir=os.path.join(root_dir, 'eval'),
      summary_interval=-1,
      name='EvalTrainActor')

  if eval_interval:
    logging.info('Evaluating.')
    img_summary(
        next(augmented_iterator)[0], eval_actor.summary_writer, train_step)
    if load_episode_data:
      contrastive_img_summary(
          next(episode_iterator), agent, eval_actor.summary_writer, train_step)
    eval_actor.run_and_log()

  logging.info('Saving operative gin config file.')
  gin_path = os.path.join(train_dir, 'train_operative_gin_config.txt')
  with tf.io.gfile.GFile(gin_path, mode='w') as f:
    f.write(gin.operative_config_str())

  logging.info('Training Staring at: %r', train_step.numpy())
  while train_step < num_train_steps:
    collect_actor.run()
    agent_learner.run(iterations=1)
    if (not eval_interval) and (train_step % 10000 == 0):
      img_summary(
          next(augmented_iterator)[0], agent_learner.train_summary_writer,
          train_step)
    if eval_interval and agent_learner.train_step_numpy % eval_interval == 0:
      logging.info('Evaluating.')
      img_summary(
          next(augmented_iterator)[0], eval_actor.summary_writer, train_step)
      if load_episode_data:
        contrastive_img_summary(next(episode_iterator), agent,
                                eval_actor.summary_writer, train_step)
      eval_actor.run_and_log()
