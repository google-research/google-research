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
"""Code for running the main experiment."""

import os.path as osp
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from meta_reward_learning.textworld import common_flags
from meta_reward_learning.textworld.lib import helpers
from meta_reward_learning.textworld.lib.agent import MetaRLAgent
from meta_reward_learning.textworld.lib.agent import RLAgent
from meta_reward_learning.textworld.lib.model import create_checkpoint_manager
from meta_reward_learning.textworld.lib.replay_buffer import BufferScorer
from meta_reward_learning.textworld.lib.replay_buffer import SampleGenerator
from tensorflow.contrib import summary as contrib_summary

FLAGS = flags.FLAGS
flags.adopt_module_key_flags(common_flags)




def train(agent, replay_buffer, dev_data, objective='mapo'):
  """Training Loop."""
  sgd_steps = 0
  train_env_dict = replay_buffer.env_dict
  train_sample_gen = SampleGenerator(
      replay_buffer,
      agent,
      objective=objective,
      explore=FLAGS.explore,
      n_samples=FLAGS.n_replay_samples,
      use_top_k_samples=FLAGS.use_top_k_samples,
      min_replay_weight=FLAGS.min_replay_weight)
  train_sample_generator = train_sample_gen.generate_samples(
      batch_size=len(train_env_dict), debug=FLAGS.is_debug)
  if FLAGS.meta_learn:
    dev_replay_buffer = dev_data
    dev_env_dict = dev_replay_buffer.env_dict
    dev_sample_gen = SampleGenerator(
        dev_replay_buffer,
        agent,
        objective=objective,
        explore=FLAGS.dev_explore)
    dev_sample_generator = dev_sample_gen.generate_samples(
        batch_size=len(dev_env_dict), debug=FLAGS.is_debug)
  else:
    dev_env_dict = dev_data

  ckpt_dir = osp.join(FLAGS.train_dir, 'model')
  if (tf.train.latest_checkpoint(ckpt_dir) is
      None) and FLAGS.pretrained_ckpt_dir:
    pretrained_ckpt_dir = osp.join(FLAGS.pretrained_ckpt_dir, 'best_model')
    # Store weights before loading the checkpoint
    if FLAGS.pretrained_load_data_only and FLAGS.meta_learn:
      pi_weights = agent.pi.get_weights()
    create_checkpoint_manager(
        agent,
        pretrained_ckpt_dir,
        restore=True,
        include_optimizer=False,
        meta_learn=False)
    # Reset the global step to 0
    tf.assign(agent.global_step, 0)
    if FLAGS.pretrained_load_data_only and FLAGS.meta_learn:
      dev_trajs = agent.sample_trajs(dev_env_dict.values(), greedy=True)
      dev_replay_buffer.save_trajs(dev_trajs)
      agent.pi.set_weights(pi_weights)
      tf.logging.info('Collected data using the pretrained checkpoint')

  ckpt_manager = create_checkpoint_manager(
      agent,
      ckpt_dir,
      restore=True,
      include_optimizer=True,
      meta_learn=FLAGS.meta_learn)
  best_ckpt_dir = osp.join(FLAGS.train_dir, 'best_model')
  best_ckpt_manager = create_checkpoint_manager(
      agent, best_ckpt_dir, restore=False, include_optimizer=False)
  # Log summaries for the accuracy results
  summary_writer = contrib_summary.create_file_writer(
      osp.join(FLAGS.train_dir, 'tb_log'), flush_millis=5000)
  max_val_acc = helpers.eval_agent(agent, dev_env_dict)

  with summary_writer.as_default(), \
    contrib_summary.always_record_summaries():
    while agent.global_step.numpy() < FLAGS.num_steps:
      if sgd_steps % FLAGS.save_every_n == 0:
        ckpt_manager.save()
        train_acc = helpers.eval_agent(agent, train_env_dict)
        val_acc = helpers.eval_agent(agent, dev_env_dict)
        contrib_summary.scalar('train_acc', train_acc)
        contrib_summary.scalar('validation_acc', val_acc)
        if val_acc > max_val_acc:
          max_val_acc = val_acc
          tf.logging.info('Best validation accuracy {}'.format(max_val_acc))
          best_ckpt_manager.save()

      # Sample environments and trajectories
      samples, contexts = next(train_sample_generator)
      if FLAGS.meta_learn:
        dev_samples, dev_contexts = next(dev_sample_generator)
        agent.update(samples, contexts, dev_samples, dev_contexts)
      else:
        # Update the policy
        agent.update(samples, contexts)
      # Update the random noise
      agent.update_eps(agent.global_step.numpy(), FLAGS.num_steps)
      sgd_steps += 1


def run_experiment():
  """Code for creating the agent and run training/evaluation."""
  agent_args = dict(
      log_summaries=FLAGS.log_summaries,
      eps=FLAGS.eps,
      entropy_reg_coeff=FLAGS.entropy_reg_coeff,
      units=FLAGS.units,
      learning_rate=FLAGS.learning_rate,
      debug=FLAGS.is_debug,
      seed=FLAGS.seed,
      gamma=FLAGS.gamma,
      use_critic=False,
      max_grad_norm=FLAGS.max_grad_norm,
      objective='mapo')
  if FLAGS.meta_learn:
    agent = MetaRLAgent(
        meta_lr=FLAGS.meta_lr, score_fn=FLAGS.score_fn, **agent_args)
  else:
    agent = RLAgent(**agent_args)

  if FLAGS.use_buffer_scorer:
    num_features = len(common_flags.PAIR_FEATURE_KEYS)
    score_weights = np.zeros(
        num_features * (num_features + 1), dtype=np.float32)
    w1, w2 = [
        getattr(FLAGS, 'score_{}'.format(x))
        for x in common_flags.PAIRWISE_WEIGHTS
    ]
    for counter, key in enumerate(common_flags.PAIR_FEATURE_KEYS):
      # Assign the weights to the first `num_features`
      score_weights[counter] = getattr(FLAGS, 'score_{}'.format(key))
      for counter2, key2 in enumerate(common_flags.PAIR_FEATURE_KEYS):
        index = (counter + 1) * num_features + counter2
        interactions = helpers.cross_product(key, key2)
        features = [getattr(FLAGS, 'score_{}'.format(i)) for i in interactions]
        # Pairwise interaction features are assumed to be a linear combination
        # of unary interaction features
        score_weights[index] = features[0] * features[-1] * w1 + w2 * features[
            1] * features[2]

    buffer_scorer = BufferScorer(score_weights)
  else:
    buffer_scorer = None

  if not FLAGS.eval_only:  # Training
    train_replay_buffer = helpers.create_replay_buffer(
        FLAGS.train_file,
        grid_size=FLAGS.grid_size,
        n_plants=FLAGS.n_train_plants,
        num_envs=FLAGS.n_train_envs,
        seed=FLAGS.seed,
        use_gold_trajs=FLAGS.use_gold_trajs,
        buffer_scorer=buffer_scorer)
    if not FLAGS.meta_learn:
      dev_env_dict = helpers.create_dataset(
          FLAGS.dev_file,
          grid_size=FLAGS.grid_size,
          n_plants=FLAGS.n_dev_plants,
          seed=FLAGS.seed,
          num_envs=FLAGS.n_dev_envs,
          return_trajs=False)
      train(agent, train_replay_buffer, dev_env_dict)
    else:
      dev_replay_buffer = helpers.create_replay_buffer(
          FLAGS.dev_file,
          grid_size=FLAGS.grid_size,
          n_plants=FLAGS.n_dev_plants,
          seed=FLAGS.seed,
          use_gold_trajs=FLAGS.use_dev_gold_trajs,
          save_trajs=not (FLAGS.pretrained_ckpt_dir and FLAGS.dev_explore),
          num_envs=FLAGS.n_dev_envs)
      train(agent, train_replay_buffer, dev_replay_buffer)
    best_ckpt_dir = osp.join(FLAGS.train_dir, 'best_model')
  else:
    best_ckpt_dir = osp.join(FLAGS.eval_dir, 'best_model')

  # Run the agent evaluation at the end
  test_env_dict = helpers.create_dataset(
      FLAGS.test_file,
      grid_size=FLAGS.grid_size,
      n_plants=FLAGS.n_test_plants,
      return_trajs=False,
      num_envs=None,
      seed=FLAGS.seed)
  create_checkpoint_manager(
      agent, best_ckpt_dir, restore=True, include_optimizer=False)
  test_accuracy = helpers.eval_agent(agent, test_env_dict)
  tf.logging.info('Final Test accuracy {}'.format(test_accuracy))


def main(argv):
  _ = argv
  tf.logging.set_verbosity(tf.logging.INFO)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)
  run_experiment()


if __name__ == '__main__':
  app.run(main)
