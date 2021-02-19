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

"""Run eco-agent experiment."""

import json
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import training_utils
from recs_ecosystem_creator_rl.environment import environment
from recs_ecosystem_creator_rl.recommender import agent
from recs_ecosystem_creator_rl.recommender import data_utils
from recs_ecosystem_creator_rl.recommender import runner
from recs_ecosystem_creator_rl.recommender import value_model

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 7e-4, 'Learning rate.')
# User value model configs.
flags.DEFINE_string('user_rnn_type', 'LSTM', 'User recurrent cell.')
flags.DEFINE_string('user_hidden_sizes', '32,32,16',
                    'Sizes of hidden layers to embed user history.')

# Creator value model configs.
flags.DEFINE_string('creator_rnn_type', 'LSTM', 'Creator recurrent cell.')
flags.DEFINE_string('creator_hidden_sizes', '32,32,16',
                    'Sizes of hidden layers to embed creator history.')
flags.DEFINE_integer('creator_id_embedding_size', 32,
                     'Size of creator id embedding.')

# Actor model configs.
flags.DEFINE_string('actor_hidden_sizes', '32,32,32',
                    'Sizes of hidden layers in actor model.')
flags.DEFINE_integer('actor_weight_size', 16,
                     'Size of weights for softmax in actor model.')
flags.DEFINE_float('actor_learning_rate', 7e-4, 'Learning rate in actor model.')
flags.DEFINE_float('actor_entropy_coeff', 0,
                   'Entropy coefficient in loss function in actor model.')
flags.DEFINE_float('social_reward_coeff', 0.0,
                   'Coefficient of social reward in actor model optimization.')
flags.DEFINE_float(
    'loss_denom_decay', 0.0,
    'Momentum for moving average of label weights normalization.')

# Environment configs.
flags.DEFINE_float('large_recommendation_reward', 2.0,
                   'Large recommendation reward for creators.')
flags.DEFINE_float('small_recommendation_reward', 0.5,
                   'Small recommendation reward for creators.')
flags.DEFINE_string(
    'copy_varied_property', None,
    'If not None, generate two identical creator groups but vary the specified property.'
)

# Runner configs.
flags.DEFINE_integer('nsteps', 20, 'Maximum length of a trajectory.')
flags.DEFINE_float('user_gamma', 0.99, 'Discount factor for user utility.')
flags.DEFINE_float('creator_gamma', 0.99,
                   'Discount factor for creator utility.')

# Training configs.
flags.DEFINE_float('random_user_accumulated_reward', 42.4,
                   'Average user accumulated reward from random agent.')
flags.DEFINE_float('random_creator_accumulated_reward', 3.9,
                   'Average creator accumulated reward from random agent.')
flags.DEFINE_string('logdir', '', 'Log directory.')
flags.DEFINE_integer('epochs', 3000,
                     'The number of epochs to run training for.')
flags.DEFINE_integer('start_save', 300,
                     'The number of epochs to run before saving models.')
flags.DEFINE_integer('save_frequency', 100, 'Frequency of saving.')
flags.DEFINE_integer('summary_frequency', 50, 'Frequency of writing summaries.')
flags.DEFINE_integer('epoch_runs', 10, 'The number of runs to collect data.')
flags.DEFINE_integer('epoch_trains', 1, 'The number of trains per epoch.')
flags.DEFINE_integer('batch_size', 32,
                     'The number of trajectories per training batch.')


def learn(env_config, user_value_model_config, creator_value_model_config,
          actor_model_config, exp_config):
  """Train and test user_value_model and creator_value_model with random agent."""

  # Random agent normalization.
  random_user_accumulated_reward = FLAGS.random_user_accumulated_reward
  random_creator_accumulated_reward = FLAGS.random_creator_accumulated_reward

  env = environment.create_gym_environment(env_config)
  user_value_model = value_model.UserValueModel(**user_value_model_config)
  creator_value_model = value_model.CreatorValueModel(
      **creator_value_model_config)
  actor_model = agent.PolicyGradientAgent(
      user_model=user_value_model,
      creator_model=creator_value_model,
      **actor_model_config)

  runner_ = runner.Runner(env, actor_model, exp_config['nsteps'])

  experience_replay = data_utils.ExperienceReplay(exp_config['nsteps'],
                                                  env_config['topic_dim'],
                                                  env_config['num_candidates'],
                                                  exp_config['user_gamma'],
                                                  exp_config['creator_gamma'])

  train_summary_dir = os.path.join(FLAGS.logdir, 'train/')
  os.makedirs(train_summary_dir)
  train_summary_writer = tf.summary.create_file_writer(train_summary_dir)

  # Train, save.
  for epoch in range(exp_config['epochs']):

    num_users = []  # Shape (sum(run_trajectory_length)).
    num_creators = []  # Shape (sum(run_trajectory_length)).
    num_documents = []  # Shape (sum(run_trajectory_length)).
    selected_probs = []  # Shape (sum(run_trajectory_length)).
    policy_probs = []  # Shape (sum(run_trajectory_length), num_candidates).
    # Collect training data.
    for _ in range(exp_config['epoch_runs']):
      (user_dict, creator_dict, preprocessed_user_candidates, _, probs, _,
       _) = runner_.run()
      experience_replay.update_experience(
          user_dict,
          creator_dict,
          preprocessed_user_candidates,
          update_actor=True)
      num_users.append(runner_.env.num_users)
      num_creators.append(runner_.env.num_creators)
      num_documents.append(runner_.env.num_documents)
      selected_probs.extend(probs['selected_probs'])
      policy_probs.extend(probs['policy_probs'])

    # Update model with training data.
    for _ in range(exp_config['epoch_trains']):
      for (inputs, label, user_utility, social_reward,
           _) in experience_replay.actor_data_generator(
               creator_value_model, batch_size=exp_config['batch_size']):
        actor_model.train_step(
            inputs, label, user_utility / random_user_accumulated_reward,
            social_reward / random_creator_accumulated_reward)

      for batch_data in experience_replay.user_data_generator(
          exp_config['batch_size']):
        user_value_model.train_step(*batch_data)
      for batch_data in experience_replay.creator_data_generator(
          exp_config['batch_size'],
          creator_id_embedding_size=creator_value_model
          .creator_id_embedding_size):
        creator_value_model.train_step(*batch_data)

    sum_user_normalized_accumulated_reward = np.sum(
        list(experience_replay.user_accumulated_reward.values())
    ) / experience_replay.num_runs / random_user_accumulated_reward

    sum_creator_normalized_accumulated_reward = np.sum(
        list(experience_replay.creator_accumulated_reward.values())
    ) / experience_replay.num_runs / random_creator_accumulated_reward

    overall_scaled_accumulated_reward = (
        (1 - actor_model_config['social_reward_coeff']) *
        sum_user_normalized_accumulated_reward +
        actor_model_config['social_reward_coeff'] *
        sum_creator_normalized_accumulated_reward)
    # Write summary statistics for tensorboard.
    if epoch % exp_config['summary_frequency'] == 0:
      ## Value model and environment summaries.
      training_utils.save_summary(train_summary_writer, user_value_model,
                                  creator_value_model, experience_replay,
                                  num_users, num_creators, num_documents,
                                  policy_probs, selected_probs,
                                  overall_scaled_accumulated_reward, epoch)
      ## Actor model summaries.
      with train_summary_writer.as_default():
        tf.summary.scalar(
            'actor_loss', actor_model.train_loss.result(), step=epoch)

        social_rewards = np.array(
            experience_replay.actor_creator_uplift_utilities)
        tf.summary.scalar('social_rewards', np.mean(social_rewards), step=epoch)
        actor_label_weights = (
            (1 - actor_model_config['social_reward_coeff']) *
            np.array(experience_replay.actor_user_utilities) /
            random_user_accumulated_reward +
            actor_model_config['social_reward_coeff'] * social_rewards /
            random_creator_accumulated_reward)
        tf.summary.scalar(
            'actor_label_weights', np.mean(actor_label_weights), step=epoch)

    # Reset.
    user_value_model.train_loss.reset_states()
    user_value_model.train_relative_loss.reset_states()
    creator_value_model.train_loss.reset_states()
    creator_value_model.train_relative_loss.reset_states()
    actor_model.train_loss.reset_states()
    actor_model.train_utility_loss.reset_states()
    actor_model.train_entropy_loss.reset_states()
    experience_replay.reset()

    if epoch >= exp_config['start_save'] and exp_config[
        'save_frequency'] > 0 and epoch % exp_config['save_frequency'] == 0:
      # Save model.
      user_value_model.save()
      creator_value_model.save()
      actor_model.save()


def main(unused_argv):
  num_users, num_creators = 50, 10
  half_creators = num_creators // 2
  env_config = {
      # Hyperparameters for environment.
      'resample_documents':
          True,
      'topic_dim':
          10,
      'choice_features':
          dict(),
      'sampling_space':
          'unit ball',
      'num_candidates':
          10,
      # Hyperparameters for users.
      'num_users':
          num_users,
      'user_quality_sensitivity': [0.3] * num_users,
      'user_topic_influence': [0.2] * num_users,
      'observation_noise_std': [0.05] * num_users,
      'user_initial_satisfaction': [10.0] * num_users,
      'user_satisfaction_decay': [1.0] * num_users,
      'user_viability_threshold': [0.0] * num_users,
      'user_model_seed':
          list(range(num_users)),
      'slate_size':
          1,
      # Hyperparameters for creators and documents.
      'num_creators':
          num_creators,
      'creator_initial_satisfaction': [5.0] * num_creators,
      'creator_viability_threshold': [0.0] * num_creators,
      'creator_no_recommendation_penalty': [1.0] * num_creators,
      'creator_new_document_margin': [20.0] * num_creators,
      'creator_recommendation_reward':
          [FLAGS.small_recommendation_reward] * half_creators +
          [FLAGS.large_recommendation_reward] * half_creators,
      'creator_user_click_reward': [0.1] * num_creators,
      'creator_satisfaction_decay': [1.0] * num_creators,
      'doc_quality_std': [0.1] * num_creators,
      'doc_quality_mean_bound': [0.8] * num_creators,
      'creator_initial_num_docs': [20] * num_creators,
      'creator_is_saturation': [False] * num_creators,
      'creator_topic_influence': [0.2] * num_creators,
      'copy_varied_property':
          FLAGS.copy_varied_property,
  }

  exp_config = {
      'nsteps': FLAGS.nsteps,
      'user_gamma': FLAGS.user_gamma,
      'creator_gamma': FLAGS.creator_gamma,
      'epochs': FLAGS.epochs,
      'epoch_runs': FLAGS.epoch_runs,
      'epoch_trains': FLAGS.epoch_trains,
      'start_save': FLAGS.start_save,
      'save_frequency': FLAGS.save_frequency,
      'batch_size': FLAGS.batch_size,
      'summary_frequency': FLAGS.summary_frequency,
  }
  ckpt_save_dir = os.path.join(FLAGS.logdir, 'ckpt/')
  user_ckpt_save_dir = os.path.join(ckpt_save_dir, 'user')
  creator_ckpt_save_dir = os.path.join(ckpt_save_dir, 'creator')
  actor_ckpt_save_dir = os.path.join(ckpt_save_dir, 'actor')
  os.makedirs(ckpt_save_dir)

  user_value_model_config = {
      'document_feature_size': env_config['topic_dim'],
      'creator_feature_size': None,
      'user_feature_size': None,
      'input_reward': False,
      'regularization_coeff': None,
      'rnn_type': FLAGS.user_rnn_type,
      'hidden_sizes': [
          int(size) for size in FLAGS.user_hidden_sizes.split(',')
      ],
      'lr': FLAGS.learning_rate,
      'model_path': user_ckpt_save_dir,
  }

  creator_value_model_config = {
      'document_feature_size': env_config['topic_dim'],
      'creator_feature_size': 1,
      'regularization_coeff': None,
      'rnn_type': FLAGS.creator_rnn_type,
      'hidden_sizes': [
          int(size) for size in FLAGS.creator_hidden_sizes.split(',')
      ],
      'lr': FLAGS.learning_rate,
      'model_path': creator_ckpt_save_dir,
      'num_creators': env_config['num_creators'],
      'creator_id_embedding_size': FLAGS.creator_id_embedding_size,
      'trajectory_length': exp_config['nsteps']
  }

  actor_model_config = {
      'slate_size': env_config['slate_size'],
      'user_embedding_size': user_value_model_config['hidden_sizes'][0],
      'document_embedding_size': env_config['topic_dim'],
      'creator_embedding_size': creator_value_model_config['hidden_sizes'][0],
      'hidden_sizes': [
          int(size) for size in FLAGS.actor_hidden_sizes.split(',')
      ],
      'weight_size': FLAGS.actor_weight_size,
      'lr': FLAGS.learning_rate,
      'entropy_coeff': FLAGS.actor_entropy_coeff,
      'regularization_coeff': 1e-5,
      'loss_denom_decay': FLAGS.loss_denom_decay,
      'social_reward_coeff': FLAGS.social_reward_coeff,
      'model_path': actor_ckpt_save_dir,
  }
  os.makedirs(FLAGS.logdir)
  with open(os.path.join(FLAGS.logdir, 'env_config.json'), 'w') as f:
    f.write(json.dumps(env_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'user_value_model_config.json'),
            'w') as f:
    f.write(json.dumps(user_value_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'creator_value_model_config.json'),
            'w') as f:
    f.write(json.dumps(creator_value_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'actor_model_config.json'), 'w') as f:
    f.write(json.dumps(actor_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'exp_config.json'), 'w') as f:
    f.write(json.dumps(exp_config, sort_keys=True, indent=0))
  learn(env_config, user_value_model_config, creator_value_model_config,
        actor_model_config, exp_config)


if __name__ == '__main__':
  app.run(main)
