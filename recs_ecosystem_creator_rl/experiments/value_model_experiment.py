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

"""Run value model optimization experiment."""

import json
import os

from absl import app
from absl import flags
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

# Runner configs.
flags.DEFINE_integer('nsteps', 20, 'Maximum length of a trajectory.')
flags.DEFINE_float('user_gamma', 0.99, 'Discount factor for user utility.')
flags.DEFINE_float('creator_gamma', 0.99,
                   'Discount factor for creator utility.')

# Environment configs.
flags.DEFINE_float('large_recommendation_reward', 2.0,
                   'Large recommendation reward for creators.')
flags.DEFINE_float('small_recommendation_reward', 0.5,
                   'Small recommendation reward for creators.')
flags.DEFINE_string(
    'copy_varied_property', None,
    'If not None, generate two identical creator groups but vary the specified property.'
)

# Training configs.
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
          exp_config):
  """Train and test user_value_model and creator_value_model with random agent."""


  env = environment.create_gym_environment(env_config)
  agent_ = agent.RandomAgent(env_config['slate_size'])
  runner_ = runner.Runner(env, agent_, exp_config['nsteps'])

  train_summary_dir = os.path.join(FLAGS.logdir, 'train/')
  os.makedirs(train_summary_dir)
  log_path = os.path.join(FLAGS.logdir, 'log')

  user_value_model = value_model.UserValueModel(**user_value_model_config)
  creator_value_model = value_model.CreatorValueModel(
      **creator_value_model_config)

  experience_replay = data_utils.ExperienceReplay(exp_config['nsteps'],
                                                  env_config['topic_dim'],
                                                  env_config['num_candidates'],
                                                  exp_config['user_gamma'],
                                                  exp_config['creator_gamma'])

  train_summary_writer = tf.summary.create_file_writer(train_summary_dir)

  # Train, save.
  for epoch in range(1, 1 + exp_config['epochs']):

    # Collect training data.
    num_users = []  # Shape (sum(run_trajectory_length)).
    num_creators = []  # Shape (sum(run_trajectory_length)).
    num_documents = []  # Shape (sum(run_trajectory_length)).
    selected_probs = []  # Shape (sum(run_trajectory_length)).
    policy_probs = []  # Shape (sum(run_trajectory_length), num_candidates).
    for _ in range(exp_config['epoch_runs']):
      (user_dict, creator_dict, _, env_record, probs, _, _) = runner_.run()
      experience_replay.update_experience(
          user_dict, creator_dict, update_actor=False)
      num_users.extend(env_record['user_num'])
      num_creators.extend(env_record['creator_num'])
      num_documents.extend(env_record['document_num'])
      selected_probs.extend(probs['selected_probs'])
      policy_probs.extend(probs['policy_probs'])

    # Update model with training data.
    for batch_data in experience_replay.user_data_generator(
        exp_config['batch_size']):
      user_value_model.train_step(*batch_data)
    for batch_data in experience_replay.creator_data_generator(
        exp_config['batch_size'],
        creator_id_embedding_size=creator_value_model.creator_id_embedding_size
    ):
      creator_value_model.train_step(*batch_data)

    # Write summaries.
    if epoch % exp_config['summary_frequency'] == 0:
      training_utils.save_summary(
          train_summary_writer,
          user_value_model,
          creator_value_model,
          experience_replay,
          num_users,
          num_creators,
          num_documents,
          policy_probs,
          selected_probs,
          epoch=epoch)

      with open(log_path, 'a') as f:
        f.write('Epoch {}, User train Loss: {}, Creator train Loss: {}.'.format(
            epoch, user_value_model.train_loss.result(),
            creator_value_model.train_loss.result()))

    # Reset.
    user_value_model.train_loss.reset_states()
    user_value_model.train_relative_loss.reset_states()
    creator_value_model.train_loss.reset_states()
    creator_value_model.train_relative_loss.reset_states()
    experience_replay.reset()

    if epoch >= exp_config['start_save'] and exp_config[
        'save_frequency'] > 0 and epoch % exp_config['save_frequency'] == 0:
      # Save model.
      user_value_model.save()
      creator_value_model.save()


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
      'start_save': FLAGS.start_save,
      'save_frequency': FLAGS.save_frequency,
      'batch_size': FLAGS.batch_size,
      'summary_frequency': FLAGS.summary_frequency,
  }
  ckpt_save_dir = os.path.join(FLAGS.logdir, 'ckpt/')
  user_ckpt_save_dir = os.path.join(ckpt_save_dir, 'user')
  creator_ckpt_save_dir = os.path.join(ckpt_save_dir, 'creator')
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

  os.makedirs(FLAGS.logdir)
  with open(os.path.join(FLAGS.logdir, 'env_config.json'), 'w') as f:
    f.write(json.dumps(env_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'user_value_model_config.json'),
            'w') as f:
    f.write(json.dumps(user_value_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'creator_value_model_config.json'),
            'w') as f:
    f.write(json.dumps(creator_value_model_config, sort_keys=True, indent=0))
  with open(os.path.join(FLAGS.logdir, 'exp_config.json'), 'w') as f:
    f.write(json.dumps(exp_config, sort_keys=True, indent=0))

  learn(env_config, user_value_model_config, creator_value_model_config,
        exp_config)


if __name__ == '__main__':
  app.run(main)
