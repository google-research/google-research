# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Run Dichotomy of Control on FrozenLake."""

import pickle
import getpass
from absl import app
from absl import flags

import gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import os

from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.estimators import estimator as estimator_lib
from dichotomy_of_control import utils
from dichotomy_of_control.envs.frozenlake_wrapper import FrozenLakeWrapper
from dichotomy_of_control.models.stochastic_decision_transformer import StochasticDecisionTransformer
from dichotomy_of_control.scripts.stochastic_decision_transformer_training import StochasticDecisionTransformerTrainer
from dichotomy_of_control.scripts.stochastic_decision_transformer_training import StochasticSequenceDataLoader
from dichotomy_of_control.scripts.stochastic_decision_transformer_evaluation import evaluate_stochastic_decision_transformer_episode

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'FrozenLake-v1', 'Environment name.')
flags.DEFINE_integer('env_seed', 0, 'Env random seed.')
flags.DEFINE_integer('num_trajectory', 100,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', -1., 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/tmp/dichotomy_of_control/',
                    'Directory to load dataset from.')

flags.DEFINE_float('prior_weight', 1., 'Weight of prior.')
flags.DEFINE_float('energy_weight', 1., 'Weight of energy loss.')

flags.DEFINE_integer('seed', 1, 'Training random seed.')
flags.DEFINE_integer('context_len', 20, 'Context length.')

flags.DEFINE_integer('max_iters', 10, 'Training iterations.')
flags.DEFINE_integer('num_steps_per_iter', 1_000, 'Steps per iteration.')


def main(argv):
  env_name = FLAGS.env_name
  env_seed = FLAGS.env_seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=env_seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset.')
  dataset = Dataset.load(directory)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  estimate = estimator_lib.get_fullbatch_average(dataset, by_steps=False)
  print('data per episode avg', estimate)

  hparam_dict = {
      'env_name': env_name,
      'alpha': alpha,
      'env_seed': env_seed,
      'context_len': FLAGS.context_len,
      'seed': FLAGS.seed,
      'num_trajectory': num_trajectory,
      'max_trajectory_length': max_trajectory_length,
      'prior_weight': FLAGS.prior_weight,
      'energy_weight': FLAGS.energy_weight,
  }

  settings = {
      'context_len': FLAGS.context_len,
      'batch_size': 64,
      'embed_dim': 128,
      'n_layer': 3,
      'n_head': 1,
      'n_positions': 1024,
      'activation_function': 'relu',
      'dropout': 0.1,
      'learning_rate': 1e-4,
      'weight_decay': 1e-4,
      'warmup_steps': 1000,
      'num_eval_episodes': 10,
      'max_iters': FLAGS.max_iters,
      'num_steps_per_iter': FLAGS.num_steps_per_iter,
  }

  max_ep_len = max_trajectory_length
  if 'FrozenLake' in FLAGS.env_name:
    dataset_spec = dataset.spec
    observation_spec = dataset_spec.observation
    action_spec = dataset_spec.action
    state_dim = observation_spec.maximum + 1
    act_dim = action_spec.maximum + 1
    env = FrozenLakeWrapper(state_dim)
    env.seed(FLAGS.env_seed)
    env_targets = [1]
    scale = 1.
  else:
    raise NotImplementedError
  context_len = settings['context_len']
  batch_size = settings['batch_size']
  num_eval_episodes = settings['num_eval_episodes']

  trajectories = utils.convert_to_np_dataset(
      dataset,
      tabular_obs=FLAGS.tabular_obs,
      tabular_act=True)
  print('observations', trajectories[0]['observations'].shape)
  print('actions', trajectories[0]['actions'].shape)
  print('rewards', trajectories[0]['rewards'].shape)
  print('dones', trajectories[0]['dones'].shape)

  # separate out num_timesteps, returns for printing statistics
  # and calculate state means and state stds
  num_timesteps = 0
  states = []
  returns = []
  for traj in trajectories:
    num_timesteps += len(traj['rewards'])
    returns.append(traj['rewards'].sum())
    states.append(traj['observations'])

  # used for input normalization
  states = np.concatenate(states, axis=0)
  state_mean = np.mean(states, axis=0)
  state_std = np.std(states, axis=0) + 1e-6
  print('state_mean', state_mean)
  print('state_std', state_std)

  print('=' * 50)
  print(f'Starting new experiment: {env_name}')
  print(f'{len(trajectories)} trajectories, {num_timesteps} timesteps found')
  print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
  print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
  print('Return histogram')
  print('[Returns range]: number of trajectories')
  hist, bin_edges = np.histogram(returns, bins='auto')
  for v, lo, hi in zip(hist, bin_edges[:-1], bin_edges[1:]):
    print(f'[{lo} -- {hi}]: {v}')
  print('=' * 50, flush=True)

  def eval_episodes(target_ret):

    def fn(model):
      returns, lengths = [], []
      for _ in range(num_eval_episodes):
        ret, length = evaluate_stochastic_decision_transformer_episode(
            env,
            state_dim,
            act_dim,
            model,
            max_ep_len,
            scale,
            state_mean,
            state_std,
            target_ret / scale,
        )
        returns.append(ret)
        lengths.append(length)
      return {
          f'target_{target_ret}_return_mean': np.mean(returns),
          f'target_{target_ret}_return_std': np.std(returns),
          f'target_{target_ret}_length_mean': np.mean(lengths),
          f'target_{target_ret}_length_std': np.std(lengths),
      }

    return fn

  eval_fns = [eval_episodes(tar) for tar in env_targets]

  # create model
  model = StochasticDecisionTransformer(
      state_dim=state_dim,
      act_dim=act_dim,
      hidden_size=settings['embed_dim'],
      context_len=context_len,
      max_ep_len=max_ep_len,
      n_layer=settings['n_layer'],
      n_head=settings['n_head'],
      n_inner=4 * settings['embed_dim'],
      activation_function=settings['activation_function'],
      n_positions=settings['n_positions'],
      resid_pdrop=settings['dropout'],
      attn_pdrop=settings['dropout'],
  )

  # create optimizer
  warmup_steps = settings['warmup_steps']
  lr = tf.keras.optimizers.schedules.PolynomialDecay(
      settings['learning_rate'] / warmup_steps,
      warmup_steps,
      end_learning_rate=settings['learning_rate'],
      power=1.0,
  )
  optimizer = tfa.optimizers.AdamW(
      learning_rate=lr,
      weight_decay=settings['weight_decay'],
      clipnorm=0.25,
  )

  # create data loader
  data_loader = StochasticSequenceDataLoader(trajectories, context_len, max_ep_len, batch_size,
                             state_mean, state_std, scale)

  # create trainer
  trainer = StochasticDecisionTransformerTrainer(
      model=model,
      optimizer=optimizer,
      data_loader=data_loader,
      loss_fn=tf.keras.losses.MeanSquaredError(),
      eval_fns=eval_fns,
      prior_weight=FLAGS.prior_weight,
      energy_weight=FLAGS.energy_weight,
  )

  # run training
  for iter in range(settings['max_iters']):
    outputs = trainer.train_iteration(
        num_steps=settings['num_steps_per_iter'],
        iter_num=iter + 1,
        print_logs=True)


if __name__ == '__main__':
  app.run(main)
