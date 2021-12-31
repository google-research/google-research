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

"""Run tabular behavioral cloning with representation learning."""
import os

from absl import app
from absl import flags
from dice_rl.data.dataset import Dataset
from dice_rl.data.tf_agents_onpolicy_dataset import TFAgentsOnpolicyDataset
import dice_rl.environments.gridworld.low_rank as low_rank
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.taxi as taxi
import dice_rl.environments.gridworld.tree as tree
from dice_rl.environments.infinite_frozenlake import InfiniteFrozenLake
from dice_rl.estimators import estimator as estimator_lib
import dice_rl.utils.common as common_utils
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from rl_repr.contrastive_fourier.tabular_bc_energy import TabularBCEnergy
from rl_repr.contrastive_fourier.tabular_bc_sgd import TabularBCSGD
from rl_repr.contrastive_fourier.tabular_bc_svd import TabularBCSVD

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'lowrank_tree', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_integer('num_trajectory', 500,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('num_expert_trajectory', 5,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 3,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', -0.125, 'How close to target policy.')
flags.DEFINE_float('alpha_expert', 1.125, 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '../tests/testdata/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', './output/', 'Directory to save result to.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')

flags.DEFINE_string('embed_learner', None, 'One of [sgd, svd, energy] or None.')
flags.DEFINE_integer('embed_dim', 64, 'Embedder dimension.')
flags.DEFINE_integer('fourier_dim', None, 'Dimension of fourier features.')
flags.DEFINE_float('embed_learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_integer('embed_pretraining_steps', 10_000,
                     'Number of pretraining training steps.')
flags.DEFINE_integer('num_steps', 100_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 1000,
                     'Number of pretraining training steps.')
flags.DEFINE_bool('finetune', False,
                  'Whether to finetune embedder during policy learning.')
flags.DEFINE_bool('latent_policy', True, 'Whether to learn a latent policy.')

flags.DEFINE_integer('batch_size', 1024, 'Batch size.')


def get_onpolicy_dataset(env_name, tabular_obs, policy_fn, policy_info_spec):
  """Gets target policy."""
  if env_name == 'taxi':
    env = taxi.Taxi(tabular_obs=tabular_obs)
  elif env_name == 'grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs)
  elif env_name == 'lowrank_tree':
    env = tree.Tree(branching=2, depth=3, duplicate=10)
  elif env_name == 'frozenlake':
    env = InfiniteFrozenLake()
  elif env_name == 'low_rank':
    env = low_rank.LowRank()
  else:
    raise ValueError('Unknown environment: %s.' % env_name)

  tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
  tf_policy = common_utils.TFAgentsWrappedPolicy(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      policy_fn,
      policy_info_spec,
      emit_log_probability=True)

  return TFAgentsOnpolicyDataset(tf_env, tf_policy)


def main(argv):
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  num_expert_trajectory = FLAGS.num_expert_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  alpha_expert = FLAGS.alpha_expert
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  gamma = FLAGS.gamma
  assert 0 <= gamma < 1.
  embed_dim = FLAGS.embed_dim
  fourier_dim = FLAGS.fourier_dim
  embed_learning_rate = FLAGS.embed_learning_rate
  learning_rate = FLAGS.learning_rate
  finetune = FLAGS.finetune
  latent_policy = FLAGS.latent_policy
  embed_learner = FLAGS.embed_learner
  num_steps = FLAGS.num_steps
  embed_pretraining_steps = FLAGS.embed_pretraining_steps
  batch_size = FLAGS.batch_size

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset.')
  dataset = Dataset.load(directory)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  estimate = estimator_lib.get_fullbatch_average(dataset, gamma=gamma)
  print('data per step avg', estimate)

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha_expert,
                    SEED=seed,
                    NUM_TRAJ=num_expert_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(load_dir, hparam_str)
  print('Loading expert dataset.')
  expert_dataset = Dataset.load(directory)
  print('num loaded expert steps', expert_dataset.num_steps)
  print('num loaded total expert steps', expert_dataset.num_total_steps)
  print('num loaded expert episodes', expert_dataset.num_episodes)
  print('num loaded total expert episodes', expert_dataset.num_total_episodes)
  expert_estimate = estimator_lib.get_fullbatch_average(
      expert_dataset, gamma=gamma)
  print('expert data per step avg', expert_estimate)

  hparam_dict = {
      'env_name': env_name,
      'alpha_expert': alpha_expert,
      'seed': seed,
      'num_trajectory': num_trajectory,
      'num_expert_trajectory': num_expert_trajectory,
      'max_trajectory_length': max_trajectory_length,
      'embed_learner': embed_learner,
      'embed_dim': embed_dim,
      'fourier_dim': fourier_dim,
      'embed_learning_rate': embed_learning_rate,
      'learning_rate': learning_rate,
      'latent_policy': latent_policy,
      'finetune': finetune,
  }
  hparam_str = ','.join(
      ['%s=%s' % (k, str(hparam_dict[k])) for k in sorted(hparam_dict.keys())])
  summary_writer = tf.summary.create_file_writer(
      os.path.join(save_dir, hparam_str, 'train'))

  if embed_learner == 'sgd' or not embed_learner:
    algo = TabularBCSGD(
        dataset.spec,
        gamma=gamma,
        embed_dim=embed_dim,
        embed_learning_rate=embed_learning_rate,
        learning_rate=learning_rate,
        finetune=finetune,
        latent_policy=latent_policy)
  elif embed_learner == 'svd':
    algo = TabularBCSVD(
        dataset.spec,
        gamma=gamma,
        embed_dim=embed_dim,
        learning_rate=learning_rate)
  elif embed_learner == 'energy':
    algo = TabularBCEnergy(
        dataset.spec,
        gamma=gamma,
        embed_dim=embed_dim,
        fourier_dim=fourier_dim,
        embed_learning_rate=embed_learning_rate,
        learning_rate=learning_rate)
  else:
    raise ValueError('embed learner %s not supported' % embed_learner)

  if embed_learner == 'svd':
    embed_dict = algo.solve(dataset)
    with summary_writer.as_default():
      for k, v in embed_dict.items():
        tf.summary.scalar(f'embed/{k}', v, step=0)
        print('embed', k, v)
  else:
    algo.prepare_datasets(dataset, expert_dataset)
    if embed_learner is not None:
      for step in range(embed_pretraining_steps):
        batch = dataset.get_step(batch_size, num_steps=2)
        embed_dict = algo.train_embed(batch)
        if step % FLAGS.eval_interval == 0:
          with summary_writer.as_default():
            for k, v in embed_dict.items():
              tf.summary.scalar(f'embed/{k}', v, step=step)
              print('embed', step, k, v)

  for step in range(num_steps):
    batch = expert_dataset.get_step(batch_size, num_steps=2)
    info_dict = algo.train_step(batch)
    if step % FLAGS.eval_interval == 0:
      with summary_writer.as_default():
        for k, v in info_dict.items():
          tf.summary.scalar(f'bc/{k}', v, step=step)
          print('bc', k, v)

      policy_fn, policy_info_spec = algo.get_policy()
      onpolicy_data = get_onpolicy_dataset(env_name, tabular_obs, policy_fn,
                                           policy_info_spec)
      onpolicy_episodes, _ = onpolicy_data.get_episode(
          100, truncate_episode_at=max_trajectory_length)
      with summary_writer.as_default():
        tf.print('eval/reward', np.mean(onpolicy_episodes.reward))
        tf.summary.scalar(
            'eval/reward', np.mean(onpolicy_episodes.reward), step=step)


if __name__ == '__main__':
  app.run(main)
