# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python2, python3
# pylint: disable=missing-docstring

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import os
import pickle
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

from edward2_autoreparam import experiments as algs
from edward2_autoreparam import models
from edward2_autoreparam import util


ALG_NAMES = ['HMC-CP', 'HMC-NCP', 'iHMC', 'c-VIP-HMC', 'd-VIP-HMC']

flags.DEFINE_integer(
    'num_leapfrog_steps', default=4, help='Number of leapfrog steps')

flags.DEFINE_integer(
    'num_optimization_steps', default=2000,
    help='Number of steps to optimize the ELBO.')

flags.DEFINE_integer(
    'num_mc_samples',
    default=64,
    help='Number of Monte Carlo samples to use in the ELBO.')

flags.DEFINE_list(
    'num_leapfrog_steps_list',
    default=[],
    help='Number of leapfrog steps (list)')

flags.DEFINE_string(
    'results_dir', default='/tmp/results', help='Directory to write results.')

flags.DEFINE_string(
    'method', default='vip',
    help='Methods to be used: VIP or baseline')  # can be baseline or vip

flags.DEFINE_string('model', default='radon_stddvs', help='Model name')

flags.DEFINE_string('dataset', default='na', help='Dataset')

flags.DEFINE_bool(
    'use_iaf_posterior',
    default=False,
    help=
    'Use an Inverse Autoregressive FLow as a variational posterior, '
    'and disable HMC sampling for cVIP.'
)

flags.DEFINE_float(
    'tau',
    default=1.0,
    help=
    'Temperature of the sigmoid parameterization for '
    'VIP centering coefficients (default: 1.).'
)

flags.DEFINE_integer('num_samples', default=5000, help='')
flags.DEFINE_integer('burnin', default=800, help='')
flags.DEFINE_integer('num_adaptation_steps', default=600, help='')


FLAGS = flags.FLAGS


def main(_):

  util.print('Loading model {} with dataset {}.'.format(
      FLAGS.model, FLAGS.dataset))

  if FLAGS.model == 'radon':
    model_config = models.get_radon(state_code=FLAGS.dataset)
  elif FLAGS.model == 'radon_stddvs':
    model_config = models.get_radon_model_stddvs(state_code=FLAGS.dataset)
  elif FLAGS.model == '8schools':
    model_config = models.get_eight_schools()
  elif FLAGS.model == 'german_credit_gammascale':
    model_config = models.get_german_credit_gammascale()
  elif FLAGS.model == 'german_credit_lognormalcentered':
    model_config = models.get_german_credit_lognormalcentered()
  else:
    raise Exception('unknown model {}'.format(FLAGS.model))

  description = FLAGS.model + '_{}'.format(FLAGS.dataset)

  experiments_dir = os.path.join(
      FLAGS.results_dir,
      'num_leapfrog_steps={}'.format(FLAGS.num_leapfrog_steps))
  if not tf.gfile.Exists(experiments_dir):
    tf.gfile.MakeDirs(experiments_dir)

  if FLAGS.method == 'baseline':
    run_baseline(
        description,
        model_config=model_config,
        experiments_dir=experiments_dir,
        num_samples=FLAGS.num_samples,
        burnin=FLAGS.burnin,
        num_adaptation_steps=FLAGS.num_adaptation_steps,
        num_optimization_steps=FLAGS.num_optimization_steps,
        tau=FLAGS.tau,
        num_leapfrog_steps=FLAGS.num_leapfrog_steps,
        description=description)
  elif FLAGS.method == 'vip':
    run_vip(
        description,
        model_config=model_config,
        experiments_dir=experiments_dir,
        use_iaf_posterior=FLAGS.use_iaf_posterior,
        num_samples=FLAGS.num_samples,
        burnin=FLAGS.burnin,
        num_adaptation_steps=FLAGS.num_adaptation_steps,
        num_optimization_steps=FLAGS.num_optimization_steps,
        num_mc_samples=FLAGS.num_mc_samples,
        tau=FLAGS.tau,
        num_leapfrog_steps=FLAGS.num_leapfrog_steps,
        description=description)
  else:
    raise Exception('No such method')


def run_baseline(experiment_name,
                 model_config,
                 experiments_dir,
                 description,
                 num_samples=2000,
                 burnin=1000,
                 num_adaptation_steps=500,
                 num_leapfrog_steps=4,
                 num_optimization_steps=2000,
                 tau=1.):

  folder_path = os.path.join(experiments_dir, experiment_name)
  if not tf.gfile.Exists(folder_path):
    tf.gfile.MakeDirs(folder_path)

  spec = 'Model name: {}\n{}\n\n'.format(experiment_name, description)
  spec += '{} samples\n'.format(num_samples)
  spec += '{} burnin\n'.format(burnin)
  spec += '{} adaptation steps\n'.format(num_adaptation_steps)
  spec += '{} leapfrog steps\n'.format(num_leapfrog_steps)
  spec += '{} optimization steps\n'.format(num_optimization_steps)

  if tau != 1.0:
    spec += 'tau = {}\n'.format(tau)

  util.print('\nRunning HMC-CP...')
  util.print(spec)
  results = algs.run_centered_hmc(
      model_config=model_config,
      num_samples=num_samples,
      burnin=burnin,
      num_leapfrog_steps=num_leapfrog_steps,
      num_adaptation_steps=num_adaptation_steps,
      num_optimization_steps=num_optimization_steps)

  util.print('  ess / leapfrogs:     {}'.format(
      min([np.amin(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  avg ess / leapfrogs: {}'.format(
      np.mean([np.mean(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  acceptance rate: {}'.format(results['acceptance_rate']))

  util.print('  VI ran for:  {} minutes'.format(results['vi_time'] / 60.))
  util.print('  HMC ran for: {} minutes'.format(
      results['sampling_time'] / 60.))

  step_size_cp = results['step_size']
  results['algorithm'] = 'HMC-CP'
  results['num_leapfrog_steps'] = num_leapfrog_steps

  save_path = os.path.join(folder_path, results['algorithm'])
  pickle_results(results, save_path)

  util.print('\nRunning HMC-NCP...')
  results = algs.run_noncentered_hmc(
      model_config=model_config,
      num_samples=num_samples,
      burnin=burnin,
      num_leapfrog_steps=num_leapfrog_steps,
      num_adaptation_steps=num_adaptation_steps,
      num_optimization_steps=num_optimization_steps)
  util.print('  ess / leapfrogs:     {}'.format(
      min([np.amin(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  avg ess / leapfrogs: {}'.format(
      np.mean([np.mean(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  acceptance rate: {}'.format(results['acceptance_rate']))
  util.print('  VI ran for:  {} minutes'.format(results['vi_time'] / 60.))
  util.print('  HMC ran for: {} minutes'.format(
      results['sampling_time'] / 60.))

  step_size_ncp = results['step_size']
  results['algorithm'] = 'HMC-NCP'
  results['num_leapfrog_steps'] = num_leapfrog_steps

  save_path = os.path.join(folder_path, results['algorithm'])
  pickle_results(results, save_path)

  util.print('\nRunning iHMC...')
  results = algs.run_interleaved_hmc(
      model_config=model_config,
      num_samples=num_samples,
      burnin=burnin,
      num_leapfrog_steps=num_leapfrog_steps,
      step_size_cp=step_size_cp,
      step_size_ncp=step_size_ncp)
  util.print('  ess / leapfrogs:     {}'.format(
      min([np.amin(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  avg ess / leapfrogs: {}'.format(
      np.mean([np.mean(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  acceptance rate: {}'.format(results['acceptance_rate']))
  util.print('  HMC ran for: {} minutes'.format(
      results['sampling_time'] / 60.))

  results['algorithm'] = 'iHMC'
  results['num_leapfrog_steps'] = num_leapfrog_steps

  save_path = os.path.join(folder_path, results['algorithm'])
  pickle_results(results, save_path)


def run_vip(experiment_name,
            model_config,
            experiments_dir,
            num_samples=2000,
            burnin=1000,
            num_adaptation_steps=500,
            num_leapfrog_steps=4,
            num_optimization_steps=2000,
            num_mc_samples=32,
            tau=1.,
            use_iaf_posterior=False,
            description=''):

  folder_path = os.path.join(experiments_dir, experiment_name)
  if not tf.gfile.Exists(folder_path):
    tf.gfile.MakeDirs(folder_path)

  spec = 'Model name: {}\n{}\n\n'.format(experiment_name, description)
  spec += '{} samples\n'.format(num_samples)
  spec += '{} burnin\n'.format(burnin)
  spec += '{} adaptation steps\n'.format(num_adaptation_steps)
  spec += '{} leapfrog steps\n'.format(num_leapfrog_steps)
  spec += '{} optimization steps\n'.format(num_optimization_steps)
  spec += '{} mc samples\n'.format(num_mc_samples)
  if tau != 1.0:
    spec += 'tau = {}\n'.format(tau)

  util.print('\nRunning c-VIP-HMC...')
  results = algs.run_vip_hmc_continuous(
      model_config=model_config,
      num_samples=num_samples,
      burnin=burnin,
      use_iaf_posterior=use_iaf_posterior,
      num_leapfrog_steps=num_leapfrog_steps,
      num_adaptation_steps=num_adaptation_steps,
      num_optimization_steps=num_optimization_steps,
      num_mc_samples=num_mc_samples,
      experiments_dir=experiments_dir,
      tau=tau,
      description=description + 'c-VIP')

  if not use_iaf_posterior:
    util.print('  ess / leapfrogs:     {}'.format(
        min([np.amin(ess) for ess in results['ess']]) / num_leapfrog_steps))
    util.print('  avg ess / leapfrogs: {}'.format(
        np.mean([np.mean(ess) for ess in results['ess']]) / num_leapfrog_steps))
    util.print('  acceptance rate: {}'.format(results['acceptance_rate']))
    util.print('  VI ran for:  {} minutes'.format(results['vi_time'] / 60.))
    util.print('  HMC ran for: {} minutes'.format(
        results['sampling_time'] / 60.))

  results['algorithm'] = 'c-VIP-HMC'
  results['num_leapfrog_steps'] = num_leapfrog_steps

  save_path = os.path.join(folder_path, results['algorithm'])
  pickle_results(results, save_path)

  parameterisation = results['parameterisation']
  discrete_parameterisation = collections.OrderedDict(
      [(key, (np.array(parameterisation[key]) >= 0.5).astype(np.float32))
       for key in parameterisation.keys()])
  vip_to_centered = model_config.make_to_centered(**discrete_parameterisation)

  model_config_dvip = model_config._replace(to_centered=vip_to_centered)

  util.print('\nRunning d-VIP-HMC...')
  results = algs.run_vip_hmc_discrete(
      model_config=model_config_dvip,
      parameterisation=discrete_parameterisation,
      num_samples=num_samples,
      burnin=burnin,
      num_leapfrog_steps=num_leapfrog_steps,
      num_adaptation_steps=num_adaptation_steps,
      num_optimization_steps=num_optimization_steps)

  util.print('  ess / leapfrogs:     {}'.format(
      min([np.amin(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  avg ess / leapfrogs: {}'.format(
      np.mean([np.mean(ess) for ess in results['ess']]) / num_leapfrog_steps))
  util.print('  acceptance rate: {}'.format(results['acceptance_rate']))
  util.print('  HMC ran for: {} minutes'.format(
      results['sampling_time'] / 60.))

  results['algorithm'] = 'd-VIP-HMC'
  results['num_leapfrog_steps'] = num_leapfrog_steps

  save_path = os.path.join(folder_path, results['algorithm'])
  pickle_results(results, save_path)


def gen_id():
  return np.random.randint(10000, 99999)


def pickle_results(results, save_path):

  if not tf.gfile.Exists(save_path):
    tf.gfile.MakeDirs(save_path)

  id_num = gen_id()
  results_path = os.path.join(save_path, 'results_' + str(id_num) + '.pkl')

  with tf.gfile.Open(results_path, 'wb') as f:
    pickle.dump(results, f)


if __name__ == '__main__':
  tf.app.run()
