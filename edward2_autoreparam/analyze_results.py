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
"""Generate text results from pickled results."""
# pylint: disable=missing-docstring,broad-except

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle

from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf


alg_names = ['HMC-CP', 'HMC-NCP', 'iHMC', 'c-VIP-HMC', 'd-VIP-HMC']

flags.DEFINE_string(
    'results_dir',
    default='/tmp/results',
    help='Directory to read results from.')
flags.DEFINE_string(
    'model_and_dataset',
    default='8schools_na',
    help='Name of the subdirectory to read results from.')

FLAGS = flags.FLAGS


def get_min_ess(esss):
  return min([np.amin(ess) for ess in esss])


def get_all_results(results_dir):

  runs = dict([(alg_name, []) for alg_name in alg_names])

  for alg in alg_names:
    alg_dir = os.path.join(results_dir, alg)
    for file_path in tf.gfile.ListDirectory(alg_dir):
      full_path = os.path.join(alg_dir, file_path)
      with tf.gfile.Open(full_path, 'rb') as f:
        try:
          results = pickle.load(f)
          runs[alg].append(results)
        except Exception:
          print('Could not unpickle {}'.format(full_path))
  return runs


def get_best_esss_results(model_name, results_dir):
  best_esss = collections.OrderedDict([(alg, [0.]) for alg in alg_names])
  best_runs = collections.OrderedDict([(alg, []) for alg in alg_names])
  best_num_leapfrog_steps = collections.OrderedDict(
      [(alg, 0) for alg in alg_names])

  for num_leapfrog_steps in [1, 2, 4, 8, 16, 32]:
    folder = os.path.join(results_dir, 'num_leapfrog_steps={}/{}'.format(
        num_leapfrog_steps, model_name))

    try:
      runs = get_all_results(folder)
    except Exception as err:
      print('Error loading results from {}: {}'.format(folder, err))
      continue

    for alg in alg_names:
      esss = [r['ess'] for r in runs[alg] if 'ess' in r]
      min_esss_this = [get_min_ess(e) / num_leapfrog_steps for e in esss]

      if (len(min_esss_this) and
          np.amax(min_esss_this) > np.amax(best_esss[alg])):
        best_esss[alg] = min_esss_this
        best_runs[alg] = runs[alg]
        best_num_leapfrog_steps[alg] = num_leapfrog_steps

  return best_esss, best_runs, best_num_leapfrog_steps


def print_ess_stats(esss):
  for alg in alg_names:
    min_esss = esss[alg]

    mean = np.mean(min_esss)
    std = np.std(min_esss)/np.sqrt(len(min_esss))

    print('{}: {} +/- {}'.format(alg, mean, std))
    print(min_esss)
    print('')


def print_parameterisation(parameterisation):
  for k, v in parameterisation.items():
    print('{}: {}'.format(k, np.array(v)))


def analyze_dataset(model_dataset_name, results_dir):
  esss, runs, num_lp = get_best_esss_results(model_dataset_name, results_dir)

  analysis_dir = os.path.join(results_dir, 'analysis')
  tf.gfile.MakeDirs(analysis_dir)
  outfile = os.path.join(analysis_dir, model_dataset_name + '_analysis.txt')
  with tf.gfile.Open(outfile, 'w') as f:

    f.write(str(num_lp) + '\n')

    for alg in alg_names:
      min_esss = esss[alg]
      mean = np.mean(min_esss)
      std = np.std(min_esss)/np.sqrt(len(min_esss))
      f.write('{}: {} +/- {}\n'.format(alg, mean, std))
      f.write(str(min_esss) + '\n')

      elbos = [d['elbo'] for d in runs[alg] if 'elbo' in d]
      f.write('elbos: {}\n\n'.format(elbos))

    f.write('d-VIP Parameterization:\n')
    try:
      for k, v in runs['d-VIP-HMC'][0]['parameterisation'].items():
        f.write('{}: {}\n'.format(k, np.array(v)))
    except Exception as err:
      f.write(str(err) + '\n')

    f.write('\nc-VIP Parameterization:\n')
    try:
      for k, v in runs['c-VIP-HMC'][0]['parameterisation'].items():
        f.write('{}: {}\n'.format(k, np.array(v)))
    except Exception as err:
      f.write(str(err) + '\n')

    f.write('\n\neverything else:\n')
    for alg, alg_runs in runs.items():
      for run_i, run_dict in enumerate(alg_runs):
        f.write('{} run {}:\n'.format(alg, run_i))

        if 'samples' in run_dict:
          try:
            run_dict['hmc_posterior_means'] = {}
            run_dict['hmc_posterior_stddevs'] = {}
            for vname, samples in run_dict['samples'].items():
              run_dict['hmc_posterior_means'][vname] = np.mean(samples, axis=0)
              run_dict['hmc_posterior_stddevs'][vname] = np.std(samples, axis=0)
            del run_dict['samples']
            del run_dict['is_accepted']

            del run_dict['log_accept_ratio']
          except Exception as err:
            f.write(str(err) + '\n')

        for k, v in run_dict.items():
          f.write('  {}: {}\n'.format(k, v))
        f.write('\n\n')


def main(_):
  analyze_dataset(FLAGS.model_and_dataset, FLAGS.results_dir)

if __name__ == '__main__':
  tf.app.run()
